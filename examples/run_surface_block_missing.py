# examples/run_surface_block_missing.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import math
import json
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.models import SpectralCAR_FullVI

# IMPORTANT: importing examples.benchmarks triggers filter registrations
import examples.benchmarks  # noqa: F401
from examples.benchmarks.registry import get_filter_spec, available_filters


# -----------------------------
# Truth surfaces on [0,1]^2
# -----------------------------
def f1(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return (
        10.0
        + 15.0 * torch.log1p(h1) / (1.0 + h2**2)
        + 11.0 * torch.sin(2.0 * math.pi * h1) * torch.cos(2.0 * math.pi * h2)
        + 5.0 * h1**2 * (1.0 - h2**2)
    )


def f2(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return (
        2.0
        + 20.0 * h1**2
        + 12.5 * torch.exp(-2.0 * h1) * (h2 - 2.0) ** 2
        - 0.25 / (h2 + 1.0)
    )


def make_grid(m: int, device: torch.device, dtype=torch.double):
    """
    m x m grid of pixel centers on [0,1]^2.

    Returns:
      H  : [n,2] coordinates
      h1 : [n]
      h2 : [n]
    """
    coords_1d = (torch.arange(m, device=device, dtype=dtype) + 0.5) / m
    h1g, h2g = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    H = torch.stack([h1g.reshape(-1), h2g.reshape(-1)], dim=1)  # [n,2]
    return H, h1g.reshape(-1), h2g.reshape(-1)


# -----------------------------
# Covariates / design matrix
# -----------------------------
def build_design_matrix(
    h1: torch.Tensor,
    h2: torch.Tensor,
    kind: str,
    *,
    z_dim: int = 0,
    z_scale: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Build X deterministically from (h1,h2) plus optional random covariates z.

    kinds:
      - intercept: [1]
      - linear:    [1, h1, h2]
      - poly2:     [1, h1, h2, h1*h2, h1^2, h2^2]
      - fourier:   [1, sin(2πh1), cos(2πh1), sin(2πh2), cos(2πh2), sin(2π(h1+h2)), cos(2π(h1+h2))]
      - linear+z:  linear + z_dim random columns
      - poly2+z:   poly2 + z_dim random columns

    Notes:
      - Random z covariates are fixed once generated (seeded) and treated as observed covariates.
      - If you pass z_dim>0 with kind="linear" (not "linear+z"), we still append z (for convenience).
    """
    n = h1.numel()
    device, dtype = h1.device, h1.dtype

    if kind == "intercept":
        base = torch.ones((n, 1), dtype=dtype, device=device)

    elif kind == "linear":
        base = torch.stack([torch.ones_like(h1), h1, h2], dim=1)

    elif kind == "poly2":
        base = torch.stack([torch.ones_like(h1), h1, h2, h1 * h2, h1**2, h2**2], dim=1)

    elif kind == "fourier":
        base = torch.stack(
            [
                torch.ones_like(h1),
                torch.sin(2.0 * math.pi * h1),
                torch.cos(2.0 * math.pi * h1),
                torch.sin(2.0 * math.pi * h2),
                torch.cos(2.0 * math.pi * h2),
                torch.sin(2.0 * math.pi * (h1 + h2)),
                torch.cos(2.0 * math.pi * (h1 + h2)),
            ],
            dim=1,
        )

    elif kind in ("linear+z", "poly2+z"):
        if kind.startswith("linear"):
            base = torch.stack([torch.ones_like(h1), h1, h2], dim=1)
        else:
            base = torch.stack([torch.ones_like(h1), h1, h2, h1 * h2, h1**2, h2**2], dim=1)

    else:
        raise ValueError(f"Unknown xkind: {kind}")

    want_z = kind.endswith("+z") or (z_dim and z_dim > 0)
    if want_z:
        if z_dim <= 0:
            raise ValueError("z_dim must be > 0 when using '+z' xkind or when z_dim is provided.")
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        z = torch.randn((n, z_dim), generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
        z = z_scale * z
        X = torch.cat([base, z], dim=1)
        return X

    return base


def parse_beta_true(beta_str: str | None, p: int, device: torch.device, dtype=torch.double) -> torch.Tensor:
    """
    beta_str format examples:
      - "1,-0.5,0.2"
      - None -> default pattern
    """
    if beta_str is None:
        # default: mild coefficients, decaying
        vals = []
        for j in range(p):
            if j == 0:
                vals.append(0.0)  # intercept handled mostly by phi surface; feel free to change
            else:
                vals.append(((-1.0) ** j) * (0.8 / (j + 1)))
        return torch.tensor(vals, dtype=dtype, device=device)

    parts = [s.strip() for s in beta_str.split(",") if s.strip() != ""]
    vals = [float(x) for x in parts]
    if len(vals) != p:
        raise ValueError(f"--beta_true has length {len(vals)} but X has p={p}. Provide exactly p values.")
    return torch.tensor(vals, dtype=dtype, device=device)


# -----------------------------
# Missing block
# -----------------------------
def make_missing_block_mask(m: int, block: int, where: str, device: torch.device):
    """
    Returns:
      obs_mask  : [n] bool
      miss_mask : [n] bool
    """
    if block <= 0 or block >= m:
        raise ValueError(f"block must be in [1, m-1], got {block} with m={m}")

    miss = torch.zeros(m, m, dtype=torch.bool, device=device)

    if where == "center":
        r0 = m // 2 - block // 2
        c0 = m // 2 - block // 2
    elif where == "topleft":
        r0, c0 = 0, 0
    elif where == "topright":
        r0, c0 = 0, m - block
    elif where == "bottomleft":
        r0, c0 = m - block, 0
    elif where == "bottomright":
        r0, c0 = m - block, m - block
    else:
        raise ValueError("where must be one of: center, topleft, topright, bottomleft, bottomright")

    miss[r0 : r0 + block, c0 : c0 + block] = True
    miss_mask = miss.reshape(-1)
    obs_mask = ~miss_mask
    return obs_mask, miss_mask


# -----------------------------
# Metrics + plotting
# -----------------------------
def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.norm() * b0.norm()).clamp_min(1e-12)
    return float((a0 @ b0 / denom).item())


def _finite_minmax(x: torch.Tensor) -> tuple[float, float]:
    """
    Torch version-agnostic replacement for nanmin/nanmax.
    """
    xf = x[torch.isfinite(x)]
    return float(xf.min().item()), float(xf.max().item())


def plot_grid(values: np.ndarray, m: int, title: str, path: Path, vmin=None, vmax=None):
    img = values.reshape(m, m)
    plt.figure(figsize=(5.6, 4.6))
    plt.imshow(img, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# -----------------------------
# Registry-powered filter builder
# -----------------------------
def _resolve_fixed_tokens(fixed: dict, *, eps_car: float) -> dict:
    """
    Your cases sometimes store tokens like {"eps_car": "eps_car"}.
    Resolve those to actual numeric values.
    """
    out = {}
    for k, v in fixed.items():
        if isinstance(v, str) and v == "eps_car":
            out[k] = float(eps_car)
        else:
            out[k] = v
    return out


def build_filter_from_registry(
    *,
    filter_name: str,
    case_id: str,
    device: torch.device,
    eps_car: float,
    tau2_true: float,
    lam_max: float,
) -> torch.nn.Module:
    """
    Build filter module via FilterSpec/CaseSpec registry.
    """
    spec = get_filter_spec(filter_name)
    if case_id not in spec.cases:
        raise ValueError(
            f"Unknown case '{case_id}' for filter '{filter_name}'. "
            f"Available: {list(spec.cases.keys())}"
        )

    case = spec.cases[case_id]
    fixed_resolved = _resolve_fixed_tokens(case.fixed, eps_car=eps_car)

    kwargs = dict(
        device=device,
        tau2_true=float(tau2_true),
        eps_car=float(eps_car),
        lam_max=float(lam_max),
        **fixed_resolved,
    )

    try:
        filt = case.build_filter(**kwargs)
    except TypeError:
        # fallback: only pass device + fixed
        kwargs2 = dict(fixed_resolved)
        kwargs2["device"] = device
        filt = case.build_filter(**kwargs2)

    return filt.to(device)


# -----------------------------
# Conjugate Gradient + conditional imputation
# -----------------------------
def cg_solve(mv, b, x0=None, tol=1e-6, max_iter=800):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - mv(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b)).clamp_min(1e-12)

    for _ in range(int(max_iter)):
        Ap = mv(p)
        denom = torch.dot(p, Ap).clamp_min(1e-18)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)

        # relative criterion
        if torch.sqrt(rsnew) / bnorm < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

@torch.no_grad()
def conditional_impute_y_plugin(
    *,
    model: SpectralCAR_FullVI,
    y_current: torch.Tensor,     # [n] with current imputation
    obs_mask: torch.Tensor,      # bool [n]
    miss_mask: torch.Tensor,     # bool [n]
    cg_tol: float = 1e-6,
    cg_max_iter: int = 800,
):
    """
    Conditional mean imputation under plugin hyperparameters:

        y_m | y_o ~ Normal( mu_m + Sigma_mo Sigma_oo^{-1} (y_o - mu_o), ... )

    where mu = X m_beta(plugin), Sigma = U diag(F+sigma2) U^T.

    Returns:
        y_new : [n] (copy of y_current with missing replaced by conditional mean)
        info  : dict diagnostics
    """
    device = y_current.device
    dtype = y_current.dtype
    n = y_current.numel()

    obs_idx = torch.where(obs_mask)[0]
    miss_idx = torch.where(miss_mask)[0]

    # ---- plugin hyperparams ----
    theta_mean = model.filter.mean_unconstrained()
    sigma2 = torch.exp(model.mu_log_sigma2).clamp_min(1e-12)         # scalar
    F_lam = model.filter.spectrum(model.lam, theta_mean)             # [n]
    var = (F_lam + sigma2).clamp_min(1e-12)                          # [n]

    # ---- Sigma matvec: v -> U (var * (U^T v)) ----
    def Sigma_mv(v_full: torch.Tensor) -> torch.Tensor:
        return model.U @ (var * (model.U.T @ v_full))

    # ---- observed sub-operator Sigma_oo matvec ----
    def Sigma_oo_mv(x_obs: torch.Tensor) -> torch.Tensor:
        v = torch.zeros(n, dtype=dtype, device=device)
        v[obs_idx] = x_obs
        Sv = Sigma_mv(v)
        return Sv[obs_idx]

    # ---- observed-only plugin beta given y_o ----
    Xo = model.X[obs_idx]                 # [n_o, p]
    yo = y_current[obs_idx]               # [n_o]
    p = Xo.shape[1]

    # helper: apply Sigma_oo^{-1} to a vector via CG
    def solve_Sigma_oo(v_obs: torch.Tensor) -> torch.Tensor:
        return cg_solve(Sigma_oo_mv, v_obs.contiguous(), tol=cg_tol, max_iter=cg_max_iter)

    # compute Sigma_oo^{-1} y_o
    invSig_yo = solve_Sigma_oo(yo)

    # compute Sigma_oo^{-1} Xo columnwise
    invSig_Xo_cols = [solve_Sigma_oo(Xo[:, j]) for j in range(p)]
    invSig_Xo = torch.stack(invSig_Xo_cols, dim=1)   # [n_o, p]

    # assemble X_o^T Sigma_oo^{-1} X_o and X_o^T Sigma_oo^{-1} y_o
    Xt_invSig_X = Xo.T @ invSig_Xo                   # [p, p]
    Xt_invSig_y = Xo.T @ invSig_yo                   # [p]

    eps = 1e-6
    V_beta_inv = model.V0_inv + Xt_invSig_X + eps * torch.eye(p, dtype=dtype, device=device)
    L = torch.linalg.cholesky(V_beta_inv)
    rhs = model.V0_inv @ model.m0 + Xt_invSig_y
    m_beta = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)

    mu = model.X @ m_beta                             # [n]

    # ---- conditional mean for missing ----
    b = (y_current[obs_idx] - mu[obs_idx]).contiguous()
    alpha = solve_Sigma_oo(b)  # alpha = Sigma_oo^{-1}(y_o - mu_o)

    # Sigma[:,obs] alpha
    v = torch.zeros(n, dtype=dtype, device=device)
    v[obs_idx] = alpha
    Sa = Sigma_mv(v)

    mu_cond_m = mu[miss_idx] + Sa[miss_idx]

    y_new = y_current.clone()
    y_new[miss_idx] = mu_cond_m

    info = {"sigma2_plugin": float(sigma2.item())}
    return y_new, info

# -----------------------------
# Fit once
# -----------------------------
def fit_model_once(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    filter_module,
    sigma2_init: float,
    vi_iters: int,
    lr: float,
    num_mc: int,
    device: torch.device,
    log_every: int = 100,
):
    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        mu_log_sigma2=math.log(max(sigma2_init, 1e-12)),
        log_std_log_sigma2=-2.3,
        num_mc=num_mc,
        sigma2_prior="logsigma2_normal",
        sigma2_prior_params={"mu": 0.0, "std": 2.0},
        kl_sigma_mc=8,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(1, vi_iters + 1):
        opt.zero_grad()
        elbo, stats = model.elbo()
        (-elbo).backward()
        opt.step()

        if log_every > 0 and (t % log_every == 0 or t == 1 or t == vi_iters):
            rho_str = ""
            tau2_str = ""

            # Preferred: filters that implement _constrain + mean_unconstrained (like LerouxCARFilterFullVI)
            if hasattr(model.filter, "_constrain") and hasattr(model.filter, "mean_unconstrained"):
                try:
                    c = model.filter._constrain(model.filter.mean_unconstrained())
                    if "tau2" in c:
                        tau2_str = f" tau2={float(c['tau2'].detach().cpu().item()):.4f}"
                    if "rho" in c:
                        rho_str = f" rho={float(c['rho'].detach().cpu().item()):.6f}"
                except Exception:
                    pass

            # Fallback if you later add mean_params_dict()
            if rho_str == "" and hasattr(model.filter, "mean_params_dict"):
                try:
                    pd = model.filter.mean_params_dict()
                    if "rho" in pd:
                        rho_str = f" rho={float(pd['rho'].detach().cpu().item()):.6f}"
                except Exception:
                    pass

            print(
                f"[VI {t:04d}] ELBO={elbo.item():.2f} "
                f"kl_filt={stats['kl_filter'].item():.2f} "
                f"kl_sig={stats['kl_sigma2'].item():.2f} "
                f"sigma2_last={stats['sigma2_last'].item():.4f}"
                + tau2_str
                + rho_str
            )

    return model


def affine_calibration_rmse(yhat: torch.Tensor, ytrue: torch.Tensor) -> tuple[float, float, float]:
    """
    Fit ytrue ≈ a + b*yhat in least squares, return (a,b,rmse_cal).
    """
    x = yhat
    y = ytrue
    x0 = x - x.mean()
    b = (x0 @ (y - y.mean())) / (x0 @ x0).clamp_min(1e-12)
    a = y.mean() - b * x.mean()
    ycal = a + b * x
    rmse_cal = float(torch.sqrt(torch.mean((ycal - y) ** 2)).item())
    return float(a.item()), float(b.item()), rmse_cal


def main():
    ap = argparse.ArgumentParser()

    # grid + graph
    ap.add_argument("--m", type=int, default=50, help="grid size m (m x m)")
    ap.add_argument("--knn", type=int, default=10, help="k for kNN graph")
    ap.add_argument("--seed", type=int, default=0)

    # latent surface / signal
    ap.add_argument("--surface", type=str, default="f1", choices=["f1", "f2"])
    ap.add_argument("--phi_scale", type=float, default=1.0, help="scale multiplier on phi_true")

    # covariates + regression
    ap.add_argument(
        "--xkind",
        type=str,
        default="intercept",
        choices=["intercept", "linear", "poly2", "fourier", "linear+z", "poly2+z"],
    )
    ap.add_argument("--z_dim", type=int, default=0, help="number of random covariates for '+z' kinds")
    ap.add_argument("--z_scale", type=float, default=1.0, help="std scale for random covariates z")
    ap.add_argument("--beta_true", type=str, default=None, help="comma-separated beta values, length must match p")

    # noise on observations y
    ap.add_argument("--sigma", type=float, default=1.0, help="noise std for observation y")

    # missing block
    ap.add_argument("--block", type=int, default=10, help="missing block size (block x block)")
    ap.add_argument(
        "--where",
        type=str,
        default="center",
        choices=["center", "topleft", "topright", "bottomleft", "bottomright"],
    )

    # filter/case (registry)
    ap.add_argument("--filter", type=str, default="matern", help=f"Filter family. Available: {available_filters()}")
    ap.add_argument("--case", type=str, default="baseline", help="Case id for chosen filter.")
    ap.add_argument("--eps_car", type=float, default=1e-3, help="eps_car token value used by Classic CAR cases")
    ap.add_argument("--tau2_true", type=float, default=1.0, help="initializer used by some build_filter cases")

    # VI / imputation loop
    ap.add_argument("--outer", type=int, default=3, help="outer imputation iterations")
    ap.add_argument("--vi_iters", type=int, default=900, help="VI iterations per outer step")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--num_mc", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=100)

    # phi extraction (used for diagnostics y_pred)
    ap.add_argument("--phi_mode", type=str, default="plugin", choices=["plugin", "mc"])
    ap.add_argument("--phi_mc", type=int, default=64)

    # imputation init
    ap.add_argument(
        "--init_missing",
        type=str,
        default="mean_obs",
        choices=["mean_obs", "zero", "surface_mean"],
        help="initial fill for missing y entries before VI",
    )

    # imputation strategy
    ap.add_argument(
        "--impute",
        type=str,
        default="conditional",
        choices=["plugin_phi", "conditional"],
        help="plugin_phi: y_m <- (X m_beta + E[phi])_m; conditional: y_m <- E[y_m | y_o] (CG)",
    )
    ap.add_argument("--cg_tol", type=float, default=1e-6)
    ap.add_argument("--cg_max_iter", type=int, default=800)

    # output
    ap.add_argument("--outdir", type=str, default="outputs/surface_block_missing")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Coordinates, X, latent phi, observed y
    # -----------------------------
    H, h1, h2 = make_grid(args.m, device=device, dtype=dtype)
    n = H.shape[0]

    # latent signal (deterministic “truth” surface)
    phi_base = f1(h1, h2) if args.surface == "f1" else f2(h1, h2)
    phi_true = args.phi_scale * phi_base

    # covariates
    X = build_design_matrix(
        h1,
        h2,
        args.xkind,
        z_dim=args.z_dim,
        z_scale=args.z_scale,
        seed=args.seed + 999,  # decouple from noise seed but deterministic
    )
    p = X.shape[1]
    beta_true = parse_beta_true(args.beta_true, p, device=device, dtype=dtype)

    # observations
    eps = args.sigma * torch.randn(n, dtype=dtype, device=device)
    y_true = (X @ beta_true) + phi_true + eps

    # masks
    obs_mask, miss_mask = make_missing_block_mask(args.m, args.block, args.where, device=device)

    # for plotting “observed with hole”
    y_obs_plot = y_true.detach().clone()
    y_obs_plot[miss_mask] = float("nan")

    # -----------------------------
    # 2) Graph Laplacian + eigendecomp on ALL nodes
    # -----------------------------
    L_out = build_laplacian_from_knn(H, k=args.knn)
    L = L_out[0] if isinstance(L_out, (tuple, list)) else L_out

    lam, U = laplacian_eigendecomp(L)
    idx = torch.argsort(lam)
    lam, U = lam[idx], U[:, idx]

    # -----------------------------
    # 3) Initialize imputation y_imp
    # -----------------------------
    y_imp = y_true.detach().clone()

    if args.init_missing == "mean_obs":
        y_imp[miss_mask] = y_true[obs_mask].mean()
    elif args.init_missing == "zero":
        y_imp[miss_mask] = 0.0
    elif args.init_missing == "surface_mean":
        mean_signal_obs = ((X @ beta_true) + phi_true)[obs_mask].mean()
        y_imp[miss_mask] = mean_signal_obs
    else:
        raise ValueError("Unknown init_missing")

    # shared color range based on observed (finite) values
    vmin, vmax = _finite_minmax(y_obs_plot)

    # -----------------------------
    # 4) Save reference plots / metadata
    # -----------------------------
    plot_grid(
        phi_true.detach().cpu().numpy(),
        args.m,
        f"Latent surface phi_true ({args.surface})",
        outdir / f"phi_true_{args.surface}_m{args.m}.png",
        vmin=vmin,
        vmax=vmax,
    )
    plot_grid(
        (X @ beta_true).detach().cpu().numpy(),
        args.m,
        f"Covariate signal X beta_true ({args.xkind})",
        outdir / f"xbeta_true_{args.surface}_{args.xkind}_m{args.m}.png",
        vmin=None,
        vmax=None,
    )
    plot_grid(
        y_obs_plot.detach().cpu().numpy(),
        args.m,
        f"Observed y with missing block ({args.where}, {args.block}x{args.block})",
        outdir / f"y_obs_hole_{args.surface}_{args.xkind}_m{args.m}_k{args.knn}.png",
        vmin=vmin,
        vmax=vmax,
    )

    meta = dict(
        surface=args.surface,
        m=args.m,
        n=n,
        knn=args.knn,
        seed=args.seed,
        sigma=args.sigma,
        phi_scale=args.phi_scale,
        xkind=args.xkind,
        z_dim=args.z_dim,
        z_scale=args.z_scale,
        beta_true=beta_true.detach().cpu().tolist(),
        block=args.block,
        where=args.where,
        filter=args.filter,
        case=args.case,
        eps_car=args.eps_car,
        tau2_true=args.tau2_true,
        outer=args.outer,
        vi_iters=args.vi_iters,
        lr=args.lr,
        num_mc=args.num_mc,
        phi_mode=args.phi_mode,
        phi_mc=args.phi_mc,
        init_missing=args.init_missing,
        impute=args.impute,
        cg_tol=args.cg_tol,
        cg_max_iter=args.cg_max_iter,
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -----------------------------
    # 5) Outer loop: fit -> impute missing -> replace
    # -----------------------------
    for it in range(1, args.outer + 1):
        print("\n==============================")
        print(f"OUTER ITER {it}/{args.outer}")
        print(f"surface={args.surface} m={args.m} knn={args.knn} block={args.block}@{args.where}")
        print(f"xkind={args.xkind} p={p}  sigma={args.sigma}")
        print(f"filter={args.filter} case={args.case}")
        print(f"impute={args.impute}")
        print("==============================")

        filter_module = build_filter_from_registry(
            filter_name=args.filter,
            case_id=args.case,
            device=device,
            eps_car=args.eps_car,
            tau2_true=args.tau2_true,
            lam_max=float(lam.max().detach().cpu()),
        )

        model = fit_model_once(
            X=X,
            y=y_imp,
            lam=lam,
            U=U,
            filter_module=filter_module,
            sigma2_init=args.sigma**2,
            vi_iters=args.vi_iters,
            lr=args.lr,
            num_mc=args.num_mc,
            device=device,
            log_every=args.log_every,
        )

        # ---- Diagnostics prediction (useful even if you impute conditionally) ----
        if args.phi_mode == "plugin" and hasattr(model, "beta_posterior_plugin"):
            m_beta_plug, _, _, _ = model.beta_posterior_plugin()
            mean_phi, _ = model.posterior_phi(mode="plugin", num_mc=args.phi_mc)
            y_pred = (X @ m_beta_plug) + mean_phi
        else:
            mean_phi, _ = model.posterior_phi(mode=args.phi_mode, num_mc=args.phi_mc)
            y_pred = (X @ model.m_beta) + mean_phi

        a, b, rmse_cal = affine_calibration_rmse(y_pred[miss_mask], y_true[miss_mask])
        print(f"Affine calib on missing (y_pred): a={a:.3f} b={b:.3f} RMSE_cal={rmse_cal:.4f}")

        # ---- Impute ----
        if args.impute == "plugin_phi":
            # old behavior: overwrite missing with current posterior mean prediction
            y_imp[miss_mask] = y_pred[miss_mask].detach()

        elif args.impute == "conditional":
            # proper conditional Gaussian imputation under plugin hyperparams
            y_imp, info = conditional_impute_y_plugin(
                model=model,
                y_current=y_imp,
                obs_mask=obs_mask,
                miss_mask=miss_mask,
                cg_tol=float(args.cg_tol),
                cg_max_iter=int(args.cg_max_iter),
            )
            print(f"Conditional impute used sigma2_plugin={info['sigma2_plugin']:.6f}")

        else:
            raise ValueError("Unknown --impute choice.")

        # plot updated imputation
        plot_grid(
            y_imp.detach().cpu().numpy(),
            args.m,
            f"Imputed y after iter {it} ({args.filter}/{args.case}, {args.impute})",
            outdir / f"y_imputed_iter{it}_{args.filter}_{args.case}_{args.surface}_{args.xkind}_m{args.m}_k{args.knn}.png",
            vmin=vmin,
            vmax=vmax,
        )

        # quick metrics per-iter
        rmse_miss_it = rmse(y_imp[miss_mask], y_true[miss_mask])
        corr_miss_it = corr(y_imp[miss_mask], y_true[miss_mask])
        print(f"[ITER {it}] RMSE(miss)={rmse_miss_it:.4f}  Corr(miss)={corr_miss_it:.4f}")

    # -----------------------------
    # 6) Final evaluation
    # -----------------------------
    y_final = y_imp.detach()
    rmse_miss = rmse(y_final[miss_mask], y_true[miss_mask])
    corr_miss = corr(y_final[miss_mask], y_true[miss_mask])

    rmse_all = rmse(y_final, y_true)
    corr_all = corr(y_final, y_true)

    print("\n==============================")
    print("FINAL METRICS")
    print("==============================")
    print(f"RMSE missing-block: {rmse_miss:.4f}   Corr missing-block: {corr_miss:.4f}")
    print(f"RMSE full-grid    : {rmse_all:.4f}   Corr full-grid    : {corr_all:.4f}")

    summary = dict(
        **meta,
        rmse_missing=rmse_miss,
        corr_missing=corr_miss,
        rmse_all=rmse_all,
        corr_all=corr_all,
    )

    (outdir / f"summary_{args.filter}_{args.case}_{args.surface}_{args.xkind}_m{args.m}_k{args.knn}_blk{args.block}_{args.where}.json").write_text(
        json.dumps(summary, indent=2)
    )
    np.save(
        outdir / f"summary_{args.filter}_{args.case}_{args.surface}_{args.xkind}_m{args.m}_k{args.knn}_blk{args.block}_{args.where}.npy",
        summary,
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()