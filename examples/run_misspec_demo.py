# examples/run_misspec_demo.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import math
import random
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional, List, Dict

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.models import SpectralCAR_FullVI
from sdmcar.mcmc import MCMCConfig, make_collapsed_mcmc_from_model
from sdmcar import diagnostics

# ensure registry is populated
from examples.benchmarks.registry import get_filter_spec, available_filters
import examples.benchmarks  # noqa: F401


torch.set_default_dtype(torch.double)


# -------------------------
# Fit specs / results
# -------------------------
@dataclass(frozen=True)
class FitSpec:
    filter: str
    case: str

    @property
    def key(self) -> str:
        return f"{self.filter}:{self.case}"


@dataclass
class FitResult:
    spec: FitSpec
    case_display: str              # e.g. case_spec.display_name
    label: str                     # e.g. "poly/deg5"
    F_vi: torch.Tensor             # [n] on CPU
    F_mcmc: Optional[torch.Tensor] # [n] on CPU or None if skipped
    rmse_phi_vi: float
    rmse_phi_mcmc: Optional[float]
    rmse_logF_vi: float
    rmse_logF_mcmc: Optional[float]


# -------------------------
# Truth spectra (misspec)
# -------------------------
def truth_spectrum(lam: torch.Tensor, kind: str, *, eps_car: float) -> torch.Tensor:
    """
    Return F_true(lam) [n] for misspecified truth.

    Recommended kinds:
      - mix2 : CAR-like + diffusion tail (two-scale)
      - floor: CAR-like + flat floor
      - bump : CAR-like with mid-frequency bump
    """
    lam = lam.clamp_min(0.0)
    kind = kind.lower()

    if kind == "mix2":
        tau1 = 0.20
        tau2 = 0.25
        a = 8.0
        F = (tau1 / (lam + eps_car)) + tau2 * torch.exp(-a * lam)
        return F.clamp_min(1e-12)

    if kind == "floor":
        tau = 0.20
        c = 0.03
        F = (tau / (lam + eps_car)) + c
        return F.clamp_min(1e-12)

    if kind == "bump":
        tau = 0.18
        mu = 0.35 * float(lam.max().detach().cpu())
        sig = 0.12 * float(lam.max().detach().cpu())
        b = 0.9
        bump = 1.0 + b * torch.exp(-0.5 * ((lam - mu) / max(sig, 1e-12)) ** 2)
        F = (tau / (lam + eps_car)) * bump
        return F.clamp_min(1e-12)
    
    if kind == "bandpass":
        mu = 0.35 * float(lam.max().detach().cpu())
        sig = 0.08 * float(lam.max().detach().cpu())
        A = 0.25
        F = A * torch.exp(-0.5 * ((lam - mu) / max(sig, 1e-12)) ** 2)
        return F.clamp_min(1e-12)

    raise ValueError(f"Unknown truth kind '{kind}'. Choose from: mix2, floor, bump.")


# -------------------------
# Spectrum diagnostics
# -------------------------
# @torch.no_grad()
# def spectrum_vi_mean(filter_module, lam: torch.Tensor) -> torch.Tensor:
#     theta_mean = filter_module.mean_unconstrained()
#     return filter_module.spectrum(lam, theta_mean)

# -------------------------
# Spectrum diagnostics
# -------------------------
@torch.no_grad()
def spectrum_vi_mc_mean(filter_module, lam: torch.Tensor, *, S: int = 256) -> torch.Tensor:
    """
    Monte Carlo estimate of E_q[F(lam;theta)] under the VI posterior q(theta).
    This is the right quantity for nonlinear filters (mixtures, splines, etc.).
    """
    acc = torch.zeros_like(lam)
    for _ in range(S):
        th = filter_module.sample_unconstrained()
        acc += filter_module.spectrum(lam, th)
    return acc / float(S)


@torch.no_grad()
def spectrum_mcmc_mean(
    filter_module,
    lam: torch.Tensor,
    theta_chain: torch.Tensor,
    *,
    batch: int = 256,
) -> torch.Tensor:
    """
    theta_chain: [S, d_theta] packed
    returns: mean over draws of F(lam; theta_s)
    """
    device = next(filter_module.parameters()).device if any(True for _ in filter_module.parameters()) else lam.device
    dtype = lam.dtype
    lam = lam.to(device=device, dtype=dtype)

    S = theta_chain.shape[0]
    acc = torch.zeros_like(lam)
    count = 0

    for i in range(0, S, batch):
        chunk = theta_chain[i:i+batch].to(device=device, dtype=dtype)
        for j in range(chunk.shape[0]):
            theta = filter_module.unpack(chunk[j])
            acc += filter_module.spectrum(lam, theta)
            count += 1

    return acc / max(count, 1)


def plot_spectrum_curves(lam_np: np.ndarray, curves: Dict[str, np.ndarray], save_path: Path, *, ylog: bool, title: str):
    plt.figure(figsize=(6.2, 4.2))
    for name, y in curves.items():
        plt.plot(lam_np, y, linewidth=2.0 if name == "truth" else 1.6, label=name)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F(\lambda)$")
    plt.title(title)
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_spectrum_overlay(*, lam: torch.Tensor, F_true: torch.Tensor, results: List[FitResult], save_dir: Path, truth_name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    lam_np = lam.detach().cpu().numpy()

    curves_linear: Dict[str, np.ndarray] = {"truth": F_true.detach().cpu().numpy()}
    for r in results:
        curves_linear[f"{r.label} (VI)"] = r.F_vi.detach().cpu().numpy()
        if r.F_mcmc is not None:
            curves_linear[f"{r.label} (MCMC)"] = r.F_mcmc.detach().cpu().numpy()

    plot_spectrum_curves(
        lam_np,
        curves_linear,
        save_dir / "spectrum_overlay_linear.png",
        ylog=False,
        title=f"Spectrum overlay (truth={truth_name})",
    )
    plot_spectrum_curves(
        lam_np,
        curves_linear,
        save_dir / "spectrum_overlay_log.png",
        ylog=True,
        title=f"Spectrum overlay (truth={truth_name})",
    )


def rmse_log_spectrum(F_hat: torch.Tensor, F_true: torch.Tensor) -> float:
    eps = 1e-12
    a = torch.log(F_hat.clamp_min(eps))
    b = torch.log(F_true.clamp_min(eps))
    return float(torch.sqrt(torch.mean((a - b) ** 2)).detach().cpu().item())


# -------------------------
# Run one fit
# -------------------------
def run_fit(
    *,
    spec_name: str,
    case_id: str,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    coords: torch.Tensor,
    phi_true: torch.Tensor,
    beta_true: torch.Tensor,
    F_true: torch.Tensor,
    sigma2_true: float,
    tau2_init: float,
    eps_car: float,
    prior_V0: torch.Tensor,
    device: torch.device,
    outdir: Path,
    vi_iters: int,
    vi_mc: int,
    vi_lr: float,
    mcmc_steps: int,
    mcmc_burnin: int,
    mcmc_thin: int,
    compare_only: bool,
    skip_mcmc: bool,
) -> Optional[FitResult]:
    spec = get_filter_spec(spec_name)
    if case_id not in spec.cases:
        print(f"  - skipping {spec_name}/{case_id} (case not defined)")
        return None

    case_spec = spec.cases[case_id]
    label = f"{spec_name}/{case_spec.display_name}"
    case_dir = outdir / spec_name / case_spec.display_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"FIT: {spec_name} | CASE: {case_spec.display_name}")
    print("=" * 80)

    # Build filter (VI)
    filter_module = case_spec.build_filter(
        tau2_true=tau2_init,
        eps_car=eps_car,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **case_spec.fixed,
    )

    # VI model
    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_true),
        log_std_log_sigma2=-2.3,
        num_mc=vi_mc,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    # Train VI
    for _ in range(vi_iters):
        opt.zero_grad()
        elbo, _ = model.elbo()
        (-elbo).backward()
        opt.step()

    # VI phi + spectrum
    with torch.no_grad():
        phi_vi, _ = model.posterior_phi(mode="mc", num_mc=64)
        rmse_phi_vi = float(torch.sqrt(torch.mean((phi_vi.detach().cpu() - phi_true.cpu()) ** 2)).item())

        # F_vi = spectrum_vi_mean(model.filter, lam).detach()
        # rmse_logF_vi = rmse_log_spectrum(F_vi, F_true)
        F_vi = spectrum_vi_mc_mean(model.filter, lam, S=256).detach()
        rmse_logF_vi = rmse_log_spectrum(F_vi, F_true)

    # define "mcmc outputs" so VI-only works cleanly
    phi_mcmc = None
    rmse_phi_mcmc = None
    F_mcmc = None
    rmse_logF_mcmc = None
    acc_s_mid = None
    acc_theta_mid = None

    F_vi_cpu = F_vi.detach().cpu()

    # MCMC becomes optional
    if (not skip_mcmc) and (mcmc_steps > 0):
        cfg = MCMCConfig(
            num_steps=mcmc_steps,
            burnin=mcmc_burnin,
            thin=mcmc_thin,
            step_s=float(case_spec.step_s),
            step_theta=case_spec.get_step_theta(model.filter),
            seed=0,
            device=device,
        )
        sampler = make_collapsed_mcmc_from_model(model, config=cfg)

        init_s = model.mu_log_sigma2.detach()
        theta0 = model.filter.mean_unconstrained()
        init_theta_vec = model.filter.pack(theta0).detach()

        out = sampler.run(
            init_s=init_s,
            init_theta_vec=init_theta_vec,
            init_from_conditional_beta=True,
            store_phi_mean=True,
            U=U,
            X=X,
            y=y,
        )

        phi_mean_chain = out["phi_mean"]  # [S,n]
        phi_mcmc = phi_mean_chain.mean(dim=0)
        rmse_phi_mcmc = float(torch.sqrt(torch.mean((phi_mcmc - phi_true.cpu()) ** 2)).item())

        theta_chain = out["theta"]  # [S,d]
        F_mcmc = spectrum_mcmc_mean(model.filter, lam, theta_chain).detach()
        rmse_logF_mcmc = rmse_log_spectrum(F_mcmc, F_true)

        acc_s_mid = float(out["acc"]["s"][2])
        acc_theta_mid = {k: float(v[2]) for k, v in out["acc"]["theta"].items()}

        F_mcmc_cpu = F_mcmc.detach().cpu()
    else:
        F_mcmc_cpu = None

    # Save per-fit plots only if not compare_only
    if not compare_only:
        lam_np = lam.detach().cpu().numpy()
        curves = {
            "truth": F_true.detach().cpu().numpy(),
            f"{label} (VI)M": F_vi_cpu.numpy(),
        }
        if F_mcmc_cpu is not None:
            curves[f"{label} (MCMC)"] = F_mcmc_cpu.numpy()

        plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_linear.png", ylog=False, title="Spectrum comparison")
        plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_log.png", ylog=True, title="Spectrum comparison")

        diagnostics.plot_phi_mean_vs_true(
            coords=coords,
            mean_phi=phi_vi.to(device),
            phi_true=phi_true,
            save_path_prefix=str(case_dir / "phi_vi"),
        )
        if phi_mcmc is not None:
            diagnostics.plot_phi_mean_vs_true(
                coords=coords,
                mean_phi=phi_mcmc.to(device=device, dtype=torch.double),
                phi_true=phi_true,
                save_path_prefix=str(case_dir / "phi_mcmc"),
            )

    # Safe printing when MCMC is skipped
    if rmse_phi_mcmc is None:
        print(f"  phi RMSE  : VI={rmse_phi_vi:.4f} | MCMC=NA")
        print(f"  logF RMSE : VI={rmse_logF_vi:.4f} | MCMC=NA")
    else:
        print(f"  phi RMSE  : VI={rmse_phi_vi:.4f} | MCMC={rmse_phi_mcmc:.4f}")
        print(f"  logF RMSE : VI={rmse_logF_vi:.4f} | MCMC={rmse_logF_mcmc:.4f}")
        if acc_s_mid is not None and acc_theta_mid is not None:
            print(
                f"  acc_s={acc_s_mid:.3f}  acc_theta="
                + ", ".join([f"{k}={v:.3f}" for k, v in acc_theta_mid.items()])
            )

    return FitResult(
        spec=FitSpec(filter=spec_name, case=case_id),
        case_display=case_spec.display_name,
        label=label,
        F_vi=F_vi_cpu,
        F_mcmc=F_mcmc_cpu,
        rmse_phi_vi=rmse_phi_vi,
        rmse_phi_mcmc=rmse_phi_mcmc,
        rmse_logF_vi=rmse_logF_vi,
        rmse_logF_mcmc=rmse_logF_mcmc,
    )


def parse_fit_specs(args) -> List[FitSpec]:
    # Preferred mode
    if args.fits is not None:
        specs: List[FitSpec] = []
        for tok in args.fits:
            if ":" not in tok:
                raise ValueError(f"--fits token must be filter:case, got '{tok}'")
            f, c = tok.split(":", 1)
            f = f.strip()
            c = c.strip()
            if not f or not c:
                raise ValueError(f"Bad --fits token '{tok}' (empty filter or case)")
            spec = get_filter_spec(f)
            if c not in spec.cases:
                raise ValueError(
                    f"Case '{c}' not defined for filter '{f}'. "
                    f"Available: {list(spec.cases.keys())}"
                )
            specs.append(FitSpec(filter=f, case=c))
        return specs

    # Legacy mode: cartesian product
    if args.filters is None:
        raise ValueError("Must provide either --fits or --filters/--cases.")

    specs: List[FitSpec] = []
    for f in args.filters:
        spec = get_filter_spec(f)  # validate filter exists
        for c in args.cases:
            if c in spec.cases:
                specs.append(FitSpec(filter=f, case=c))
            else:
                print(f"  - skipping {f}/{c} (case not defined)")
    if not specs:
        raise ValueError("No valid fits found from --filters/--cases.")
    return specs


def _sort_key(r: FitResult) -> float:
    return r.rmse_logF_mcmc if r.rmse_logF_mcmc is not None else r.rmse_logF_vi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, choices=["mix2", "floor", "bump", "bandpass"])

    # Legacy mode (optional now)
    p.add_argument(
        "--filters",
        nargs="+",
        default=None,
        help=f"Legacy mode: filter families to fit. Available: {available_filters()}",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=["baseline"],
        help="Legacy mode: case IDs to try for each filter (skips missing).",
    )

    p.add_argument("--outdir", default=str(Path("examples") / "figures" / "misspec"))

    p.add_argument("--vi_iters", type=int, default=2000)
    p.add_argument("--vi_mc", type=int, default=10)
    p.add_argument("--vi_lr", type=float, default=1e-2)
    p.add_argument("--mcmc_steps", type=int, default=30000)
    p.add_argument("--mcmc_burnin", type=int, default=10000)
    p.add_argument("--mcmc_thin", type=int, default=10)

    p.add_argument(
        "--fits",
        nargs="+",
        default=None,
        help=(
            "Preferred mode: explicit list of fits as filter:case tokens. "
            "Example: --fits classic_car:baseline leroux:learn_rho poly:deg5"
        ),
    )
    p.add_argument(
        "--compare_only",
        action="store_true",
        help="Skip per-fit plots; still writes COMPARE overlay plots + leaderboard.",
    )
    p.add_argument("--skip_mcmc", action="store_true", help="Skip MCMC for all fits (VI only).")

    args = p.parse_args()

    # Basic arg sanity
    if args.fits is None and args.filters is None:
        raise ValueError("Provide either --fits or --filters (legacy).")

    # Seeds + device
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    outdir = Path(args.outdir) / f"truth_{args.truth}"
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # 1) Grid + Laplacian + eigendecomp
    # --------------------------------------------
    nx, ny = 40, 40
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    n = coords.size(0)

    L, W = build_laplacian_from_knn(coords, k=8, gamma=0.2, rho=0.95)
    lam, U = laplacian_eigendecomp(L)
    lam = lam.to(device)
    U = U.to(device)

    # --------------------------------------------
    # 2) Generate misspecified truth
    # --------------------------------------------
    eps_car = 1e-3
    F_true = truth_spectrum(lam, args.truth, eps_car=eps_car).to(device)

    z_true = torch.sqrt(F_true) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true

    x_coord = coords[:, 0]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device), x_coord], dim=1)
    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)

    sigma2_true = 0.10
    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(n, dtype=torch.double, device=device)

    # Prior on beta
    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(X.shape[1], dtype=torch.double, device=device)

    # print("\nAVAILABLE FILTERS/CASES:")
    # for f in available_filters():
    #     spec = get_filter_spec(f)
    #     print(f"  {f}: {list(spec.cases.keys())}")
    # print()


    # --------------------------------------------
    # 3) Fit requested specs
    # --------------------------------------------
    tau2_init = 0.4  # just an initialization hint
    fit_specs = parse_fit_specs(args)

    results: List[FitResult] = []
    for fs in fit_specs:
        r = run_fit(
            spec_name=fs.filter,
            case_id=fs.case,
            X=X, y=y, lam=lam, U=U, coords=coords,
            phi_true=phi_true, beta_true=beta_true,
            F_true=F_true,
            sigma2_true=sigma2_true,
            tau2_init=tau2_init,
            eps_car=eps_car,
            prior_V0=prior_V0,
            device=device,
            outdir=outdir,
            vi_iters=args.vi_iters,
            vi_mc=args.vi_mc,
            vi_lr=args.vi_lr,
            mcmc_steps=args.mcmc_steps,
            mcmc_burnin=args.mcmc_burnin,
            mcmc_thin=args.mcmc_thin,
            compare_only=args.compare_only,
            skip_mcmc=args.skip_mcmc,
        )
        if r is not None:
            results.append(r)

    # --------------------------------------------
    # 4) COMPARE overlay plots
    # --------------------------------------------
    if results:
        compare_dir = outdir / "COMPARE"
        plot_spectrum_overlay(lam=lam, F_true=F_true, results=results, save_dir=compare_dir, truth_name=args.truth)

    # --------------------------------------------
    # 5) Print leaderboard
    # --------------------------------------------
    if results:
        print("\n" + "=" * 80)
        print(f"MISSPEC SUMMARY (truth={args.truth})")
        print("=" * 80)
        for r in sorted(results, key=_sort_key):
            mcmc_logF = f"{r.rmse_logF_mcmc:.4f}" if r.rmse_logF_mcmc is not None else "NA"
            mcmc_phi = f"{r.rmse_phi_mcmc:.4f}" if r.rmse_phi_mcmc is not None else "NA"
            print(
                f"{r.label:<32} | "
                f"logF_RMSE(VI)={r.rmse_logF_vi:.4f}  logF_RMSE(MCMC)={mcmc_logF} | "
                f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={mcmc_phi}"
            )

        # also write to file for paper artifacts
        compare_dir = outdir / "COMPARE"
        compare_dir.mkdir(parents=True, exist_ok=True)
        with open(compare_dir / "leaderboard.txt", "w", encoding="utf-8") as f:
            f.write(f"MISSPEC SUMMARY (truth={args.truth})\n")
            for r in sorted(results, key=_sort_key):
                mcmc_logF = f"{r.rmse_logF_mcmc:.4f}" if r.rmse_logF_mcmc is not None else "NA"
                mcmc_phi = f"{r.rmse_phi_mcmc:.4f}" if r.rmse_phi_mcmc is not None else "NA"
                f.write(
                    f"{r.label} | "
                    f"logF_RMSE(VI)={r.rmse_logF_vi:.4f}  logF_RMSE(MCMC)={mcmc_logF} | "
                    f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={mcmc_phi}\n"
                )


if __name__ == "__main__":
    main()
