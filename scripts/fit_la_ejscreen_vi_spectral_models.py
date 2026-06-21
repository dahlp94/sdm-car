# scripts/fit_la_ejscreen_vi_spectral_models.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import sparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sdmcar.models import SpectralCAR_FullVI
from sdmcar.filters import (
    LerouxCARFilterFullVI,
    AnchoredBSplineSpectrumFullVI,
    UnanchoredBSplineSpectrumFullVI,
    PartiallyAnchoredBSplineSpectrumFullVI,
)


torch.set_default_dtype(torch.double)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def safe_name(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def seconds_to_str(x: float) -> str:
    if x < 60:
        return f"{x:.2f}s"
    if x < 3600:
        return f"{x / 60:.2f}min"
    return f"{x / 3600:.2f}hr"


def tensor_to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


def tensor_to_py(x):
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr.tolist()
    if isinstance(x, dict):
        return {k: tensor_to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def make_paths(
    *,
    out_dir: Path,
    response: str,
    transform: str,
    prefix: str | None = None,
) -> dict[str, Path]:
    if prefix is None:
        stem = f"la_ejscreen_{response.lower()}_{transform.lower()}"
        base = out_dir / stem
    else:
        base = Path(prefix)

    return {
        "base": base,
        "gpkg": Path(str(base) + "_tracts.gpkg"),
        "csv": Path(str(base) + "_tracts.csv"),
        "adj": Path(str(base) + "_queen_adjacency.npz"),
        "metadata": Path(str(base) + "_metadata.json"),
    }


def infer_y_col(
    gdf: gpd.GeoDataFrame,
    *,
    metadata_path: Path | None,
    response: str,
    transform: str,
    y_col: str | None = None,
) -> str:
    if y_col is not None:
        if y_col not in gdf.columns:
            raise ValueError(f"--y-col {y_col!r} not found in GPKG columns.")
        return y_col

    if metadata_path is not None and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "y_col" in meta and meta["y_col"] in gdf.columns:
            return meta["y_col"]

    candidate = f"y_{response.lower()}_{transform.lower()}"
    if candidate in gdf.columns:
        return candidate

    y_cols = [c for c in gdf.columns if c.lower().startswith("y_")]
    if len(y_cols) == 1:
        return y_cols[0]

    raise ValueError(
        "Could not infer response column. "
        f"Candidate y_ columns: {y_cols}. Use --y-col explicitly."
    )


def default_covariate_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    candidates = [
        "ACSTOTPOP",
        "ACSTOTHH",
        "ACSTOTHU",
        "PEOPCOLORPCT",
        "LOWINCPCT",
        "UNEMPPCT",
        "DISABILITYPCT",
        "LINGISOPCT",
        "LESSHSPCT",
        "UNDER5PCT",
        "OVER64PCT",
    ]
    return [c for c in candidates if c in gdf.columns]


# ---------------------------------------------------------------------
# Graph and design matrix
# ---------------------------------------------------------------------
def build_normalized_laplacian(A: sparse.spmatrix) -> np.ndarray:
    """
    Build normalized graph Laplacian:

        L = I - D^{-1/2} A D^{-1/2}

    This matches the diagnostic scripts we used for the EJScreen spectra.
    """
    A = A.tocsr()
    A = 0.5 * (A + A.T)
    A = A.tocsr()

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    if np.any(deg <= 0):
        bad = np.where(deg <= 0)[0]
        raise ValueError(f"Found isolated nodes in adjacency. First bad indices: {bad[:10]}")

    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sparse.diags(inv_sqrt_deg)
    S = D_inv_sqrt @ A @ D_inv_sqrt

    n = A.shape[0]
    L = sparse.eye(n, format="csr") - S
    L = 0.5 * (L + L.T)

    return L.toarray()


def train_test_split_random(
    n: int,
    *,
    test_frac: float,
    seed: int,
) -> np.ndarray:
    """
    Returns boolean mask where True = held out / test.
    """
    if test_frac <= 0:
        return np.zeros(n, dtype=bool)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test-frac must be in [0,1).")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(test_frac * n))
    n_test = max(1, min(n - 1, n_test))

    is_holdout = np.zeros(n, dtype=bool)
    is_holdout[idx[:n_test]] = True
    return is_holdout


def standardize_train(
    Z: np.ndarray,
    train_mask: np.ndarray,
    *,
    names: list[str],
) -> tuple[np.ndarray, dict]:
    """
    Standardize columns using training rows only.
    Missing values are filled by training median before standardization.
    """
    Z = np.asarray(Z, dtype=float)
    out = Z.copy()

    med = np.nanmedian(out[train_mask], axis=0)
    inds = np.where(~np.isfinite(out))
    if len(inds[0]) > 0:
        out[inds] = np.take(med, inds[1])

    mean = out[train_mask].mean(axis=0)
    sd = out[train_mask].std(axis=0, ddof=1)
    sd = np.where(sd <= 1e-12, 1.0, sd)

    Zs = (out - mean) / sd

    info = {
        "names": names,
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "median_for_missing": med.tolist(),
    }
    return Zs, info


def build_design_matrix(
    gdf: gpd.GeoDataFrame,
    *,
    train_mask: np.ndarray,
    use_covariates: bool = True,
    include_coords: bool = True,
    poly_degree: int = 2,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Build X = [intercept, standardized covariates, standardized coordinate polynomial].
    """
    n = len(gdf)
    blocks = []
    names = []
    standardization = {}

    if use_covariates:
        covar_names = default_covariate_columns(gdf)
        if len(covar_names) == 0:
            raise ValueError(
                "No default covariates found in GPKG. "
                "Rebuild the processed dataset with covariates or pass --no-covariates."
            )

        Z_cov = np.column_stack([
            pd.to_numeric(gdf[c], errors="coerce").to_numpy(dtype=float)
            for c in covar_names
        ])
        Z_cov_s, info_cov = standardize_train(Z_cov, train_mask, names=covar_names)
        blocks.append(Z_cov_s)
        names.extend(covar_names)
        standardization["covariates"] = info_cov

    if include_coords:
        cent = gdf.geometry.centroid
        cx = cent.x.to_numpy(dtype=float)
        cy = cent.y.to_numpy(dtype=float)

        coord_raw = np.column_stack([cx, cy])
        coord_s, info_coord = standardize_train(coord_raw, train_mask, names=["coord_x", "coord_y"])
        x = coord_s[:, 0]
        y = coord_s[:, 1]

        coord_blocks = [x, y]
        coord_names = ["coord_x", "coord_y"]

        if poly_degree >= 2:
            coord_blocks.extend([x**2, x * y, y**2])
            coord_names.extend(["coord_x2", "coord_xy", "coord_y2"])

        Z_coord = np.column_stack(coord_blocks)
        blocks.append(Z_coord)
        names.extend(coord_names)
        standardization["coords"] = info_coord
        standardization["coord_poly_degree"] = int(poly_degree)

    if len(blocks) == 0:
        X_no_intercept = np.empty((n, 0))
    else:
        X_no_intercept = np.column_stack(blocks)

    X = np.column_stack([np.ones(n), X_no_intercept])
    X_names = ["intercept"] + names

    return X, X_names, standardization


def standardize_response(
    y: np.ndarray,
    *,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=float)
    mean = float(y[train_mask].mean())
    sd = float(y[train_mask].std(ddof=1))
    if sd <= 1e-12:
        raise ValueError("Training response has near-zero standard deviation.")
    ys = (y - mean) / sd
    return ys, {"mean": mean, "sd": sd}


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------
def ols_fit(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    Xtr = X[mask]
    ytr = y[mask]
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid = y - X @ beta
    dof = max(int(mask.sum()) - X.shape[1], 1)
    sigma2 = float(np.sum(resid[mask] ** 2) / dof)
    return beta, resid, sigma2


def build_leroux_filter(
    *,
    init_tau2: float,
    init_rho_raw: float,
    fixed_rho: float | None,
    device: torch.device,
) -> nn.Module:
    return LerouxCARFilterFullVI(
        mu_log_tau2=math.log(max(init_tau2, 1e-8)),
        log_std_log_tau2=-2.3,
        mu_rho_raw=float(init_rho_raw),
        log_std_rho_raw=-2.3,
        fixed_rho=fixed_rho,
        rho_eps=1e-4,
    ).to(device)


def build_spline_filter(
    *,
    spline_filter: str,
    lam_max: float,
    init_tau2: float,
    init_slope: float,
    degree: int,
    n_internal_knots: int,
    prior_std_theta: float,
    prior_std_w: float,
    log_std0: float,
    init_w: float,
    logF_min: float,
    logF_max: float,
    anchor_strength: float,
    device: torch.device,
) -> nn.Module:
    init_theta = [math.log(max(init_tau2, 1e-8)), float(init_slope)]

    if spline_filter == "anchored_bspline":
        cls = AnchoredBSplineSpectrumFullVI
        kwargs = {}

    elif spline_filter == "unanchored_bspline":
        cls = UnanchoredBSplineSpectrumFullVI
        kwargs = {}

    elif spline_filter == "partially_anchored_bspline":
        cls = PartiallyAnchoredBSplineSpectrumFullVI
        kwargs = {"anchor_strength": float(anchor_strength)}

    else:
        raise ValueError(
            "--spline-filter must be one of "
            "{anchored_bspline, unanchored_bspline, partially_anchored_bspline}."
        )

    return cls(
        lam_max=float(lam_max),
        degree=int(degree),
        n_internal_knots=int(n_internal_knots),
        prior_std_theta=float(prior_std_theta),
        prior_std_w=float(prior_std_w),
        log_std0=float(log_std0),
        init_theta=init_theta,
        init_w=float(init_w),
        logF_min=float(logF_min),
        logF_max=float(logF_max),
        **kwargs,
    ).to(device)


def build_vi_model(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    filter_module: nn.Module,
    prior_V0: torch.Tensor,
    sigma2_init: float,
    fixed_sigma2: float | None,
    is_holdout: torch.Tensor | None,
    vi_mc: int,
) -> SpectralCAR_FullVI:
    return SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(max(sigma2_init, 1e-8)),
        log_std_log_sigma2=-2.3,
        num_mc=int(vi_mc),
        fixed_sigma2=fixed_sigma2,
        is_holdout=is_holdout,
    )


# ---------------------------------------------------------------------
# VI training and prediction
# ---------------------------------------------------------------------
def train_vi(
    *,
    model: SpectralCAR_FullVI,
    out_dir: Path,
    label: str,
    vi_iters: int,
    vi_lr: float,
    log_every: int,
    grad_clip: float | None,
) -> list[float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=float(vi_lr))
    elbo_hist = []

    t0 = time.perf_counter()

    for it in range(int(vi_iters)):
        opt.zero_grad()
        elbo, stats = model.elbo()
        loss = -elbo
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        opt.step()
        elbo_hist.append(float(elbo.detach().cpu().item()))

        if (it + 1) % int(log_every) == 0 or (it + 1) == int(vi_iters):
            with torch.no_grad():
                theta_u, theta_c, sigma2_plugin = model.plugin_hyperparams()

                msg = (
                    f"[{label} VI {it + 1:05d}] "
                    f"ELBO={elbo.item():.3f} "
                    f"loglik={stats['mc_loglik'].item():.3f} "
                    f"KLbeta={stats['mc_kl_beta'].item():.3f} "
                    f"KLfilter={stats['kl_filter'].item():.3f} "
                    f"KLsigma={stats['kl_sigma2'].item():.3f} "
                    f"sigma2={tensor_to_float(sigma2_plugin):.6g}"
                )

                if "rho" in theta_c:
                    msg += f" rho={tensor_to_float(theta_c['rho']):.6g}"
                if "tau2" in theta_c:
                    msg += f" tau2={tensor_to_float(theta_c['tau2']):.6g}"
                if "theta" in theta_c:
                    th = theta_c["theta"].detach().cpu().numpy()
                    msg += " theta=[" + ", ".join(f"{v:.3f}" for v in th[:2]) + "]"
                if "w" in theta_c:
                    w = theta_c["w"].detach().cpu().numpy()
                    msg += f" max|w|={np.max(np.abs(w)):.3f}"

                print(msg)

    elapsed = time.perf_counter() - t0
    print(f"[{label}] VI training time: {seconds_to_str(elapsed)}")

    # Save ELBO trace
    hist_df = pd.DataFrame({"iter": np.arange(1, len(elbo_hist) + 1), "elbo": elbo_hist})
    hist_df.to_csv(out_dir / f"{label}_elbo_trace.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(hist_df["iter"], hist_df["elbo"])
    plt.xlabel("VI iteration")
    plt.ylabel("ELBO")
    plt.title(f"VI ELBO trace: {label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_elbo_trace.png", dpi=220)
    plt.close()

    return elbo_hist


@torch.no_grad()
def predict_plugin(model: SpectralCAR_FullVI) -> dict[str, torch.Tensor]:
    theta_u, _, sigma2 = model.plugin_hyperparams()

    F = model.filter.spectrum(model.lam, theta_u).reshape(-1).clamp_min(0.0)
    var = (F + sigma2.reshape(())).clamp_min(1e-12)
    inv_var = 1.0 / var

    y_curr, y_tilde_curr = model.get_dynamic_y()

    m_beta, V_beta = model._beta_update(inv_var, y_tilde_curr, return_Xt_invSig_X=False)

    r_tilde = y_tilde_curr - model.X_tilde @ m_beta
    w = F / var
    mu_z = w * r_tilde

    mean_phi = model.U @ mu_z
    var_phi_diag = (model.U ** 2) @ (F * sigma2.reshape(()) / var)

    yhat = model.X @ m_beta + mean_phi

    beta_var_diag = torch.sum((model.X @ V_beta) * model.X, dim=1)
    yhat_var_diag = (var_phi_diag + beta_var_diag + sigma2.reshape(())).clamp_min(1e-12)

    return {
        "yhat": yhat.detach(),
        "yhat_var_diag": yhat_var_diag.detach(),
        "mean_phi": mean_phi.detach(),
        "var_phi_diag": var_phi_diag.detach(),
        "m_beta": m_beta.detach(),
        "V_beta": V_beta.detach(),
        "F": F.detach(),
        "sigma2": sigma2.detach(),
        "theta": {k: v.detach() for k, v in theta_u.items()},
    }


@torch.no_grad()
def predict_mc_mean(model: SpectralCAR_FullVI, *, num_mc: int) -> dict[str, torch.Tensor]:
    K = int(num_mc)
    if K <= 0:
        raise ValueError("num_mc must be positive.")

    y_acc = torch.zeros_like(model.y_obs)
    phi_acc = torch.zeros_like(model.y_obs)
    var_acc = torch.zeros_like(model.y_obs)
    F_acc = torch.zeros_like(model.lam)
    sigma2_acc = torch.zeros((), dtype=model.y_obs.dtype, device=model.y_obs.device)

    y_curr, y_tilde_curr = model.get_dynamic_y()

    for _ in range(K):
        if model.fixed_sigma2 is not None:
            sigma2 = torch.tensor(
                model.fixed_sigma2,
                dtype=model.y_obs.dtype,
                device=model.y_obs.device,
            ).clamp_min(1e-12)
        else:
            eps = torch.randn_like(model.mu_log_sigma2)
            s = model.mu_log_sigma2 + torch.exp(model.log_std_log_sigma2) * eps
            sigma2 = torch.exp(s).reshape(()).clamp_min(1e-12)

        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(model.lam, theta).reshape(-1).clamp_min(0.0)

        var = (F + sigma2).clamp_min(1e-12)
        inv_var = 1.0 / var

        m_beta, V_beta = model._beta_update(inv_var, y_tilde_curr, return_Xt_invSig_X=False)

        r_tilde = y_tilde_curr - model.X_tilde @ m_beta
        w = F / var
        mu_z = w * r_tilde

        mean_phi = model.U @ mu_z
        var_phi_diag = (model.U ** 2) @ (F * sigma2 / var)
        beta_var_diag = torch.sum((model.X @ V_beta) * model.X, dim=1)

        yhat = model.X @ m_beta + mean_phi
        yhat_var_diag = (var_phi_diag + beta_var_diag + sigma2).clamp_min(1e-12)

        y_acc += yhat
        phi_acc += mean_phi
        var_acc += yhat_var_diag
        F_acc += F
        sigma2_acc += sigma2

    return {
        "yhat": (y_acc / K).detach(),
        "yhat_var_diag": (var_acc / K).detach(),
        "mean_phi": (phi_acc / K).detach(),
        "F": (F_acc / K).detach(),
        "sigma2": (sigma2_acc / K).detach(),
    }


def regression_metrics(
    *,
    y_true_std: np.ndarray,
    yhat_std: np.ndarray,
    yvar_std: np.ndarray | None,
    mask: np.ndarray,
    y_mean: float,
    y_sd: float,
) -> dict[str, float]:
    y = np.asarray(y_true_std)[mask]
    yh = np.asarray(yhat_std)[mask]

    rmse_std = float(np.sqrt(np.mean((yh - y) ** 2)))
    mae_std = float(np.mean(np.abs(yh - y)))

    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - yh) ** 2) / denom) if denom > 1e-12 else float("nan")

    y_resp = y_mean + y_sd * y
    yh_resp = y_mean + y_sd * yh

    rmse_response = float(np.sqrt(np.mean((yh_resp - y_resp) ** 2)))
    mae_response = float(np.mean(np.abs(yh_resp - y_resp)))

    out = {
        "n": int(mask.sum()),
        "rmse_std": rmse_std,
        "mae_std": mae_std,
        "r2": r2,
        "rmse_response_scale": rmse_response,
        "mae_response_scale": mae_response,
    }

    if yvar_std is not None:
        v = np.asarray(yvar_std)[mask]
        v = np.maximum(v, 1e-12)
        lpd = -0.5 * np.mean(np.log(2.0 * np.pi * v) + (y - yh) ** 2 / v)
        out["diag_lpd_std"] = float(lpd)

    return out


def compute_all_metrics(
    *,
    y_true_std: np.ndarray,
    pred: dict[str, torch.Tensor],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    y_standardization: dict,
) -> dict[str, dict[str, float]]:
    yhat = pred["yhat"].detach().cpu().numpy()
    yvar = pred.get("yhat_var_diag", None)
    yvar_np = None if yvar is None else yvar.detach().cpu().numpy()

    y_mean = float(y_standardization["mean"])
    y_sd = float(y_standardization["sd"])

    metrics = {
        "train": regression_metrics(
            y_true_std=y_true_std,
            yhat_std=yhat,
            yvar_std=yvar_np,
            mask=train_mask,
            y_mean=y_mean,
            y_sd=y_sd,
        )
    }

    if int(test_mask.sum()) > 0:
        metrics["test"] = regression_metrics(
            y_true_std=y_true_std,
            yhat_std=yhat,
            yvar_std=yvar_np,
            mask=test_mask,
            y_mean=y_mean,
            y_sd=y_sd,
        )

    return metrics


# ---------------------------------------------------------------------
# Spectrum helpers and plots
# ---------------------------------------------------------------------
def empirical_binned_spectrum(
    *,
    lam: torch.Tensor,
    U: torch.Tensor,
    y_std: torch.Tensor,
    X: torch.Tensor,
    beta_ols: np.ndarray,
    n_bins: int,
    drop_zero: bool = True,
) -> pd.DataFrame:
    beta_t = torch.tensor(beta_ols, dtype=y_std.dtype, device=y_std.device)
    resid = y_std - X @ beta_t
    z = U.T @ resid
    energy = (z ** 2).detach().cpu().numpy()
    lam_np = lam.detach().cpu().numpy()

    mask = np.isfinite(lam_np) & np.isfinite(energy)
    if drop_zero:
        mask &= lam_np > 1e-10

    lam_use = lam_np[mask]
    e_use = energy[mask]

    order = np.argsort(lam_use)
    lam_use = lam_use[order]
    e_use = e_use[order]

    bins = np.array_split(np.arange(len(lam_use)), int(n_bins))

    rows = []
    for b, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append({
            "bin": b,
            "count": int(len(idx)),
            "lambda_min": float(lam_use[idx].min()),
            "lambda_mean": float(lam_use[idx].mean()),
            "lambda_max": float(lam_use[idx].max()),
            "energy_mean": float(e_use[idx].mean()),
            "energy_median": float(np.median(e_use[idx])),
            "energy_q25": float(np.quantile(e_use[idx], 0.25)),
            "energy_q75": float(np.quantile(e_use[idx], 0.75)),
        })

    return pd.DataFrame(rows)


def save_spectrum_curve_csv(
    *,
    lam: torch.Tensor,
    model_name: str,
    plugin_pred: dict[str, torch.Tensor],
    mc_pred: dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    lam_np = lam.detach().cpu().numpy()

    rows = pd.DataFrame({
        "lambda": lam_np,
        f"{model_name}_F_plugin": plugin_pred["F"].detach().cpu().numpy(),
        f"{model_name}_sigma2_plugin": float(plugin_pred["sigma2"].detach().cpu().reshape(-1)[0]),
        f"{model_name}_total_var_plugin": (
            plugin_pred["F"] + plugin_pred["sigma2"].reshape(())
        ).detach().cpu().numpy(),
        f"{model_name}_F_mc_mean": mc_pred["F"].detach().cpu().numpy(),
        f"{model_name}_sigma2_mc_mean": float(mc_pred["sigma2"].detach().cpu().reshape(-1)[0]),
        f"{model_name}_total_var_mc_mean": (
            mc_pred["F"] + mc_pred["sigma2"].reshape(())
        ).detach().cpu().numpy(),
    })

    rows.to_csv(out_path, index=False)


def plot_spectrum_overlay(
    *,
    binned: pd.DataFrame,
    lam: torch.Tensor,
    leroux_plugin: dict[str, torch.Tensor],
    leroux_mc: dict[str, torch.Tensor],
    spline_plugin: dict[str, torch.Tensor],
    spline_mc: dict[str, torch.Tensor],
    spline_label: str,
    out_path: Path,
    use_total_variance: bool = True,
    logy: bool = True,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    order = np.argsort(lam_np)
    x = lam_np[order]

    if use_total_variance:
        y_leroux = (
            leroux_mc["F"] + leroux_mc["sigma2"].reshape(())
        ).detach().cpu().numpy()[order]
        y_spline = (
            spline_mc["F"] + spline_mc["sigma2"].reshape(())
        ).detach().cpu().numpy()[order]
        ylabel = r"Modal variance: $F(\lambda)+\sigma^2$"
        empirical_label = "Empirical binned residual energy"
    else:
        y_leroux = leroux_mc["F"].detach().cpu().numpy()[order]
        y_spline = spline_mc["F"].detach().cpu().numpy()[order]
        ylabel = r"Spatial spectrum: $F(\lambda)$"
        empirical_label = "Empirical binned residual energy"

    y_leroux = np.clip(y_leroux, 1e-12, None)
    y_spline = np.clip(y_spline, 1e-12, None)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.scatter(
        binned["lambda_mean"],
        np.clip(binned["energy_mean"], 1e-12, None),
        s=35,
        alpha=0.85,
        label=empirical_label,
    )

    ax.plot(
        x,
        y_leroux,
        linewidth=2.2,
        linestyle="--",
        label="Leroux VI MC mean",
    )

    ax.plot(
        x,
        y_spline,
        linewidth=2.2,
        linestyle="-",
        label=f"{spline_label} VI MC mean",
    )

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"Graph eigenvalue $\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title("Empirical residual spectrum vs fitted VI spectra")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_prediction_scatter(
    *,
    y_true: np.ndarray,
    yhat: np.ndarray,
    mask: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    yt = y_true[mask]
    yp = yhat[mask]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(yt, yp, s=12, alpha=0.65)

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)

    ax.set_xlabel("Observed response")
    ax.set_ylabel("Predicted response")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_prediction_map(
    *,
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=True,
        linewidth=0.03,
        edgecolor="black",
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
@torch.no_grad()
def summarize_vi_model(
    *,
    model: SpectralCAR_FullVI,
    num_mc: int,
) -> dict:
    beta = model.beta_posterior_vi(num_mc=num_mc, return_draws=False)
    sigma2 = model.sigma2_posterior_vi(num_mc=num_mc, return_draws=False)
    theta = model.theta_posterior_vi(num_mc=num_mc, return_draws=False)
    spectrum = model.spectrum_posterior_vi(num_mc=num_mc, return_draws=False)

    out = {
        "beta": tensor_to_py(beta),
        "sigma2": tensor_to_py(sigma2),
        "theta": tensor_to_py(theta),
        "spectrum_plugin_min": float(spectrum["plugin"].min().item()),
        "spectrum_plugin_max": float(spectrum["plugin"].max().item()),
        "spectrum_mc_mean_min": float(spectrum["mc"]["mean"].min().item()),
        "spectrum_mc_mean_max": float(spectrum["mc"]["mean"].max().item()),
    }

    if hasattr(model.filter, "shrinkage_summary"):
        out["filter_shrinkage"] = tensor_to_py(model.filter.shrinkage_summary())

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--response", type=str, default="ptraf")
    parser.add_argument("--transform", type=str, default="yeojohnson")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--y-col", type=str, default=None)

    # Design
    parser.add_argument("--no-covariates", action="store_true")
    parser.add_argument("--no-coords", action="store_true")
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--beta-prior-var", type=float, default=10.0)

    # Split
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1)

    # VI
    parser.add_argument("--vi-iters", type=int, default=2000)
    parser.add_argument("--vi-mc", type=int, default=5)
    parser.add_argument("--summary-mc", type=int, default=128)
    parser.add_argument("--pred-mc", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    # Sigma2
    parser.add_argument(
        "--fix-sigma2",
        action="store_true",
        help="Fix sigma2 instead of learning it.",
    )
    parser.add_argument(
        "--fixed-sigma2-value",
        type=float,
        default=None,
        help="Fixed sigma2 on standardized response scale. If omitted, uses --fixed-sigma2-frac * OLS residual variance.",
    )
    parser.add_argument("--fixed-sigma2-frac", type=float, default=0.25)

    # Leroux
    parser.add_argument("--leroux-fixed-rho", type=str, default="none")
    parser.add_argument("--leroux-init-rho-raw", type=float, default=2.0)

    # Spline
    parser.add_argument(
        "--spline-filter",
        type=str,
        default="anchored_bspline",
        choices=["anchored_bspline", "unanchored_bspline", "partially_anchored_bspline"],
    )
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-internal-knots", type=int, default=8)
    parser.add_argument("--prior-std-theta", type=float, default=2.0)
    parser.add_argument("--prior-std-w", type=float, default=0.50)
    parser.add_argument("--log-std0", type=float, default=-2.3)
    parser.add_argument("--init-slope", type=float, default=-3.0)
    parser.add_argument("--init-w", type=float, default=0.0)
    parser.add_argument("--logF-min", type=float, default=-30.0)
    parser.add_argument("--logF-max", type=float, default=30.0)
    parser.add_argument("--anchor-strength", type=float, default=0.75)

    # Plotting
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--plot-spatial-only", action="store_true")
    parser.add_argument("--save-maps", action="store_true")

    # Output/device
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: examples/figures/la_ejscreen_vi_<response>_<transform>",
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    if args.outdir is None:
        args.outdir = Path("examples") / "figures" / f"la_ejscreen_vi_{args.response.lower()}_{args.transform.lower()}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = make_paths(
        out_dir=args.processed_dir,
        response=args.response,
        transform=args.transform,
        prefix=args.prefix,
    )

    print("\n[FILES]")
    print(f"  GPKG = {paths['gpkg']}")
    print(f"  ADJ  = {paths['adj']}")
    print(f"  OUT  = {args.outdir}")

    if not paths["gpkg"].exists():
        raise FileNotFoundError(paths["gpkg"])
    if not paths["adj"].exists():
        raise FileNotFoundError(paths["adj"])

    gdf = gpd.read_file(paths["gpkg"], layer="tracts")
    A = sparse.load_npz(paths["adj"]).tocsr()

    if A.shape[0] != len(gdf):
        raise ValueError(f"Adjacency shape {A.shape} does not match GPKG rows {len(gdf)}.")

    y_col = infer_y_col(
        gdf,
        metadata_path=paths["metadata"],
        response=args.response,
        transform=args.transform,
        y_col=args.y_col,
    )

    y_raw = pd.to_numeric(gdf[y_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_raw).all():
        raise ValueError(f"Response column {y_col} contains non-finite values.")

    n = len(gdf)
    is_holdout_np = train_test_split_random(n, test_frac=args.test_frac, seed=args.seed)
    train_mask = ~is_holdout_np
    test_mask = is_holdout_np

    print("\n[DATA]")
    print(f"  n = {n}")
    print(f"  response column = {y_col}")
    print(f"  train n = {train_mask.sum()}")
    print(f"  test n  = {test_mask.sum()}")

    y_std_np, y_standardization = standardize_response(y_raw, train_mask=train_mask)

    X_np, X_names, X_standardization = build_design_matrix(
        gdf,
        train_mask=train_mask,
        use_covariates=not args.no_covariates,
        include_coords=not args.no_coords,
        poly_degree=args.poly_degree,
    )

    beta_ols, resid_ols, resid_var_ols = ols_fit(X_np, y_std_np, train_mask)

    print("\n[MEAN MODEL / INITIALIZATION]")
    print(f"  X shape = {X_np.shape}")
    print(f"  OLS residual variance on train = {resid_var_ols:.6g}")
    print(f"  beta_ols first 8 = {np.round(beta_ols[:8], 4)}")

    init_tau2 = max(0.50 * resid_var_ols, 1e-6)
    sigma2_init = max(0.25 * resid_var_ols, 1e-6)

    fixed_sigma2 = None
    if args.fix_sigma2:
        if args.fixed_sigma2_value is not None:
            fixed_sigma2 = float(args.fixed_sigma2_value)
        else:
            fixed_sigma2 = float(args.fixed_sigma2_frac * resid_var_ols)
        fixed_sigma2 = max(fixed_sigma2, 1e-8)
        print(f"  fixed sigma2 = {fixed_sigma2:.6g}")

    # Build Laplacian/eigenbasis
    print("\n[GRAPH]")
    print("  building normalized Laplacian...")
    L_np = build_normalized_laplacian(A)

    print("  eigendecomposition...")
    t0 = time.perf_counter()
    L_t = torch.tensor(L_np, dtype=torch.double, device=device)
    lam, U = torch.linalg.eigh(L_t)
    eig_time = time.perf_counter() - t0

    print(f"  eig time = {seconds_to_str(eig_time)}")
    print(f"  lambda min/max = {lam.min().item():.6g} / {lam.max().item():.6g}")

    # Convert data to torch
    X = torch.tensor(X_np, dtype=torch.double, device=device)
    y = torch.tensor(y_std_np, dtype=torch.double, device=device)

    is_holdout_t = None
    if test_mask.sum() > 0:
        is_holdout_t = torch.tensor(is_holdout_np, dtype=torch.bool, device=device)

    p = X.shape[1]
    prior_V0 = float(args.beta_prior_var) * torch.eye(p, dtype=torch.double, device=device)

    # Empirical binned spectrum from OLS residuals
    binned = empirical_binned_spectrum(
        lam=lam,
        U=U,
        y_std=y,
        X=X,
        beta_ols=beta_ols,
        n_bins=args.n_bins,
        drop_zero=True,
    )
    binned_path = args.outdir / "empirical_binned_ols_residual_spectrum.csv"
    binned.to_csv(binned_path, index=False)

    # -----------------------------------------------------------------
    # Build and train Leroux
    # -----------------------------------------------------------------
    if args.leroux_fixed_rho.lower() in {"none", "null", "na"}:
        leroux_fixed_rho = None
    else:
        leroux_fixed_rho = float(args.leroux_fixed_rho)

    leroux_filter = build_leroux_filter(
        init_tau2=init_tau2,
        init_rho_raw=args.leroux_init_rho_raw,
        fixed_rho=leroux_fixed_rho,
        device=device,
    )

    leroux_model = build_vi_model(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=leroux_filter,
        prior_V0=prior_V0,
        sigma2_init=sigma2_init,
        fixed_sigma2=fixed_sigma2,
        is_holdout=is_holdout_t,
        vi_mc=args.vi_mc,
    ).to(device)

    train_vi(
        model=leroux_model,
        out_dir=args.outdir,
        label="leroux",
        vi_iters=args.vi_iters,
        vi_lr=args.lr,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
    )

    # -----------------------------------------------------------------
    # Build and train spline
    # -----------------------------------------------------------------
    spline_filter = build_spline_filter(
        spline_filter=args.spline_filter,
        lam_max=float(lam.max().item()),
        init_tau2=init_tau2,
        init_slope=args.init_slope,
        degree=args.degree,
        n_internal_knots=args.n_internal_knots,
        prior_std_theta=args.prior_std_theta,
        prior_std_w=args.prior_std_w,
        log_std0=args.log_std0,
        init_w=args.init_w,
        logF_min=args.logF_min,
        logF_max=args.logF_max,
        anchor_strength=args.anchor_strength,
        device=device,
    )

    spline_model = build_vi_model(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=spline_filter,
        prior_V0=prior_V0,
        sigma2_init=sigma2_init,
        fixed_sigma2=fixed_sigma2,
        is_holdout=is_holdout_t,
        vi_mc=args.vi_mc,
    ).to(device)

    train_vi(
        model=spline_model,
        out_dir=args.outdir,
        label=args.spline_filter,
        vi_iters=args.vi_iters,
        vi_lr=args.lr,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
    )

    # -----------------------------------------------------------------
    # Predictions and metrics
    # -----------------------------------------------------------------
    print("\n[SUMMARIES / PREDICTIONS]")
    leroux_plugin = predict_plugin(leroux_model)
    spline_plugin = predict_plugin(spline_model)

    leroux_mc = predict_mc_mean(leroux_model, num_mc=args.pred_mc)
    spline_mc = predict_mc_mean(spline_model, num_mc=args.pred_mc)

    leroux_metrics_plugin = compute_all_metrics(
        y_true_std=y_std_np,
        pred=leroux_plugin,
        train_mask=train_mask,
        test_mask=test_mask,
        y_standardization=y_standardization,
    )
    leroux_metrics_mc = compute_all_metrics(
        y_true_std=y_std_np,
        pred=leroux_mc,
        train_mask=train_mask,
        test_mask=test_mask,
        y_standardization=y_standardization,
    )

    spline_metrics_plugin = compute_all_metrics(
        y_true_std=y_std_np,
        pred=spline_plugin,
        train_mask=train_mask,
        test_mask=test_mask,
        y_standardization=y_standardization,
    )
    spline_metrics_mc = compute_all_metrics(
        y_true_std=y_std_np,
        pred=spline_mc,
        train_mask=train_mask,
        test_mask=test_mask,
        y_standardization=y_standardization,
    )

    metrics = {
        "response": args.response,
        "transform": args.transform,
        "y_col": y_col,
        "n": int(n),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "y_standardization": y_standardization,
        "X_names": X_names,
        "X_standardization": X_standardization,
        "ols": {
            "resid_var_train_std": float(resid_var_ols),
            "beta_ols": beta_ols.tolist(),
        },
        "initialization": {
            "init_tau2": float(init_tau2),
            "sigma2_init": float(sigma2_init),
            "fixed_sigma2": None if fixed_sigma2 is None else float(fixed_sigma2),
        },
        "leroux": {
            "metrics_plugin": leroux_metrics_plugin,
            "metrics_mc": leroux_metrics_mc,
            "summary": summarize_vi_model(model=leroux_model, num_mc=args.summary_mc),
        },
        args.spline_filter: {
            "metrics_plugin": spline_metrics_plugin,
            "metrics_mc": spline_metrics_mc,
            "summary": summarize_vi_model(model=spline_model, num_mc=args.summary_mc),
        },
    }

    with open(args.outdir / "vi_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Flatten key metric table
    rows = []
    for model_name, m_plugin, m_mc in [
        ("leroux", leroux_metrics_plugin, leroux_metrics_mc),
        (args.spline_filter, spline_metrics_plugin, spline_metrics_mc),
    ]:
        for pred_type, mm in [("plugin", m_plugin), ("mc_mean", m_mc)]:
            for split, vals in mm.items():
                row = {"model": model_name, "prediction": pred_type, "split": split}
                row.update(vals)
                rows.append(row)

    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(args.outdir / "vi_predictive_metrics.csv", index=False)

    print("\n[PREDICTIVE METRICS]")
    print(metric_df)

    # -----------------------------------------------------------------
    # Save fitted spectra and overlay
    # -----------------------------------------------------------------
    save_spectrum_curve_csv(
        lam=lam,
        model_name="leroux",
        plugin_pred=leroux_plugin,
        mc_pred=leroux_mc,
        out_path=args.outdir / "leroux_fitted_spectrum.csv",
    )

    save_spectrum_curve_csv(
        lam=lam,
        model_name=args.spline_filter,
        plugin_pred=spline_plugin,
        mc_pred=spline_mc,
        out_path=args.outdir / f"{args.spline_filter}_fitted_spectrum.csv",
    )

    plot_spectrum_overlay(
        binned=binned,
        lam=lam,
        leroux_plugin=leroux_plugin,
        leroux_mc=leroux_mc,
        spline_plugin=spline_plugin,
        spline_mc=spline_mc,
        spline_label=args.spline_filter,
        out_path=args.outdir / "empirical_vs_leroux_vs_spline_spectrum_overlay.png",
        use_total_variance=not args.plot_spatial_only,
        logy=True,
    )

    # -----------------------------------------------------------------
    # Save prediction CSV
    # -----------------------------------------------------------------
    y_mean = y_standardization["mean"]
    y_sd = y_standardization["sd"]

    pred_df = pd.DataFrame({
        "row": np.arange(n),
        "is_holdout": is_holdout_np,
        "y_response_scale": y_raw,
        "y_std": y_std_np,
        "ols_resid_std": resid_ols,
    })

    for id_col in ["GEOID", "ID", "TRACTCE", "GEOID20"]:
        if id_col in gdf.columns:
            pred_df[id_col] = gdf[id_col].astype(str).to_numpy()

    for model_name, pred_plugin, pred_mc in [
        ("leroux", leroux_plugin, leroux_mc),
        (args.spline_filter, spline_plugin, spline_mc),
    ]:
        for tag, pred in [("plugin", pred_plugin), ("mc_mean", pred_mc)]:
            yhat_std = pred["yhat"].detach().cpu().numpy()
            yhat_response = y_mean + y_sd * yhat_std

            pred_df[f"{model_name}_{tag}_yhat_std"] = yhat_std
            pred_df[f"{model_name}_{tag}_resid_std"] = y_std_np - yhat_std
            pred_df[f"{model_name}_{tag}_yhat_response_scale"] = yhat_response
            pred_df[f"{model_name}_{tag}_resid_response_scale"] = y_raw - yhat_response

    pred_path = args.outdir / "vi_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Prediction scatter plots on response scale
    for model_name in ["leroux", args.spline_filter]:
        yhat_resp = pred_df[f"{model_name}_mc_mean_yhat_response_scale"].to_numpy(dtype=float)

        plot_prediction_scatter(
            y_true=y_raw,
            yhat=yhat_resp,
            mask=train_mask,
            title=f"{model_name}: train predictions",
            out_path=args.outdir / f"{model_name}_train_prediction_scatter.png",
        )

        if test_mask.sum() > 0:
            plot_prediction_scatter(
                y_true=y_raw,
                yhat=yhat_resp,
                mask=test_mask,
                title=f"{model_name}: held-out predictions",
                out_path=args.outdir / f"{model_name}_test_prediction_scatter.png",
            )

    # Optional maps
    if args.save_maps:
        gdf_out = gdf.copy()
        for col in pred_df.columns:
            if col not in gdf_out.columns:
                gdf_out[col] = pred_df[col].to_numpy()

        gpkg_out = args.outdir / "vi_prediction_maps.gpkg"
        gdf_out.to_file(gpkg_out, layer="tracts", driver="GPKG")

        for model_name in ["leroux", args.spline_filter]:
            save_prediction_map(
                gdf=gdf_out,
                column=f"{model_name}_mc_mean_yhat_response_scale",
                title=f"{model_name}: predicted response",
                out_path=args.outdir / f"{model_name}_predicted_response_map.png",
                cmap="viridis",
            )
            save_prediction_map(
                gdf=gdf_out,
                column=f"{model_name}_mc_mean_resid_response_scale",
                title=f"{model_name}: residual response",
                out_path=args.outdir / f"{model_name}_residual_response_map.png",
                cmap="coolwarm",
            )

    print("\n[SAVED]")
    print(f"  metrics JSON: {args.outdir / 'vi_metrics_summary.json'}")
    print(f"  metrics CSV : {args.outdir / 'vi_predictive_metrics.csv'}")
    print(f"  predictions : {pred_path}")
    print(f"  binned spec : {binned_path}")
    print(f"  overlay plot: {args.outdir / 'empirical_vs_leroux_vs_spline_spectrum_overlay.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()