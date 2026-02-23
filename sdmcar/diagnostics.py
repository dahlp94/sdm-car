# sdmcar/diagnostics.py

from __future__ import annotations

import math
import torch
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Existing plots (kept)
# -------------------------

def plot_elbo(history, save_path: str | None = None):
    plt.figure()
    plt.plot(history["step"], history["elbo"])
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO (Full VI)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_phi_mean_vs_true(coords, mean_phi, phi_true, save_path_prefix: str | None = None):
    n_side = int(math.sqrt(coords.size(0)))
    phi_post = mean_phi.reshape(n_side, n_side).detach().cpu().numpy()
    phi_true_grid = phi_true.reshape(n_side, n_side).detach().cpu().numpy()

    plt.figure()
    plt.imshow(phi_post, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title("Posterior mean φ")
    plt.tight_layout()
    if save_path_prefix is not None:
        plt.savefig(save_path_prefix + "_phi_post.png", bbox_inches="tight", dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(phi_true_grid, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title("True φ")
    plt.tight_layout()
    if save_path_prefix is not None:
        plt.savefig(save_path_prefix + "_phi_true.png", bbox_inches="tight", dpi=200)
    plt.close()


def plot_residual_spectrum(lam, U, X, y, model, save_path: str | None = None):
    """
    Residual power in spectral domain using VI plugin hyperparameters.

    NOTE: This assumes the model exposes:
      - model.filter.mean_unconstrained()
      - model.filter.spectrum(lam, theta)
      - model._beta_update(inv_var)
    """
    with torch.no_grad():
        theta_mean = model.filter.mean_unconstrained()
        F_lam = model.filter.spectrum(lam, theta_mean)
        sigma2_m = torch.exp(model.mu_log_sigma2)
        var = (F_lam + sigma2_m).clamp_min(1e-12)
        inv_var = 1.0 / var

        m_beta, V_beta = model._beta_update(inv_var)
        g = (U.T @ y) - (U.T @ X) @ m_beta
        res_power = (g ** 2).detach().cpu().numpy()

    plt.figure()
    plt.scatter(lam.detach().cpu().numpy(), res_power, s=8)
    plt.xlabel("λ")
    plt.ylabel("g_i^2")
    plt.title("Residual spectrum (VI plugin means)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_beta_intervals(model, save_path: str | None = None):
    with torch.no_grad():
        beta_hat = model.m_beta
        beta_se = torch.sqrt(torch.diag(model.V_beta))
        z = 1.96
        idx = list(range(beta_hat.numel()))
        centers = beta_hat.detach().cpu().numpy()
        errs = (z * beta_se).detach().cpu().numpy()

    plt.figure()
    plt.errorbar(idx, centers, yerr=errs, fmt="o")
    plt.xlabel("β index")
    plt.ylabel("Value")
    plt.title("Posterior β (95% intervals)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


# -------------------------
# Spectrum recovery
# -------------------------

@torch.no_grad()
def _mcmc_spectrum_summary(
    *,
    lam_sorted: torch.Tensor,                 # [n]
    filter_module,
    theta_mat: np.ndarray,                    # [S,d] (unconstrained packed)
    theta_names: list[str],                   # len d (kept for validation)
    max_draws: int = 2000,
    band: bool = True,
    # NEW:
    band_space: str = "log",                  # "log" (default) or "linear"
    q_lo: float = 0.025,
    q_hi: float = 0.975,
    clamp_min: float = 1e-12,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (mean, lo, hi) for F(lam) over MCMC theta draws.

    band_space:
      - "log": compute quantiles of log F, then exponentiate back to F-space.
      - "linear": compute quantiles of F directly.
    """
    if theta_mat.ndim != 2:
        raise ValueError(f"theta_mat must be [S,d], got {theta_mat.shape}")
    S, d = theta_mat.shape
    if len(theta_names) != d:
        raise ValueError(f"theta_names length {len(theta_names)} != d {d}")

    # thin for speed
    if S > max_draws:
        keep = np.linspace(0, S - 1, num=max_draws).astype(int)
        theta_mat = theta_mat[keep, :]
        S = theta_mat.shape[0]

    device, dtype = lam_sorted.device, lam_sorted.dtype
    theta_t = torch.as_tensor(theta_mat, device=device, dtype=dtype)  # [S,d]

    F_draws = []
    for i in range(S):
        theta_dict = filter_module.unpack(theta_t[i])   # dict (unconstrained)
        Fi = filter_module.spectrum(lam_sorted, theta_dict).reshape(-1).clamp_min(clamp_min)
        F_draws.append(Fi)

    F_draws = torch.stack(F_draws, dim=0)  # [S, n]
    mean = F_draws.mean(dim=0)

    if not band:
        return mean, None, None

    band_space = str(band_space).lower().strip()
    if band_space not in {"log", "linear"}:
        raise ValueError("band_space must be one of {'log','linear'}")

    if band_space == "linear":
        lo = torch.quantile(F_draws, q_lo, dim=0)
        hi = torch.quantile(F_draws, q_hi, dim=0)
        return mean, lo, hi

    # log-space band (default)
    logF = torch.log(F_draws)
    lo = torch.exp(torch.quantile(logF, q_lo, dim=0))
    hi = torch.exp(torch.quantile(logF, q_hi, dim=0))
    return mean, lo, hi

@torch.no_grad()
def _vi_spectrum_summary(
    *,
    lam_sorted: torch.Tensor,                 # [n]
    filter_module,
    vi_theta_mean: Optional[Dict[str, torch.Tensor]] = None,
    num_draws: int = 2000,
    band: bool = True,
    # NEW:
    band_space: str = "log",                  # "log" (default) or "linear"
    q_lo: float = 0.025,
    q_hi: float = 0.975,
    clamp_min: float = 1e-12,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (mean, lo, hi) for F(lam) under VI (q(theta)).

    mean:
      - if vi_theta_mean provided: plugin curve F(lam; E_q[theta]) (your current VI line)
      - else: MC mean across draws from q(theta)

    band_space:
      - "log": quantiles in log-space then exp back (default)
      - "linear": quantiles in linear F-space
    """
    device, dtype = lam_sorted.device, lam_sorted.dtype

    # plugin mean curve (optional)
    F_plugin = None
    if vi_theta_mean is not None:
        theta_mean = {k: v.to(device=device, dtype=dtype).reshape(-1) for k, v in vi_theta_mean.items()}
        F_plugin = filter_module.spectrum(lam_sorted, theta_mean).reshape(-1).clamp_min(clamp_min)

    if (not band) and (F_plugin is not None):
        return F_plugin, None, None

    if not hasattr(filter_module, "sample_unconstrained"):
        raise AttributeError("filter_module must implement sample_unconstrained() for VI bands.")

    S = int(num_draws)
    if S <= 0:
        raise ValueError("num_draws must be positive.")

    F_draws = []
    for _ in range(S):
        theta = filter_module.sample_unconstrained()
        theta = {k: v.to(device=device, dtype=dtype).reshape(-1) for k, v in theta.items()}
        Fi = filter_module.spectrum(lam_sorted, theta).reshape(-1).clamp_min(clamp_min)
        F_draws.append(Fi)

    F_draws = torch.stack(F_draws, dim=0)  # [S,n]
    mean_mc = F_draws.mean(dim=0)
    mean = F_plugin if (F_plugin is not None) else mean_mc

    if not band:
        return mean, None, None

    band_space = str(band_space).lower().strip()
    if band_space not in {"log", "linear"}:
        raise ValueError("band_space must be one of {'log','linear'}")

    if band_space == "linear":
        lo = torch.quantile(F_draws, q_lo, dim=0)
        hi = torch.quantile(F_draws, q_hi, dim=0)
        return mean, lo, hi

    # log-space band (default)
    logF = torch.log(F_draws)
    lo = torch.exp(torch.quantile(logF, q_lo, dim=0))
    hi = torch.exp(torch.quantile(logF, q_hi, dim=0))
    return mean, lo, hi


@torch.no_grad()
def plot_spectrum_recovery(
    *,
    lam: torch.Tensor,  # [n]
    F_true: torch.Tensor,  # [n]
    filter_module,
    vi_theta: Optional[Dict[str, torch.Tensor]] = None,
    mcmc_theta: Optional[np.ndarray] = None,
    mcmc_theta_names: Optional[list[str]] = None,
    title: str = "Spectrum recovery",
    save_path: str | Path = "spectrum_recovery.png",
    xscale: str = "linear",
    yscale: str = "log",
    mcmc_band: bool = True,
    mcmc_max_draws: int = 2000,

    # VI
    vi_band: bool = True,
    vi_num_draws: int = 2000,

    # band policy
    band_space_default: str = "log",       # default: log bands
    vi_band_space: Optional[str] = None,   # if None -> use band_space_default
    mcmc_band_space: Optional[str] = None, # if None -> use band_space_default
    band_q_lo: float = 0.025,
    band_q_hi: float = 0.975,
) -> Dict[str, Any]:

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lam = lam.reshape(-1)
    lam_sorted, idx = torch.sort(lam)
    device, dtype = lam_sorted.device, lam_sorted.dtype

    # True spectrum (sorted)
    F_true = F_true.to(device=device, dtype=dtype).reshape(-1)
    F_true_sorted = F_true[idx].reshape(-1)

    # pick band spaces
    vi_space = band_space_default if vi_band_space is None else vi_band_space
    mcmc_space = band_space_default if mcmc_band_space is None else mcmc_band_space

    # ---- VI summary ----
    F_vi = None
    F_vi_lo = None
    F_vi_hi = None
    if (vi_theta is not None) or vi_band:
        F_vi, F_vi_lo, F_vi_hi = _vi_spectrum_summary(
            lam_sorted=lam_sorted,
            filter_module=filter_module,
            vi_theta_mean=vi_theta,
            num_draws=vi_num_draws,
            band=vi_band,
            band_space=vi_space,
            q_lo=band_q_lo,
            q_hi=band_q_hi,
        )

    # ---- MCMC summary ----
    F_mcmc_mean = None
    F_mcmc_lo = None
    F_mcmc_hi = None
    if mcmc_theta is not None:
        if mcmc_theta_names is None:
            raise ValueError("mcmc_theta_names must be provided when mcmc_theta is provided.")
        F_mcmc_mean, F_mcmc_lo, F_mcmc_hi = _mcmc_spectrum_summary(
            lam_sorted=lam_sorted,
            filter_module=filter_module,
            theta_mat=mcmc_theta,
            theta_names=mcmc_theta_names,
            max_draws=mcmc_max_draws,
            band=mcmc_band,
            band_space=mcmc_space,
            q_lo=band_q_lo,
            q_hi=band_q_hi,
        )

    # ---- Plot ----
    xs = lam_sorted.detach().cpu().numpy()
    plt.figure(figsize=(7, 4))

    plt.plot(xs, F_true_sorted.detach().cpu().numpy(), linewidth=2.2, linestyle="-", label="True F(λ)")

    if F_vi is not None:
        plt.plot(xs, F_vi.detach().cpu().numpy(), linewidth=2, label="VI mean F(λ)")
        if (F_vi_lo is not None) and (F_vi_hi is not None):
            plt.fill_between(
                xs,
                F_vi_lo.detach().cpu().numpy(),
                F_vi_hi.detach().cpu().numpy(),
                alpha=0.2,
                label=f"VI 95% band ({vi_space})",
            )

    if F_mcmc_mean is not None:
        plt.plot(xs, F_mcmc_mean.detach().cpu().numpy(), linewidth=2, label="MCMC mean F(λ)")
        if (F_mcmc_lo is not None) and (F_mcmc_hi is not None):
            plt.fill_between(
                xs,
                F_mcmc_lo.detach().cpu().numpy(),
                F_mcmc_hi.detach().cpu().numpy(),
                alpha=0.2,
                label=f"MCMC 95% band ({mcmc_space})",
            )

    plt.title(title)
    plt.xlabel("λ")
    plt.ylabel("F(λ)")
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

    return {
        "save_path": str(save_path),
        "lam_sorted": lam_sorted.detach().cpu(),
        "F_true": F_true_sorted.detach().cpu(),
        "F_vi": None if F_vi is None else F_vi.detach().cpu(),
        "F_vi_lo": None if F_vi_lo is None else F_vi_lo.detach().cpu(),
        "F_vi_hi": None if F_vi_hi is None else F_vi_hi.detach().cpu(),
        "F_mcmc_mean": None if F_mcmc_mean is None else F_mcmc_mean.detach().cpu(),
        "F_mcmc_lo": None if F_mcmc_lo is None else F_mcmc_lo.detach().cpu(),
        "F_mcmc_hi": None if F_mcmc_hi is None else F_mcmc_hi.detach().cpu(),
        "band_space_default": band_space_default,
        "vi_band_space": vi_space,
        "mcmc_band_space": mcmc_space,
    }

@torch.no_grad()
def spectrum_error_log_l1(
    lam: torch.Tensor,
    F_true: torch.Tensor,
    *,
    model=None,
    filter_module=None,
    theta_mcmc: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Log-scale L1 error between true spectrum and estimated spectrum.

    Exactly one of:
        - model (for VI)
        - (filter_module + theta_mcmc) (for MCMC)

    Parameters
    ----------
    lam : (n,) torch.Tensor
        Laplacian eigenvalues
    F_true : (n,) torch.Tensor
        True spectral variance
    model : SpectralCAR_FullVI, optional
        Uses VI posterior mean of hyperparameters
    filter_module : filter object, optional
        Used for MCMC spectrum averaging
    theta_mcmc : (S, d) torch.Tensor, optional
        Packed MCMC samples of unconstrained filter parameters
    eps : float
        Numerical stability

    Returns
    -------
    float
        Mean absolute log-spectrum error
    """

    if model is not None:
        # -------- VI mean spectrum --------
        theta_mean = model.filter.mean_unconstrained()
        F_est = model.filter.spectrum(lam, theta_mean)

    elif filter_module is not None and theta_mcmc is not None:
        # -------- MCMC mean spectrum --------
        S = theta_mcmc.shape[0]
        F_acc = torch.zeros_like(lam)

        for s in range(S):
            theta_s = filter_module.unpack(theta_mcmc[s])
            F_acc += filter_module.spectrum(lam, theta_s)

        F_est = F_acc / float(S)

    else:
        raise ValueError("Must provide either model or (filter_module + theta_mcmc).")

    # -------- log-scale error --------
    log_diff = torch.abs(
        torch.log(F_est.clamp_min(eps))
        - torch.log(F_true.clamp_min(eps))
    )

    return float(log_diff.mean().cpu().item())

def posterior_corr_matrix(
    theta_draws: np.ndarray,
    theta_names: list[str],
    *,
    drop: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute posterior correlation matrix for MCMC draws.

    Args:
        theta_draws: [S, d] array of unconstrained draws.
        theta_names: list of length d, names for columns.
        drop: optional set of names to remove (e.g., {"const"}).

    Returns:
        R: [d', d'] correlation matrix
        names: filtered names corresponding to R
    """
    if theta_draws.ndim != 2:
        raise ValueError(f"theta_draws must be 2D [S,d], got shape {theta_draws.shape}")
    _, d = theta_draws.shape # S is unused
    if len(theta_names) != d:
        raise ValueError(f"theta_names length {len(theta_names)} != d {d}")

    keep_idx = list(range(d))
    if drop:
        keep_idx = [j for j, nm in enumerate(theta_names) if nm not in drop]

    X = theta_draws[:, keep_idx].astype(np.float64)
    names = [theta_names[j] for j in keep_idx]

    # Guard against constant columns
    std = X.std(axis=0, ddof=1)
    good = std > 1e-12
    if not np.all(good):
        # bad_names = [names[i] for i in range(len(names)) if not good[i]]
        # remove constant cols
        X = X[:, good]
        names = [nm for nm, ok in zip(names, good) if ok]

    # corrcoef expects variables in rows if rowvar=True; we want columns as vars
    R = np.corrcoef(X, rowvar=False)
    return R, names


def ridge_strength_summary(R: np.ndarray, names: list[str], *, topk: int = 5) -> dict:
    """
    Simple ridge strength metrics from correlation matrix.
    Robust to d=0 / d=1 cases where np.corrcoef can return a scalar.
    """
    R = np.asarray(R)

    # ---- handle scalar correlation (single parameter) ----
    # np.corrcoef(x) with x shape (S,) returns scalar 1.0 (ndim=0)
    if R.ndim == 0:
        R = R.reshape(1, 1)

    # ---- handle empty / degenerate cases ----
    d = R.shape[0] if R.ndim >= 2 else 0
    if d <= 1:
        return {
            "max_abs_corr": 0.0,
            "mean_abs_corr": 0.0,
            "top_pairs": [],
        }

    if R.shape != (d, d):
        raise ValueError(f"R must be square, got shape {R.shape}")

    # off-diagonal absolute correlations
    absR = np.abs(R.copy())
    np.fill_diagonal(absR, 0.0)

    # top-k pairs
    iu = np.triu_indices(d, k=1)
    vals = absR[iu]
    order = np.argsort(vals)[::-1]
    topk = min(topk, len(vals))

    top_pairs = []
    for idx in order[:topk]:
        i = iu[0][idx]
        j = iu[1][idx]
        top_pairs.append((names[i], names[j], float(R[i, j]), float(absR[i, j])))

    return {
        "max_abs_corr": float(vals.max()) if len(vals) else 0.0,
        "mean_abs_corr": float(vals.mean()) if len(vals) else 0.0,
        "top_pairs": top_pairs,
    }


import numpy as np
import matplotlib.pyplot as plt

def plot_corr_heatmap(
    R: np.ndarray,
    names: list[str],
    *,
    title: str,
    save_path: str,
    max_vars: int = 18,
):
    """
    Plot correlation heatmap for a correlation matrix R.
    Robust to the single-parameter case where R can be a scalar.
    """
    R = np.asarray(R)

    # Handle scalar corrcoef output (single variable)
    if R.ndim == 0:
        R = R.reshape(1, 1)

    d = R.shape[0] if R.ndim >= 2 else 0
    if d <= 1:
        # nothing meaningful to plot; still write a tiny figure so pipeline doesn't break
        plt.figure(figsize=(4, 2.2))
        plt.title(title + " (d<=1)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return

    if R.shape != (d, d):
        raise ValueError(f"R must be square, got shape {R.shape}")

    # Optionally truncate to max_vars
    if d > max_vars:
        R = R[:max_vars, :max_vars]
        names = names[:max_vars]
        d = max_vars

    plt.figure(figsize=(0.45 * d + 3, 0.45 * d + 2))
    plt.imshow(R, vmin=-1.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(d), names, rotation=90, fontsize=8)
    plt.yticks(range(d), names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def ridge_report(
    theta_draws: np.ndarray,
    theta_names: list[str],
    *,
    drop: set[str] | None = None,
    topk: int = 8,
    highlight_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """
    Convenience wrapper: correlation matrix + top pairs + optional highlighted pairs.
    """
    R, names = posterior_corr_matrix(theta_draws, theta_names, drop=drop)
    summ = ridge_strength_summary(R, names, topk=topk)

    highlights = []
    if highlight_pairs:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        for a, b in highlight_pairs:
            if a in name_to_idx and b in name_to_idx:
                i, j = name_to_idx[a], name_to_idx[b]
                highlights.append((a, b, float(R[i, j])))

    summ["R"] = R
    summ["names"] = names
    summ["highlights"] = highlights
    return summ


def spectrum_draw_sd(
    lam: torch.Tensor,
    filter_module,
    theta_draws: torch.Tensor,
    theta_names: list[str],
):
    """
    Computes SD of log F(λ) across MCMC draws.

    Returns:
        mean_sd_logF : scalar
        sd_curve     : np.ndarray shape [n]
    """
    device = lam.device
    dtype = lam.dtype

    S, d = theta_draws.shape
    n = lam.numel()

    logF = torch.zeros(S, n, dtype=dtype, device=device)

    for i in range(S):
        # For each draw do:
        theta_dict = {
            theta_names[j]: theta_draws[i, j].reshape(1)
            for j in range(d)
        }
        # Build spectral variance for that draw
        F = filter_module.spectrum(lam, theta_dict).clamp_min(1e-12)
        logF[i] = torch.log(F)

    sd_curve = logF.std(dim=0)
    mean_sd = sd_curve.mean()

    return float(mean_sd.item()), sd_curve.detach().cpu().numpy()
