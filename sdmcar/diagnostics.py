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
# New: Spectrum recovery
# -------------------------

@torch.no_grad()
def _mcmc_spectrum_summary(
    *,
    lam_sorted: torch.Tensor,                 # [n]
    filter_module,
    theta_mat: np.ndarray,                    # [S,d] (unconstrained packed)
    theta_names: list[str],                   # len d
    max_draws: int = 2000,
    band: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (mean, lo, hi) for F(lam) over MCMC theta draws.
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
        theta_dict = filter_module.unpack(theta_t[i])   # <-- correct shapes/keys
        Fi = filter_module.spectrum(lam_sorted, theta_dict).reshape(-1)
        F_draws.append(Fi)

    F_draws = torch.stack(F_draws, dim=0)  # [S, n]
    mean = F_draws.mean(dim=0)

    if not band:
        return mean, None, None

    lo = torch.quantile(F_draws, 0.025, dim=0)
    hi = torch.quantile(F_draws, 0.975, dim=0)
    return mean, lo, hi


@torch.no_grad()
def plot_spectrum_recovery(
    *,
    lam: torch.Tensor,  # [n]
    F_true: torch.Tensor,  # [n] aligned with lam (same order)
    filter_module,
    vi_theta: Optional[Dict[str, torch.Tensor]] = None,
    mcmc_theta: Optional[np.ndarray] = None,
    mcmc_theta_names: Optional[list[str]] = None,  # can keep for now, or remove later
    title: str = "Spectrum recovery",
    save_path: str | Path = "spectrum_recovery.png",
    xscale: str = "linear",
    yscale: str = "log",
    mcmc_band: bool = True,
    mcmc_max_draws: int = 2000,
) -> Dict[str, Any]:

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lam = lam.reshape(-1)
    lam_sorted, idx = torch.sort(lam)
    device, dtype = lam_sorted.device, lam_sorted.dtype

    # True spectrum: just sort the provided F_true using idx
    F_true = F_true.to(device=device, dtype=dtype).reshape(-1)
    F_true_sorted = F_true[idx].reshape(-1)

    # VI plugin
    F_vi = None
    if vi_theta is not None:
        vi_theta_ = {k: v.to(device=device, dtype=dtype).reshape(-1) for k, v in vi_theta.items()}
        F_vi = filter_module.spectrum(lam_sorted, vi_theta_).reshape(-1)

    # MCMC
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
        )

    # Plot
    xs = lam_sorted.detach().cpu().numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(xs, F_true_sorted.detach().cpu().numpy(), linewidth=2, linestyle="--", label="True F(λ)")

    if F_vi is not None:
        plt.plot(xs, F_vi.detach().cpu().numpy(), linewidth=2, label="VI mean F(λ)")

    if F_mcmc_mean is not None:
        plt.plot(xs, F_mcmc_mean.detach().cpu().numpy(), linewidth=2, label="MCMC mean F(λ)")
        if (F_mcmc_lo is not None) and (F_mcmc_hi is not None):
            plt.fill_between(
                xs,
                F_mcmc_lo.detach().cpu().numpy(),
                F_mcmc_hi.detach().cpu().numpy(),
                alpha=0.2,
                label="MCMC 95% band",
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
        "F_mcmc_mean": None if F_mcmc_mean is None else F_mcmc_mean.detach().cpu(),
        "F_mcmc_lo": None if F_mcmc_lo is None else F_mcmc_lo.detach().cpu(),
        "F_mcmc_hi": None if F_mcmc_hi is None else F_mcmc_hi.detach().cpu(),
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
