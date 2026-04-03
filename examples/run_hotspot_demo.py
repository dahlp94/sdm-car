from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.benchmarks.registry import get_filter_spec, available_filters
import examples.benchmarks  # noqa: F401  # ensure registry is populated

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.models import SpectralCAR_FullVI
from sdmcar.utils import set_default_dtype, set_seed


DTYPE = torch.double


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

def rmse_on_index(a: torch.Tensor, b: torch.Tensor, idx: np.ndarray) -> float:
    ii = torch.as_tensor(idx, device=a.device)
    return float(torch.sqrt(torch.mean((a[ii] - b[ii]) ** 2)).item())


def summarize_residual_rmse(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[float, float]:
    tr = torch.as_tensor(train_idx, device=y.device)
    te = torch.as_tensor(test_idx, device=y.device)

    train_rmse = float(torch.sqrt(torch.mean((y[tr] - y_hat[tr]) ** 2)).item())
    test_rmse = float(torch.sqrt(torch.mean((y[te] - y_hat[te]) ** 2)).item())
    return train_rmse, test_rmse


def parse_fit_strings(fits: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in fits:
        if ":" in item:
            filt, variant = item.split(":", 1)
        else:
            filt, variant = item, "default"
        out.append((filt, variant))
    return out


def parse_centers(text: str) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        x, y = part.split(",")
        centers.append((float(x), float(y)))
    return centers


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(";") if x.strip()]


def parse_point(text: str) -> tuple[float, float]:
    x, y = text.split(",")
    return float(x), float(y)


def make_grid_2d_coords(
    nx: int,
    ny: int,
    *,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    xs = torch.arange(nx, dtype=dtype, device=device)
    ys = torch.arange(ny, dtype=dtype, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def make_split(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(test_frac * n)
    n_test = max(1, min(n - 1, n_test))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


# ---------------------------------------------------------------------
# Hotspot truth
# ---------------------------------------------------------------------
def gaussian_bump(
    coords: torch.Tensor,
    *,
    center: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive.")

    cx, cy = center
    c = torch.tensor([cx, cy], dtype=coords.dtype, device=coords.device)
    d2 = torch.sum((coords - c) ** 2, dim=1)
    return torch.exp(-0.5 * d2 / (radius * radius))


def mexican_hat_bump(
    coords: torch.Tensor,
    *,
    center: tuple[float, float],
    r_inner: float,
    r_outer: float,
) -> torch.Tensor:
    if not (0.0 < r_inner < r_outer):
        raise ValueError("Need 0 < r_inner < r_outer for mexican_hat_bump.")
    inner = gaussian_bump(coords, center=center, radius=r_inner)
    outer = gaussian_bump(coords, center=center, radius=r_outer)
    return inner - outer


def make_hotspot_field(
    coords: torch.Tensor,
    centers: list[tuple[float, float]],
    radii: list[float],
    amplitudes: list[float],
    *,
    normalize_to_sd: float | None = None,
    add_ring: bool = False,
    ring_center: tuple[float, float] = (16.0, 24.0),
    ring_inner: float = 1.6,
    ring_outer: float = 3.8,
    ring_amplitude: float = 1.25,
    center_field: bool = True,
) -> torch.Tensor:
    """
    Build a more spectrally challenging hotspot field.

    Base component:
        phi_i = sum_k a_k exp(-||s_i - c_k||^2 / (2 r_k^2))

    Optional addition:
        localized Mexican-hat feature to inject local sign changes.
    """
    if not (len(centers) == len(radii) == len(amplitudes)):
        raise ValueError("centers, radii, and amplitudes must have same length.")

    phi = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)

    for center, r, a in zip(centers, radii, amplitudes):
        phi = phi + a * gaussian_bump(coords, center=center, radius=r)

    if add_ring:
        phi = phi + ring_amplitude * mexican_hat_bump(
            coords,
            center=ring_center,
            r_inner=ring_inner,
            r_outer=ring_outer,
        )

    if center_field:
        phi = phi - torch.mean(phi)

    if normalize_to_sd is not None:
        sd = torch.std(phi)
        if float(sd) > 0.0:
            phi = phi / sd * normalize_to_sd

    return phi


def simulate_hotspot_dataset(
    coords: torch.Tensor,
    *,
    beta0: float,
    sigma: float,
    centers: list[tuple[float, float]],
    radii: list[float],
    amplitudes: list[float],
    normalize_phi_sd: float | None,
    device: torch.device,
    add_ring: bool = False,
    ring_center: tuple[float, float] = (16.0, 24.0),
    ring_inner: float = 1.6,
    ring_outer: float = 3.8,
    ring_amplitude: float = 1.25,
) -> dict[str, torch.Tensor]:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n = coords.shape[0]
    X = torch.ones((n, 1), dtype=DTYPE, device=device)
    beta_true = torch.tensor([beta0], dtype=DTYPE, device=device)

    phi_true = make_hotspot_field(
        coords,
        centers=centers,
        radii=radii,
        amplitudes=amplitudes,
        normalize_to_sd=normalize_phi_sd,
        add_ring=add_ring,
        ring_center=ring_center,
        ring_inner=ring_inner,
        ring_outer=ring_outer,
        ring_amplitude=ring_amplitude,
    )
    eta_true = (X @ beta_true).reshape(-1) + phi_true
    y = eta_true + sigma * torch.randn(n, dtype=DTYPE, device=device)

    return {
        "X": X,
        "y": y,
        "phi_true": phi_true,
        "eta_true": eta_true,
        "beta_true": beta_true,
    }


def empirical_hotspot_spectrum(U: torch.Tensor, phi_true: torch.Tensor) -> torch.Tensor:
    z = U.T @ phi_true
    return z ** 2

def spectral_band_summary(
    lam: torch.Tensor,
    energy: torch.Tensor,
) -> dict[str, float]:
    lam_np = lam.detach().cpu().numpy()
    e_np = energy.detach().cpu().numpy()

    total = float(np.sum(e_np)) + 1e-12

    q1 = float(np.quantile(lam_np, 0.33))
    q2 = float(np.quantile(lam_np, 0.67))

    low = float(np.sum(e_np[lam_np <= q1]) / total)
    mid = float(np.sum(e_np[(lam_np > q1) & (lam_np <= q2)]) / total)
    high = float(np.sum(e_np[lam_np > q2]) / total)

    order = np.argsort(lam_np)
    e_sorted = e_np[order]
    cume = np.cumsum(e_sorted) / total
    n = len(cume)

    def frac_energy_at(p: float) -> float:
        j = max(0, min(n - 1, int(np.floor(p * (n - 1)))))
        return float(cume[j])

    return {
        "low_band_frac": low,
        "mid_band_frac": mid,
        "high_band_frac": high,
        "cum_energy_10pct_eigs": frac_energy_at(0.10),
        "cum_energy_25pct_eigs": frac_energy_at(0.25),
        "cum_energy_50pct_eigs": frac_energy_at(0.50),
        "cum_energy_75pct_eigs": frac_energy_at(0.75),
    }


def print_spectral_summary(name: str, summary: dict[str, float]) -> None:
    print(
        f"[SPECTRUM] {name} | "
        f"low={summary['low_band_frac']:.3f}, "
        f"mid={summary['mid_band_frac']:.3f}, "
        f"high={summary['high_band_frac']:.3f} | "
        f"cum@10%={summary['cum_energy_10pct_eigs']:.3f}, "
        f"cum@25%={summary['cum_energy_25pct_eigs']:.3f}, "
        f"cum@50%={summary['cum_energy_50pct_eigs']:.3f}, "
        f"cum@75%={summary['cum_energy_75pct_eigs']:.3f}"
    )

def print_spectral_band_energy(
    lam: torch.Tensor,
    empirical_energy: torch.Tensor,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    e_np = empirical_energy.detach().cpu().numpy()

    total = float(np.sum(e_np)) + 1e-12
    q1 = float(np.quantile(lam_np, 0.33))
    q2 = float(np.quantile(lam_np, 0.67))

    low = float(np.sum(e_np[lam_np <= q1]) / total)
    mid = float(np.sum(e_np[(lam_np > q1) & (lam_np <= q2)]) / total)
    high = float(np.sum(e_np[lam_np > q2]) / total)

    print(
        "[SPECTRUM] Empirical hotspot energy fractions -> "
        f"low: {low:.3f}, mid: {mid:.3f}, high: {high:.3f}"
    )


# ---------------------------------------------------------------------
# Spectrum / predictive helpers (adapted from Experiment 3)
# ---------------------------------------------------------------------
@torch.no_grad()
def spectrum_vi_mc_mean(filter_module, lam: torch.Tensor, *, S: int = 256) -> torch.Tensor:
    acc = torch.zeros_like(lam)
    for _ in range(S):
        th = filter_module.sample_unconstrained()
        acc += filter_module.spectrum(lam, th).clamp_min(1e-12)
    return acc / float(S)


def _k_blocks_from_spectrum(
    U: torch.Tensor,
    F: torch.Tensor,
    sigma2: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    device = U.device
    dtype = U.dtype

    tr = torch.as_tensor(train_idx, device=device)
    te = torch.as_tensor(test_idx, device=device)

    Utr = U[tr, :]
    Ute = U[te, :]

    F_row = F.to(device=device, dtype=dtype).clamp_min(1e-12).reshape(1, -1)

    K_tt = (Utr * F_row) @ Utr.T
    K_st = (Ute * F_row) @ Utr.T
    K_ss = (Ute * F_row) @ Ute.T

    K_tt = K_tt + sigma2 * torch.eye(K_tt.shape[0], device=device, dtype=dtype)
    K_ss = K_ss + sigma2 * torch.eye(K_ss.shape[0], device=device, dtype=dtype)

    return K_tt, K_st, K_ss


def conditional_predictive_loglik(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    beta: torch.Tensor,
    U: torch.Tensor,
    F: torch.Tensor,
    sigma2: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    device = y.device
    dtype = y.dtype

    tr = torch.as_tensor(train_idx, device=device)
    te = torch.as_tensor(test_idx, device=device)

    y_tr = y[tr]
    y_te = y[te]
    X_tr = X[tr, :]
    X_te = X[te, :]

    mu_tr = X_tr @ beta
    mu_te = X_te @ beta
    r_tr = y_tr - mu_tr

    K_tt, K_st, K_ss = _k_blocks_from_spectrum(U, F, sigma2, train_idx, test_idx)

    L = torch.linalg.cholesky(K_tt)
    alpha = torch.cholesky_solve(r_tr.reshape(-1, 1), L).reshape(-1)
    cond_mean = mu_te + (K_st @ alpha)

    V = torch.cholesky_solve(K_st.T, L)
    cond_cov = K_ss - (K_st @ V)
    cond_cov = 0.5 * (cond_cov + cond_cov.T)

    e = (y_te - cond_mean).reshape(-1, 1)

    jitter = 1e-8
    cond_cov = cond_cov + jitter * torch.eye(cond_cov.shape[0], device=device, dtype=dtype)

    Lc = torch.linalg.cholesky(cond_cov)
    sol = torch.cholesky_solve(e, Lc)
    quad = float((e.T @ sol).reshape(()).detach().cpu())
    logdet = float((2.0 * torch.log(torch.diag(Lc))).sum().detach().cpu())
    n_te = cond_cov.shape[0]

    ll = -0.5 * (quad + logdet + n_te * math.log(2.0 * math.pi))
    return float(ll)


@torch.no_grad()
def predictive_loglik_vi_heldout(
    *,
    model: SpectralCAR_FullVI,
    y: torch.Tensor,
    X: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    S: int = 16,
    use_plugin_beta: bool = True,
    sample_sigma2: bool = False,
) -> float:
    m_beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta = m_beta_plugin.reshape(-1)

    logliks = []

    for _ in range(int(S)):
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(lam, theta).clamp_min(1e-12)

        if sample_sigma2:
            eps = torch.randn_like(model.mu_log_sigma2)
            s = model.mu_log_sigma2 + torch.exp(model.log_std_log_sigma2) * eps
            sigma2 = float(torch.exp(s).clamp_min(1e-12).detach().cpu())
        else:
            sigma2 = float(sigma2_plugin.detach().cpu())

        if not use_plugin_beta:
            denom = (F + torch.tensor(sigma2, device=F.device, dtype=F.dtype)).clamp_min(1e-12)
            inv_var = 1.0 / denom
            beta, _ = model._beta_update(inv_var, return_Xt_invSig_X=False)
            beta = beta.reshape(-1)

        ll = conditional_predictive_loglik(
            y=y, X=X, beta=beta, U=U, F=F, sigma2=sigma2,
            train_idx=train_idx, test_idx=test_idx,
        )
        logliks.append(ll)

    a = max(logliks)
    pll = a + math.log(sum(math.exp(v - a) for v in logliks) / float(len(logliks)))
    return float(pll)


@torch.no_grad()
def phi_full_from_train(
    *,
    U_train: torch.Tensor,
    U_full: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    beta: torch.Tensor,
    F: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    r_tr = y_train - X_train @ beta
    Ut_r = U_train.T @ r_tr
    shrink = (F / (F + sigma2)).clamp(0.0, 1.0)
    mu_z = shrink * Ut_r
    return U_full @ mu_z


@torch.no_grad()
def compute_phi_vi_full(
    *,
    model: SpectralCAR_FullVI,
    lam: torch.Tensor,
    U_train: torch.Tensor,
    U_full: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mode: str = "plugin",
    S: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    beta_mean, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2 = float(sigma2_plugin.detach().cpu())

    if mode == "plugin":
        F_used = spectrum_vi_mc_mean(model.filter, lam, S=S).detach()
        phi = phi_full_from_train(
            U_train=U_train,
            U_full=U_full,
            X_train=X_train,
            y_train=y_train,
            beta=beta_mean,
            F=F_used.to(lam.device),
            sigma2=sigma2,
        )
        return phi, F_used.detach(), sigma2

    if mode == "posterior":
        acc_phi = torch.zeros(U_full.shape[0], device=lam.device, dtype=lam.dtype)
        acc_F = torch.zeros_like(lam)

        for _ in range(int(S)):
            th = model.filter.sample_unconstrained()
            F = model.filter.spectrum(lam, th).clamp_min(1e-12)
            phi = phi_full_from_train(
                U_train=U_train,
                U_full=U_full,
                X_train=X_train,
                y_train=y_train,
                beta=beta_mean,
                F=F,
                sigma2=sigma2,
            )
            acc_phi += phi
            acc_F += F

        return acc_phi / float(S), acc_F / float(S), sigma2

    raise ValueError(f"Unknown VI phi mode '{mode}'.")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def save_heatmap(
    field: torch.Tensor,
    *,
    nx: int,
    ny: int,
    title: str,
    outpath: Path,
) -> None:
    arr = field.detach().cpu().reshape(nx, ny).numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, origin="lower", aspect="equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_compare_fields(
    *,
    phi_true: torch.Tensor,
    phi_hats: dict[str, torch.Tensor],
    nx: int,
    ny: int,
    outpath: Path,
) -> None:
    panels = [("Truth", phi_true)] + list(phi_hats.items())
    m = len(panels)

    fig, axes = plt.subplots(1, m, figsize=(4.5 * m, 4.2))
    if m == 1:
        axes = [axes]

    arrays = [x.detach().cpu().reshape(nx, ny).numpy() for _, x in panels]
    vmin = min(arr.min() for arr in arrays)
    vmax = max(arr.max() for arr in arrays)

    for ax, (title, _), arr in zip(axes, panels, arrays):
        im = ax.imshow(arr, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_empirical_vs_learned_spectra(
    *,
    lam: torch.Tensor,
    empirical_energy: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    emp_np = empirical_energy.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(lam_np, emp_np, label="Empirical hotspot energy", linewidth=2.0)

    for label, spec in learned.items():
        ax.plot(lam_np, spec.detach().cpu().numpy(), label=label, linewidth=2.0)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Spectral energy / learned F")
    ax.set_title("Empirical hotspot graph spectrum vs learned spectra")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def plot_empirical_vs_learned_spectra_logy(
    *,
    lam: torch.Tensor,
    empirical_energy: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    emp_np = np.clip(empirical_energy.detach().cpu().numpy(), 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(lam_np, emp_np, label="Empirical hotspot energy", linewidth=2.0)

    for label, spec in learned.items():
        spec_np = np.clip(spec.detach().cpu().numpy(), 1e-12, None)
        ax.plot(lam_np, spec_np, label=label, linewidth=2.0)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("log spectral energy / learned F")
    ax.set_title("Spectrum comparison (log-y scale)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_empirical_vs_learned_spectra_zoom(
    *,
    lam: torch.Tensor,
    empirical_energy: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
    lam_quantile: float = 0.15,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    emp_np = empirical_energy.detach().cpu().numpy()

    cutoff = float(np.quantile(lam_np, lam_quantile))
    mask = lam_np <= cutoff

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(lam_np[mask], emp_np[mask], label="Empirical hotspot energy", linewidth=2.0)

    for label, spec in learned.items():
        spec_np = spec.detach().cpu().numpy()
        ax.plot(lam_np[mask], spec_np[mask], label=label, linewidth=2.0)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Spectral energy / learned F")
    ax.set_title(f"Spectrum comparison (lowest {int(100 * lam_quantile)}% of eigenvalues)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class FitResult:
    filter_name: str
    variant: str
    label: str

    pll_vi: float
    pll_vi_per_test: float

    rmse_phi_vi: float
    rmse_eta_vi: float
    rmse_phi_vi_test: float
    rmse_eta_vi_test: float

    y_rmse_train: float
    y_rmse_test: float

    spectrum_mean_vi: torch.Tensor
    phi_hat_vi: torch.Tensor
    eta_hat_vi: torch.Tensor

# ---------------------------------------------------------------------
# Model fitting wrapper
# ---------------------------------------------------------------------
def fit_one_model(
    *,
    filter_name: str,
    variant: str,
    X: torch.Tensor,
    y: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    phi_true: torch.Tensor,
    eta_true: torch.Tensor,
    vi_iters: int,
    vi_mc: int,
    vi_lr: float,
    device: torch.device,
) -> FitResult:
    spec = get_filter_spec(filter_name)
    if variant not in spec.cases:
        raise ValueError(
            f"Case '{variant}' not defined for filter '{filter_name}'. "
            f"Available: {list(spec.cases.keys())}"
        )

    case_spec = spec.cases[variant]

    tr = torch.as_tensor(train_idx, device=device)
    X_tr = X[tr, :]
    y_tr = y[tr]
    U_tr = U[tr, :]

    tau2_init = float(torch.var(phi_true).detach().cpu())
    if tau2_init <= 0.0:
        tau2_init = 1.0
    eps_car = 1e-3

    filter_module = case_spec.build_filter(
        tau2_true=tau2_init,
        eps_car=eps_car,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **case_spec.fixed,
    )

    prior_V0 = 10.0 * torch.eye(X.shape[1], dtype=DTYPE, device=device)
    sigma2_init = float(torch.var(y_tr).detach().cpu()) * 0.1
    sigma2_init = max(sigma2_init, 1e-3)

    model = SpectralCAR_FullVI(
        X=X_tr,
        y=y_tr,
        lam=lam,
        U=U_tr,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_init),
        log_std_log_sigma2=-2.3,
        num_mc=vi_mc,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    print(f"\n[MODEL] Starting {filter_name}:{variant}")
    print(f"[MODEL] Training size: {len(train_idx)}")

    for it in range(vi_iters):
        opt.zero_grad()
        elbo, _ = model.elbo()
        loss = -elbo
        loss.backward()
        opt.step()

        if (it + 1) % 100 == 0 or it == 0:
            print(
                f"[VI] {filter_name}:{variant} | iter {it+1}/{vi_iters} | "
                f"ELBO = {elbo.item():.3f}"
            )
    print(f"[MODEL] Finished {filter_name}:{variant}\n")

    phi_hat_vi, F_vi_used, _ = compute_phi_vi_full(
        model=model,
        lam=lam,
        U_train=U_tr,
        U_full=U,
        X_train=X_tr,
        y_train=y_tr,
        mode="plugin",
        S=64,
    )

    beta_mean, _, _, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    eta_hat_vi = (X @ beta_mean) + phi_hat_vi

    rmse_phi_full = rmse(phi_hat_vi.cpu(), phi_true.cpu())
    rmse_eta_full = rmse(eta_hat_vi.cpu(), eta_true.cpu())

    rmse_phi_test = rmse_on_index(phi_hat_vi, phi_true, test_idx)
    rmse_eta_test = rmse_on_index(eta_hat_vi, eta_true, test_idx)

    y_rmse_train, y_rmse_test = summarize_residual_rmse(
        y=y,
        y_hat=eta_hat_vi,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    pll_vi = predictive_loglik_vi_heldout(
        model=model,
        y=y,
        X=X,
        U=U,
        lam=lam,
        train_idx=train_idx,
        test_idx=test_idx,
        S=16,
        use_plugin_beta=True,
        sample_sigma2=False,
    )
    pll_vi_per_test = pll_vi / float(len(test_idx))

    print(
        f"[RESULT] {filter_name}:{variant} | "
        f"PLL/test = {pll_vi_per_test:.4f} | "
        f"phi_RMSE(full) = {rmse_phi_full:.4f} | "
        f"eta_RMSE(full) = {rmse_eta_full:.4f} | "
        f"phi_RMSE(test) = {rmse_phi_test:.4f} | "
        f"eta_RMSE(test) = {rmse_eta_test:.4f} | "
        f"y_RMSE(train) = {y_rmse_train:.4f} | "
        f"y_RMSE(test) = {y_rmse_test:.4f}"
    )

    return FitResult(
        filter_name=filter_name,
        variant=variant,
        label=f"{filter_name}/{case_spec.display_name}",
        pll_vi=float(pll_vi),
        pll_vi_per_test=float(pll_vi_per_test),
        rmse_phi_vi=rmse_phi_full,
        rmse_eta_vi=rmse_eta_full,
        rmse_phi_vi_test=rmse_phi_test,
        rmse_eta_vi_test=rmse_eta_test,
        y_rmse_train=y_rmse_train,
        y_rmse_test=y_rmse_test,
        spectrum_mean_vi=F_vi_used.detach().cpu(),
        phi_hat_vi=phi_hat_vi.detach().cpu(),
        eta_hat_vi=eta_hat_vi.detach().cpu(),
    )


# ---------------------------------------------------------------------
# Leaderboard / serialization
# ---------------------------------------------------------------------
def write_leaderboard(results: list[FitResult], outpath: Path) -> None:
    lines = ["HOTSPOT DEMO SUMMARY"]
    for r in results:
        line = (
            f"{r.label:30s} | "
            f"PLL(VI)={r.pll_vi:8.2f}  PLL/test(VI)={r.pll_vi_per_test:8.4f} | "
            f"phi_RMSE(full)={r.rmse_phi_vi:7.4f}  eta_RMSE(full)={r.rmse_eta_vi:7.4f} | "
            f"phi_RMSE(test)={r.rmse_phi_vi_test:7.4f}  eta_RMSE(test)={r.rmse_eta_vi_test:7.4f} | "
            f"y_RMSE(train)={r.y_rmse_train:7.4f}  y_RMSE(test)={r.y_rmse_test:7.4f}"
        )
        lines.append(line)

    text = "\n".join(lines)
    outpath.write_text(text)
    print(text)


def save_metrics_json(
    *,
    results: list[FitResult],
    outpath: Path,
) -> None:
    payload: list[dict[str, Any]] = []
    for r in results:
        payload.append(
            {
                "filter_name": r.filter_name,
                "variant": r.variant,
                "label": r.label,
                "pll_vi": r.pll_vi,
                "pll_vi_per_test": r.pll_vi_per_test,
                "rmse_phi_vi_full": r.rmse_phi_vi,
                "rmse_eta_vi_full": r.rmse_eta_vi,
                "rmse_phi_vi_test": r.rmse_phi_vi_test,
                "rmse_eta_vi_test": r.rmse_eta_vi_test,
                "y_rmse_train": r.y_rmse_train,
                "y_rmse_test": r.y_rmse_test,
            }
        )
    outpath.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hotspot demo for SDM-CAR vs CAR baselines.")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.35)
    p.add_argument("--rho", type=float, default=0.95)

    p.add_argument("--beta0", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.4)
    p.add_argument("--normalize_phi_sd", type=float, default=1.0)

    # Mixed-sign, mixed-scale defaults for a stronger non-CAR hotspot truth
    p.add_argument("--centers", type=str, default="9,10;29,30;31,9")
    p.add_argument("--radii", type=str, default="1.6;3.8;1.8")
    p.add_argument("--amplitudes", type=str, default="2.6;1.7;-2.2")

    # Optional local ring feature
    p.add_argument("--add_ring", action="store_true")
    p.add_argument("--ring_center", type=str, default="16,24")
    p.add_argument("--ring_inner", type=float, default=1.6)
    p.add_argument("--ring_outer", type=float, default=3.8)
    p.add_argument("--ring_amplitude", type=float, default=1.25)

    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--fits",
        nargs="+",
        default=[
            "classic_car:baseline",
            "leroux:learn_rho",
            "rational:flex_21",
        ],
        help=f"Fits to run as filter:case tokens. Available filters: {available_filters()}",
    )
    p.add_argument("--vi_iters", type=int, default=1500)
    p.add_argument("--vi_mc", type=int, default=10)
    p.add_argument("--vi_lr", type=float, default=1e-2)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="examples/figures/hotspot_demo")
    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    set_default_dtype(DTYPE)
    set_seed(args.seed)

    device = torch.device(args.device)
    outdir = ensure_dir(args.outdir)

    print("\n[INFO] Starting Hotspot Demo")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Grid: {args.nx} x {args.ny}")
    print(f"[INFO] Fits: {args.fits}")
    print(f"[INFO] VI iters: {args.vi_iters}\n")

    coords = make_grid_2d_coords(args.nx, args.ny, device=device, dtype=DTYPE)
    L, _W = build_laplacian_from_knn(coords, k=args.k, gamma=args.gamma, rho=args.rho)
    lam, U = laplacian_eigendecomp(L)

    print("[INFO] Graph built")
    print(f"[INFO] Number of nodes: {coords.shape[0]}")
    print(f"[INFO] Eigenvalues range: [{lam.min().item():.4f}, {lam.max().item():.4f}]\n")

    centers = parse_centers(args.centers)
    radii = parse_float_list(args.radii)
    amplitudes = parse_float_list(args.amplitudes)
    ring_center = parse_point(args.ring_center)

    data = simulate_hotspot_dataset(
        coords=coords,
        beta0=args.beta0,
        sigma=args.sigma,
        centers=centers,
        radii=radii,
        amplitudes=amplitudes,
        normalize_phi_sd=args.normalize_phi_sd,
        device=device,
        add_ring=args.add_ring,
        ring_center=ring_center,
        ring_inner=args.ring_inner,
        ring_outer=args.ring_outer,
        ring_amplitude=args.ring_amplitude,
    )

    X = data["X"]
    y = data["y"]
    phi_true = data["phi_true"]
    eta_true = data["eta_true"]

    print("[INFO] Data generated")
    print(f"[INFO] y mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")
    print(f"[INFO] phi_true std: {phi_true.std().item():.4f}\n")

    empirical_energy = empirical_hotspot_spectrum(U, phi_true)

    emp_summary = spectral_band_summary(lam, empirical_energy)
    print_spectral_summary("empirical_hotspot", emp_summary)
    (outdir / "empirical_spectral_summary.json").write_text(json.dumps(emp_summary, indent=2))

    print_spectral_band_energy(lam, empirical_energy)

    train_idx, test_idx = make_split(n=X.shape[0], test_frac=args.test_frac, seed=args.seed)

    print("[INFO] Train/Test split")
    print(f"[INFO] Train size: {len(train_idx)}")
    print(f"[INFO] Test size: {len(test_idx)}\n")

    save_heatmap(
        phi_true,
        nx=args.nx,
        ny=args.ny,
        title="True hotspot field",
        outpath=outdir / "true_phi.png",
    )
    save_heatmap(
        y,
        nx=args.nx,
        ny=args.ny,
        title="Observed y",
        outpath=outdir / "observed_y.png",
    )

    fit_specs = parse_fit_strings(args.fits)
    results: list[FitResult] = []

    for filter_name, variant in fit_specs:
        model_dir = ensure_dir(outdir / f"{filter_name}__{variant}")

        res = fit_one_model(
            filter_name=filter_name,
            variant=variant,
            X=X,
            y=y,
            U=U,
            lam=lam,
            train_idx=train_idx,
            test_idx=test_idx,
            phi_true=phi_true,
            eta_true=eta_true,
            vi_iters=args.vi_iters,
            vi_mc=args.vi_mc,
            vi_lr=args.vi_lr,
            device=device,
        )
        results.append(res)

        save_heatmap(
            res.phi_hat_vi,
            nx=args.nx,
            ny=args.ny,
            title=f"{res.label} phi_hat (VI)",
            outpath=model_dir / "phi_hat_vi.png",
        )
        save_heatmap(
            res.eta_hat_vi,
            nx=args.nx,
            ny=args.ny,
            title=f"{res.label} eta_hat (VI)",
            outpath=model_dir / "eta_hat_vi.png",
        )

        plot_empirical_vs_learned_spectra(
            lam=lam,
            empirical_energy=empirical_energy,
            learned={"VI mean": res.spectrum_mean_vi},
            outpath=model_dir / "spectrum_empirical_vs_learned.png",
        )

        plot_empirical_vs_learned_spectra_logy(
            lam=lam,
            empirical_energy=empirical_energy,
            learned={"VI mean": res.spectrum_mean_vi},
            outpath=model_dir / "spectrum_empirical_vs_learned_logy.png",
        )

        plot_empirical_vs_learned_spectra_zoom(
            lam=lam,
            empirical_energy=empirical_energy,
            learned={"VI mean": res.spectrum_mean_vi},
            outpath=model_dir / "spectrum_empirical_vs_learned_zoom.png",
            lam_quantile=0.15,
        )

    compare_dir = ensure_dir(outdir / "COMPARE")

    phi_hats_vi = {r.label: r.phi_hat_vi for r in results}
    plot_compare_fields(
        phi_true=phi_true,
        phi_hats=phi_hats_vi,
        nx=args.nx,
        ny=args.ny,
        outpath=compare_dir / "phi_compare_vi.png",
    )

    learned_vi = {r.label: r.spectrum_mean_vi for r in results}
    plot_empirical_vs_learned_spectra(
        lam=lam,
        empirical_energy=empirical_energy,
        learned=learned_vi,
        outpath=compare_dir / "spectrum_compare_vi.png",
    )

    plot_empirical_vs_learned_spectra_logy(
        lam=lam,
        empirical_energy=empirical_energy,
        learned=learned_vi,
        outpath=compare_dir / "spectrum_compare_vi_logy.png",
    )

    plot_empirical_vs_learned_spectra_zoom(
        lam=lam,
        empirical_energy=empirical_energy,
        learned=learned_vi,
        outpath=compare_dir / "spectrum_compare_vi_zoom.png",
        lam_quantile=0.15,
    )

    write_leaderboard(results, outdir / "leaderboard.txt")
    save_metrics_json(results=results, outpath=outdir / "metrics.json")

    config = {
        "nx": args.nx,
        "ny": args.ny,
        "k": args.k,
        "gamma": args.gamma,
        "rho": args.rho,
        "beta0": args.beta0,
        "sigma": args.sigma,
        "normalize_phi_sd": args.normalize_phi_sd,
        "centers": centers,
        "radii": radii,
        "amplitudes": amplitudes,
        "add_ring": args.add_ring,
        "ring_center": ring_center,
        "ring_inner": args.ring_inner,
        "ring_outer": args.ring_outer,
        "ring_amplitude": args.ring_amplitude,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "fits": args.fits,
        "vi_iters": args.vi_iters,
        "vi_mc": args.vi_mc,
        "vi_lr": args.vi_lr,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    print("\n[INFO] Hotspot demo completed successfully!")
    print(f"[INFO] Results saved to: {outdir}\n")


if __name__ == "__main__":
    main()