from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def log_spectrum_rmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = torch.log(a.clamp_min(eps))
    bb = torch.log(b.clamp_min(eps))
    return float(torch.sqrt(torch.mean((aa - bb) ** 2)).item())


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


# ---------------------------------------------------------------------
# Hard-coded matched spectral family
# ---------------------------------------------------------------------
def true_spectral_density(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (3.0 + 2.0 * x).pow(4)


class SimplePowerDecayFilter(nn.Module):
    """
    Variational filter family

        F_theta(x) = tau2 / (t1 + t2 x)^4

    with t1 > 0, t2 > 0, and tau2 > 0.
    """
    def __init__(
        self,
        *,
        mu_t1_raw: float = math.log(math.exp(3.0) - 1.0),
        mu_t2_raw: float = math.log(math.exp(2.0) - 1.0),
        log_std0: float = -2.3,
    ):
        super().__init__()
        self.mu_t1_raw = nn.Parameter(torch.tensor(float(mu_t1_raw), dtype=DTYPE))
        self.mu_t2_raw = nn.Parameter(torch.tensor(float(mu_t2_raw), dtype=DTYPE))
        self.log_std_t1_raw = nn.Parameter(torch.tensor(float(log_std0), dtype=DTYPE))
        self.log_std_t2_raw = nn.Parameter(torch.tensor(float(log_std0), dtype=DTYPE))

        self.mu_log_tau2 = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor(float(log_std0), dtype=DTYPE))

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        eps1 = torch.randn((), dtype=DTYPE, device=self.mu_t1_raw.device)
        eps2 = torch.randn((), dtype=DTYPE, device=self.mu_t2_raw.device)
        eps3 = torch.randn((), dtype=DTYPE, device=self.mu_t1_raw.device)

        t1_raw = self.mu_t1_raw + torch.exp(self.log_std_t1_raw) * eps1
        t2_raw = self.mu_t2_raw + torch.exp(self.log_std_t2_raw) * eps2
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps3

        return {
            "t1_raw": t1_raw,
            "t2_raw": t2_raw,
            "log_tau2": log_tau2,
        }

    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        return {
            "t1_raw": self.mu_t1_raw,
            "t2_raw": self.mu_t2_raw,
            "log_tau2": self.mu_log_tau2,
        }

    def posterior_mean_unconstrained(self) -> dict[str, torch.Tensor]:
        return self.mean_unconstrained()

    def posterior_mean_constrained(self) -> dict[str, torch.Tensor]:
        return self._constrain(self.mean_unconstrained())

    def constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        t1 = F.softplus(theta["t1_raw"]).clamp_min(1e-8).reshape(())
        t2 = F.softplus(theta["t2_raw"]).clamp_min(1e-8).reshape(())
        tau2 = torch.exp(theta["log_tau2"]).reshape(())
        return {"t1": t1, "t2": t2, "tau2": tau2}

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.constrain(theta)

    def spectrum(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.constrain(theta)
        t1 = c["t1"]
        t2 = c["t2"]
        tau2 = c["tau2"]

        lam_max = lam.max().clamp_min(1e-12)
        x = (lam / lam_max).clamp(0.0, 1.0)
        return tau2 / (t1 + t2 * x).pow(4)

    def kl_q_p(self) -> torch.Tensor:
        mu1 = self.mu_t1_raw
        mu2 = self.mu_t2_raw
        mu3 = self.mu_log_tau2

        s1 = torch.exp(self.log_std_t1_raw)
        s2 = torch.exp(self.log_std_t2_raw)
        s3 = torch.exp(self.log_std_log_tau2)

        kl1 = 0.5 * (mu1.pow(2) + s1.pow(2) - 1.0 - 2.0 * self.log_std_t1_raw)
        kl2 = 0.5 * (mu2.pow(2) + s2.pow(2) - 1.0 - 2.0 * self.log_std_t2_raw)
        kl3 = 0.5 * (mu3.pow(2) + s3.pow(2) - 1.0 - 2.0 * self.log_std_log_tau2)

        return kl1 + kl2 + kl3


# ---------------------------------------------------------------------
# Truth generation
# ---------------------------------------------------------------------
def simulate_hardcoded_truth(
    U: torch.Tensor,
    lam: torch.Tensor,
    *,
    beta0: float,
    sigma: float,
    device: torch.device,
    normalize_phi_sd: float | None = None,
) -> dict[str, torch.Tensor]:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n = U.shape[0]
    X = torch.ones((n, 1), dtype=DTYPE, device=device)
    beta_true = torch.tensor([beta0], dtype=DTYPE, device=device)

    lam_max = lam.max().clamp_min(1e-12)
    x = (lam / lam_max).clamp(0.0, 1.0)
    F_true = true_spectral_density(x).clamp_min(1e-12)

    z_true = torch.sqrt(F_true) * torch.randn(n, dtype=DTYPE, device=device)
    phi_true = U @ z_true
    phi_true = phi_true - torch.mean(phi_true)

    if normalize_phi_sd is not None:
        sd = torch.std(phi_true)
        if float(sd) > 0.0:
            scale = normalize_phi_sd / sd
            phi_true = phi_true * scale
            z_true = U.T @ phi_true
            F_true = F_true * (scale ** 2)

    eta_true = (X @ beta_true).reshape(-1) + phi_true
    y = eta_true + sigma * torch.randn(n, dtype=DTYPE, device=device)
    empirical_energy = z_true ** 2

    return {
        "X": X,
        "y": y,
        "phi_true": phi_true,
        "eta_true": eta_true,
        "beta_true": beta_true,
        "z_true": z_true,
        "F_true": F_true,
        "empirical_energy": empirical_energy,
    }


@torch.no_grad()
def monte_carlo_mean_true_energy(lam: torch.Tensor, *, R: int) -> tuple[torch.Tensor, torch.Tensor]:
    lam_max = lam.max().clamp_min(1e-12)
    x = (lam / lam_max).clamp(0.0, 1.0)
    F_true = true_spectral_density(x).clamp_min(1e-12)
    acc = torch.zeros_like(F_true)
    for _ in range(int(R)):
        z = torch.sqrt(F_true) * torch.randn_like(F_true)
        acc += z ** 2
    return F_true, acc / float(R)


# ---------------------------------------------------------------------
# Spectrum / predictive helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def spectrum_vi_mc_mean(filter_module, lam: torch.Tensor, *, S: int = 256) -> torch.Tensor:
    acc = torch.zeros_like(lam)
    for _ in range(S):
        th = filter_module.sample_unconstrained()
        acc += filter_module.spectrum(lam, th).clamp_min(1e-12)
    return acc / float(S)


@torch.no_grad()
def filter_parameter_mc_mean(filter_module, *, S: int = 1024) -> dict[str, float]:
    acc_t1 = 0.0
    acc_t2 = 0.0
    acc_tau2 = 0.0

    for _ in range(S):
        th = filter_module.sample_unconstrained()
        c = filter_module.constrain(th)
        acc_t1 += float(c["t1"].detach().cpu())
        acc_t2 += float(c["t2"].detach().cpu())
        acc_tau2 += float(c["tau2"].detach().cpu())

    return {
        "t1_mc_mean": acc_t1 / float(S),
        "t2_mc_mean": acc_t2 / float(S),
        "tau2_mc_mean": acc_tau2 / float(S),
    }

def bandwise_energy_diagnostic(
    lam: torch.Tensor,
    F_true: torch.Tensor,
    F_hat: torch.Tensor,
) -> dict[str, float]:
    lam_cpu = lam.detach().cpu()
    F_true_cpu = F_true.detach().cpu()
    F_hat_cpu = F_hat.detach().cpu()

    q1 = torch.quantile(lam_cpu, 1.0 / 3.0)
    q2 = torch.quantile(lam_cpu, 2.0 / 3.0)

    low = lam_cpu <= q1
    mid = (lam_cpu > q1) & (lam_cpu <= q2)
    high = lam_cpu > q2

    def safe_sum(x: torch.Tensor, mask: torch.Tensor) -> float:
        return float(x[mask].sum().item())

    low_true = safe_sum(F_true_cpu, low)
    mid_true = safe_sum(F_true_cpu, mid)
    high_true = safe_sum(F_true_cpu, high)

    low_hat = safe_sum(F_hat_cpu, low)
    mid_hat = safe_sum(F_hat_cpu, mid)
    high_hat = safe_sum(F_hat_cpu, high)

    total_true = float(F_true_cpu.sum().item())
    total_hat = float(F_hat_cpu.sum().item())

    def safe_div(a: float, b: float) -> float:
        return a / b if abs(b) > 1e-12 else float("nan")

    return {
        "low_band_frac_true": safe_div(low_true, total_true),
        "mid_band_frac_true": safe_div(mid_true, total_true),
        "high_band_frac_true": safe_div(high_true, total_true),
        "low_band_frac_hat": safe_div(low_hat, total_hat),
        "mid_band_frac_hat": safe_div(mid_hat, total_hat),
        "high_band_frac_hat": safe_div(high_hat, total_hat),
        "low_band_ratio_hat_to_true": safe_div(low_hat, low_true),
        "mid_band_ratio_hat_to_true": safe_div(mid_hat, mid_true),
        "high_band_ratio_hat_to_true": safe_div(high_hat, high_true),
        "low_band_mass_true": low_true,
        "mid_band_mass_true": mid_true,
        "high_band_mass_true": high_true,
        "low_band_mass_hat": low_hat,
        "mid_band_mass_hat": mid_hat,
        "high_band_mass_hat": high_hat,
    }

def _k_blocks_from_spectrum(
    U: torch.Tensor,
    F_spec: torch.Tensor,
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
    F_row = F_spec.to(device=device, dtype=dtype).clamp_min(1e-12).reshape(1, -1)

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
    F_spec: torch.Tensor,
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

    K_tt, K_st, K_ss = _k_blocks_from_spectrum(U, F_spec, sigma2, train_idx, test_idx)

    L = torch.linalg.cholesky(K_tt)
    alpha = torch.cholesky_solve(r_tr.reshape(-1, 1), L).reshape(-1)
    cond_mean = mu_te + (K_st @ alpha)

    V = torch.cholesky_solve(K_st.T, L)
    cond_cov = K_ss - (K_st @ V)
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    cond_cov = cond_cov + 1e-8 * torch.eye(cond_cov.shape[0], device=device, dtype=dtype)

    e = (y_te - cond_mean).reshape(-1, 1)
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
) -> float:
    m_beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta = m_beta_plugin.reshape(-1)

    logliks = []
    for _ in range(int(S)):
        theta = model.filter.sample_unconstrained()
        F_spec = model.filter.spectrum(lam, theta).clamp_min(1e-12)
        sigma2 = float(sigma2_plugin.detach().cpu())
        ll = conditional_predictive_loglik(
            y=y,
            X=X,
            beta=beta,
            U=U,
            F_spec=F_spec,
            sigma2=sigma2,
            train_idx=train_idx,
            test_idx=test_idx,
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
    F_spec: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    r_tr = y_train - X_train @ beta
    Ut_r = U_train.T @ r_tr
    shrink = (F_spec / (F_spec + sigma2)).clamp(0.0, 1.0)
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
            F_spec=F_used.to(lam.device),
            sigma2=sigma2,
        )
        return phi, F_used.detach(), sigma2

    if mode == "posterior":
        acc_phi = torch.zeros(U_full.shape[0], device=lam.device, dtype=lam.dtype)
        acc_F = torch.zeros_like(lam)
        for _ in range(int(S)):
            th = model.filter.sample_unconstrained()
            F_spec = model.filter.spectrum(lam, th).clamp_min(1e-12)
            phi = phi_full_from_train(
                U_train=U_train,
                U_full=U_full,
                X_train=X_train,
                y_train=y_train,
                beta=beta_mean,
                F_spec=F_spec,
                sigma2=sigma2,
            )
            acc_phi += phi
            acc_F += F_spec
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


def plot_true_vs_empirical(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    empirical_energy: torch.Tensor,
    outpath: Path,
    title: str,
    logy: bool = False,
    use_markers: bool = True,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    F_np = F_true.detach().cpu().numpy()
    e_np = empirical_energy.detach().cpu().numpy()

    if logy:
        F_np = np.clip(F_np, 1e-12, None)
        e_np = np.clip(e_np, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if use_markers:
        ax.plot(
            lam_np, F_np,
            marker="o", linestyle="none",
            label="True $F_0(x)$", markersize=4, alpha=0.9,
        )
        ax.plot(
            lam_np, e_np,
            marker="x", linestyle="none",
            label="Empirical energy $z^2$", markersize=4, alpha=0.7,
        )
    else:
        ax.plot(lam_np, F_np, label="True $F_0(x)$", linewidth=2.5)
        ax.plot(lam_np, e_np, label="Empirical energy $z^2$", linewidth=1.8)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Spectral value")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_true_vs_learned_spectra(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
    title: str,
    logy: bool = False,
    use_markers: bool = True,
) -> None:
    lam_np = lam.detach().cpu().numpy()
    true_np = F_true.detach().cpu().numpy()

    if logy:
        true_np = np.clip(true_np, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if use_markers:
        ax.plot(
            lam_np, true_np,
            marker="o", linestyle="none",
            label="True $F_0(x)$", markersize=4,
        )
    else:
        ax.plot(lam_np, true_np, label="True $F_0(x)$", linewidth=2.5)

    for label, spec in learned.items():
        spec_np = spec.detach().cpu().numpy()
        if logy:
            spec_np = np.clip(spec_np, 1e-12, None)

        if use_markers:
            ax.plot(
                lam_np, spec_np,
                marker="x", linestyle="none",
                label=label, markersize=4, alpha=0.8,
            )
        else:
            ax.plot(lam_np, spec_np, label=label, linewidth=2.0)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Spectral value")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Fit wrapper
# ---------------------------------------------------------------------
def fit_model(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    phi_true: torch.Tensor,
    eta_true: torch.Tensor,
    sigma_true: float,
    vi_iters: int,
    vi_mc: int,
    vi_lr: float,
    device: torch.device,
) -> tuple[SpectralCAR_FullVI, dict]:
    tr = torch.as_tensor(train_idx, device=device)
    X_tr = X[tr, :]
    y_tr = y[tr]
    U_tr = U[tr, :]

    filter_module = SimplePowerDecayFilter()

    prior_V0 = 10.0 * torch.eye(X.shape[1], dtype=DTYPE, device=device)

    #sigma2_init = max(float(torch.var(y_tr).detach().cpu()) * 0.25, 1e-3)
    sigma2_init = sigma_true ** 2

    model = SpectralCAR_FullVI(
        X=X_tr,
        y=y_tr,
        lam=lam,
        U=U_tr,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_init),
        log_std_log_sigma2=-1.5,
        num_mc=vi_mc,
    ).to(device)

    print(f"[DEBUG] Learning sigma^2 freely")
    print(f"[DEBUG] sigma_true^2 = {sigma_true ** 2:.6f}")
    print(f"[DEBUG] sigma2_init  = {sigma2_init:.6f}")

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    for it in range(vi_iters):
        opt.zero_grad()
        elbo, _ = model.elbo()
        loss = -elbo
        loss.backward()
        opt.step()

        if (it + 1) % 100 == 0 or it == 0:
            with torch.no_grad():
                _, _, sigma2_plugin, _ = model.beta_posterior_plugin()
                sigma2_now = float(sigma2_plugin.detach().cpu())
                sigma_now = math.sqrt(max(sigma2_now, 1e-12))
            print(
                f"[VI] iter {it+1}/{vi_iters} | "
                f"ELBO={elbo.item():.3f} | "
                f"sigma2_plugin={sigma2_now:.6f} | sigma_plugin={sigma_now:.6f}"
            )

    phi_hat_plugin, F_vi_plugin, sigma2_plugin = compute_phi_vi_full(
        model=model,
        lam=lam,
        U_train=U_tr,
        U_full=U,
        X_train=X_tr,
        y_train=y_tr,
        mode="plugin",
        S=64,
    )
    phi_hat_post, F_vi_post, sigma2_post = compute_phi_vi_full(
        model=model,
        lam=lam,
        U_train=U_tr,
        U_full=U,
        X_train=X_tr,
        y_train=y_tr,
        mode="posterior",
        S=128,
    )

    beta_mean, _, sigma2_plugin_from_beta, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2_plugin_scalar = float(sigma2_plugin_from_beta.detach().cpu())

    eta_hat_plugin = (X @ beta_mean) + phi_hat_plugin
    eta_hat_post = (X @ beta_mean) + phi_hat_post

    y_rmse_train_plugin, y_rmse_test_plugin = summarize_residual_rmse(
        y, eta_hat_plugin, train_idx, test_idx
    )
    y_rmse_train_post, y_rmse_test_post = summarize_residual_rmse(
        y, eta_hat_post, train_idx, test_idx
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
    )

    constrained_plugin = model.filter.posterior_mean_constrained()
    parameter_mc = filter_parameter_mc_mean(model.filter, S=2048)

    metrics = {
        "pll_vi_per_test": pll_vi / float(len(test_idx)),
        "phi_rmse_plugin": rmse(phi_hat_plugin.cpu(), phi_true.cpu()),
        "phi_rmse_post": rmse(phi_hat_post.cpu(), phi_true.cpu()),
        "eta_rmse_plugin": rmse(eta_hat_plugin.cpu(), eta_true.cpu()),
        "eta_rmse_post": rmse(eta_hat_post.cpu(), eta_true.cpu()),
        "y_rmse_train_plugin": y_rmse_train_plugin,
        "y_rmse_test_plugin": y_rmse_test_plugin,
        "y_rmse_train_post": y_rmse_train_post,
        "y_rmse_test_post": y_rmse_test_post,
        "logF_rmse_plugin": None,
        "logF_rmse_post": None,
        "t1_plugin": float(constrained_plugin["t1"].detach().cpu()),
        "t2_plugin": float(constrained_plugin["t2"].detach().cpu()),
        "tau2_plugin": float(constrained_plugin["tau2"].detach().cpu()),
        "t1_mc_mean": parameter_mc["t1_mc_mean"],
        "t2_mc_mean": parameter_mc["t2_mc_mean"],
        "tau2_mc_mean": parameter_mc["tau2_mc_mean"],
        "sigma_true": float(sigma_true),
        "sigma2_true": float(sigma_true ** 2),
        "sigma_plugin": math.sqrt(max(sigma2_plugin_scalar, 1e-12)),
        "sigma2_plugin": sigma2_plugin_scalar,
        "abs_err_sigma_plugin": abs(math.sqrt(max(sigma2_plugin_scalar, 1e-12)) - float(sigma_true)),
        "abs_err_sigma2_plugin": abs(sigma2_plugin_scalar - float(sigma_true ** 2)),
        "spectrum_mean_vi_plugin": F_vi_plugin.detach().cpu(),
        "spectrum_mean_vi_post": F_vi_post.detach().cpu(),
        "phi_hat_vi_plugin": phi_hat_plugin.detach().cpu(),
        "phi_hat_vi_post": phi_hat_post.detach().cpu(),
    }
    return model, metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matched-family spectral recovery demo with learned sigma.")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.35)
    p.add_argument("--rho", type=float, default=0.95)

    p.add_argument("--beta0", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--normalize_phi_sd", type=float, default=None)
    p.add_argument("--mc_reps", type=int, default=500)

    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--vi_iters", type=int, default=2000)
    p.add_argument("--vi_mc", type=int, default=10)
    p.add_argument("--vi_lr", type=float, default=5e-3)

    p.add_argument("--use_markers", action="store_true")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--outdir",
        type=str,
        default="examples/figures/hardcoded_power_decay_recovery_learn_sigma",
    )
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

    print("\n[INFO] Starting matched-family recovery demo with learned sigma")
    print("[INFO] True spectral density: F0(x) = 1 / (3 + 2x)^4")
    print("[INFO] Fitted family: F_theta(x) = tau2 / (t1 + t2 x)^4")
    print("[INFO] sigma^2 is learned, not fixed")

    coords = make_grid_2d_coords(args.nx, args.ny, device=device, dtype=DTYPE)
    L, _W = build_laplacian_from_knn(coords, k=args.k, gamma=args.gamma, rho=args.rho)
    lam, U = laplacian_eigendecomp(L)

    data = simulate_hardcoded_truth(
        U=U,
        lam=lam,
        beta0=args.beta0,
        sigma=args.sigma,
        device=device,
        normalize_phi_sd=args.normalize_phi_sd,
    )

    X = data["X"]
    y = data["y"]
    phi_true = data["phi_true"]
    eta_true = data["eta_true"]
    F_true = data["F_true"]
    empirical_energy = data["empirical_energy"]

    lam_max = lam.max().clamp_min(1e-12)
    x = (lam / lam_max).clamp(0.0, 1.0)
    F_formula = 1.0 / (3.0 + 2.0 * x).pow(4)
    z_from_phi = U.T @ phi_true
    empirical_from_phi = z_from_phi ** 2

    print("\n[DEBUG] Truth diagnostics")
    print(f"mean(F_true)           = {float(F_true.mean()):.6e}")
    print(f"mean(F_formula)        = {float(F_formula.mean()):.6e}")
    print(f"mean(empirical_energy) = {float(empirical_energy.mean()):.6e}")
    print(f"mean(z_from_phi^2)     = {float(empirical_from_phi.mean()):.6e}")
    print(f"phi_true std           = {float(phi_true.std()):.6e}")
    print(f"sigma_true             = {float(args.sigma):.6e}")
    print(f"sigma2_true            = {float(args.sigma ** 2):.6e}")

    F_true_mc, mc_mean_energy = monte_carlo_mean_true_energy(lam, R=args.mc_reps)

    plot_true_vs_empirical(
        lam=lam,
        F_true=F_true,
        empirical_energy=empirical_energy,
        outpath=outdir / "truth_vs_empirical.png",
        title="True spectrum vs empirical energy",
        logy=False,
        use_markers=args.use_markers,
    )
    plot_true_vs_empirical(
        lam=lam,
        F_true=F_true,
        empirical_energy=empirical_energy,
        outpath=outdir / "truth_vs_empirical_logy.png",
        title="True spectrum vs empirical energy (log-y)",
        logy=True,
        use_markers=args.use_markers,
    )
    plot_true_vs_empirical(
        lam=lam,
        F_true=F_true_mc,
        empirical_energy=mc_mean_energy,
        outpath=outdir / "truth_vs_mc_mean.png",
        title=f"True spectrum vs MC mean energy (R={args.mc_reps})",
        logy=False,
        use_markers=args.use_markers,
    )
    plot_true_vs_empirical(
        lam=lam,
        F_true=F_true_mc,
        empirical_energy=mc_mean_energy,
        outpath=outdir / "truth_vs_mc_mean_logy.png",
        title=f"True spectrum vs MC mean energy (R={args.mc_reps}, log-y)",
        logy=True,
        use_markers=args.use_markers,
    )

    save_heatmap(
        phi_true,
        nx=args.nx,
        ny=args.ny,
        title="True spatial field $\\phi$",
        outpath=outdir / "true_phi.png",
    )
    save_heatmap(
        y,
        nx=args.nx,
        ny=args.ny,
        title="Observed y",
        outpath=outdir / "observed_y.png",
    )

    train_idx, test_idx = make_split(n=X.shape[0], test_frac=args.test_frac, seed=args.seed)
    model, metrics = fit_model(
        X=X,
        y=y,
        U=U,
        lam=lam,
        train_idx=train_idx,
        test_idx=test_idx,
        phi_true=phi_true,
        eta_true=eta_true,
        sigma_true=args.sigma,
        vi_iters=args.vi_iters,
        vi_mc=args.vi_mc,
        vi_lr=args.vi_lr,
        device=device,
    )

    print("\n[DEBUG] Learned parameter summaries")
    print(f"t1_plugin    = {metrics['t1_plugin']:.6f}")
    print(f"t2_plugin    = {metrics['t2_plugin']:.6f}")
    print(f"tau2_plugin  = {metrics['tau2_plugin']:.6f}")
    print(f"t1_mc_mean   = {metrics['t1_mc_mean']:.6f}")
    print(f"t2_mc_mean   = {metrics['t2_mc_mean']:.6f}")
    print(f"tau2_mc_mean = {metrics['tau2_mc_mean']:.6f}")
    print(f"sigma_plugin = {metrics['sigma_plugin']:.6f}")
    print(f"sigma2_plugin = {metrics['sigma2_plugin']:.6f}")

    metrics["logF_rmse_plugin"] = log_spectrum_rmse(metrics["spectrum_mean_vi_plugin"].to(F_true.device), F_true)
    metrics["logF_rmse_post"] = log_spectrum_rmse(metrics["spectrum_mean_vi_post"].to(F_true.device), F_true)

    band_plugin = bandwise_energy_diagnostic(
        lam=lam,
        F_true=F_true,
        F_hat=metrics["spectrum_mean_vi_plugin"].to(F_true.device),
    )
    band_post = bandwise_energy_diagnostic(
        lam=lam,
        F_true=F_true,
        F_hat=metrics["spectrum_mean_vi_post"].to(F_true.device),
    )

    ratio_plugin = metrics["spectrum_mean_vi_plugin"].to(F_true.device) / F_true
    ratio_post = metrics["spectrum_mean_vi_post"].to(F_true.device) / F_true

    print("\n[DEBUG] Spectrum ratios")
    print(f"plugin ratio mean = {float(ratio_plugin.mean()):.6f}")
    print(f"plugin ratio std  = {float(ratio_plugin.std()):.6f}")
    print(f"post ratio mean   = {float(ratio_post.mean()):.6f}")
    print(f"post ratio std    = {float(ratio_post.std()):.6f}")

    print("\n[DEBUG] Bandwise energy diagnostic (plugin)")
    print(f"low  frac true={band_plugin['low_band_frac_true']:.4f} | hat={band_plugin['low_band_frac_hat']:.4f} | ratio={band_plugin['low_band_ratio_hat_to_true']:.4f}")
    print(f"mid  frac true={band_plugin['mid_band_frac_true']:.4f} | hat={band_plugin['mid_band_frac_hat']:.4f} | ratio={band_plugin['mid_band_ratio_hat_to_true']:.4f}")
    print(f"high frac true={band_plugin['high_band_frac_true']:.4f} | hat={band_plugin['high_band_frac_hat']:.4f} | ratio={band_plugin['high_band_ratio_hat_to_true']:.4f}")

    print("\n[DEBUG] Bandwise energy diagnostic (posterior)")
    print(f"low  frac true={band_post['low_band_frac_true']:.4f} | hat={band_post['low_band_frac_hat']:.4f} | ratio={band_post['low_band_ratio_hat_to_true']:.4f}")
    print(f"mid  frac true={band_post['mid_band_frac_true']:.4f} | hat={band_post['mid_band_frac_hat']:.4f} | ratio={band_post['mid_band_ratio_hat_to_true']:.4f}")
    print(f"high frac true={band_post['high_band_frac_true']:.4f} | hat={band_post['high_band_frac_hat']:.4f} | ratio={band_post['high_band_ratio_hat_to_true']:.4f}")

    plot_true_vs_learned_spectra(
        lam=lam,
        F_true=F_true,
        learned={
            "VI plugin": metrics["spectrum_mean_vi_plugin"],
            "VI posterior mean": metrics["spectrum_mean_vi_post"],
        },
        outpath=outdir / "true_vs_learned.png",
        title="True vs learned spectrum",
        logy=False,
        use_markers=args.use_markers,
    )
    plot_true_vs_learned_spectra(
        lam=lam,
        F_true=F_true,
        learned={
            "VI plugin": metrics["spectrum_mean_vi_plugin"],
            "VI posterior mean": metrics["spectrum_mean_vi_post"],
        },
        outpath=outdir / "true_vs_learned_logy.png",
        title="True vs learned spectrum (log-y)",
        logy=True,
        use_markers=args.use_markers,
    )

    save_heatmap(
        metrics["phi_hat_vi_plugin"],
        nx=args.nx,
        ny=args.ny,
        title="Recovered $\\phi$ (VI plugin)",
        outpath=outdir / "phi_hat_vi_plugin.png",
    )
    save_heatmap(
        metrics["phi_hat_vi_post"],
        nx=args.nx,
        ny=args.ny,
        title="Recovered $\\phi$ (VI posterior mean)",
        outpath=outdir / "phi_hat_vi_post.png",
    )

    summary = {
        "truth": {
            "t1_true": 3.0,
            "t2_true": 2.0,
            "tau2_true": 1.0,
            "sigma_true": float(args.sigma),
            "sigma2_true": float(args.sigma ** 2),
            "formula": "F0(x) = 1 / (3 + 2x)^4",
            "fitted_family": "F_theta(x) = tau2 / (t1 + t2 x)^4",
            "noise_learning": "sigma2 learned",
        },
        "recovery": {
            "t1_plugin": metrics["t1_plugin"],
            "t2_plugin": metrics["t2_plugin"],
            "tau2_plugin": metrics["tau2_plugin"],
            "t1_mc_mean": metrics["t1_mc_mean"],
            "t2_mc_mean": metrics["t2_mc_mean"],
            "tau2_mc_mean": metrics["tau2_mc_mean"],
            "sigma_plugin": metrics["sigma_plugin"],
            "sigma2_plugin": metrics["sigma2_plugin"],
            "abs_err_t1_plugin": abs(metrics["t1_plugin"] - 3.0),
            "abs_err_t2_plugin": abs(metrics["t2_plugin"] - 2.0),
            "abs_err_tau2_plugin": abs(metrics["tau2_plugin"] - 1.0),
            "abs_err_t1_mc_mean": abs(metrics["t1_mc_mean"] - 3.0),
            "abs_err_t2_mc_mean": abs(metrics["t2_mc_mean"] - 2.0),
            "abs_err_tau2_mc_mean": abs(metrics["tau2_mc_mean"] - 1.0),
            "abs_err_sigma_plugin": metrics["abs_err_sigma_plugin"],
            "abs_err_sigma2_plugin": metrics["abs_err_sigma2_plugin"],
        },
        "metrics": {
            "pll_vi_per_test": metrics["pll_vi_per_test"],
            "phi_rmse_plugin": metrics["phi_rmse_plugin"],
            "phi_rmse_post": metrics["phi_rmse_post"],
            "eta_rmse_plugin": metrics["eta_rmse_plugin"],
            "eta_rmse_post": metrics["eta_rmse_post"],
            "y_rmse_train_plugin": metrics["y_rmse_train_plugin"],
            "y_rmse_test_plugin": metrics["y_rmse_test_plugin"],
            "y_rmse_train_post": metrics["y_rmse_train_post"],
            "y_rmse_test_post": metrics["y_rmse_test_post"],
            "logF_rmse_plugin": metrics["logF_rmse_plugin"],
            "logF_rmse_post": metrics["logF_rmse_post"],
            "bandwise_plugin": band_plugin,
            "bandwise_post": band_post,
        },
        "config": {
            "nx": args.nx,
            "ny": args.ny,
            "k": args.k,
            "gamma": args.gamma,
            "rho": args.rho,
            "beta0": args.beta0,
            "sigma": args.sigma,
            "normalize_phi_sd": args.normalize_phi_sd,
            "mc_reps": args.mc_reps,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "vi_iters": args.vi_iters,
            "vi_mc": args.vi_mc,
            "vi_lr": args.vi_lr,
            "use_markers": bool(args.use_markers),
        },
    }

    (outdir / "recovery_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[RESULT] Parameter recovery")
    print(
        f"  true t1 = 3.0000 | "
        f"plugin = {metrics['t1_plugin']:.4f} | "
        f"MC mean = {metrics['t1_mc_mean']:.4f}"
    )
    print(
        f"  true t2 = 2.0000 | "
        f"plugin = {metrics['t2_plugin']:.4f} | "
        f"MC mean = {metrics['t2_mc_mean']:.4f}"
    )
    print(
        f"  true tau2 = 1.0000 | "
        f"plugin = {metrics['tau2_plugin']:.4f} | "
        f"MC mean = {metrics['tau2_mc_mean']:.4f}"
    )
    print(
        f"  true sigma = {args.sigma:.4f} | "
        f"plugin = {metrics['sigma_plugin']:.4f}"
    )
    print(
        f"  true sigma2 = {args.sigma ** 2:.4f} | "
        f"plugin = {metrics['sigma2_plugin']:.4f}"
    )
    print(f"  logF_RMSE(plugin) = {metrics['logF_rmse_plugin']:.4f}")
    print(f"  logF_RMSE(post)   = {metrics['logF_rmse_post']:.4f}")
    print(f"\n[INFO] Results saved to: {outdir}\n")


if __name__ == "__main__":
    main()