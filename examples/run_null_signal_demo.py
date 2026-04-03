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
    case_display: str
    label: str
    F_vi: torch.Tensor
    F_mcmc: Optional[torch.Tensor]

    rmse_phi_vi: float
    rmse_phi_mcmc: Optional[float]

    rmse_logF_vi: Optional[float]
    rmse_logF_mcmc: Optional[float]

    pll_vi: Optional[float] = None
    pll_mcmc: Optional[float] = None
    pll_vi_per_test: Optional[float] = None
    pll_mcmc_per_test: Optional[float] = None

    mass_vi: float | None = None
    mass_mcmc: float | None = None

    flatness_vi: float | None = None
    flatness_mcmc: float | None = None

    spec_sdlog_vi: float | None = None
    spec_sdlog_mcmc: float | None = None

    phi_energy_vi: float | None = None
    phi_energy_mcmc: float | None = None

    vi_band: Optional[dict[str, torch.Tensor]] = None
    mcmc_band: Optional[dict[str, torch.Tensor]] = None


# -------------------------
# DGP: null / weak spatial signal
# -------------------------
def car_like_truth_spectrum(lam: torch.Tensor, *, eps_car: float, tau: float = 1.0) -> torch.Tensor:
    lam = lam.clamp_min(0.0)
    F = tau / (lam + eps_car)
    return F.clamp_min(1e-12)


def multiscale_truth_spectrum(
    lam: torch.Tensor,
    *,
    eps_car: float,
    tau1: float = 0.7,
    tau2: float = 0.3,
    decay: float = 6.0,
) -> torch.Tensor:
    lam = lam.clamp_min(0.0)
    lowfreq = tau1 / (lam + eps_car)
    midfreq = tau2 * torch.exp(-decay * lam)
    F = lowfreq + midfreq
    return F.clamp_min(1e-12)


def make_truth_spectrum(
    lam: torch.Tensor,
    *,
    truth_shape: str,
    eps_car: float,
) -> torch.Tensor:
    if truth_shape == "car":
        return car_like_truth_spectrum(lam, eps_car=eps_car, tau=1.0)
    if truth_shape == "multiscale":
        return multiscale_truth_spectrum(lam, eps_car=eps_car)
    raise ValueError(f"Unknown truth_shape='{truth_shape}'")


def make_split(n: int, test_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(test_frac * n)
    n_test = max(1, min(n - 1, n_test))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


# -------------------------
# Spectrum diagnostics
# -------------------------
@torch.no_grad()
def spectrum_vi_mc_mean(filter_module, lam: torch.Tensor, *, S: int = 256) -> torch.Tensor:
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
    device = next(filter_module.parameters()).device if any(True for _ in filter_module.parameters()) else lam.device
    dtype = lam.dtype
    lam = lam.to(device=device, dtype=dtype)

    S = theta_chain.shape[0]
    acc = torch.zeros_like(lam)
    count = 0

    for i in range(0, S, batch):
        chunk = theta_chain[i:i + batch].to(device=device, dtype=dtype)
        for j in range(chunk.shape[0]):
            theta = filter_module.unpack(chunk[j])
            acc += filter_module.spectrum(lam, theta)
            count += 1

    return acc / max(count, 1)


def rmse_log_spectrum(F_hat: torch.Tensor, F_true: torch.Tensor) -> float:
    eps = 1e-12
    a = torch.log(F_hat.clamp_min(eps))
    b = torch.log(F_true.clamp_min(eps))
    return float(torch.sqrt(torch.mean((a - b) ** 2)).detach().cpu().item())


def spectrum_mass(F: torch.Tensor) -> float:
    return float(F.mean().detach().cpu())


def spectrum_flatness(F: torch.Tensor, eps: float = 1e-12) -> float:
    m = F.mean()
    s = F.std(unbiased=False)
    return float((s / (m + eps)).detach().cpu())


def spectrum_sdlog(F: torch.Tensor, eps: float = 1e-12) -> float:
    logF = torch.log(F.clamp_min(eps))
    return float(logF.std(unbiased=False).detach().cpu())


def phi_energy(phi: torch.Tensor) -> float:
    return float(torch.mean(phi.detach().cpu() ** 2).item())


# posterior band helpers
@torch.no_grad()
def spectrum_vi_draws(filter_module, lam: torch.Tensor, *, S: int = 256) -> torch.Tensor:
    draws = []
    for _ in range(S):
        th = filter_module.sample_unconstrained()
        F = filter_module.spectrum(lam, th).clamp_min(1e-12)
        draws.append(F.detach().cpu())
    return torch.stack(draws, dim=0)


@torch.no_grad()
def spectrum_mcmc_draws(filter_module, lam: torch.Tensor, theta_chain: torch.Tensor, *, max_draws: int = 256) -> torch.Tensor:
    T = theta_chain.shape[0]
    picks = np.arange(T) if T <= max_draws else np.linspace(0, T - 1, max_draws).astype(int)
    draws = []
    for i in picks:
        theta = filter_module.unpack(theta_chain[i])
        F = filter_module.spectrum(lam, theta).clamp_min(1e-12)
        draws.append(F.detach().cpu())
    return torch.stack(draws, dim=0)


def summarize_spectrum_draws(F_draws: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "q10": torch.quantile(F_draws, 0.10, dim=0),
        "q50": torch.quantile(F_draws, 0.50, dim=0),
        "q90": torch.quantile(F_draws, 0.90, dim=0),
    }


def plot_spectrum_curves(
    lam_np: np.ndarray,
    curves: Dict[str, np.ndarray],
    save_path: Path,
    *,
    ylog: bool,
    title: str
):
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


def plot_spectrum_band(
    lam_np: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    save_path: Path,
    *,
    ylog: bool,
    title: str,
    truth: Optional[np.ndarray] = None,
):
    plt.figure(figsize=(6.2, 4.2))
    if truth is not None:
        plt.plot(lam_np, truth, linewidth=2.0, label="truth")
    plt.fill_between(lam_np, q10, q90, alpha=0.25, label="80% band")
    plt.plot(lam_np, q50, linewidth=2.0, label="median")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F(\lambda)$")
    plt.title(title)
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_spectrum_overlay(*, lam: torch.Tensor, F_true: torch.Tensor, results: List[FitResult], save_dir: Path, title_tag: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    lam_np = lam.detach().cpu().numpy()

    curves: Dict[str, np.ndarray] = {"truth": F_true.detach().cpu().numpy()}
    for r in results:
        curves[f"{r.label} (VI)"] = r.F_vi.detach().cpu().numpy()
        if r.F_mcmc is not None:
            curves[f"{r.label} (MCMC)"] = r.F_mcmc.detach().cpu().numpy()

    plot_spectrum_curves(lam_np, curves, save_dir / "spectrum_overlay_linear.png", ylog=False, title=f"Spectrum overlay ({title_tag})")
    plot_spectrum_curves(lam_np, curves, save_dir / "spectrum_overlay_log.png", ylog=True, title=f"Spectrum overlay ({title_tag})")


def plot_compare_vi_bands(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    band_map: Dict[str, dict[str, torch.Tensor]],
    save_path: Path,
    title: str,
):
    lam_np = lam.detach().cpu().numpy()
    truth_np = F_true.detach().cpu().numpy()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(lam_np, truth_np, linewidth=2.2, label="truth")

    for label, band in band_map.items():
        q10 = band["q10"].detach().cpu().numpy()
        q50 = band["q50"].detach().cpu().numpy()
        q90 = band["q90"].detach().cpu().numpy()
        plt.fill_between(lam_np, q10, q90, alpha=0.12)
        plt.plot(lam_np, q50, linewidth=1.8, label=label)

    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F(\lambda)$")
    plt.yscale("log")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# -------------------------
# Predictive log-likelihood (exact conditional blocks)
# -------------------------
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

    F_row = F.clamp_min(1e-12).reshape(1, -1)

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
    beta_mode: str = "plugin",
    sigma2_mode: str = "plugin",
) -> float:
    m_beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_plugin = m_beta_plugin.reshape(-1)

    logliks = []
    for _ in range(int(S)):
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(lam, theta).clamp_min(1e-12)

        if sigma2_mode == "draw":
            eps = torch.randn_like(model.mu_log_sigma2)
            s = model.mu_log_sigma2 + torch.exp(model.log_std_log_sigma2) * eps
            sigma2 = float(torch.exp(s).clamp_min(1e-12).detach().cpu())
        else:
            sigma2 = float(sigma2_plugin.detach().cpu())

        if beta_mode == "draw":
            denom = (F + torch.tensor(sigma2, device=F.device, dtype=F.dtype)).clamp_min(1e-12)
            inv_var = 1.0 / denom
            beta, _ = model._beta_update(inv_var, return_Xt_invSig_X=False)
            beta = beta.reshape(-1)
        else:
            beta = beta_plugin

        ll = conditional_predictive_loglik(
            y=y, X=X, beta=beta, U=U, F=F, sigma2=sigma2,
            train_idx=train_idx, test_idx=test_idx,
        )
        logliks.append(ll)

    a = max(logliks)
    pll = a + math.log(sum(math.exp(v - a) for v in logliks) / float(len(logliks)))
    return float(pll)


@torch.no_grad()
def predictive_loglik_mcmc_heldout(
    *,
    model: SpectralCAR_FullVI,
    theta_chain: torch.Tensor,
    s_chain: Optional[torch.Tensor],
    y: torch.Tensor,
    X: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    S: int = 16,
    sigma2_mode: str = "chain",
    beta_mode: str = "plugin",
) -> float:
    beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_plugin = beta_plugin.reshape(-1)

    T = theta_chain.shape[0]
    picks = np.arange(T) if T <= S else np.linspace(0, T - 1, S).astype(int)

    logliks = []
    for i in picks:
        theta = model.filter.unpack(theta_chain[i])
        F = model.filter.spectrum(lam, theta).clamp_min(1e-12)

        if sigma2_mode == "chain" and (s_chain is not None):
            sigma2 = float(torch.exp(s_chain[i]).clamp_min(1e-12).detach().cpu())
        else:
            sigma2 = float(sigma2_plugin.detach().cpu())

        beta = beta_plugin

        ll = conditional_predictive_loglik(
            y=y, X=X, beta=beta, U=U, F=F, sigma2=sigma2,
            train_idx=train_idx, test_idx=test_idx,
        )
        logliks.append(ll)

    a = max(logliks)
    pll = a + math.log(sum(math.exp(v - a) for v in logliks) / float(len(logliks)))
    return float(pll)


# -------------------------
# Phi full-domain mean (train-conditioned), VI and MCMC
# -------------------------
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
        F_used = spectrum_vi_mc_mean(model.filter, lam, S=256).detach()
        phi = phi_full_from_train(
            U_train=U_train, U_full=U_full,
            X_train=X_train, y_train=y_train,
            beta=beta_mean, F=F_used.to(lam.device), sigma2=sigma2,
        )
        return phi.detach(), F_used.detach(), sigma2

    if mode == "posterior":
        acc_phi = torch.zeros(U_full.shape[0], device=lam.device, dtype=lam.dtype)
        acc_F = torch.zeros_like(lam)

        for _ in range(int(S)):
            th = model.filter.sample_unconstrained()
            F = model.filter.spectrum(lam, th).clamp_min(1e-12)
            phi = phi_full_from_train(
                U_train=U_train, U_full=U_full,
                X_train=X_train, y_train=y_train,
                beta=beta_mean, F=F, sigma2=sigma2,
            )
            acc_phi += phi
            acc_F += F

        return (acc_phi / float(S)).detach(), (acc_F / float(S)).detach(), sigma2

    raise ValueError(f"Unknown VI phi mode '{mode}'. Choose from: plugin, posterior.")


@torch.no_grad()
def compute_phi_mcmc_full(
    *,
    model: SpectralCAR_FullVI,
    lam: torch.Tensor,
    theta_chain: torch.Tensor,
    s_chain: Optional[torch.Tensor],
    U_train: torch.Tensor,
    U_full: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mode: str = "plugin",
    max_draws: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    beta_mean, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2_plug = float(sigma2_plugin.detach().cpu())

    F_mean = spectrum_mcmc_mean(model.filter, lam, theta_chain).detach()

    if mode == "plugin":
        phi = phi_full_from_train(
            U_train=U_train, U_full=U_full,
            X_train=X_train, y_train=y_train,
            beta=beta_mean, F=F_mean.to(lam.device), sigma2=sigma2_plug,
        )
        return phi.detach(), F_mean.detach(), sigma2_plug

    if mode == "posterior":
        T = theta_chain.shape[0]
        S = min(int(max_draws), int(T))
        picks = np.arange(T) if T <= S else np.linspace(0, T - 1, S).astype(int)

        acc_phi = torch.zeros(U_full.shape[0], device=lam.device, dtype=lam.dtype)
        acc_F = torch.zeros_like(lam)

        for i in picks:
            theta_i = model.filter.unpack(theta_chain[i])
            F_i = model.filter.spectrum(lam, theta_i).clamp_min(1e-12)

            if s_chain is not None:
                sigma2_i = float(torch.exp(s_chain[i]).clamp_min(1e-12).detach().cpu())
            else:
                sigma2_i = sigma2_plug

            phi_i = phi_full_from_train(
                U_train=U_train, U_full=U_full,
                X_train=X_train, y_train=y_train,
                beta=beta_mean, F=F_i, sigma2=sigma2_i,
            )
            acc_phi += phi_i
            acc_F += F_i

        return (acc_phi / float(len(picks))).detach(), (acc_F / float(len(picks))).detach(), sigma2_plug

    raise ValueError(f"Unknown MCMC phi mode '{mode}'. Choose from: plugin, posterior.")


# -------------------------
# Fit specs parsing
# -------------------------
def parse_fit_specs(args) -> List[FitSpec]:
    if args.fits is not None:
        specs: List[FitSpec] = []
        for tok in args.fits:
            if ":" not in tok:
                raise ValueError(f"--fits token must be filter:case, got '{tok}'")
            f, c = tok.split(":", 1)
            f = f.strip()
            c = c.strip()
            spec = get_filter_spec(f)
            if c not in spec.cases:
                raise ValueError(
                    f"Case '{c}' not defined for filter '{f}'. "
                    f"Available: {list(spec.cases.keys())}"
                )
            specs.append(FitSpec(filter=f, case=c))
        return specs

    if args.filters is None:
        raise ValueError("Must provide either --fits or --filters/--cases.")

    specs: List[FitSpec] = []
    for f in args.filters:
        spec = get_filter_spec(f)
        for c in args.cases:
            if c in spec.cases:
                specs.append(FitSpec(filter=f, case=c))
            else:
                print(f"  - skipping {f}/{c} (case not defined)")
    if not specs:
        raise ValueError("No valid fits found from --filters/--cases.")
    return specs


def _sort_key(r: FitResult) -> float:
    if r.pll_mcmc is not None:
        return -r.pll_mcmc
    if r.pll_vi is not None:
        return -r.pll_vi
    if r.rmse_logF_mcmc is not None:
        return r.rmse_logF_mcmc
    if r.rmse_logF_vi is not None:
        return r.rmse_logF_vi
    return 0.0


# -------------------------
# Run one fit
# -------------------------
def run_fit(
    *,
    spec_name: str,
    case_id: str,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    U_full: torch.Tensor,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    U_tr: torch.Tensor,
    lam: torch.Tensor,
    coords: torch.Tensor,
    phi_true: torch.Tensor,
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
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pll_mc: int,
    vi_log_every: int,
    phi_vi_mode: str,
    phi_mcmc_mode: str,
    phi_draws: int,
    vi_pll_beta_mode: str,
    vi_pll_sigma2_mode: str,
    mcmc_pll_beta_mode: str,
    mcmc_pll_sigma2_mode: str,
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

    filter_module = case_spec.build_filter(
        tau2_true=tau2_init,
        eps_car=eps_car,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **case_spec.fixed,
    )

    model = SpectralCAR_FullVI(
        X=X_tr,
        y=y_tr,
        lam=lam,
        U=U_tr,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_true),
        log_std_log_sigma2=-2.3,
        num_mc=vi_mc,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    for it in range(1, vi_iters + 1):
        opt.zero_grad()
        elbo, _ = model.elbo()
        (-elbo).backward()
        opt.step()
        if vi_log_every and (it % vi_log_every == 0 or it == 1 or it == vi_iters):
            print(f"  [VI] iter {it:>5}/{vi_iters}  ELBO={float(elbo.detach().cpu()):.3f}")

    vi_band = None
    with torch.no_grad():
        phi_vi_full, F_vi_used, _ = compute_phi_vi_full(
            model=model,
            lam=lam,
            U_train=U_tr,
            U_full=U_full,
            X_train=X_tr,
            y_train=y_tr,
            mode=phi_vi_mode,
            S=max(1, int(phi_draws)),
        )

        rmse_phi_vi = float(torch.sqrt(torch.mean((phi_vi_full.cpu() - phi_true.cpu()) ** 2)).item())
        phi_energy_vi = phi_energy(phi_vi_full)

        is_null_truth = bool(torch.all(F_true == 0).item())
        rmse_logF_vi = None if is_null_truth else rmse_log_spectrum(F_vi_used, F_true)

        mass_vi = spectrum_mass(F_vi_used)
        flatness_vi = spectrum_flatness(F_vi_used)
        spec_sdlog_vi = spectrum_sdlog(F_vi_used)

        pll_vi = predictive_loglik_vi_heldout(
            model=model,
            y=y_full, X=X_full, U=U_full, lam=lam,
            train_idx=train_idx, test_idx=test_idx,
            S=pll_mc,
            beta_mode=vi_pll_beta_mode,
            sigma2_mode=vi_pll_sigma2_mode,
        )
        pll_vi_per_test = float(pll_vi) / float(len(test_idx))

    F_vi_cpu = F_vi_used.detach().cpu()

    mcmc_band = None
    phi_mcmc_full = None
    rmse_phi_mcmc = None
    F_mcmc_cpu = None
    rmse_logF_mcmc = None
    pll_mcmc = None
    acc_s_mid = None
    acc_theta_mid = None

    phi_energy_mcmc = None
    spec_sdlog_mcmc = None
    pll_mcmc_per_test = None
    mass_mcmc = None
    flatness_mcmc = None

    if (not skip_mcmc) and (mcmc_steps > 0):
        cfg = MCMCConfig(
            num_steps=mcmc_steps,
            burnin=mcmc_burnin,
            thin=mcmc_thin,
            step_s=float(case_spec.step_s),
            step_theta=case_spec.get_step_theta(model.filter),
            seed=0,
            device=device,
            print_every=5000,
        )
        sampler = make_collapsed_mcmc_from_model(model, config=cfg)

        init_s = model.mu_log_sigma2.detach()
        theta0 = model.filter.mean_unconstrained()
        init_theta_vec = model.filter.pack(theta0).detach()

        print(
            f"  [MCMC] starting: steps={mcmc_steps}, burnin={mcmc_burnin}, thin={mcmc_thin} "
            f"(kept draws ~ {(mcmc_steps - mcmc_burnin)//mcmc_thin if mcmc_steps > mcmc_burnin else 0})"
        )

        out = sampler.run(
            init_s=init_s,
            init_theta_vec=init_theta_vec,
            init_from_conditional_beta=True,
            store_phi_mean=True,
            U=U_tr,
            X=X_tr,
            y=y_tr,
        )

        print("  [MCMC] finished")

        theta_chain = out["theta"]
        s_chain = out.get("s", None)

        F_mcmc_draws = spectrum_mcmc_draws(model.filter, lam, theta_chain, max_draws=max(64, int(phi_draws)))
        mcmc_band = summarize_spectrum_draws(F_mcmc_draws)

        pll_mcmc = predictive_loglik_mcmc_heldout(
            model=model,
            theta_chain=theta_chain,
            s_chain=s_chain,
            y=y_full, X=X_full, U=U_full, lam=lam,
            train_idx=train_idx, test_idx=test_idx,
            S=pll_mc,
            beta_mode=mcmc_pll_beta_mode,
            sigma2_mode=mcmc_pll_sigma2_mode,
        )

        F_mcmc = spectrum_mcmc_mean(model.filter, lam, theta_chain).detach()
        rmse_logF_mcmc = None if is_null_truth else rmse_log_spectrum(F_mcmc, F_true)
        F_mcmc_cpu = F_mcmc.detach().cpu()

        mass_mcmc = spectrum_mass(F_mcmc)
        flatness_mcmc = spectrum_flatness(F_mcmc)
        spec_sdlog_mcmc = spectrum_sdlog(F_mcmc)

        phi_mcmc_full, _, _ = compute_phi_mcmc_full(
            model=model,
            lam=lam,
            theta_chain=theta_chain,
            s_chain=s_chain,
            U_train=U_tr,
            U_full=U_full,
            X_train=X_tr,
            y_train=y_tr,
            mode=phi_mcmc_mode,
            max_draws=max(1, int(phi_draws)),
        )
        rmse_phi_mcmc = float(torch.sqrt(torch.mean((phi_mcmc_full.cpu() - phi_true.cpu()) ** 2)).item())
        phi_energy_mcmc = phi_energy(phi_mcmc_full)

        pll_mcmc_per_test = float(pll_mcmc) / float(len(test_idx)) if pll_mcmc is not None else None

        acc_s_mid = float(out["acc"]["s"][2])
        acc_theta_mid = {k: float(v[2]) for k, v in out["acc"]["theta"].items()}

    if not compare_only:
        lam_np = lam.detach().cpu().numpy()
        curves = {"truth": F_true.clamp_min(1e-12).detach().cpu().numpy(), f"{label} (VI)": F_vi_cpu.numpy()}
        if F_mcmc_cpu is not None:
            curves[f"{label} (MCMC)"] = F_mcmc_cpu.numpy()

        plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_linear.png", ylog=False, title="Spectrum comparison")
        plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_log.png", ylog=True, title="Spectrum comparison")

        diagnostics.plot_phi_mean_vs_true(
            coords=coords,
            mean_phi=phi_vi_full.to(device),
            phi_true=phi_true,
            save_path_prefix=str(case_dir / "phi_vi"),
        )
        if phi_mcmc_full is not None:
            diagnostics.plot_phi_mean_vs_true(
                coords=coords,
                mean_phi=phi_mcmc_full.to(device=device, dtype=torch.double),
                phi_true=phi_true,
                save_path_prefix=str(case_dir / "phi_mcmc"),
            )

        F_vi_draws = spectrum_vi_draws(model.filter, lam, S=max(64, int(phi_draws)))
        vi_band = summarize_spectrum_draws(F_vi_draws)
        plot_spectrum_band(
            lam_np,
            vi_band["q10"].numpy(),
            vi_band["q50"].numpy(),
            vi_band["q90"].numpy(),
            case_dir / "spectrum_vi_band_linear.png",
            ylog=False,
            title="VI spectrum posterior band",
            truth=F_true.clamp_min(1e-12).detach().cpu().numpy(),
        )
        plot_spectrum_band(
            lam_np,
            vi_band["q10"].numpy(),
            vi_band["q50"].numpy(),
            vi_band["q90"].numpy(),
            case_dir / "spectrum_vi_band_log.png",
            ylog=True,
            title="VI spectrum posterior band",
            truth=F_true.clamp_min(1e-12).detach().cpu().numpy(),
        )

        if mcmc_band is not None:
            plot_spectrum_band(
                lam_np,
                mcmc_band["q10"].numpy(),
                mcmc_band["q50"].numpy(),
                mcmc_band["q90"].numpy(),
                case_dir / "spectrum_mcmc_band_linear.png",
                ylog=False,
                title="MCMC spectrum posterior band",
                truth=F_true.clamp_min(1e-12).detach().cpu().numpy(),
            )
            plot_spectrum_band(
                lam_np,
                mcmc_band["q10"].numpy(),
                mcmc_band["q50"].numpy(),
                mcmc_band["q90"].numpy(),
                case_dir / "spectrum_mcmc_band_log.png",
                ylog=True,
                title="MCMC spectrum posterior band",
                truth=F_true.clamp_min(1e-12).detach().cpu().numpy(),
            )

    if rmse_phi_mcmc is None:
        print(f"  phi RMSE  : VI={rmse_phi_vi:.4f} | MCMC=NA")
        if rmse_logF_vi is None:
            print("  logF RMSE : VI=NA | MCMC=NA")
        else:
            print(f"  logF RMSE : VI={rmse_logF_vi:.4f} | MCMC=NA")
    else:
        print(f"  phi RMSE  : VI={rmse_phi_vi:.4f} | MCMC={rmse_phi_mcmc:.4f}")
        if rmse_logF_vi is None or rmse_logF_mcmc is None:
            print("  logF RMSE : VI=NA | MCMC=NA")
        else:
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

        pll_vi=float(pll_vi) if pll_vi is not None else None,
        pll_mcmc=float(pll_mcmc) if pll_mcmc is not None else None,
        pll_vi_per_test=pll_vi_per_test,
        pll_mcmc_per_test=pll_mcmc_per_test,

        mass_vi=mass_vi,
        mass_mcmc=mass_mcmc,

        flatness_vi=flatness_vi,
        flatness_mcmc=flatness_mcmc,

        spec_sdlog_vi=spec_sdlog_vi,
        spec_sdlog_mcmc=spec_sdlog_mcmc,

        phi_energy_vi=phi_energy_vi,
        phi_energy_mcmc=phi_energy_mcmc,

        vi_band=vi_band,
        mcmc_band=mcmc_band,
    )


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--regime",
        choices=["null", "weak"],
        required=True,
        help="Experiment 3 regime: null signal or weak spatial signal.",
    )
    p.add_argument(
        "--truth_shape",
        choices=["car", "multiscale"],
        default="multiscale",
        help="Base truth spectrum used when regime=weak.",
    )

    p.add_argument(
        "--filters",
        nargs="+",
        default=None,
        help=f"Legacy mode: filter families to fit. Available: {available_filters()}",
    )
    p.add_argument("--cases", nargs="+", default=["baseline"], help="Legacy mode cases.")

    p.add_argument(
        "--fits",
        nargs="+",
        default=None,
        help="Preferred mode: explicit list as filter:case tokens. Example: --fits classic_car:baseline rational:rat_num2_den1",
    )

    p.add_argument("--outdir", default=str(Path("examples") / "figures" / "null_signal"))

    p.add_argument("--vi_iters", type=int, default=2000)
    p.add_argument("--vi_mc", type=int, default=10)
    p.add_argument("--vi_lr", type=float, default=1e-2)

    p.add_argument("--mcmc_steps", type=int, default=30000)
    p.add_argument("--mcmc_burnin", type=int, default=10000)
    p.add_argument("--mcmc_thin", type=int, default=10)

    p.add_argument("--skip_mcmc", action="store_true")
    p.add_argument("--compare_only", action="store_true")

    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--pll_mc", type=int, default=16)
    p.add_argument("--vi_log_every", type=int, default=200)

    p.add_argument("--phi_vi_mode", choices=["plugin", "posterior"], default="plugin")
    p.add_argument("--phi_mcmc_mode", choices=["plugin", "posterior"], default="plugin")
    p.add_argument("--phi_draws", type=int, default=128)

    p.add_argument("--vi_pll_beta", choices=["plugin", "draw"], default="plugin")
    p.add_argument("--vi_pll_sigma2", choices=["plugin", "draw"], default="plugin")
    p.add_argument("--mcmc_pll_beta", choices=["plugin"], default="plugin")
    p.add_argument("--mcmc_pll_sigma2", choices=["plugin", "chain"], default="chain")

    p.add_argument("--tau2_weak", type=float, default=0.01, help="Amplitude multiplier for weak spatial signal.")
    p.add_argument("--sigma2_true", type=float, default=0.10, help="Noise variance in DGP.")
    p.add_argument("--eps_car", type=float, default=1e-3, help="Epsilon used in CAR-like truth spectrum.")

    args = p.parse_args()

    if args.fits is None and args.filters is None:
        raise ValueError("Provide either --fits or --filters (legacy).")

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    outdir = Path(args.outdir) / f"regime_{args.regime}"
    if args.regime == "weak":
        outdir = outdir / f"truth_{args.truth_shape}" / f"tau2weak_{args.tau2_weak:g}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) grid + graph + eigendecomp
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

    # 2) covariates + truth beta
    x_coord = coords[:, 0]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device), x_coord], dim=1)
    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)

    sigma2_true = float(args.sigma2_true)

    # 3) DGP
    eps_car = float(args.eps_car)

    if args.regime == "null":
        F_true = torch.zeros_like(lam)
        phi_true = torch.zeros(n, dtype=torch.double, device=device)
    else:
        F_base = make_truth_spectrum(
            lam,
            truth_shape=args.truth_shape,
            eps_car=eps_car,
        )
        F_true = float(args.tau2_weak) * F_base
        z_true = torch.sqrt(F_true.clamp_min(1e-12)) * torch.randn(n, dtype=torch.double, device=device)
        phi_true = U @ z_true

    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(n, dtype=torch.double, device=device)

    # 4) split and slice train
    train_idx, test_idx = make_split(n, test_frac=args.test_frac, seed=args.split_seed)
    tr = torch.as_tensor(train_idx, device=device)
    X_tr = X[tr, :]
    y_tr = y[tr]
    U_tr = U[tr, :]

    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(X.shape[1], dtype=torch.double, device=device)

    tau2_init = 0.4
    fit_specs = parse_fit_specs(args)

    results: List[FitResult] = []
    for fs in fit_specs:
        r = run_fit(
            spec_name=fs.filter,
            case_id=fs.case,
            X_full=X, y_full=y, U_full=U,
            X_tr=X_tr, y_tr=y_tr, U_tr=U_tr,
            lam=lam, coords=coords,
            phi_true=phi_true,
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
            train_idx=train_idx,
            test_idx=test_idx,
            pll_mc=args.pll_mc,
            vi_log_every=args.vi_log_every,
            phi_vi_mode=args.phi_vi_mode,
            phi_mcmc_mode=args.phi_mcmc_mode,
            phi_draws=args.phi_draws,
            vi_pll_beta_mode=args.vi_pll_beta,
            vi_pll_sigma2_mode=args.vi_pll_sigma2,
            mcmc_pll_beta_mode=args.mcmc_pll_beta,
            mcmc_pll_sigma2_mode=args.mcmc_pll_sigma2,
        )
        if r is not None:
            results.append(r)

    if results:
        compare_dir = outdir / "COMPARE"
        compare_dir.mkdir(parents=True, exist_ok=True)

        plot_spectrum_overlay(
            lam=lam,
            F_true=F_true.clamp_min(1e-12),
            results=results,
            save_dir=compare_dir,
            title_tag=f"regime={args.regime}, truth={args.truth_shape if args.regime == 'weak' else 'null'}",
        )

        vi_band_map = {r.label: r.vi_band for r in results if r.vi_band is not None}
        if vi_band_map:
            plot_compare_vi_bands(
                lam=lam,
                F_true=F_true.clamp_min(1e-12),
                band_map=vi_band_map,
                save_path=compare_dir / "compare_vi_bands_log.png",
                title=f"VI posterior spectrum bands ({args.regime})",
            )

    if results:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT 3 SUMMARY (regime={args.regime}, truth_shape={args.truth_shape if args.regime == 'weak' else 'null'})")
        print("=" * 80)

        for r in sorted(results, key=_sort_key):
            pll_vi = f"{r.pll_vi:.2f}" if r.pll_vi is not None else "NA"
            pll_mcmc = f"{r.pll_mcmc:.2f}" if r.pll_mcmc is not None else "NA"
            pll_vi_pt = f"{r.pll_vi_per_test:.4f}" if r.pll_vi_per_test is not None else "NA"
            pll_mcmc_pt = f"{r.pll_mcmc_per_test:.4f}" if r.pll_mcmc_per_test is not None else "NA"

            phi_rmse_mcmc = f"{r.rmse_phi_mcmc:.4f}" if r.rmse_phi_mcmc is not None else "NA"
            phi_eng_vi = f"{r.phi_energy_vi:.4e}" if r.phi_energy_vi is not None else "NA"
            phi_eng_mcmc = f"{r.phi_energy_mcmc:.4e}" if r.phi_energy_mcmc is not None else "NA"

            flat_vi = f"{r.flatness_vi:.3f}" if r.flatness_vi is not None else "NA"
            flat_mcmc = f"{r.flatness_mcmc:.3f}" if r.flatness_mcmc is not None else "NA"
            sdlog_vi = f"{r.spec_sdlog_vi:.3f}" if r.spec_sdlog_vi is not None else "NA"
            sdlog_mcmc = f"{r.spec_sdlog_mcmc:.3f}" if r.spec_sdlog_mcmc is not None else "NA"

            print(
                f"{r.label:<32} | "
                f"PLL(VI)={pll_vi}  PLL/test(VI)={pll_vi_pt} | "
                f"PLL(MCMC)={pll_mcmc}  PLL/test(MCMC)={pll_mcmc_pt} | "
                f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={phi_rmse_mcmc} | "
                f"phi_energy(VI)={phi_eng_vi}  phi_energy(MCMC)={phi_eng_mcmc} | "
                f"flat(VI)={flat_vi}  flat(MCMC)={flat_mcmc} | "
                f"sdlog(VI)={sdlog_vi}  sdlog(MCMC)={sdlog_mcmc}"
            )

        with open(compare_dir / "leaderboard.txt", "w", encoding="utf-8") as f:
            f.write(
                f"EXPERIMENT 3 SUMMARY (regime={args.regime}, "
                f"truth_shape={args.truth_shape if args.regime == 'weak' else 'null'})\n"
            )
            for r in sorted(results, key=_sort_key):
                pll_vi = f"{r.pll_vi:.2f}" if r.pll_vi is not None else "NA"
                pll_mcmc = f"{r.pll_mcmc:.2f}" if r.pll_mcmc is not None else "NA"
                pll_vi_pt = f"{r.pll_vi_per_test:.4f}" if r.pll_vi_per_test is not None else "NA"
                pll_mcmc_pt = f"{r.pll_mcmc_per_test:.4f}" if r.pll_mcmc_per_test is not None else "NA"

                phi_rmse_mcmc = f"{r.rmse_phi_mcmc:.4f}" if r.rmse_phi_mcmc is not None else "NA"
                phi_eng_vi = f"{r.phi_energy_vi:.4e}" if r.phi_energy_vi is not None else "NA"
                phi_eng_mcmc = f"{r.phi_energy_mcmc:.4e}" if r.phi_energy_mcmc is not None else "NA"

                flat_vi = f"{r.flatness_vi:.3f}" if r.flatness_vi is not None else "NA"
                flat_mcmc = f"{r.flatness_mcmc:.3f}" if r.flatness_mcmc is not None else "NA"
                sdlog_vi = f"{r.spec_sdlog_vi:.3f}" if r.spec_sdlog_vi is not None else "NA"
                sdlog_mcmc = f"{r.spec_sdlog_mcmc:.3f}" if r.spec_sdlog_mcmc is not None else "NA"

                f.write(
                    f"{r.label:<32} | "
                    f"PLL(VI)={pll_vi}  PLL/test(VI)={pll_vi_pt} | "
                    f"PLL(MCMC)={pll_mcmc}  PLL/test(MCMC)={pll_mcmc_pt} | "
                    f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={phi_rmse_mcmc} | "
                    f"phi_energy(VI)={phi_eng_vi}  phi_energy(MCMC)={phi_eng_mcmc} | "
                    f"flat(VI)={flat_vi}  flat(MCMC)={flat_mcmc} | "
                    f"sdlog(VI)={sdlog_vi}  sdlog(MCMC)={sdlog_mcmc}\n"
                )


if __name__ == "__main__":
    main()