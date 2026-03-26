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
    pll_vi: Optional[float] = None  # store predictive metrics
    pll_mcmc: Optional[float] = None    # store predictive metrics

# -------------------------
# Truth spectra (misspec)
# -------------------------
def truth_spectrum(lam: torch.Tensor, kind: str, *, eps_car: float) -> torch.Tensor:
    """
    Canonical spectral truth families for Experiment 2.

    These define variance allocation across graph frequencies (fixed U),
    enabling controlled spectral misspecification experiments.

    All spectra are:
    - strictly positive
    - normalized to mean 1 (separates shape from scale)

    Available kinds:

    - multiscale (PRIMARY)
        Two-band spectrum: low-frequency + mid-frequency peaks.
        Represents global + regional structure.

    - bandpass
        Single mid-frequency bump.
        Clean non-CAR counterexample (no low-frequency dominance).

    - car (optional baseline)
        Classical CAR spectrum for reference.

    Notes:
    - Uses quantiles of λ for graph-adaptive parameterization.
    - Avoids dependence on absolute λ scale.
    """

    lam = lam.clamp_min(0.0)
    kind = kind.lower()

    # Convert once for quantiles
    lam_np = lam.detach().cpu().numpy()
    lam_min = float(lam.min())
    lam_max = float(lam.max())
    lam_range = max(lam_max - lam_min, 1e-8)

    # -------------------------
    # 1. Multiscale (PRIMARY)
    # -------------------------
    if kind == "multiscale":
        mu1 = np.quantile(lam_np, 0.10)   # low-frequency peak
        mu2 = np.quantile(lam_np, 0.40)   # mid-frequency peak

        s1 = 0.05 * lam_range
        s2 = 0.07 * lam_range

        a1 = 1.0
        a2 = 0.8
        delta = 0.03

        F = (
            delta
            + a1 * torch.exp(-0.5 * ((lam - mu1) / max(s1, 1e-12)) ** 2)
            + a2 * torch.exp(-0.5 * ((lam - mu2) / max(s2, 1e-12)) ** 2)
        )

        F = F / F.mean()  # normalize
        return F.clamp_min(1e-12)

    # -------------------------
    # 2. Band-pass
    # -------------------------
    if kind == "bandpass":
        mu = np.quantile(lam_np, 0.40)   # centered in mid frequencies
        s = 0.08 * lam_range
        A = 1.0

        F = A * torch.exp(-0.5 * ((lam - mu) / max(s, 1e-12)) ** 2)

        F = F / F.mean()
        return F.clamp_min(1e-12)

    # -------------------------
    # 3. CAR baseline (optional)
    # -------------------------
    if kind == "car":
        tau = 1.0
        F = tau / (lam + eps_car)

        F = F / F.mean()
        return F.clamp_min(1e-12)

    raise ValueError(
        f"Unknown truth kind '{kind}'. "
        f"Choose from: multiscale, bandpass, car."
    )

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
        acc += filter_module.spectrum(lam, th).clamp_min(1e-12)
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
    params = list(filter_module.parameters())
    device = params[0].device if params else lam.device
    dtype = lam.dtype
    lam = lam.to(device=device, dtype=dtype)

    S = theta_chain.shape[0]
    acc = torch.zeros_like(lam)
    count = 0

    for i in range(0, S, batch):
        chunk = theta_chain[i:i+batch].to(device=device, dtype=dtype)
        for j in range(chunk.shape[0]):
            theta = filter_module.unpack(chunk[j])
            acc += filter_module.spectrum(lam, theta).clamp_min(1e-12)
            count += 1

    return acc / max(count, 1)


def plot_spectrum_curves(lam_np: np.ndarray, curves: Dict[str, np.ndarray], save_path: Path, *, ylog: bool, title: str):
    plt.figure(figsize=(6.2, 4.2))
    idx = np.argsort(lam_np)
    lam_sorted = lam_np[idx]
    for name, y in curves.items():
        plt.plot(lam_sorted, y[idx], linewidth=2.0 if name == "truth" else 1.6, label=name)
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


def make_split(n: int, test_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(test_frac * n)
    n_test = max(1, min(n - 1, n_test))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx

def _k_blocks_from_spectrum(
    U: torch.Tensor,
    F: torch.Tensor,
    sigma2: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """
    Build covariance blocks for y = X beta + phi + eps,
    where phi ~ N(0, U diag(F) U^T), eps ~ N(0, sigma2 I).

    Returns:
      K_tt, K_st, K_ss
    """
    device = U.device
    dtype = U.dtype

    tr = torch.as_tensor(train_idx, device=device)
    te = torch.as_tensor(test_idx, device=device)

    Utr = U[tr, :]  # [n_tr, n]
    Ute = U[te, :]  # [n_te, n]

    F_row = F.to(device=device, dtype=dtype).clamp_min(1e-12).reshape(1, -1)

    # K_tt = Utr diag(F) Utr^T + sigma2 I
    K_tt = (Utr * F_row) @ Utr.T

    # K_st = Ute diag(F) Utr^T
    K_st = (Ute * F_row) @ Utr.T

    # K_ss = Ute diag(F) Ute^T + sigma2 I
    K_ss = (Ute * F_row) @ Ute.T

    K_tt = K_tt + sigma2 * torch.eye(K_tt.shape[0], device=device, dtype=dtype)
    K_ss = K_ss + sigma2 * torch.eye(K_ss.shape[0], device=device, dtype=dtype)

    return K_tt, K_st, K_ss


def conditional_predictive_loglik(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    beta: torch.Tensor,     # [p]
    U: torch.Tensor,
    F: torch.Tensor,        # [n]
    sigma2: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    """
    Compute log p(y_test | y_train, beta, sigma2, F) under the exact Gaussian conditional.

    Returns a scalar float (CPU).
    """
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
    r_tr = y_tr - mu_tr  # residuals

    K_tt, K_st, K_ss = _k_blocks_from_spectrum(U, F, sigma2, train_idx, test_idx)

    # Solve K_tt^{-1} r_tr and K_tt^{-1} K_ts via Cholesky
    L = torch.linalg.cholesky(K_tt)

    # alpha = K_tt^{-1} r_tr
    alpha = torch.cholesky_solve(r_tr.reshape(-1, 1), L).reshape(-1)

    # Conditional mean: mu_te + K_st K_tt^{-1} r_tr
    cond_mean = mu_te + (K_st @ alpha)

    # Conditional cov: K_ss - K_st K_tt^{-1} K_ts
    # First solve V = K_tt^{-1} K_ts, where K_ts = K_st^T
    V = torch.cholesky_solve(K_st.T, L)  # [n_tr, n_te]

    # numerically symmetric
    cond_cov = K_ss - (K_st @ V)
    cond_cov = 0.5 * (cond_cov + cond_cov.T)

    # MVN logpdf: -0.5*( (y-m)^T C^{-1} (y-m) + logdet(C) + n log(2π) )
    e = (y_te - cond_mean).reshape(-1, 1)

    # Stabilize (tiny jitter)
    jitter = 1e-8
    cond_cov = cond_cov + jitter * torch.eye(cond_cov.shape[0], device=device, dtype=dtype)

    Lc = torch.linalg.cholesky(cond_cov)
    sol = torch.cholesky_solve(e, Lc)  # C^{-1} e
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
    """
    Monte Carlo estimate of held-out predictive log-likelihood:
        log p(y_test | y_train) under VI posterior.

    We MC over theta ~ q(theta). Optionally MC over s=log sigma2 ~ q(s).
    For stability, we typically use plugin beta (posterior mean under plugin hypers).
    """
    # plugin beta mean is stable and consistent with our collapsed approach
    m_beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta = m_beta_plugin.reshape(-1)

    logliks = []

    for _ in range(int(S)):
        # sample theta from q(theta)
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(lam, theta).clamp_min(1e-12)

        # choose sigma2
        if sample_sigma2:
            eps = torch.randn_like(model.mu_log_sigma2)
            s = model.mu_log_sigma2 + torch.exp(model.log_std_log_sigma2) * eps
            sigma2 = float(torch.exp(s).clamp_min(1e-12).detach().cpu())
        else:
            sigma2 = float(sigma2_plugin.detach().cpu())

        # optionally: draw-consistent beta (more expensive, more variance)
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

    # log-mean-exp to approximate p(y_test|y_train) = E_q[ p(y_test|y_train,theta,s) ]
    a = max(logliks)
    pll = a + math.log(sum(math.exp(v - a) for v in logliks) / float(len(logliks)))
    return float(pll)

@torch.no_grad()
def predictive_loglik_mcmc_heldout(
    *,
    model: SpectralCAR_FullVI,
    theta_chain: torch.Tensor,         # [T, d]
    s_chain: Optional[torch.Tensor],   # [T] log sigma2, if available
    y: torch.Tensor,
    X: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    S: int = 16,
) -> float:
    """
    Held-out predictive log-likelihood using MCMC draws.
    Uses plugin beta (from VI model) for stability and to avoid storing beta chain.
    """
    # stable beta
    beta, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta = beta.reshape(-1)

    T = theta_chain.shape[0]
    if T <= S:
        picks = np.arange(T)
    else:
        picks = np.linspace(0, T - 1, S).astype(int)

    logliks = []
    for i in picks:
        theta = model.filter.unpack(theta_chain[i])
        F = model.filter.spectrum(lam, theta).clamp_min(1e-12)

        if s_chain is not None:
            sigma2 = float(torch.exp(s_chain[i]).clamp_min(1e-12).detach().cpu())
        else:
            sigma2 = float(sigma2_plugin.detach().cpu())

        ll = conditional_predictive_loglik(
            y=y, X=X, beta=beta, U=U, F=F, sigma2=sigma2,
            train_idx=train_idx, test_idx=test_idx,
        )
        logliks.append(ll)

    a = max(logliks)
    pll = a + math.log(sum(math.exp(v - a) for v in logliks) / float(len(logliks)))
    return float(pll)


@torch.no_grad()
def compute_phi_vi_full(
    *,
    model: SpectralCAR_FullVI,
    lam: torch.Tensor,
    U_train: torch.Tensor,
    U_full: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mode: str = "plugin",      # "plugin" or "posterior"
    S: int = 64,               # only used for "posterior"
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
      phi_full: [n] full-domain posterior mean of phi under VI (depending on mode)
      F_used  : [n] spectrum summary used (E_q[F] for plugin; average of draws for posterior)
      sigma2  : float (plugin sigma2)
    """
    beta_mean, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2 = float(sigma2_plugin.detach().cpu())

    if mode == "plugin":
        # E_q[F]
        F_used = spectrum_vi_mc_mean(model.filter, lam, S=S).detach()
        phi = phi_full_from_train(
            U_train=U_train, U_full=U_full,
            X_train=X_train, y_train=y_train,
            beta=beta_mean, F=F_used.to(lam.device), sigma2=sigma2,
        )
        return phi, F_used.detach(), sigma2

    if mode == "posterior":
        # Average phi over theta ~ q(theta)
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

        phi = acc_phi / float(S)
        F_used = acc_F / float(S)
        return phi.detach(), F_used.detach(), sigma2

    raise ValueError(f"Unknown VI phi mode '{mode}'. Choose from: plugin, posterior.")

@torch.no_grad()
def compute_phi_mcmc_full(
    *,
    model: SpectralCAR_FullVI,
    lam: torch.Tensor,
    theta_chain: torch.Tensor,          # [T, d]
    s_chain: Optional[torch.Tensor],    # [T] or None (log sigma2)
    U_train: torch.Tensor,
    U_full: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    mode: str = "plugin",               # "plugin" or "posterior"
    max_draws: int = 256,               # cap draws for posterior mode
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
      phi_full: [n]
      F_used  : [n]  (mean spectrum over draws for both modes)
      sigma2  : float (plugin sigma2 unless s_chain provided & mode=='posterior' uses per-draw)
    """
    # plugin beta/sigma2 from the TRAIN-fit model (clean, stable)
    beta_mean, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2_plug = float(sigma2_plugin.detach().cpu())

    # mean spectrum over chain (useful to report regardless)
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
        if T <= S:
            picks = np.arange(T)
        else:
            picks = np.linspace(0, T - 1, S).astype(int)

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

        phi = acc_phi / float(len(picks))
        F_used = acc_F / float(len(picks))
        # return sigma2_plug just as a reference (actual used may vary if s_chain exists)
        return phi.detach(), F_used.detach(), sigma2_plug

    raise ValueError(f"Unknown MCMC phi mode '{mode}'. Choose from: plugin, posterior.")

@torch.no_grad()
def phi_full_from_train(
    *,
    U_train: torch.Tensor,   # [n_tr, n]
    U_full: torch.Tensor,    # [n, n]
    X_train: torch.Tensor,   # [n_tr, p]
    y_train: torch.Tensor,   # [n_tr]
    beta: torch.Tensor,      # [p]
    F: torch.Tensor,         # [n]
    sigma2: float,
) -> torch.Tensor:
    """
    Posterior mean of full phi given TRAIN observations only and fixed (beta, F, sigma2).

    Model:
        phi = U_full z
        z ~ N(0, diag(F))
        y_train = X_train beta + U_train z + eps, eps ~ N(0, sigma2 I)

    Then:
        mu_z = (F / (F + sigma2)) ⊙ (U_train^T r_train)
        r_train = y_train - X_train beta

        E[phi | y_train, beta, F, sigma2] = U_full mu_z
    """
    r_tr = y_train - X_train @ beta                  # [n_tr]
    Ut_r = U_train.T @ r_tr                          # [n]
    shrink = (F / (F + sigma2)).clamp(0.0, 1.0)      # [n]
    mu_z = shrink * Ut_r                             # [n]
    return U_full @ mu_z                             # [n]



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
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pll_mc: int,
    vi_log_every: int,
    phi_vi_mode: str,
    phi_mcmc_mode: str,
    phi_draws: int,
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

    # Train VI
    for it in range(1, vi_iters + 1):
        opt.zero_grad()
        elbo, _ = model.elbo()
        loss = -elbo
        loss.backward()
        opt.step()

        if vi_log_every and (it % vi_log_every == 0 or it == 1 or it == vi_iters):
            print(f"  [VI] iter {it:>5}/{vi_iters}  ELBO={float(elbo.detach().cpu()):.3f}")

    # VI phi + spectrum
    # VI phi + spectrum (switch)
    with torch.no_grad():
        # 1) phi + spectrum summary in one place
        phi_vi_full, F_vi_used, _sigma2 = compute_phi_vi_full(
            model=model,
            lam=lam,
            U_train=U_tr,
            U_full=U_full,
            X_train=X_tr,
            y_train=y_tr,
            mode=phi_vi_mode,                # <-- SWITCH HERE
            S=max(1, int(phi_draws)),        # draws if posterior mode
        )

        rmse_phi_vi = float(torch.sqrt(torch.mean((phi_vi_full.cpu() - phi_true.cpu()) ** 2)).item())
        rmse_logF_vi = rmse_log_spectrum(F_vi_used, F_true)

        pll_vi = predictive_loglik_vi_heldout(
            model=model,
            y=y_full, X=X_full, U=U_full, lam=lam,
            train_idx=train_idx, test_idx=test_idx,
            S=pll_mc,
            use_plugin_beta=True,
            sample_sigma2=False,
        )

    F_vi_cpu = F_vi_used.detach().cpu()

    # define "mcmc outputs" so VI-only works cleanly
    phi_mcmc = None
    rmse_phi_mcmc = None
    F_mcmc = None
    rmse_logF_mcmc = None
    acc_s_mid = None
    acc_theta_mid = None


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
            print_every=5000
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

        theta_chain = out["theta"]                 # [S, d]
        s_chain = out.get("s", None)               # [S] or None

        # PLL (full y/X/U, conditioned on train/test)
        pll_mcmc = predictive_loglik_mcmc_heldout(
            model=model,
            theta_chain=theta_chain,
            s_chain=s_chain,
            y=y_full, X=X_full, U=U_full, lam=lam,
            train_idx=train_idx, test_idx=test_idx,
            S=pll_mc,
        )

        # Spectrum summary (keep this; used for RMSE + plots)
        F_mcmc = spectrum_mcmc_mean(model.filter, lam, theta_chain).detach()
        rmse_logF_mcmc = rmse_log_spectrum(F_mcmc, F_true)
        F_mcmc_cpu = F_mcmc.detach().cpu()

        # Phi summary (SWITCH HERE)
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
            f"{label} (VI)": F_vi_cpu.numpy(),
        }
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
        if (not skip_mcmc) and (mcmc_steps > 0) and (phi_mcmc_full is not None):
            diagnostics.plot_phi_mean_vs_true(
                coords=coords,
                mean_phi=phi_mcmc_full.to(device=device, dtype=torch.double),
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
        pll_vi=float(pll_vi) if "pll_vi" in locals() else None,
        pll_mcmc=float(pll_mcmc) if "pll_mcmc" in locals() else None,
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
    # sort descending PLL; fallback to logF RMSE
    if r.pll_mcmc is not None:
        return -r.pll_mcmc
    if r.pll_vi is not None:
        return -r.pll_vi
    return r.rmse_logF_mcmc if r.rmse_logF_mcmc is not None else r.rmse_logF_vi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, choices=["multiscale", "bandpass", "car"])

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
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--pll_mc", type=int, default=16, help="MC draws for predictive log-lik (VI/MCMC). Keep small.")
    p.add_argument("--vi_log_every", type=int, default=200, help="Print VI iteration every k steps (0 disables).")

    p.add_argument(
        "--phi_vi_mode",
        choices=["plugin", "posterior"],
        default="plugin",
        help="How to compute VI phi mean on full domain.",
    )
    p.add_argument(
        "--phi_mcmc_mode",
        choices=["plugin", "posterior"],
        default="plugin",
        help="How to compute MCMC phi mean on full domain.",
    )
    p.add_argument(
        "--phi_draws",
        type=int,
        default=128,
        help="Number of draws used when phi_*_mode=posterior (caps MCMC draws).",
    )

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
    train_idx, test_idx = make_split(n, test_frac=args.test_frac, seed=args.split_seed)

    tr = torch.as_tensor(train_idx, device=device)
    X_tr = X[tr, :]
    y_tr = y[tr]
    U_tr = U[tr, :]

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
            X_full=X, y_full=y, U_full=U,
            X_tr=X_tr, y_tr=y_tr, U_tr=U_tr,
            lam=lam, coords=coords,
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
            train_idx=train_idx,
            test_idx=test_idx,
            pll_mc=args.pll_mc,
            vi_log_every=args.vi_log_every,
            phi_vi_mode=args.phi_vi_mode,
            phi_mcmc_mode=args.phi_mcmc_mode,
            phi_draws=args.phi_draws,
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
            mcmc_phi  = f"{r.rmse_phi_mcmc:.4f}"  if r.rmse_phi_mcmc  is not None else "NA"

            pll_vi   = f"{r.pll_vi:.2f}"   if r.pll_vi   is not None else "NA"
            pll_mcmc = f"{r.pll_mcmc:.2f}" if r.pll_mcmc is not None else "NA"

            print(
                f"{r.label:<32} | "
                f"logF_RMSE(VI)={r.rmse_logF_vi:.4f}  logF_RMSE(MCMC)={mcmc_logF} | "
                f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={mcmc_phi} | "
                f"PLL(VI)={pll_vi}  PLL(MCMC)={pll_mcmc}"
            )

        # also write to file for paper artifacts
        compare_dir = outdir / "COMPARE"
        compare_dir.mkdir(parents=True, exist_ok=True)
        with open(compare_dir / "leaderboard.txt", "w", encoding="utf-8") as f:
            f.write(f"MISSPEC SUMMARY (truth={args.truth})\n")
            for r in sorted(results, key=_sort_key):
                mcmc_logF = f"{r.rmse_logF_mcmc:.4f}" if r.rmse_logF_mcmc is not None else "NA"
                mcmc_phi = f"{r.rmse_phi_mcmc:.4f}" if r.rmse_phi_mcmc is not None else "NA"
                pll_vi = f"{r.pll_vi:.2f}" if r.pll_vi is not None else "NA"
                pll_mcmc = f"{r.pll_mcmc:.2f}" if r.pll_mcmc is not None else "NA"
                f.write(
                    f"{r.label} | "
                    f"logF_RMSE(VI)={r.rmse_logF_vi:.4f}  logF_RMSE(MCMC)={mcmc_logF} | "
                    f"phi_RMSE(VI)={r.rmse_phi_vi:.4f}  phi_RMSE(MCMC)={mcmc_phi} | "
                    f"PLL(VI)={pll_vi}  PLL(MCMC)={pll_mcmc}\n"
                )


if __name__ == "__main__":
    main()
