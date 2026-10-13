"""
experiment3_fixed_beta_sensitivity.py

TARGETED SENSITIVITY: fixed intercept beta=0, gamma=1 by default.
The simulation data generation is IDENTICAL to the original experiment.
For fitting and oracle prediction, the design matrix is set to zero.
This eliminates the intercept from the observation and prediction equations,
while retaining a harmless one-column zero design compatible with the
SpectralCAR_HoldoutVI interface and its Gaussian-beta marginalization.
Since X beta = 0 identically, this is mathematically equivalent to fixing
beta=0 in the likelihood and predictive distribution.
This sensitivity must be stored separately from the main experiment.

experiment3_fixed_beta_sensitivity.py

Covariance + joint predictive experiment for the controlled power-warp comparison:
    Leroux CAR vs Adaptive Precision P-spline SDM-CAR.

The design deliberately keeps the graph eigenvectors U fixed and warps only
the eigenvalue coordinate supplied to the fitted models:

    mu_k = lambda_max * (lambda_k / lambda_max)**gamma.

Data are always generated from the same unwarped Leroux covariance

    Sigma_true = U diag(F_true) U^T,
    F_true(lambda_k) = tau2_true / [(1-rho_true) + rho_true lambda_k].

For each fitted model and posterior draw, this script reconstructs:

1. Latent spatial covariance
       Sigma = U diag(F) U^T

2. Latent correlation matrix
       R = D^{-1/2} Sigma D^{-1/2}

3. Full response covariance
       C = Sigma + sigma2 I

4. Conditional covariance of the held-out RESPONSE block
       V_y(H|O) = C_HH - C_HO C_OO^{-1} C_OH

5. Positive-mode latent covariance and correlation, with the graph's
   zero/constant Laplacian mode removed:
       Sigma_+ = U_+ diag(F_+) U_+^T
       R_+ = Corr(Sigma_+)

   For the connected rook graph this removes exactly one constant mode.
   Equivalently, Sigma_+ = P Sigma P with
       P = I - (1/n) 11^T.

The conditional covariance is computed draw-by-draw and only then averaged.
This is preferable to plugging the posterior-mean covariance into the
nonlinear Schur-complement formula.

Primary recovery metric:
    relative Frobenius error = ||A_hat - A_true||_F / ||A_true||_F

Secondary metrics:
    relative operator norm error
    diagonal RMSE
    off-diagonal RMSE
    off-diagonal correlation

The script saves:
    - per-run covariance metric tables
    - full-mode and positive-mode covariance/correlation matrices
    - posterior-mean response/conditional-covariance matrices
    - heatmaps for those matrices and their errors
    - across-seed summaries by gamma and model
    - paired SDM-minus-Leroux differences
    - trend plots versus POWER_GAMMA
    - aggregate 4x4 and four 2x2 predictive means, variances, and 95% intervals
    - paired degradation from gamma=1 for each model and its cross-model contrast
    - ELBO history for every fit, with selected convergence / spectral plots
    - across-seed paired t confidence intervals for selected metrics

Detailed 144x144 matrix CSVs and heatmaps are saved for selected representative
seed/gamma combinations by default; use --detailed-output all for every run.
A --pilot option supplies very short optimization/draw settings ONLY to test the
pipeline; pilot outputs are not suitable for scientific comparisons.

Example
-------
python experiment3_fixed_beta_sensitivity.py \
    --project-root "C:\\Users\\pd006\\Desktop\\internship_search\\sdm-car" \
    --seeds 111 222 333 444 555 \
    --gammas 1.0 \
    --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable
from statistics import NormalDist
import hashlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# LOCKED EXPERIMENT CONFIGURATION
# =============================================================================

N_ROWS = 12
N_COLS = 12
N = N_ROWS * N_COLS

TAU2_TRUE = 0.70
RHO_TRUE = 0.92
BETA_TRUE = 0.0
SIGMA2_TRUE = 0.15

ZERO_TOL = 1e-10

HOLDOUT_SIDE = 4
HOLDOUT_PATTERN = "middle"

FIX_SIGMA2 = True
NUM_MC = 8
GRAD_CLIP = 10.0
JITTER = 1e-8
BETA_PRIOR_VAR = 10.0

LEROUX_ITERS = 3000
LEROUX_LR = 1e-3

PSPLINE_ITERS = 12000
PSPLINE_LR = 3e-4

POSTERIOR_COV_DRAWS = 512

# Joint predictive evaluation
PREDICTIVE_DRAWS = 256
ORACLE_KL_MC = 1024
VARIOGRAM_P = 0.5

# Leroux initialization
INIT_LEROUX_TAU2 = 0.50
INIT_LEROUX_RHO = 0.80
LEROUX_RHO_EPS = 1e-4
LEROUX_LOG_STD_INIT = -3.0

# Adaptive precision P-spline initialization/prior
INIT_Q_LEFT = 0.20
INIT_Q_RIGHT = 20.0

ADAPTIVE_PSPLINE_KWARGS = {
    "degree": 3,
    "n_internal_knots": 4,
    "prior_std_log_q": 3.0,
    "global_scale_d2": 2.0,
    "prior_std_log_lambda": 2.0,
    "mu_log_lambda_init": 0.0,
    "log_std_log_lambda": -3.0,
    "log_std_log_q": -3.0,
    "log_std_d2": -3.0,
    "init_d2": 0.0,
    "log_q_min": -20.0,
    "log_q_max": 20.0,
}

DEFAULT_SEEDS = [111, 222, 333, 444, 555]
DEFAULT_GAMMAS = [1.00]
OUTPUT_SCHEMA_VERSION = 3  # avoid treating older results as current

DEFAULT_PROJECT_ROOT = Path.cwd()

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.double)

LerouxCARFilterFullVI = None
AdaptivePrecisionPSplineFullVI = None
SpectralCAR_HoldoutVI = None


# =============================================================================
# GRAPH / DATA / WARP
# =============================================================================

def build_rook_adjacency(n_rows: int, n_cols: int) -> np.ndarray:
    n = n_rows * n_cols
    W = np.zeros((n, n), dtype=float)

    def idx(r, c):
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols):
            i = idx(r, c)

            if c + 1 < n_cols:
                j = idx(r, c + 1)
                W[i, j] = W[j, i] = 1.0

            if r + 1 < n_rows:
                j = idx(r + 1, c)
                W[i, j] = W[j, i] = 1.0

    return W


def graph_laplacian(W: np.ndarray) -> np.ndarray:
    D = np.diag(W.sum(axis=1))
    L = D - W
    return 0.5 * (L + L.T)


def power_warp(lam: np.ndarray, gamma: float) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam_max = float(lam.max())

    mu = lam_max * (
        np.clip(lam, 0.0, None) / lam_max
    ) ** gamma
    mu[lam <= ZERO_TOL] = 0.0
    return mu


def holdout_block_mask(
    n_rows: int,
    n_cols: int,
    side: int,
    pattern: str = "middle",
) -> np.ndarray:
    if pattern != "middle":
        raise ValueError("This covariance-only script currently uses middle holdout.")

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    r0 = (n_rows - side) // 2
    c0 = (n_cols - side) // 2
    mask[r0:r0 + side, c0:c0 + side] = True
    return mask.ravel()


def generate_data(
    seed: int,
    lam_true: np.ndarray,
    U_true: np.ndarray,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.ones((N, 1), dtype=float)

    F_true = TAU2_TRUE / (
        (1.0 - RHO_TRUE) + RHO_TRUE * lam_true
    )

    phi_true = U_true @ (
        np.sqrt(F_true) * rng.normal(size=N)
    )

    eta_true = X[:, 0] * BETA_TRUE + phi_true

    y = eta_true + rng.normal(
        loc=0.0,
        scale=np.sqrt(SIGMA2_TRUE),
        size=N,
    )

    Sigma_true = covariance_from_spectrum(U_true, F_true)
    C_true = Sigma_true + SIGMA2_TRUE * np.eye(N)

    return {
        "X": X,
        "F_true": F_true,
        "phi_true": phi_true,
        "eta_true": eta_true,
        "y": y,
        "Sigma_true": Sigma_true,
        "C_true": C_true,
    }


# =============================================================================
# MODEL CONSTRUCTORS / VI
# =============================================================================

def rho_to_raw(rho: float, rho_eps: float = LEROUX_RHO_EPS) -> float:
    p = rho / (1.0 - rho_eps)
    if not 0.0 < p < 1.0:
        raise ValueError("rho incompatible with rho_eps.")
    return math.log(p / (1.0 - p))


def make_leroux_filter():
    return LerouxCARFilterFullVI(
        mu_log_tau2=math.log(INIT_LEROUX_TAU2),
        log_std_log_tau2=LEROUX_LOG_STD_INIT,
        mu_rho_raw=rho_to_raw(INIT_LEROUX_RHO),
        log_std_rho_raw=LEROUX_LOG_STD_INIT,
        fixed_rho=None,
        rho_eps=LEROUX_RHO_EPS,
    ).to(DEVICE)


def make_pspline_filter(lam_t: torch.Tensor):
    return AdaptivePrecisionPSplineFullVI(
        lam_max=float(lam_t.max().item()),
        mu_log_q_left=math.log(INIT_Q_LEFT),
        mu_log_q_right=math.log(INIT_Q_RIGHT),
        **ADAPTIVE_PSPLINE_KWARGS,
    ).to(DEVICE)


def fit_spectral_vi(
    *,
    label: str,
    filter_module,
    iterations: int,
    learning_rate: float,
    seed: int,
    X_t: torch.Tensor,
    y_fit_t: torch.Tensor,
    lam_warp_t: torch.Tensor,
    U_warp_t: torch.Tensor,
    is_holdout_t: torch.Tensor,
    prior_V0: torch.Tensor,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SpectralCAR_HoldoutVI(
        X=X_t,
        y=y_fit_t,
        lam=lam_warp_t,
        U=U_warp_t,
        filter_module=filter_module,
        is_holdout=is_holdout_t,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(SIGMA2_TRUE),
        log_std_log_sigma2=-2.3,
        num_mc=NUM_MC,
        fixed_sigma2=SIGMA2_TRUE,
        jitter=JITTER,
    ).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    history = []
    start_time = time.perf_counter()

    for it in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        elbo, _ = model.elbo()
        loss = -elbo

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{label}: non-finite loss at iteration {it}."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()

        history.append(float(elbo.detach().cpu()))

        if it % 500 == 0:
            print(
                f"      {label}: {it}/{iterations}, "
                f"mean last 100 ELBO={np.mean(history[-100:]):.3f}",
                flush=True,
            )

    return {
        "label": label,
        "model": model,
        "history": np.asarray(history, dtype=float),
        "fit_seconds": time.perf_counter() - start_time,
    }


# =============================================================================
# COVARIANCE MATHEMATICS
# =============================================================================

def covariance_from_spectrum(
    U: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    F = np.asarray(F, dtype=float).reshape(-1)

    Sigma = (U * F[None, :]) @ U.T
    return 0.5 * (Sigma + Sigma.T)


def covariance_to_correlation(Sigma: np.ndarray) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-15, None))
    R = Sigma / np.outer(d, d)
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return 0.5 * (R + R.T)


def stable_cholesky_np(
    A: np.ndarray,
    base_jitter: float = 1e-10,
    max_tries: int = 8,
) -> np.ndarray:
    A = 0.5 * (A + A.T)
    eye = np.eye(A.shape[0])
    scale = max(float(np.mean(np.abs(np.diag(A)))), 1.0)

    last_error = None
    for k in range(max_tries):
        jitter = base_jitter * (10.0 ** k) * scale
        try:
            return np.linalg.cholesky(A + jitter * eye)
        except np.linalg.LinAlgError as exc:
            last_error = exc

    raise np.linalg.LinAlgError("stable_cholesky_np failed") from last_error


def solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    L = stable_cholesky_np(A)
    y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, y)


def conditional_response_covariance(
    C: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """
    V_y(H|O) = C_HH - C_HO C_OO^{-1} C_OH
    """
    C_OO = C[np.ix_(train_idx, train_idx)]
    C_HO = C[np.ix_(test_idx, train_idx)]
    C_OH = C_HO.T
    C_HH = C[np.ix_(test_idx, test_idx)]

    solved = solve_spd(C_OO, C_OH)
    V = C_HH - C_HO @ solved
    return 0.5 * (V + V.T)


def conditional_response_covariance_from_modes(
    U_test: np.ndarray,
    response_mode_variances: np.ndarray,
) -> np.ndarray:
    """Exact Cov(y_H|y_O) by inverting a held-out precision block.

    Since C = U diag(v) U.T and U is orthonormal,
        Cov(y_H | y_O) = [U_H diag(1/v) U_H.T]^{-1}.
    Avoids a Cholesky factorization of the much larger observed block.
    Regression coefficients are held fixed here, as in the original script.
    """
    inv_v = 1.0 / np.maximum(np.asarray(response_mode_variances), 1e-12)
    precision_hh = (U_test * inv_v[None, :]) @ U_test.T
    precision_hh = 0.5 * (precision_hh + precision_hh.T)
    return solve_spd(precision_hh, np.eye(U_test.shape[0]))


# =============================================================================
# KL DIVERGENCE + JOINT PREDICTIVE SCORING
# =============================================================================

def gaussian_kl_same_basis_per_dim(
    truth_variances: np.ndarray,
    fitted_variances: np.ndarray,
) -> float:
    """
    KL[N(0, diag(truth)) || N(0, diag(fitted))] per dimension.

    This is exact whenever truth and fitted covariance operators share the same
    orthonormal eigenvectors, which is true in this controlled power-warp
    experiment.
    """
    v0 = np.clip(np.asarray(truth_variances, dtype=float), 1e-12, None)
    v1 = np.clip(np.asarray(fitted_variances, dtype=float), 1e-12, None)
    if v0.shape != v1.shape:
        raise ValueError("truth_variances and fitted_variances must match.")

    terms = v0 / v1 - 1.0 + np.log(v1 / v0)
    return float(max(0.0, 0.5 * np.mean(terms)))


def gaussian_kl_covariance_per_dim(
    truth_cov: np.ndarray,
    fitted_cov: np.ndarray,
) -> float:
    """KL[N(0, truth_cov) || N(0, fitted_cov)] per dimension."""
    truth_cov = 0.5 * (np.asarray(truth_cov) + np.asarray(truth_cov).T)
    fitted_cov = 0.5 * (np.asarray(fitted_cov) + np.asarray(fitted_cov).T)
    d = truth_cov.shape[0]

    solved = solve_spd(fitted_cov, truth_cov)
    trace_term = float(np.trace(solved))

    L0 = stable_cholesky_np(truth_cov)
    L1 = stable_cholesky_np(fitted_cov)
    logdet0 = 2.0 * np.log(np.diag(L0)).sum()
    logdet1 = 2.0 * np.log(np.diag(L1)).sum()

    kl = 0.5 * (trace_term - d + logdet1 - logdet0)
    return float(max(0.0, kl / d))


def stable_cholesky_torch(
    matrix: torch.Tensor,
    base_jitter: float = 1e-10,
    max_tries: int = 8,
) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.T)
    eye = torch.eye(
        matrix.shape[0],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    scale = torch.diagonal(matrix).abs().mean().detach().clamp_min(1.0)
    last_error = None

    for attempt in range(max_tries):
        jitter = base_jitter * (10.0 ** attempt) * scale
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
        except RuntimeError as error:
            last_error = error

    raise RuntimeError("stable_cholesky_torch failed.") from last_error


def oracle_response_predictive(
    *,
    C_true: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prior_V0: np.ndarray,
    prior_m0: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Oracle Bayesian posterior predictive distribution for y_H | y_O using the
    TRUE response covariance C_true and the SAME beta prior as the fitted model.

    Beta uncertainty is integrated in the same way as for the fitted models.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    prior_V0 = np.asarray(prior_V0, dtype=float)

    p = X.shape[1]
    if prior_m0 is None:
        prior_m0 = np.zeros(p, dtype=float)
    prior_m0 = np.asarray(prior_m0, dtype=float).reshape(-1)

    X_O = X[train_idx]
    X_H = X[test_idx]
    y_O = y[train_idx]

    C_OO = C_true[np.ix_(train_idx, train_idx)]
    C_HO = C_true[np.ix_(test_idx, train_idx)]
    C_OH = C_HO.T
    C_HH = C_true[np.ix_(test_idx, test_idx)]

    Cinv_X = solve_spd(C_OO, X_O)
    Cinv_y = solve_spd(C_OO, y_O)
    gain = solve_spd(C_OO, C_OH).T

    V0_inv = np.linalg.inv(prior_V0)
    V_beta = np.linalg.inv(V0_inv + X_O.T @ Cinv_X)
    m_beta = V_beta @ (
        V0_inv @ prior_m0 + X_O.T @ Cinv_y
    )

    A_beta = X_H - gain @ X_O
    mean = gain @ y_O + A_beta @ m_beta

    conditional_cov = C_HH - gain @ C_OH
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)

    covariance = conditional_cov + A_beta @ V_beta @ A_beta.T
    covariance = 0.5 * (covariance + covariance.T)

    return {
        "mean": mean,
        "cov": covariance,
        "conditional_cov_given_beta": conditional_cov,
        "beta_mean": m_beta,
        "beta_cov": V_beta,
    }


@torch.no_grad()
def posterior_joint_response_prediction(
    model,
    *,
    num_mc: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Joint posterior predictive mixture for the full held-out response vector.

    Each VI hyperparameter draw gives one multivariate Gaussian component

        y_H | y_O, theta^(k) ~ N(m_k, V_k),

    with beta uncertainty integrated analytically.
    """
    K = int(num_mc)
    if K < 2:
        raise ValueError("num_mc must be at least 2.")

    torch.manual_seed(seed)

    component_means = []
    component_covs = []
    predictive_draws = []

    for _ in range(K):
        sigma2 = model._sample_sigma2()
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(
            model.lam, theta
        ).clamp_min(model.min_variance)

        terms = model._observed_terms(F, sigma2)
        m_beta, V_beta = model._beta_update_from_terms(terms)

        response_precision_modes = 1.0 / (F + sigma2)

        P_HH = (
            model.U_test * response_precision_modes.unsqueeze(0)
        ) @ model.U_test.T
        P_HO = (
            model.U_test * response_precision_modes.unsqueeze(0)
        ) @ model.U_train.T

        chol_PHH = stable_cholesky_torch(
            P_HH,
            base_jitter=max(model.jitter, 1e-10),
        )

        # B = P_HH^{-1} P_HO; covariance-form gain is -B.
        B = torch.cholesky_solve(P_HO, chol_PHH)
        conditional_cov = torch.cholesky_inverse(chol_PHH)
        conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)

        A_beta = model.X_test + B @ model.X_train
        pred_mean = -B @ model.y_train + A_beta @ m_beta

        pred_cov = conditional_cov + A_beta @ V_beta @ A_beta.T
        pred_cov = 0.5 * (pred_cov + pred_cov.T)

        chol_pred = stable_cholesky_torch(
            pred_cov,
            base_jitter=max(model.jitter, 1e-10),
        )
        draw = pred_mean + chol_pred @ torch.randn(
            model.n_test,
            dtype=pred_mean.dtype,
            device=pred_mean.device,
        )

        component_means.append(pred_mean)
        component_covs.append(pred_cov)
        predictive_draws.append(draw)

    means = torch.stack(component_means, dim=0)
    covs = torch.stack(component_covs, dim=0)
    draws = torch.stack(predictive_draws, dim=0)

    mean = means.mean(dim=0)
    second = (
        covs + means.unsqueeze(2) * means.unsqueeze(1)
    ).mean(dim=0)
    covariance = second - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)

    return {
        "mean": mean,
        "cov": covariance,
        "component_means": means,
        "component_covs": covs,
        "draws": draws,
        "test_idx": model.test_idx,
    }


def mixture_logpdf_torch(
    samples: torch.Tensor,
    component_means: torch.Tensor,
    component_covs: torch.Tensor,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    """Evaluate an equally weighted Gaussian-mixture log density."""
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)

    K, h = component_means.shape
    eye = torch.eye(
        h, dtype=component_covs.dtype, device=component_covs.device
    )

    scale = (
        torch.diagonal(component_covs, dim1=-2, dim2=-1)
        .abs()
        .mean(dim=1)
        .clamp_min(1.0)
    )
    base_covs = 0.5 * (
        component_covs + component_covs.transpose(-1, -2)
    )
    last_error = None
    chol = None
    for attempt in range(8):
        jitter = (1e-10 * (10.0 ** attempt) * scale)[:, None, None]
        try:
            chol = torch.linalg.cholesky(base_covs + jitter * eye)
            break
        except RuntimeError as error:
            last_error = error
    if chol is None:
        raise RuntimeError(
            "Batched Cholesky failed for predictive mixture covariances."
        ) from last_error
    precision = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)
    ).sum(dim=1)

    constant = h * math.log(2.0 * math.pi)
    outputs = []

    for start in range(0, samples.shape[0], batch_size):
        x = samples[start:start + batch_size]
        residual = x[:, None, :] - component_means[None, :, :]
        quadratic = torch.einsum(
            "bki,kij,bkj->bk",
            residual,
            precision,
            residual,
        )
        component_logpdf = -0.5 * (
            constant + logdet[None, :] + quadratic
        )
        outputs.append(
            torch.logsumexp(component_logpdf, dim=1) - math.log(K)
        )

    return torch.cat(outputs, dim=0)


def gaussian_logpdf_torch(
    samples: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)

    h = mean.numel()
    chol = stable_cholesky_torch(covariance)
    residual = samples - mean.unsqueeze(0)
    solved = torch.linalg.solve_triangular(
        chol, residual.T, upper=False
    )
    quadratic = solved.square().sum(dim=0)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()

    return -0.5 * (
        h * math.log(2.0 * math.pi) + logdet + quadratic
    )


def energy_score(
    target: torch.Tensor,
    draws: torch.Tensor,
) -> float:
    """Multivariate CRPS analogue; lower is better."""
    target = target.reshape(-1)
    first = torch.linalg.vector_norm(
        draws - target.unsqueeze(0), dim=1
    ).mean()
    pairwise = torch.cdist(draws, draws, p=2)
    second = 0.5 * pairwise.mean()
    return float((first - second).detach().cpu())


def heldout_variogram_pairs(
    test_idx: np.ndarray,
    n_cols: int,
):
    """Inverse-grid-distance pair weights normalized to sum to one."""
    test_idx = np.asarray(test_idx, dtype=int)
    rows = test_idx // n_cols
    cols = test_idx % n_cols

    ii, jj = np.triu_indices(test_idx.size, k=1)
    distances = np.sqrt(
        (rows[ii] - rows[jj]) ** 2
        + (cols[ii] - cols[jj]) ** 2
    )
    weights = 1.0 / np.clip(distances, 1e-12, None)
    weights = weights / weights.sum()
    return ii, jj, weights


def variogram_score(
    target: torch.Tensor,
    draws: torch.Tensor,
    *,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_weights: np.ndarray,
    p: float = VARIOGRAM_P,
) -> float:
    """Spatial variogram score; lower is better."""
    device = draws.device
    dtype = draws.dtype

    i = torch.as_tensor(pair_i, dtype=torch.long, device=device)
    j = torch.as_tensor(pair_j, dtype=torch.long, device=device)
    w = torch.as_tensor(pair_weights, dtype=dtype, device=device)

    observed = torch.abs(target[i] - target[j]).pow(p)
    predicted = torch.abs(
        draws[:, i] - draws[:, j]
    ).pow(p).mean(dim=0)

    score = torch.sum(w * (observed - predicted).square())
    return float(score.detach().cpu())


def draw_oracle_samples(
    oracle_mean: np.ndarray,
    oracle_cov: np.ndarray,
    *,
    draws: int,
    seed: int,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    torch.manual_seed(seed)
    mean = torch.tensor(oracle_mean, dtype=dtype, device=device)
    cov = torch.tensor(oracle_cov, dtype=dtype, device=device)
    chol = stable_cholesky_torch(cov)
    z = torch.randn(draws, mean.numel(), dtype=dtype, device=device)
    return mean.unsqueeze(0) + z @ chol.T


def joint_predictive_metrics(
    *,
    target: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    oracle: Dict[str, np.ndarray],
    oracle_samples: torch.Tensor,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_weights: np.ndarray,
) -> Dict[str, float]:
    target_t = torch.as_tensor(
        target,
        dtype=prediction["mean"].dtype,
        device=prediction["mean"].device,
    ).reshape(-1)

    h = target_t.numel()
    error = prediction["mean"] - target_t
    rmse = torch.sqrt(torch.mean(error.square()))

    log_density = mixture_logpdf_torch(
        target_t,
        prediction["component_means"],
        prediction["component_covs"],
    )[0]
    joint_nlpd_per_site = -log_density / float(h)

    es = energy_score(target_t, prediction["draws"])
    vs = variogram_score(
        target_t,
        prediction["draws"],
        pair_i=pair_i,
        pair_j=pair_j,
        pair_weights=pair_weights,
        p=VARIOGRAM_P,
    )

    oracle_mean_t = torch.as_tensor(
        oracle["mean"], dtype=target_t.dtype, device=target_t.device
    )
    oracle_cov_t = torch.as_tensor(
        oracle["cov"], dtype=target_t.dtype, device=target_t.device
    )

    log_p0 = gaussian_logpdf_torch(
        oracle_samples, oracle_mean_t, oracle_cov_t
    )
    log_pm = mixture_logpdf_torch(
        oracle_samples,
        prediction["component_means"],
        prediction["component_covs"],
    )
    log_ratio = log_p0 - log_pm

    kl_per_site = log_ratio.mean() / float(h)
    kl_mcse_per_site = (
        log_ratio.std(unbiased=True)
        / math.sqrt(log_ratio.numel())
        / float(h)
    )

    oracle_cov_np = np.asarray(oracle["cov"], dtype=float)
    pred_cov_np = prediction["cov"].detach().cpu().numpy()
    predictive_cov_rel_frob = float(
        np.linalg.norm(pred_cov_np - oracle_cov_np, ord="fro")
        / np.linalg.norm(oracle_cov_np, ord="fro")
    )

    return {
        "response_RMSE": float(rmse.detach().cpu()),
        "joint_response_NLPD_per_site": float(
            joint_nlpd_per_site.detach().cpu()
        ),
        "energy_score": es,
        f"variogram_score_p{str(VARIOGRAM_P).replace('.', 'p')}": vs,
        "oracle_conditional_KL_per_site": float(
            kl_per_site.detach().cpu()
        ),
        "oracle_conditional_KL_MCSE_per_site": float(
            kl_mcse_per_site.detach().cpu()
        ),
        "joint_predictive_cov_relative_frobenius": predictive_cov_rel_frob,
    }


# =============================================================================
# MATRIX RECOVERY METRICS
# =============================================================================

def matrix_metrics(
    estimated: np.ndarray,
    target: np.ndarray,
    *,
    include_operator: bool = True,
) -> Dict[str, float]:
    estimated = np.asarray(estimated, dtype=float)
    target = np.asarray(target, dtype=float)

    if estimated.shape != target.shape:
        raise ValueError("estimated and target must have the same shape.")

    n = target.shape[0]
    delta = estimated - target
    eps = np.finfo(float).eps

    frob = np.linalg.norm(delta, ord="fro")
    target_frob = np.linalg.norm(target, ord="fro")

    # For symmetric matrices, the operator norm is max |eigenvalue|.
    # This expensive diagnostic is omitted on individual posterior draws.
    if include_operator:
        op = float(np.max(np.abs(np.linalg.eigvalsh(delta))))
        target_op = float(np.max(np.abs(np.linalg.eigvalsh(target))))

    diag_delta = np.diag(delta)
    diagonal_rmse = float(
        np.sqrt(np.mean(diag_delta ** 2))
    )

    off_mask = ~np.eye(n, dtype=bool)
    off_delta = delta[off_mask]
    offdiagonal_rmse = float(
        np.sqrt(np.mean(off_delta ** 2))
    )

    est_off = estimated[off_mask]
    target_off = target[off_mask]

    if np.std(est_off) > 1e-14 and np.std(target_off) > 1e-14:
        offdiag_corr = float(
            np.corrcoef(est_off, target_off)[0, 1]
        )
    else:
        offdiag_corr = np.nan

    out = {
        "relative_frobenius": float(
            frob / max(target_frob, eps)
        ),
        "diagonal_rmse": diagonal_rmse,
        "offdiagonal_rmse": offdiagonal_rmse,
        "offdiagonal_correlation": offdiag_corr,
    }
    if include_operator:
        out["relative_operator_norm"] = op / max(target_op, eps)
    return out


def summarize_draw_metric_rows(
    metric_rows: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    df = pd.DataFrame(list(metric_rows))
    out = {}

    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        out[f"{col}__draw_mean"] = float(np.nanmean(values))
        out[f"{col}__draw_sd"] = float(np.nanstd(values, ddof=1))
        out[f"{col}__draw_median"] = float(np.nanmedian(values))
        out[f"{col}__draw_q025"] = float(np.nanquantile(values, 0.025))
        out[f"{col}__draw_q975"] = float(np.nanquantile(values, 0.975))

    return out


# =============================================================================
# POSTERIOR COVARIANCE RECOVERY
# =============================================================================

@torch.no_grad()
def posterior_covariance_recovery(
    result,
    *,
    U: np.ndarray,
    positive_mode_mask: np.ndarray,
    Sigma_true: np.ndarray,
    C_true: np.ndarray,
    R_true: np.ndarray,
    V_cond_true: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    draws: int,
    seed: int,
):
    """
    Compute covariance quantities draw-by-draw.

    In addition to the original full-mode matrices, this function removes the
    zero/constant Laplacian mode and evaluates covariance recovery on the
    positive-mode spatial subspace:

        Sigma_+ = U_+ diag(F_+) U_+^T.

    On a connected graph the removed zero mode is proportional to the constant
    vector 1, so Sigma_+ is the covariance of spatial contrasts rather than the
    global/common component.

    Returns posterior-mean matrices plus:
      - metrics of posterior-mean matrix vs truth
      - summaries of draw-level matrix errors
      - full-mode and positive-mode spectral/Frobenius identity checks
    """
    model = result["model"]
    torch.manual_seed(seed)

    U = np.asarray(U, dtype=float)
    positive_mode_mask = np.asarray(positive_mode_mask, dtype=bool).reshape(-1)

    n = U.shape[0]
    h = len(test_idx)

    if U.shape[1] != positive_mode_mask.size:
        raise ValueError(
            "positive_mode_mask must have one entry for every column of U."
        )
    if not np.any(positive_mode_mask):
        raise ValueError("positive_mode_mask contains no positive modes.")

    U_positive = U[:, positive_mode_mask]
    U_test = U[np.asarray(test_idx, dtype=int)]

    # Reconstruct the target modal covariance in the SAME graph-eigenvector
    # ordering. This is important because eigvalsh(Sigma_true) sorts values
    # independently and therefore does not preserve graph-mode labels.
    F_true_modal = np.diag(U.T @ Sigma_true @ U)
    F_true_positive = F_true_modal[positive_mode_mask]

    Sigma_true_positive = covariance_from_spectrum(
        U_positive,
        F_true_positive,
    )
    R_true_positive = covariance_to_correlation(
        Sigma_true_positive
    )

    Sigma_sum = np.zeros((n, n), dtype=float)
    C_sum = np.zeros((n, n), dtype=float)
    R_sum = np.zeros((n, n), dtype=float)
    V_cond_sum = np.zeros((h, h), dtype=float)
    F_sum = np.zeros(n, dtype=float)

    Sigma_positive_sum = np.zeros((n, n), dtype=float)
    R_positive_sum = np.zeros((n, n), dtype=float)

    sigma_metric_rows = []
    response_metric_rows = []
    corr_metric_rows = []
    conditional_metric_rows = []
    sigma_positive_metric_rows = []
    corr_positive_metric_rows = []

    # Exact orthogonal-basis identity checks:
    #
    # full modes:
    #   ||Sigma_hat - Sigma_true||_F
    #       = ||F_hat - F_true||_2
    #
    # positive modes:
    #   ||Sigma_hat,+ - Sigma_true,+||_F
    #       = ||F_hat,+ - F_true,+||_2
    spectral_vs_matrix_identity_error = []
    positive_spectral_vs_matrix_identity_error = []

    for _ in range(int(draws)):
        theta = model.filter.sample_unconstrained()
        F_t = model.filter.spectrum(
            model.lam, theta
        ).clamp_min(1e-12)

        F = F_t.detach().cpu().numpy()

        # ------------------------------------------------------------------
        # Full-mode covariance quantities
        # ------------------------------------------------------------------
        Sigma = covariance_from_spectrum(U, F)
        C = Sigma + SIGMA2_TRUE * np.eye(n)
        R = covariance_to_correlation(Sigma)
        V_cond = conditional_response_covariance_from_modes(
            U_test, F + SIGMA2_TRUE
        )

        # ------------------------------------------------------------------
        # Positive-mode covariance quantities: remove the zero/constant mode
        # ------------------------------------------------------------------
        F_positive = F[positive_mode_mask]
        Sigma_positive = covariance_from_spectrum(
            U_positive,
            F_positive,
        )
        R_positive = covariance_to_correlation(
            Sigma_positive
        )

        Sigma_sum += Sigma
        C_sum += C
        R_sum += R
        V_cond_sum += V_cond
        F_sum += F

        Sigma_positive_sum += Sigma_positive
        R_positive_sum += R_positive

        sigma_metrics = matrix_metrics(Sigma, Sigma_true, include_operator=False)
        sigma_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(F_true_modal, F)
        )
        sigma_metric_rows.append(sigma_metrics)

        response_metrics = matrix_metrics(C, C_true, include_operator=False)
        response_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(
                F_true_modal + SIGMA2_TRUE,
                F + SIGMA2_TRUE,
            )
        )
        response_metric_rows.append(response_metrics)

        corr_metric_rows.append(matrix_metrics(R, R_true, include_operator=False))

        conditional_metrics = matrix_metrics(V_cond, V_cond_true, include_operator=False)
        conditional_metrics["gaussian_kl_covariance_only_per_dim"] = (
            gaussian_kl_covariance_per_dim(V_cond_true, V_cond)
        )
        conditional_metric_rows.append(conditional_metrics)

        sigma_positive_metrics = matrix_metrics(
            Sigma_positive,
            Sigma_true_positive,
            include_operator=False,
        )
        sigma_positive_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(
                F_true_positive, F_positive
            )
        )
        sigma_positive_metric_rows.append(sigma_positive_metrics)

        corr_positive_metric_rows.append(
            matrix_metrics(R_positive, R_true_positive, include_operator=False)
        )

        lhs = np.linalg.norm(
            Sigma - Sigma_true,
            ord="fro",
        )
        rhs = np.linalg.norm(
            F - F_true_modal
        )
        spectral_vs_matrix_identity_error.append(
            abs(lhs - rhs)
        )

        lhs_positive = np.linalg.norm(
            Sigma_positive - Sigma_true_positive,
            ord="fro",
        )
        rhs_positive = np.linalg.norm(
            F_positive - F_true_positive
        )
        positive_spectral_vs_matrix_identity_error.append(
            abs(lhs_positive - rhs_positive)
        )

    K = float(draws)

    Sigma_mean = Sigma_sum / K
    C_mean = C_sum / K
    R_mean = R_sum / K
    V_cond_mean = V_cond_sum / K
    F_mean = F_sum / K

    Sigma_positive_mean = Sigma_positive_sum / K
    R_positive_mean = R_positive_sum / K

    posterior_mean_metrics = {
        "latent_covariance": matrix_metrics(
            Sigma_mean, Sigma_true
        ),
        "latent_covariance_positive_modes": matrix_metrics(
            Sigma_positive_mean,
            Sigma_true_positive,
        ),
        "response_covariance": matrix_metrics(
            C_mean, C_true
        ),
        "latent_correlation": matrix_metrics(
            R_mean, R_true
        ),
        "latent_correlation_positive_modes": matrix_metrics(
            R_positive_mean,
            R_true_positive,
        ),
        "conditional_response_covariance": matrix_metrics(
            V_cond_mean, V_cond_true
        ),
    }

    # Gaussian KL evaluated at the posterior-mean spectrum/covariance.
    posterior_mean_metrics["latent_covariance"][
        "gaussian_kl_truth_to_fit_per_dim"
    ] = gaussian_kl_same_basis_per_dim(
        F_true_modal, F_mean
    )
    posterior_mean_metrics["latent_covariance_positive_modes"][
        "gaussian_kl_truth_to_fit_per_dim"
    ] = gaussian_kl_same_basis_per_dim(
        F_true_positive, F_mean[positive_mode_mask]
    )
    posterior_mean_metrics["response_covariance"][
        "gaussian_kl_truth_to_fit_per_dim"
    ] = gaussian_kl_same_basis_per_dim(
        F_true_modal + SIGMA2_TRUE,
        F_mean + SIGMA2_TRUE,
    )
    posterior_mean_metrics["conditional_response_covariance"][
        "gaussian_kl_covariance_only_per_dim"
    ] = gaussian_kl_covariance_per_dim(
        V_cond_true, V_cond_mean
    )

    draw_metric_summaries = {
        "latent_covariance": summarize_draw_metric_rows(
            sigma_metric_rows
        ),
        "latent_covariance_positive_modes": summarize_draw_metric_rows(
            sigma_positive_metric_rows
        ),
        "response_covariance": summarize_draw_metric_rows(
            response_metric_rows
        ),
        "latent_correlation": summarize_draw_metric_rows(
            corr_metric_rows
        ),
        "latent_correlation_positive_modes": summarize_draw_metric_rows(
            corr_positive_metric_rows
        ),
        "conditional_response_covariance": summarize_draw_metric_rows(
            conditional_metric_rows
        ),
    }

    return {
        "F_mean": F_mean,
        "Sigma_mean": Sigma_mean,
        "C_mean": C_mean,
        "R_mean": R_mean,
        "V_cond_mean": V_cond_mean,
        "Sigma_positive_mean": Sigma_positive_mean,
        "R_positive_mean": R_positive_mean,
        "Sigma_true_positive": Sigma_true_positive,
        "R_true_positive": R_true_positive,
        "posterior_mean_metrics": posterior_mean_metrics,
        "draw_metric_summaries": draw_metric_summaries,
        "max_spectral_frobenius_identity_error": float(
            np.max(spectral_vs_matrix_identity_error)
        ),
        "max_positive_mode_spectral_frobenius_identity_error": float(
            np.max(positive_spectral_vs_matrix_identity_error)
        ),
    }

# =============================================================================
# SPATIAL AGGREGATE UNCERTAINTY (RESPONSE, BETA + PARAMETER UNCERTAINTY)
# =============================================================================

def make_holdout_aggregate_weights(
    test_idx: np.ndarray,
    n_cols: int = N_COLS,
) -> list[dict]:
    """Full held-out block and its four equal 2x2 subblocks.

    The selection is made in GLOBAL lattice coordinates, while each returned
    normalized weight vector is indexed in the held-out response order.
    """
    test_idx = np.asarray(test_idx, dtype=int)
    rows, cols = test_idx // n_cols, test_idx % n_cols
    r_vals, c_vals = np.unique(rows), np.unique(cols)
    if (r_vals.size != HOLDOUT_SIDE or c_vals.size != HOLDOUT_SIDE
            or len(test_idx) != HOLDOUT_SIDE ** 2 or HOLDOUT_SIDE % 2):
        raise ValueError("Aggregates require the full even-sided square holdout.")
    ans = []
    specifications = [(f"full_{HOLDOUT_SIDE}x{HOLDOUT_SIDE}", "full_block",
                       np.ones(len(test_idx), dtype=bool))]
    half = HOLDOUT_SIDE // 2
    for ri in range(2):
        for ci in range(2):
            mask = (np.isin(rows, r_vals[ri*half:(ri+1)*half])
                    & np.isin(cols, c_vals[ci*half:(ci+1)*half]))
            specifications.append((f"quadrant_{ri+1}{ci+1}", "subblock", mask))
    for name, scale, selected in specifications:
        k = int(selected.sum())
        if k != (HOLDOUT_SIDE ** 2 if scale == "full_block" else half ** 2):
            raise AssertionError("Unexpected aggregate site count")
        weights = selected.astype(float) / k
        ans.append({"aggregate_id": name, "aggregate_scale": scale,
                    "n_sites": k, "weights": weights})
    return ans


def normal_mixture_quantile(
    means: torch.Tensor,
    variances: torch.Tensor,
    probability: float,
) -> float:
    """Numerically invert the 1D Gaussian mixture CDF (not draw quantiles)."""
    means = means.detach().cpu().double().reshape(-1)
    sd = variances.detach().cpu().double().clamp_min(1e-14).sqrt().reshape(-1)
    lo = float((means - 12.0 * sd).min().item())
    hi = float((means + 12.0 * sd).max().item())
    for _ in range(65):
        mid = (lo + hi) / 2
        cdf = torch.special.ndtr((mid - means) / sd).mean().item()
        if cdf < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def aggregate_prediction_rows(
    *,
    seed: int,
    gamma: float,
    model_label: str,
    y_test: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    oracle: Dict[str, np.ndarray],
    aggregates: list[dict],
) -> list[dict]:
    means = prediction["component_means"].detach().cpu().double()
    covs = prediction["component_covs"].detach().cpu().double()
    z95 = NormalDist().inv_cdf(0.975)
    rows = []
    for spec in aggregates:
        w_np = spec["weights"]
        w = torch.tensor(w_np, dtype=means.dtype)
        component_mean = means @ w
        component_var = torch.einsum("i,kij,j->k", w, covs, w).clamp_min(1e-14)
        fitted_mean = float(component_mean.mean())
        # E[Var(A|theta)] + Var(E[A|theta]); finite mixture moments.
        fitted_var = float(component_var.mean() + component_mean.var(unbiased=False))
        fitted_low = normal_mixture_quantile(component_mean, component_var, 0.025)
        fitted_high = normal_mixture_quantile(component_mean, component_var, 0.975)
        oracle_mean = float(w_np @ oracle["mean"])
        oracle_var = float(w_np @ oracle["cov"] @ w_np)
        oracle_sd = math.sqrt(max(oracle_var, 0.0))
        oracle_low = oracle_mean - z95 * oracle_sd
        oracle_high = oracle_mean + z95 * oracle_sd
        actual = float(w_np @ y_test)
        rel_var = fitted_var / max(oracle_var, 1e-14) - 1
        rows.append({
            "seed": seed, "power_gamma": gamma, "model": model_label,
            "aggregate_id": spec["aggregate_id"],
            "aggregate_scale": spec["aggregate_scale"],
            "n_sites": spec["n_sites"],
            "observed_aggregate": actual,
            "oracle_mean": oracle_mean, "oracle_variance": oracle_var,
            "oracle_lower95": oracle_low, "oracle_upper95": oracle_high,
            "oracle_covered95": int(oracle_low <= actual <= oracle_high),
            "predicted_mean": fitted_mean,
            "predicted_variance": fitted_var,
            "predicted_lower95": fitted_low,
            "predicted_upper95": fitted_high,
            "variance_relative_error": rel_var,
            "variance_absolute_relative_error": abs(rel_var),
            "aggregate_mean_absolute_error": abs(fitted_mean - actual),
            "aggregate_mean_squared_error": (fitted_mean - actual)**2,
            "covered95": int(fitted_low <= actual <= fitted_high),
            "interval_width95": fitted_high - fitted_low,
        })
    return rows


def save_elbo_diagnostics(
    *,
    leroux_result: dict,
    sdm_result: dict,
    output_dir: Path,
    prefix: str,
    make_plot: bool,
    dpi: int,
) -> None:
    tracks = {"Leroux CAR": leroux_result, "Adaptive P-spline SDM-CAR": sdm_result}
    history_frame = pd.DataFrame({
        "leroux_elbo": pd.Series(leroux_result["history"]),
        "sdm_elbo": pd.Series(sdm_result["history"]),
    })
    history_frame.insert(0, "iteration", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(output_dir / f"{prefix}__elbo_history.csv", index=False)
    diagnostics = {}
    for label, result in tracks.items():
        hist = np.asarray(result["history"])
        tail = hist[-min(100, len(hist)):]
        diagnostics[label] = {
            "iterations": len(hist), "elapsed_seconds": result["fit_seconds"],
            "final_elbo": float(hist[-1]),
            "last_100_elbo_mean": float(tail.mean()),
            "last_100_elbo_sd": float(tail.std(ddof=1)) if len(tail) > 1 else 0.0,
        }
    (output_dir / f"{prefix}__fit_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    if make_plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        for ax, (label, result) in zip(axs, tracks.items()):
            hist = np.asarray(result["history"])
            rolling = pd.Series(hist).rolling(min(100, len(hist)), min_periods=1).mean()
            ax.plot(np.arange(1, len(hist) + 1), hist, alpha=0.25, label="ELBO")
            ax.plot(np.arange(1, len(hist) + 1), rolling, lw=1.8, label="rolling mean")
            ax.set_title(label)
            ax.set_ylabel("ELBO")
            ax.set_xlabel("Iteration")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
        fig.savefig(output_dir / f"{prefix}__elbo_convergence.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_spectral_diagnostics(
    *,
    output_dir: Path,
    prefix: str,
    spectral: pd.DataFrame,
    dpi: int,
) -> None:
    work = spectral.sort_values("mu_warped")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(work["mu_warped"], work["F_true"], "k-", lw=2, label="True spectrum")
    ax.plot(work["mu_warped"], work["F_leroux_mean"], lw=1.8,
            label="Leroux posterior mean")
    ax.plot(work["mu_warped"], work["F_sdm_mean"], lw=1.8,
            label="SDM-CAR posterior mean")
    ax.set_xlabel("Warped Laplacian eigenvalue μ")
    ax.set_ylabel("Covariance eigenvalue F")
    ax.set_title("Spectral recovery using the warped eigenvalue coordinate")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / f"{prefix}__spectral_recovery.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# OUTPUT / PLOTTING
# =============================================================================

def gamma_token(gamma: float) -> str:
    return f"{float(gamma):.3f}".replace(".", "p")


def run_prefix(seed: int, gamma: float) -> str:
    return f"seed_{seed:05d}__power_gamma_{gamma_token(gamma)}"


def save_matrix_csv(
    matrix: np.ndarray,
    path: Path,
):
    pd.DataFrame(matrix).to_csv(path, index=False)


def matrix_panel_figure(
    target: np.ndarray,
    leroux: np.ndarray,
    sdm: np.ndarray,
    *,
    title: str,
    value_label: str,
):
    leroux_error = leroux - target
    sdm_error = sdm - target

    fig, axes = plt.subplots(
        1, 5, figsize=(24, 4.8), constrained_layout=True
    )

    main_limit = max(
        np.max(np.abs(target)),
        np.max(np.abs(leroux)),
        np.max(np.abs(sdm)),
    )

    error_limit = max(
        np.max(np.abs(leroux_error)),
        np.max(np.abs(sdm_error)),
        1e-12,
    )

    main_mats = [
        (target, "Truth"),
        (leroux, "Leroux"),
        (sdm, "SDM-CAR"),
    ]

    main_images = []
    for ax, (mat, label) in zip(axes[:3], main_mats):
        im = ax.imshow(
            mat,
            cmap="coolwarm",
            vmin=-main_limit,
            vmax=main_limit,
            interpolation="nearest",
        )
        main_images.append(im)
        ax.set_title(label)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")

    error_mats = [
        (leroux_error, "Leroux - truth"),
        (sdm_error, "SDM-CAR - truth"),
    ]

    error_images = []
    for ax, (mat, label) in zip(axes[3:], error_mats):
        im = ax.imshow(
            mat,
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            interpolation="nearest",
        )
        error_images.append(im)
        ax.set_title(label)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")

    fig.colorbar(
        main_images[0],
        ax=axes[:3],
        shrink=0.8,
        label=value_label,
    )
    fig.colorbar(
        error_images[0],
        ax=axes[3:],
        shrink=0.8,
        label="Error",
    )

    fig.suptitle(title)
    return fig


def metric_rows_for_model(
    model_name: str,
    recovery: Dict,
) -> pd.DataFrame:
    rows = []

    for quantity, mean_metrics in recovery[
        "posterior_mean_metrics"
    ].items():
        row = {
            "model": model_name,
            "quantity": quantity,
            "max_spectral_frobenius_identity_error":
                recovery["max_spectral_frobenius_identity_error"],
            "max_positive_mode_spectral_frobenius_identity_error":
                recovery[
                    "max_positive_mode_spectral_frobenius_identity_error"
                ],
        }

        for key, value in mean_metrics.items():
            row[f"{key}__posterior_mean_matrix"] = value

        row.update(
            recovery["draw_metric_summaries"][quantity]
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# ONE RUN
# =============================================================================

def run_one(
    *,
    seed: int,
    gamma: float,
    common: Dict[str, np.ndarray],
    output_dir: Path,
    dpi: int,
    detailed: bool,
):
    prefix = run_prefix(seed, gamma)
    run_dir = output_dir / "per_run" / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    lam_true = common["lam_true"]
    U_true = common["U_true"]
    positive_mode_mask = common["positive_mode_mask"]
    test_mask = common["test_mask"]
    train_mask = ~test_mask

    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    data = generate_data(seed, lam_true, U_true)

    # Fixed-beta sensitivity: KEEP the same simulation y for this seed, but
    # remove the intercept from the fitted observation/prediction equations.
    # A zero-column (n x 1) design is used instead of an n x 0 design to
    # maintain compatibility with the existing model's beta update routines.
    # X @ beta is identically zero, so beta is exactly absent from the model.
    X = np.zeros_like(data["X"])
    assert np.array_equal(X, np.zeros((N, 1)))
    y = data["y"]
    Sigma_true = data["Sigma_true"]
    C_true = data["C_true"]
    R_true = covariance_to_correlation(Sigma_true)

    V_cond_true = conditional_response_covariance_from_modes(
        U_true[test_idx], data["F_true"] + SIGMA2_TRUE
    )

    lam_warp = power_warp(lam_true, gamma)
    U_warp = U_true.copy()

    # Exact controlled design: eigenvectors are unchanged.
    np.testing.assert_allclose(
        U_warp, U_true, rtol=0.0, atol=0.0
    )

    X_t = torch.tensor(
        X, dtype=torch.double, device=DEVICE
    )
    y_all_t = torch.tensor(
        y, dtype=torch.double, device=DEVICE
    )
    is_holdout_t = torch.tensor(
        test_mask, dtype=torch.bool, device=DEVICE
    )

    y_fit_t = y_all_t.clone()
    y_fit_t[is_holdout_t] = torch.nan

    lam_warp_t = torch.tensor(
        lam_warp, dtype=torch.double, device=DEVICE
    )
    U_warp_t = torch.tensor(
        U_warp, dtype=torch.double, device=DEVICE
    )

    prior_V0 = BETA_PRIOR_VAR * torch.eye(
        X_t.shape[1],
        dtype=torch.double,
        device=DEVICE,
    )

    print("    fitting Leroux CAR", flush=True)
    leroux_result = fit_spectral_vi(
        label="Leroux CAR",
        filter_module=make_leroux_filter(),
        iterations=LEROUX_ITERS,
        learning_rate=LEROUX_LR,
        seed=seed,
        X_t=X_t,
        y_fit_t=y_fit_t,
        lam_warp_t=lam_warp_t,
        U_warp_t=U_warp_t,
        is_holdout_t=is_holdout_t,
        prior_V0=prior_V0,
    )

    print("    fitting Adaptive P-spline SDM-CAR", flush=True)
    sdm_result = fit_spectral_vi(
        label="Adaptive P-spline SDM-CAR",
        filter_module=make_pspline_filter(lam_warp_t),
        iterations=PSPLINE_ITERS,
        learning_rate=PSPLINE_LR,
        seed=seed,
        X_t=X_t,
        y_fit_t=y_fit_t,
        lam_warp_t=lam_warp_t,
        U_warp_t=U_warp_t,
        is_holdout_t=is_holdout_t,
        prior_V0=prior_V0,
    )

    save_elbo_diagnostics(
        leroux_result=leroux_result, sdm_result=sdm_result,
        output_dir=run_dir, prefix=prefix, make_plot=detailed, dpi=dpi,
    )

    leroux_recovery = posterior_covariance_recovery(
        leroux_result,
        U=U_warp,
        positive_mode_mask=positive_mode_mask,
        Sigma_true=Sigma_true,
        C_true=C_true,
        R_true=R_true,
        V_cond_true=V_cond_true,
        train_idx=train_idx,
        test_idx=test_idx,
        draws=POSTERIOR_COV_DRAWS,
        seed=seed + 30_001,
    )

    sdm_recovery = posterior_covariance_recovery(
        sdm_result,
        U=U_warp,
        positive_mode_mask=positive_mode_mask,
        Sigma_true=Sigma_true,
        C_true=C_true,
        R_true=R_true,
        V_cond_true=V_cond_true,
        train_idx=train_idx,
        test_idx=test_idx,
        draws=POSTERIOR_COV_DRAWS,
        seed=seed + 30_002,
    )

    # ------------------------------------------------------------------
    # Save metric table
    # ------------------------------------------------------------------
    metrics = pd.concat(
        [
            metric_rows_for_model(
                "Leroux CAR", leroux_recovery
            ),
            metric_rows_for_model(
                "Adaptive P-spline SDM-CAR", sdm_recovery
            ),
        ],
        ignore_index=True,
    )

    metrics.insert(0, "power_gamma", gamma)
    metrics.insert(0, "seed", seed)

    metrics.to_csv(
        run_dir / f"{prefix}__covariance_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Joint response-vector predictive evaluation
    # ------------------------------------------------------------------
    prior_V0_np = BETA_PRIOR_VAR * np.eye(X.shape[1])
    oracle_predictive = oracle_response_predictive(
        C_true=C_true,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        prior_V0=prior_V0_np,
    )

    pair_i, pair_j, pair_weights = heldout_variogram_pairs(
        test_idx, N_COLS
    )

    # The SAME oracle draws are used for both models so the Monte Carlo noise
    # in the KL comparison is paired within each seed/gamma run.
    oracle_samples = draw_oracle_samples(
        oracle_predictive["mean"],
        oracle_predictive["cov"],
        draws=ORACLE_KL_MC,
        seed=seed + 40_000,
    )

    leroux_joint_prediction = posterior_joint_response_prediction(
        leroux_result["model"],
        num_mc=PREDICTIVE_DRAWS,
        seed=seed + 41_001,
    )
    sdm_joint_prediction = posterior_joint_response_prediction(
        sdm_result["model"],
        num_mc=PREDICTIVE_DRAWS,
        seed=seed + 41_002,
    )

    y_test = y[test_idx]

    leroux_joint_metrics = joint_predictive_metrics(
        target=y_test,
        prediction=leroux_joint_prediction,
        oracle=oracle_predictive,
        oracle_samples=oracle_samples,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_weights=pair_weights,
    )
    sdm_joint_metrics = joint_predictive_metrics(
        target=y_test,
        prediction=sdm_joint_prediction,
        oracle=oracle_predictive,
        oracle_samples=oracle_samples,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_weights=pair_weights,
    )

    predictive_table = pd.DataFrame(
        [
            {"model": "Leroux CAR", **leroux_joint_metrics},
            {
                "model": "Adaptive P-spline SDM-CAR",
                **sdm_joint_metrics,
            },
        ]
    )
    predictive_table.insert(0, "power_gamma", gamma)
    predictive_table.insert(0, "seed", seed)
    predictive_table.to_csv(
        run_dir / f"{prefix}__joint_predictive_metrics.csv",
        index=False,
    )

    # Analytic 1D Gaussian-mixture intervals for each held-out spatial average.
    # Compute moments/quantiles from the JOINT predictive mixture, not
    # V_cond which excludes beta and parameter uncertainty.
    aggregates = make_holdout_aggregate_weights(test_idx, N_COLS)
    aggregate_rows = []
    for label, prediction in (
        ("Leroux CAR", leroux_joint_prediction),
        ("Adaptive P-spline SDM-CAR", sdm_joint_prediction),
    ):
        aggregate_rows.extend(aggregate_prediction_rows(
            seed=seed, gamma=gamma, model_label=label,
            y_test=y_test, prediction=prediction,
            oracle=oracle_predictive, aggregates=aggregates,
        ))
    pd.DataFrame(aggregate_rows).to_csv(
        run_dir / f"{prefix}__aggregate_uncertainty.csv", index=False,
    )

    predictive_summary = pd.DataFrame(
        {
            "node": test_idx,
            "y_observed": y_test,
            "oracle_mean": oracle_predictive["mean"],
            "leroux_mean": leroux_joint_prediction["mean"].cpu().numpy(),
            "sdm_mean": sdm_joint_prediction["mean"].cpu().numpy(),
        }
    )
    predictive_summary.to_csv(
        run_dir / f"{prefix}__joint_predictive_means.csv",
        index=False,
    )

    # Compact spectral recovery table for EVERY seed/gamma, regardless of
    # --detailed-output. This is also input to representative spectral plots.
    spectral_table = pd.DataFrame({
        "lambda_true": lam_true,
        "mu_warped": lam_warp,
        "F_true": data["F_true"],
        "F_leroux_mean": leroux_recovery["F_mean"],
        "F_sdm_mean": sdm_recovery["F_mean"],
    })
    spectral_table.to_csv(
        run_dir / f"{prefix}__spectral_means_for_covariance_check.csv",
        index=False,
    )

    if detailed:
        matrix_dir = run_dir / "matrices"
        matrix_dir.mkdir(exist_ok=True)
        comparisons = [
            ("latent_covariance", "Sigma", Sigma_true,
             leroux_recovery["Sigma_mean"], sdm_recovery["Sigma_mean"]),
            ("latent_correlation", "R", R_true,
             leroux_recovery["R_mean"], sdm_recovery["R_mean"]),
            ("positive_mode_latent_covariance", "Sigma_positive",
             leroux_recovery["Sigma_true_positive"],
             leroux_recovery["Sigma_positive_mean"],
             sdm_recovery["Sigma_positive_mean"]),
            ("positive_mode_latent_correlation", "R_positive",
             leroux_recovery["R_true_positive"],
             leroux_recovery["R_positive_mean"],
             sdm_recovery["R_positive_mean"]),
            ("response_covariance", "C", C_true,
             leroux_recovery["C_mean"], sdm_recovery["C_mean"]),
            ("conditional_response_covariance", "V_cond_y", V_cond_true,
             leroux_recovery["V_cond_mean"], sdm_recovery["V_cond_mean"]),
            ("joint_predictive_covariance", "V_joint_predictive",
             oracle_predictive["cov"],
             leroux_joint_prediction["cov"].cpu().numpy(),
             sdm_joint_prediction["cov"].cpu().numpy()),
        ]
        for plot_name, matrix_name, target, leroux, sdm in comparisons:
            for suffix, matrix in (("true", target), ("leroux_mean", leroux),
                                   ("sdm_mean", sdm)):
                save_matrix_csv(matrix, matrix_dir / f"{prefix}__{matrix_name}_{suffix}.csv")
            fig = matrix_panel_figure(
                target, leroux, sdm,
                title=f"{plot_name}: seed={seed}, gamma={gamma:g}",
                value_label=f"{matrix_name} entry",
            )
            fig.savefig(run_dir / f"{prefix}__{plot_name}_heatmaps.png",
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        save_spectral_diagnostics(
            output_dir=run_dir, prefix=prefix, spectral=spectral_table, dpi=dpi,
        )

    metadata = {
        "seed": int(seed),
        "power_gamma": float(gamma),
        "n": N,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "detail_matrices_and_plots_saved": detailed,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "tau2_true": TAU2_TRUE,
        "rho_true": RHO_TRUE,
        "sigma2_true": SIGMA2_TRUE,
        "posterior_covariance_draws": POSTERIOR_COV_DRAWS,
        "predictive_draws": PREDICTIVE_DRAWS,
        "oracle_kl_mc_draws": ORACLE_KL_MC,
        "variogram_p": VARIOGRAM_P,
        "variogram_weights": "inverse grid distance, normalized to sum to 1",
        "oracle_predictive_beta_treatment": (
            "fixed beta=0; identically zero design ensures beta makes no contribution"
        ),
        "leroux_iterations": LEROUX_ITERS,
        "sdm_iterations": PSPLINE_ITERS,
        "leroux_learning_rate": LEROUX_LR,
        "sdm_learning_rate": PSPLINE_LR,
        "num_mc_elbo": NUM_MC,
        "adaptive_pspline_kwargs": ADAPTIVE_PSPLINE_KWARGS,
        "beta_prior_variance": BETA_PRIOR_VAR,
        "fixed_beta_zero_sensitivity": True,
        "beta_design_matrix": "all zero; beta has no effect on likelihood or prediction",
        "data_generation_design": "intercept of ones, beta_true=0; identical y to main Experiment 3",
        "gamma_transform": "lambda_max * (lambda / lambda_max)**power_gamma",
        "truth_is_unwarped_leroux": True,
        "aggregate_definitions": "full block + four disjoint half-side subblocks; response averages",
        "aggregate_interval_method": "one-dimensional Gaussian mixture quantile inversion",
        "same_eigenvectors_for_truth_and_fit": True,
        "n_zero_modes": int((~positive_mode_mask).sum()),
        "n_positive_modes": int(positive_mode_mask.sum()),
        "positive_mode_metrics_remove_zero_constant_mode": True,
    }

    (
        run_dir / f"{prefix}__metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (
        run_dir / f"{prefix}__COMPLETE.txt"
    ).write_text(
        "Covariance + joint predictive + KL run complete.\n",
        encoding="utf-8",
    )

    del leroux_result, sdm_result


# =============================================================================
# CROSS-RUN SUMMARIES
# =============================================================================

def collect_run_metrics(output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__covariance_metrics.csv"
        )
    )

    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def summarize_across_seeds(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = ["power_gamma", "model", "quantity"]

    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"seed", *id_cols}
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]

    grouped = metrics.groupby(
        id_cols, sort=True
    )[numeric_cols]

    return pd.concat(
        [
            grouped.mean().add_suffix("__mean"),
            grouped.std(ddof=1).add_suffix("__sd"),
            grouped.median().add_suffix("__median"),
            grouped.quantile(0.025).add_suffix("__q025"),
            grouped.quantile(0.975).add_suffix("__q975"),
        ],
        axis=1,
    ).reset_index()


def paired_differences(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = ["seed", "power_gamma", "quantity"]

    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"model", *id_cols}
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]

    leroux = (
        metrics[metrics["model"] == "Leroux CAR"]
        .set_index(id_cols)[numeric_cols]
    )

    sdm = (
        metrics[
            metrics["model"]
            == "Adaptive P-spline SDM-CAR"
        ]
        .set_index(id_cols)[numeric_cols]
    )

    common = leroux.index.intersection(sdm.index)

    diff = sdm.loc[common] - leroux.loc[common]
    diff.columns = [
        f"sdm_minus_leroux__{c}"
        for c in diff.columns
    ]

    return diff.reset_index()


def trend_plot(
    metrics: pd.DataFrame,
    *,
    quantity: str,
    metric_col: str,
    output_path: Path,
    dpi: int,
):
    work = metrics[metrics["quantity"] == quantity].copy()
    if work.empty or metric_col not in work.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for model_name, model_df in work.groupby(
        "model", sort=False
    ):
        summary = (
            model_df.groupby("power_gamma")[metric_col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("power_gamma")
        )

        ax.plot(
            summary["power_gamma"],
            summary["mean"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )

        lower = summary["mean"] - summary["std"].fillna(0.0)
        upper = summary["mean"] + summary["std"].fillna(0.0)

        ax.fill_between(
            summary["power_gamma"],
            lower,
            upper,
            alpha=0.12,
        )

    ax.set_xlabel("POWER_GAMMA")
    ax.set_ylabel(metric_col)
    ax.set_title(
        f"{quantity}: {metric_col} vs POWER_GAMMA"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def collect_joint_predictive_metrics(output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__joint_predictive_metrics.csv"
        )
    )
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_predictive_across_seeds(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = ["power_gamma", "model"]
    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"seed", *id_cols}
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]
    grouped = metrics.groupby(id_cols, sort=True)[numeric_cols]
    return pd.concat(
        [
            grouped.mean().add_suffix("__mean"),
            grouped.std(ddof=1).add_suffix("__sd"),
            grouped.median().add_suffix("__median"),
            grouped.quantile(0.025).add_suffix("__q025"),
            grouped.quantile(0.975).add_suffix("__q975"),
        ],
        axis=1,
    ).reset_index()


def paired_predictive_differences(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = ["seed", "power_gamma"]
    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"model", *id_cols}
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]

    leroux = (
        metrics[metrics["model"] == "Leroux CAR"]
        .set_index(id_cols)[numeric_cols]
    )
    sdm = (
        metrics[metrics["model"] == "Adaptive P-spline SDM-CAR"]
        .set_index(id_cols)[numeric_cols]
    )
    common = leroux.index.intersection(sdm.index)
    diff = sdm.loc[common] - leroux.loc[common]
    diff.columns = [
        f"sdm_minus_leroux__{c}" for c in diff.columns
    ]
    return diff.reset_index()


def predictive_trend_plot(
    metrics: pd.DataFrame,
    *,
    metric_col: str,
    output_path: Path,
    dpi: int,
):
    if metrics.empty or metric_col not in metrics.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model_name, model_df in metrics.groupby("model", sort=False):
        summary = (
            model_df.groupby("power_gamma")[metric_col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("power_gamma")
        )
        ax.plot(
            summary["power_gamma"],
            summary["mean"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )
        lower = summary["mean"] - summary["std"].fillna(0.0)
        upper = summary["mean"] + summary["std"].fillna(0.0)
        ax.fill_between(
            summary["power_gamma"], lower, upper, alpha=0.12
        )

    ax.set_xlabel("POWER_GAMMA")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} vs POWER_GAMMA")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PAIRED MODEL COMPARISONS AND DEGRADATION FROM GAMMA = 1
# =============================================================================

def summarize_confidence_intervals(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """A group/metric/contrast-level table with uncertainty across SEEDS.

    For a paired comparison, each row entering here must represent ONE seed,
    i.e. aggregation over correlated subblocks is done before this function.
    CI uses a Student-t interval for the mean paired difference, n >= 2.
    """
    results = []
    for key, frame in df.groupby(group_cols, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        header = dict(zip(group_cols, key))
        for metric in value_cols:
            x = frame[metric].dropna().to_numpy(dtype=float)
            n = len(x)
            avg = float(x.mean()) if n else np.nan
            sd = float(x.std(ddof=1)) if n >= 2 else np.nan
            radius = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n >= 2 else np.nan
            results.append({**header, "contrast": metric, "n_seeds": n,
                            "mean": avg, "sd": sd,
                            "ci95_low": avg-radius if n >= 2 else np.nan,
                            "ci95_high": avg+radius if n >= 2 else np.nan})
    return pd.DataFrame(results)


def paired_robustness_analysis(
    *,
    df: pd.DataFrame,
    dimensions: list[str],
    metrics: list[str],
    output_prefix: str,
    master_dir: Path,
) -> pd.DataFrame:
    """Within-seed pairings both across models and across gamma.

    Degradation D_m(gamma) = M_m(gamma) - M_m(1).
    Difference in degradation = D_SDM(gamma) - D_Leroux(gamma).
    The plain SDM-minus-Leroux difference is computed even if gamma=1 is absent.
    """
    keep = [x for x in metrics if x in df.columns]
    if not keep:
        return pd.DataFrame()
    ids = ["seed", "power_gamma", "model"] + dimensions
    long = df[ids + keep].melt(id_vars=ids, value_vars=keep,
                                var_name="metric", value_name="value")
    base_ids = ["seed", "model"] + dimensions + ["metric"]
    baseline = long[np.isclose(long["power_gamma"], 1.0)][base_ids + ["value"]].copy()
    baseline = baseline.rename(columns={"value": "gamma1_value"})
    if baseline.empty:
        print(f"WARNING: no gamma=1 baseline for {output_prefix}; "
              "degradation columns will be unavailable.")
    long = long.merge(baseline, on=base_ids, how="left", validate="many_to_one")
    long["degradation_from_gamma1"] = long["value"] - long["gamma1_value"]
    long.to_csv(master_dir / f"all_runs__{output_prefix}__degradation.csv", index=False)
    index_cols = ["seed", "power_gamma"] + dimensions + ["metric"]
    wide = long.pivot(index=index_cols, columns="model",
                      values=["value", "gamma1_value", "degradation_from_gamma1"])
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    leroux, sdm = "Leroux CAR", "Adaptive P-spline SDM-CAR"
    for prefix in ["value", "gamma1_value", "degradation_from_gamma1"]:
        wide[f"sdm_minus_leroux__{prefix}"] = (
            wide[f"{prefix}__{sdm}"] - wide[f"{prefix}__{leroux}"]
        )
    wide.to_csv(master_dir / f"paired__{output_prefix}__robustness.csv", index=False)
    group_cols = ["power_gamma"] + dimensions + ["metric"]
    summary = summarize_confidence_intervals(
        wide, group_cols,
        ["sdm_minus_leroux__value", "sdm_minus_leroux__degradation_from_gamma1",
         f"degradation_from_gamma1__{leroux}",
         f"degradation_from_gamma1__{sdm}"],
    )
    summary.to_csv(master_dir / f"summary__paired__{output_prefix}__robustness.csv",
                   index=False)
    return summary


def plot_degradation_comparison(
    summary: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    dpi: int,
    quantity: str | None = None,
    scale: str | None = None,
) -> None:
    """For each model: paired degradation from gamma=1, with t intervals."""
    if summary.empty:
        return
    work = summary[summary["metric"] == metric].copy()
    if quantity is not None:
        work = work[work["quantity"] == quantity]
    if scale is not None:
        work = work[work["aggregate_scale"] == scale]
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, label in [
        ("Leroux CAR", "Leroux CAR"),
        ("Adaptive P-spline SDM-CAR", "SDM-CAR"),
    ]:
        subset = work[work["contrast"] == f"degradation_from_gamma1__{model}"]
        subset = subset.sort_values("power_gamma")
        if subset.empty or subset["mean"].isna().all():
            continue
        line, = ax.plot(subset["power_gamma"], subset["mean"],
                        marker="o", label=label)
        if subset["ci95_low"].notna().any():
            ax.fill_between(subset["power_gamma"].to_numpy(dtype=float),
                            subset["ci95_low"].to_numpy(dtype=float),
                            subset["ci95_high"].to_numpy(dtype=float),
                            color=line.get_color(), alpha=0.12)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set(xlabel="Power warp gamma", ylabel=f"Degradation from gamma=1: {metric}",
           title=f"Paired robustness: {quantity or scale or 'joint prediction'}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def aggregate_scale_results(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Mean four quadrant metrics WITHIN a seed, avoiding pseudo-replication."""
    keys = ["seed", "power_gamma", "model", "aggregate_scale"]
    value_cols = [
        "variance_relative_error", "variance_absolute_relative_error",
        "aggregate_mean_absolute_error", "aggregate_mean_squared_error",
        "covered95", "interval_width95", "oracle_covered95",
    ]
    return aggregates.groupby(keys, as_index=False)[value_cols].mean()


def simple_group_summary(df: pd.DataFrame, group_cols: list[str],
                         metric_cols: list[str]) -> pd.DataFrame:
    group = df.groupby(group_cols, sort=True)[metric_cols]
    return pd.concat([
        group.mean().add_suffix("__mean"),
        group.std(ddof=1).add_suffix("__sd"),
        group.size().rename("n_seeds"),
    ], axis=1).reset_index()


def build_cross_run_outputs(
    output_dir: Path,
    dpi: int,
):
    metrics = collect_run_metrics(output_dir)
    if metrics.empty:
        print("No completed covariance metric files found.")
        return

    master_dir = output_dir / "master_tables"
    plot_dir = output_dir / "summary_plots"

    master_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    metrics = metrics.sort_values(
        ["quantity", "power_gamma", "seed", "model"]
    ).reset_index(drop=True)

    metrics.to_csv(
        master_dir / "all_runs__covariance_metrics.csv",
        index=False,
    )

    summary = summarize_across_seeds(metrics)
    summary.to_csv(
        master_dir / "summary_across_seeds__covariance_metrics.csv",
        index=False,
    )

    paired = paired_differences(metrics)
    paired.to_csv(
        master_dir / "paired_sdm_minus_leroux__covariance_metrics.csv",
        index=False,
    )

    # Primary trend plots.
    primary_metric = "relative_frobenius__posterior_mean_matrix"

    quantities = [
        "latent_covariance",
        "latent_covariance_positive_modes",
        "response_covariance",
        "latent_correlation",
        "latent_correlation_positive_modes",
        "conditional_response_covariance",
    ]

    for quantity in quantities:
        trend_plot(
            metrics,
            quantity=quantity,
            metric_col=primary_metric,
            output_path=(
                plot_dir
                / f"{quantity}__relative_frobenius_vs_gamma.png"
            ),
            dpi=dpi,
        )

    # Useful secondary plot: off-diagonal covariance recovery.
    trend_plot(
        metrics,
        quantity="latent_covariance",
        metric_col="offdiagonal_rmse__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "latent_covariance__offdiagonal_rmse_vs_gamma.png"
        ),
        dpi=dpi,
    )

    trend_plot(
        metrics,
        quantity="latent_correlation",
        metric_col="offdiagonal_rmse__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "latent_correlation__offdiagonal_rmse_vs_gamma.png"
        ),
        dpi=dpi,
    )


    # Positive-mode secondary plots. These isolate spatial contrasts after
    # removing the global constant mode.
    trend_plot(
        metrics,
        quantity="latent_covariance_positive_modes",
        metric_col="offdiagonal_rmse__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "latent_covariance_positive_modes__offdiagonal_rmse_vs_gamma.png"
        ),
        dpi=dpi,
    )

    trend_plot(
        metrics,
        quantity="latent_correlation_positive_modes",
        metric_col="offdiagonal_rmse__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "latent_correlation_positive_modes__offdiagonal_rmse_vs_gamma.png"
        ),
        dpi=dpi,
    )


    # ------------------------------------------------------------------
    # Joint predictive score master tables and trend plots
    # ------------------------------------------------------------------
    predictive = collect_joint_predictive_metrics(output_dir)
    if not predictive.empty:
        predictive = predictive.sort_values(
            ["power_gamma", "seed", "model"]
        ).reset_index(drop=True)
        predictive.to_csv(
            master_dir / "all_runs__joint_predictive_metrics.csv",
            index=False,
        )

        predictive_summary = summarize_predictive_across_seeds(
            predictive
        )
        predictive_summary.to_csv(
            master_dir / "summary_across_seeds__joint_predictive_metrics.csv",
            index=False,
        )

        predictive_paired = paired_predictive_differences(
            predictive
        )
        predictive_paired.to_csv(
            master_dir / "paired_sdm_minus_leroux__joint_predictive_metrics.csv",
            index=False,
        )

        predictive_metric_cols = [
            "response_RMSE",
            "joint_response_NLPD_per_site",
            "energy_score",
            f"variogram_score_p{str(VARIOGRAM_P).replace('.', 'p')}",
            "oracle_conditional_KL_per_site",
            "joint_predictive_cov_relative_frobenius",
        ]
        for metric_col in predictive_metric_cols:
            safe = metric_col.replace(".", "p")
            predictive_trend_plot(
                predictive,
                metric_col=metric_col,
                output_path=(
                    plot_dir
                    / f"joint_predictive__{safe}__vs_gamma.png"
                ),
                dpi=dpi,
            )

    # KL trend plots for covariance recovery.
    trend_plot(
        metrics,
        quantity="latent_covariance_positive_modes",
        metric_col="gaussian_kl_truth_to_fit_per_dim__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "latent_covariance_positive_modes__gaussian_KL_vs_gamma.png"
        ),
        dpi=dpi,
    )

    trend_plot(
        metrics,
        quantity="response_covariance",
        metric_col="gaussian_kl_truth_to_fit_per_dim__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "response_covariance__gaussian_KL_vs_gamma.png"
        ),
        dpi=dpi,
    )

    trend_plot(
        metrics,
        quantity="conditional_response_covariance",
        metric_col="gaussian_kl_covariance_only_per_dim__posterior_mean_matrix",
        output_path=(
            plot_dir
            / "conditional_response_covariance__gaussian_KL_vs_gamma.png"
        ),
        dpi=dpi,
    )

    # Primary paired robustness contrasts, including degeneration from gamma=1.
    # Only a small number of scientific outcomes is highlighted; the original
    # full metric tables and heatmaps remain available for diagnostics.
    covariance_robustness = paired_robustness_analysis(
        df=metrics, dimensions=["quantity"],
        metrics=["relative_frobenius__posterior_mean_matrix",
                 "gaussian_kl_truth_to_fit_per_dim__posterior_mean_matrix"],
        output_prefix="covariance", master_dir=master_dir,
    )
    plot_degradation_comparison(
        covariance_robustness,
        quantity="latent_covariance_positive_modes",
        metric="relative_frobenius__posterior_mean_matrix",
        output_path=plot_dir / "paired_degradation__positive_mode_covariance.png",
        dpi=dpi,
    )
    if not predictive.empty:
        predictive_robustness = paired_robustness_analysis(
            df=predictive, dimensions=[],
            metrics=["response_RMSE", "joint_response_NLPD_per_site",
                     "energy_score", f"variogram_score_p{str(VARIOGRAM_P).replace('.', 'p')}",
                     "oracle_conditional_KL_per_site",
                     "joint_predictive_cov_relative_frobenius"],
            output_prefix="predictive", master_dir=master_dir,
        )
        plot_degradation_comparison(
            predictive_robustness, metric="joint_response_NLPD_per_site",
            output_path=plot_dir / "paired_degradation__joint_predictive_NLPD.png",
            dpi=dpi,
        )

    aggregate_paths = sorted((output_dir / "per_run").rglob(
        "*__aggregate_uncertainty.csv"))
    if aggregate_paths:
        aggregates = pd.concat([pd.read_csv(p) for p in aggregate_paths],
                               ignore_index=True)
        aggregates = aggregates.sort_values(
            ["aggregate_scale", "aggregate_id", "power_gamma", "seed", "model"]
        ).reset_index(drop=True)
        aggregates.to_csv(master_dir / "all_runs__aggregate_uncertainty.csv", index=False)
        scale_results = aggregate_scale_results(aggregates)
        scale_results.to_csv(master_dir / "all_runs__aggregate_scale_metrics.csv", index=False)
        scale_summary = simple_group_summary(
            scale_results, ["power_gamma", "model", "aggregate_scale"],
            ["variance_relative_error", "variance_absolute_relative_error",
             "aggregate_mean_absolute_error", "aggregate_mean_squared_error",
             "covered95", "interval_width95", "oracle_covered95"],
        )
        scale_summary.to_csv(master_dir / "summary__aggregate_scale_metrics.csv", index=False)
        aggregate_robustness = paired_robustness_analysis(
            df=scale_results, dimensions=["aggregate_scale"],
            metrics=["variance_relative_error", "variance_absolute_relative_error",
                     "aggregate_mean_absolute_error", "aggregate_mean_squared_error",
                     "covered95", "interval_width95"],
            output_prefix="aggregate_scale", master_dir=master_dir,
        )
        for scale in ["full_block", "subblock"]:
            plot_degradation_comparison(
                aggregate_robustness, scale=scale,
                metric="variance_absolute_relative_error",
                output_path=plot_dir / f"paired_degradation__aggregate_{scale}.png",
                dpi=dpi,
            )
        # Plot mean absolute relative aggregate-variance error by gamma and scale.
        for scale, frame in scale_results.groupby("aggregate_scale"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model_name, model_frame in frame.groupby("model"):
                grp = model_frame.groupby("power_gamma")[
                    "variance_absolute_relative_error"].mean().sort_index()
                ax.plot(grp.index, grp.values, marker="o", label=model_name)
            ax.set(xlabel="Power warp gamma", ylabel="Mean |relative variance error|",
                   title=f"Spatial aggregate uncertainty: {scale}")
            ax.grid(alpha=0.2)
            ax.legend()
            fig.savefig(plot_dir / f"aggregate_{scale}__variance_error_vs_gamma.png",
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def run_has_current_outputs(
    run_dir: Path,
    prefix: str,
    *,
    expect_detailed: bool,
) -> bool:
    """Resume only if the current output schema AND run settings match.

    Prevents silent reuse of pre-aggregate results or scientific full runs
    being mixed with short pilot settings in the same output directory.
    """
    required = [
        "__COMPLETE.txt", "__covariance_metrics.csv",
        "__joint_predictive_metrics.csv", "__aggregate_uncertainty.csv",
        "__fit_diagnostics.json", "__elbo_history.csv",
        "__spectral_means_for_covariance_check.csv", "__metadata.json",
    ]
    if not all((run_dir / f"{prefix}{suffix}").exists() for suffix in required):
        return False
    try:
        metadata = json.loads((run_dir / f"{prefix}__metadata.json").read_text())
        if metadata.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
            return False
        if metadata.get("code_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            return False
        for field, value in (
            ("posterior_covariance_draws", POSTERIOR_COV_DRAWS),
            ("predictive_draws", PREDICTIVE_DRAWS),
            ("oracle_kl_mc_draws", ORACLE_KL_MC),
            ("leroux_iterations", LEROUX_ITERS),
            ("sdm_iterations", PSPLINE_ITERS),
            ("num_mc_elbo", NUM_MC),
        ):
            if metadata.get(field) != value:
                return False
        if expect_detailed and not metadata.get("detail_matrices_and_plots_saved"):
            return False
        return True
    except (OSError, ValueError, TypeError):
        return False


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fixed beta=0 sensitivity under power warping: "
            "Leroux CAR vs Adaptive P-spline SDM-CAR."
        )
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=DEFAULT_GAMMAS,
        help="Warp levels; default 1.0 for fixed-beta diagnostic",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiment3_fixed_beta_zero_sensitivity_results"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--detailed-output", choices=["representative", "all", "none"],
        default="representative",
        help="Save matrix CSVs, heatmaps, spectral and ELBO figures for selected runs, all or none.",
    )
    parser.add_argument(
        "--cov-draws", type=int, default=POSTERIOR_COV_DRAWS,
        help="Posterior draws for covariance matrix recovery.",
    )
    parser.add_argument(
        "--predictive-draws", type=int, default=PREDICTIVE_DRAWS,
    )
    parser.add_argument(
        "--oracle-kl-draws", type=int, default=ORACLE_KL_MC,
    )
    parser.add_argument(
        "--leroux-iters", type=int, default=LEROUX_ITERS,
    )
    parser.add_argument(
        "--sdm-iters", type=int, default=PSPLINE_ITERS,
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Very short fits and fewer draws. For testing ONLY, never final results.",
    )

    return parser.parse_args()


def main():
    global LerouxCARFilterFullVI
    global AdaptivePrecisionPSplineFullVI
    global SpectralCAR_HoldoutVI
    global NUM_MC, POSTERIOR_COV_DRAWS, PREDICTIVE_DRAWS, ORACLE_KL_MC
    global LEROUX_ITERS, PSPLINE_ITERS

    args = parse_args()
    if args.pilot:
        print("WARNING: --pilot gives short nonconverged fits; NOT scientific results")
        NUM_MC = 2
        POSTERIOR_COV_DRAWS = 8
        PREDICTIVE_DRAWS = 16
        ORACLE_KL_MC = 32
        LEROUX_ITERS = 30
        PSPLINE_ITERS = 50
    else:
        POSTERIOR_COV_DRAWS = args.cov_draws
        PREDICTIVE_DRAWS = args.predictive_draws
        ORACLE_KL_MC = args.oracle_kl_draws
        LEROUX_ITERS = args.leroux_iters
        PSPLINE_ITERS = args.sdm_iters
    if min(POSTERIOR_COV_DRAWS, PREDICTIVE_DRAWS, ORACLE_KL_MC,
           LEROUX_ITERS, PSPLINE_ITERS) < 2:
        raise ValueError("All draw counts and optimization iteration counts must be >= 2")

    if not FIX_SIGMA2:
        raise ValueError(
            "This experiment assumes sigma^2 is fixed."
        )

    for gamma in args.gammas:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(
                f"POWER_GAMMA must be in (0,1]; got {gamma}"
            )

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sdmcar.filters import (
        LerouxCARFilterFullVI as _LerouxCARFilterFullVI,
        AdaptivePrecisionPSplineFullVI
        as _AdaptivePrecisionPSplineFullVI,
    )

    from sdmcar.models_holdout import (
        SpectralCAR_HoldoutVI
        as _SpectralCAR_HoldoutVI,
    )

    LerouxCARFilterFullVI = _LerouxCARFilterFullVI
    AdaptivePrecisionPSplineFullVI = (
        _AdaptivePrecisionPSplineFullVI
    )
    SpectralCAR_HoldoutVI = _SpectralCAR_HoldoutVI

    # Isolate intentionally underfit pilot output from the dissertation results.
    output_dir = args.output_dir.resolve()
    if args.pilot:
        output_dir = output_dir / "pilot_DO_NOT_USE_FOR_INFERENCE"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed graph / eigendecomposition / holdout for all runs.
    W_true = build_rook_adjacency(N_ROWS, N_COLS)
    L_true = graph_laplacian(W_true)

    lam_true, U_true = np.linalg.eigh(L_true)
    lam_true[np.abs(lam_true) < ZERO_TOL] = 0.0
    lam_true = np.clip(lam_true, 0.0, None)

    positive_mode_mask = lam_true > ZERO_TOL
    n_zero_modes = int((~positive_mode_mask).sum())

    # The 12x12 rook graph is connected, so it should have exactly one
    # zero/constant Laplacian mode.
    if n_zero_modes != 1:
        raise RuntimeError(
            "Expected exactly one zero Laplacian mode for the connected "
            f"rook graph, but found {n_zero_modes}."
        )

    test_mask = holdout_block_mask(
        N_ROWS,
        N_COLS,
        HOLDOUT_SIDE,
        HOLDOUT_PATTERN,
    )

    common = {
        "W_true": W_true,
        "L_true": L_true,
        "lam_true": lam_true,
        "U_true": U_true,
        "positive_mode_mask": positive_mode_mask,
        "test_mask": test_mask,
    }

    print("SENSITIVITY: beta fixed at zero via zero design matrix.")
    print("Data generation and y are unchanged relative to main experiment.")
    print("Project root:", project_root)
    print("Output directory:", output_dir)
    print("Device:", DEVICE)
    print("Seeds:", args.seeds)
    print("POWER_GAMMA values:", args.gammas)
    print("Posterior covariance draws:", POSTERIOR_COV_DRAWS)
    print("Joint predictive mixture draws:", PREDICTIVE_DRAWS)
    print("Oracle KL Monte Carlo draws:", ORACLE_KL_MC)
    print("Zero modes removed in positive-mode analysis:", n_zero_modes)
    print("Positive modes retained:", int(positive_mode_mask.sum()))
    print(
        "Primary outcome: posterior-mean covariance "
        "relative Frobenius error."
    )
    print(
        "Additional outcome: positive-mode covariance/correlation "
        "recovery after removing the constant mode."
    )

    # Selected representative diagnostic plots: first seed, identity, closest
    # gamma to 0.5, and most distorted gamma. Other runs save compact tables.
    representative_gamma = {
        max(args.gammas), min(args.gammas),
        min(args.gammas, key=lambda x: abs(x - 0.5)),
    }
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.gammas)) != len(args.gammas):
        raise ValueError("Seeds and gamma values must be unique")
    for seed in args.seeds:
        for gamma in args.gammas:
            gamma = float(gamma)
            detailed = (args.detailed_output == "all" or (
                args.detailed_output == "representative"
                and seed == args.seeds[0] and gamma in representative_gamma
            ))

            prefix = run_prefix(seed, gamma)
            run_dir = (
                output_dir
                / "per_run"
                / prefix
            )
            complete = (
                run_dir
                / f"{prefix}__COMPLETE.txt"
            )

            print(
                f"\n=== seed={seed}, gamma={gamma:g} ===",
                flush=True,
            )

            if args.resume and run_has_current_outputs(
                run_dir, prefix, expect_detailed=detailed,
            ):
                print(
                    "    SKIP: current complete outputs and settings match.",
                    flush=True,
                )
                continue

            if args.resume and complete.exists():
                print(
                    "    Existing results are outdated or have different settings; rerunning.",
                    flush=True,
                )

            if run_dir.exists():
                shutil.rmtree(run_dir)

            run_one(
                seed=int(seed),
                gamma=gamma,
                common=common,
                output_dir=output_dir,
                dpi=args.dpi,
                detailed=detailed,
            )

    print(
        "\nBuilding cross-run covariance summaries...",
        flush=True,
    )
    build_cross_run_outputs(
        output_dir,
        dpi=args.dpi,
    )

    print("\n=== COVARIANCE + JOINT PREDICTIVE EXPERIMENT COMPLETE ===")
    print("Per-run results:", output_dir / "per_run")
    print("Master tables:", output_dir / "master_tables")
    print("Summary plots:", output_dir / "summary_plots")


if __name__ == "__main__":
    main()
