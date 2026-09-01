"""
covariance_power_warp_experiment.py

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

Example
-------
python covariance_power_warp_experiment.py \
    --project-root "C:\\Users\\pd006\\Desktop\\internship_search\\sdm-car" \
    --seeds 111 222 333 444 555 \
    --gammas 1.0 0.90 0.75 0.50 0.35 \
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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

PSPLINE_ITERS = 6000
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
    "global_scale_d2": 0.50,
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
DEFAULT_GAMMAS = [1.00, 0.90, 0.75, 0.50, 0.35]

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\pd006\Desktop\internship_search\sdm-car"
)

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

    op = np.linalg.norm(delta, ord=2)
    target_op = np.linalg.norm(target, ord=2)

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

    return {
        "relative_frobenius": float(
            frob / max(target_frob, eps)
        ),
        "relative_operator_norm": float(
            op / max(target_op, eps)
        ),
        "diagonal_rmse": diagonal_rmse,
        "offdiagonal_rmse": offdiagonal_rmse,
        "offdiagonal_correlation": offdiag_corr,
    }


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
        V_cond = conditional_response_covariance(
            C, train_idx, test_idx
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

        sigma_metrics = matrix_metrics(Sigma, Sigma_true)
        sigma_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(F_true_modal, F)
        )
        sigma_metric_rows.append(sigma_metrics)

        response_metrics = matrix_metrics(C, C_true)
        response_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(
                F_true_modal + SIGMA2_TRUE,
                F + SIGMA2_TRUE,
            )
        )
        response_metric_rows.append(response_metrics)

        corr_metric_rows.append(matrix_metrics(R, R_true))

        conditional_metrics = matrix_metrics(V_cond, V_cond_true)
        conditional_metrics["gaussian_kl_covariance_only_per_dim"] = (
            gaussian_kl_covariance_per_dim(V_cond_true, V_cond)
        )
        conditional_metric_rows.append(conditional_metrics)

        sigma_positive_metrics = matrix_metrics(
            Sigma_positive,
            Sigma_true_positive,
        )
        sigma_positive_metrics["gaussian_kl_truth_to_fit_per_dim"] = (
            gaussian_kl_same_basis_per_dim(
                F_true_positive, F_positive
            )
        )
        sigma_positive_metric_rows.append(sigma_positive_metrics)

        corr_positive_metric_rows.append(
            matrix_metrics(R_positive, R_true_positive)
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

    X = data["X"]
    y = data["y"]
    Sigma_true = data["Sigma_true"]
    C_true = data["C_true"]
    R_true = covariance_to_correlation(Sigma_true)

    V_cond_true = conditional_response_covariance(
        C_true,
        train_idx,
        test_idx,
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

    # ------------------------------------------------------------------
    # Save matrices
    # ------------------------------------------------------------------
    matrices = {
        "Sigma_true": Sigma_true,
        "Sigma_leroux_mean": leroux_recovery["Sigma_mean"],
        "Sigma_sdm_mean": sdm_recovery["Sigma_mean"],
        "R_true": R_true,
        "R_leroux_mean": leroux_recovery["R_mean"],
        "R_sdm_mean": sdm_recovery["R_mean"],
        "Sigma_positive_true":
            leroux_recovery["Sigma_true_positive"],
        "Sigma_positive_leroux_mean":
            leroux_recovery["Sigma_positive_mean"],
        "Sigma_positive_sdm_mean":
            sdm_recovery["Sigma_positive_mean"],
        "R_positive_true":
            leroux_recovery["R_true_positive"],
        "R_positive_leroux_mean":
            leroux_recovery["R_positive_mean"],
        "R_positive_sdm_mean":
            sdm_recovery["R_positive_mean"],
        "C_true": C_true,
        "C_leroux_mean": leroux_recovery["C_mean"],
        "C_sdm_mean": sdm_recovery["C_mean"],
        "V_cond_y_true": V_cond_true,
        "V_cond_y_leroux_mean": leroux_recovery["V_cond_mean"],
        "V_cond_y_sdm_mean": sdm_recovery["V_cond_mean"],
        "V_joint_predictive_oracle": oracle_predictive["cov"],
        "V_joint_predictive_leroux": (
            leroux_joint_prediction["cov"].cpu().numpy()
        ),
        "V_joint_predictive_sdm": (
            sdm_joint_prediction["cov"].cpu().numpy()
        ),
    }

    matrix_dir = run_dir / "matrices"
    matrix_dir.mkdir(exist_ok=True)

    for name, matrix in matrices.items():
        save_matrix_csv(
            matrix,
            matrix_dir / f"{prefix}__{name}.csv",
        )

    # ------------------------------------------------------------------
    # Save spectral means too, only to verify covariance link.
    # This is not treated as the main outcome.
    # ------------------------------------------------------------------
    pd.DataFrame(
        {
            "lambda_true": lam_true,
            "mu_warped": lam_warp,
            "F_true": data["F_true"],
            "F_leroux_mean": leroux_recovery["F_mean"],
            "F_sdm_mean": sdm_recovery["F_mean"],
        }
    ).to_csv(
        run_dir / f"{prefix}__spectral_means_for_covariance_check.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------
    fig = matrix_panel_figure(
        Sigma_true,
        leroux_recovery["Sigma_mean"],
        sdm_recovery["Sigma_mean"],
        title=(
            f"Latent covariance recovery: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label=r"$\Sigma$ entry",
    )
    fig.savefig(
        run_dir / f"{prefix}__latent_covariance_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = matrix_panel_figure(
        R_true,
        leroux_recovery["R_mean"],
        sdm_recovery["R_mean"],
        title=(
            f"Latent correlation recovery: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label="Correlation",
    )
    fig.savefig(
        run_dir / f"{prefix}__latent_correlation_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Positive-mode covariance: zero/constant graph mode removed.
    fig = matrix_panel_figure(
        leroux_recovery["Sigma_true_positive"],
        leroux_recovery["Sigma_positive_mean"],
        sdm_recovery["Sigma_positive_mean"],
        title=(
            f"Positive-mode latent covariance recovery: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label=r"$\Sigma_+$ entry",
    )
    fig.savefig(
        run_dir / f"{prefix}__latent_covariance_positive_modes_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Positive-mode correlation: dependence among spatial contrasts.
    fig = matrix_panel_figure(
        leroux_recovery["R_true_positive"],
        leroux_recovery["R_positive_mean"],
        sdm_recovery["R_positive_mean"],
        title=(
            f"Positive-mode latent correlation recovery: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label="Positive-mode correlation",
    )
    fig.savefig(
        run_dir / f"{prefix}__latent_correlation_positive_modes_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = matrix_panel_figure(
        V_cond_true,
        leroux_recovery["V_cond_mean"],
        sdm_recovery["V_cond_mean"],
        title=(
            f"Held-out response conditional covariance: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label=r"$V_{y_H|y_O}$ entry",
    )
    fig.savefig(
        run_dir / f"{prefix}__conditional_response_covariance_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


    fig = matrix_panel_figure(
        oracle_predictive["cov"],
        leroux_joint_prediction["cov"].cpu().numpy(),
        sdm_joint_prediction["cov"].cpu().numpy(),
        title=(
            f"Joint posterior predictive response covariance: "
            f"seed={seed}, gamma={gamma:g}"
        ),
        value_label="Predictive covariance entry",
    )
    fig.savefig(
        run_dir / f"{prefix}__joint_predictive_covariance_heatmaps.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    metadata = {
        "seed": int(seed),
        "power_gamma": float(gamma),
        "n": N,
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
            "true covariance with same Gaussian beta prior; beta integrated posteriorly"
        ),
        "leroux_iterations": LEROUX_ITERS,
        "sdm_iterations": PSPLINE_ITERS,
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


def run_has_positive_mode_outputs(
    run_dir: Path,
    prefix: str,
) -> bool:
    """
    Return True only if a completed run already contains the NEW positive-mode
    covariance and correlation rows.

    This prevents --resume from silently accepting result folders produced by
    the older version of the script, which had no zero-mode-removed metrics.
    """
    complete = run_dir / f"{prefix}__COMPLETE.txt"
    metrics_path = run_dir / f"{prefix}__covariance_metrics.csv"

    if not complete.exists() or not metrics_path.exists():
        return False

    try:
        quantities = set(
            pd.read_csv(
                metrics_path,
                usecols=["quantity"],
            )["quantity"].astype(str)
        )
    except Exception:
        return False

    required = {
        "latent_covariance_positive_modes",
        "latent_correlation_positive_modes",
    }
    predictive_path = run_dir / f"{prefix}__joint_predictive_metrics.csv"
    return required.issubset(quantities) and predictive_path.exists()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Covariance + joint predictive power-warp experiment: "
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
            "covariance_power_warp_joint_scores_kl_results"
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

    return parser.parse_args()


def main():
    global LerouxCARFilterFullVI
    global AdaptivePrecisionPSplineFullVI
    global SpectralCAR_HoldoutVI

    args = parse_args()

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

    output_dir = args.output_dir.resolve()
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

    for seed in args.seeds:
        for gamma in args.gammas:
            gamma = float(gamma)

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

            if args.resume and run_has_positive_mode_outputs(
                run_dir, prefix
            ):
                print(
                    "    SKIP: complete joint-predictive artifact set already exists.",
                    flush=True,
                )
                continue

            if args.resume and complete.exists():
                print(
                    "    Existing run predates positive-mode diagnostics; rerunning.",
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
