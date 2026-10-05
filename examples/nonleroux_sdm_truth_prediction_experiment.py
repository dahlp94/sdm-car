"""

nonleroux_sdm_truth_prediction_experiment.py
Experiment 2: SDM-CAR truth outside the Leroux spectral family.
Design
------
- Use one fixed 12 x 12 rook graph.
- Use the SAME graph eigenvectors/eigenvalues for data generation, fitted Leroux,
  and fitted Adaptive P-spline SDM-CAR.
- Generate the latent spatial effect from the exact AdaptivePrecisionPSplineFullVI
  spectral family with nonzero second-difference curvature.
- Keep the SDM truth anchored to the same endpoint precisions as a reference
  Leroux spectrum with rho_ref=0.92 and tau2_ref=0.70.
- Fit Leroux CAR and Adaptive P-spline SDM-CAR to the same held-out datasets.
- Compare held-out predictive performance and spectral/covariance recovery.
- Compute the best possible Leroux approximation to the true spectrum as a
  diagnostic showing that the truth is meaningfully outside the Leroux family.
There is NO graph misspecification and NO power warping in this experiment.
The only deliberate misspecification is the Leroux spectral form.
Example
-------
python nonleroux_sdm_truth_prediction_experiment.py \\
    --project-root "C:/path/to/sdm-car" \\
    --seeds 111 222 333 444 555 \\
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
from typing import Dict

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
# Reference Leroux endpoints used only to anchor the SDM truth.
# The actual truth is NOT Leroux because it has nonzero spline curvature.
REFERENCE_TAU2 = 0.70
REFERENCE_RHO = 0.92
# Target residual pattern on the 8 spline coefficients.  The endpoint residuals
# are fixed at zero by the model parameterization.  The corresponding six
# second differences are computed below and used to generate the exact SDM truth.
TRUE_RESIDUAL_COEFFICIENTS = np.array(
    [0.0, -0.30, -1.05, -1.80, -1.80, -1.05, -0.30, 0.0],
    dtype=float,
)
BETA_TRUE = 0.0
SIGMA2_TRUE = 0.15
ZERO_TOL = 1e-10
# Spatial block holdout
HOLDOUT_SIDE = 4
HOLDOUT_PATTERN = "middle"
# Inference
FIX_SIGMA2 = True
NUM_MC = 8
GRAD_CLIP = 10.0
JITTER = 1e-8
BETA_PRIOR_VAR = 10.0
LEROUX_ITERS = 3000
LEROUX_LR = 1e-3
PSPLINE_ITERS = 12000 #6000
PSPLINE_LR = 3e-4
# Posterior / predictive Monte Carlo
SPECTRUM_DRAWS = 512
PREDICTIVE_DRAWS = 256
ORACLE_KL_MC = 1024
# Leroux initialization
INIT_LEROUX_TAU2 = 0.50
INIT_LEROUX_RHO = 0.80
LEROUX_RHO_EPS = 1e-4
LEROUX_LOG_STD_INIT = -3.0
# Adaptive precision P-spline initialization / prior
INIT_Q_LEFT = 0.20
INIT_Q_RIGHT = 20.0
ADAPTIVE_PSPLINE_KWARGS = {
    "degree": 3,
    "n_internal_knots": 4,
    "prior_std_log_q": 3.0,
    "global_scale_d2": 2.0,#1.0, #0.50,
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
DEFAULT_PROJECT_ROOT = Path.cwd()
DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.double)
LerouxCARFilterFullVI = None
AdaptivePrecisionPSplineFullVI = None
SpectralCAR_HoldoutVI = None

# =============================================================================
# GRAPH / DATA

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

def holdout_block_mask(
    n_rows: int,
    n_cols: int,
    side: int,
    pattern: str = "middle",
) -> np.ndarray:
    if pattern != "middle":
        raise ValueError("Only the middle holdout pattern is implemented.")
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    r0 = (n_rows - side) // 2
    c0 = (n_cols - side) // 2
    mask[r0:r0 + side, c0:c0 + side] = True
    return mask.ravel()

def covariance_from_spectrum(
    U: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    F = np.asarray(F, dtype=float).reshape(-1)
    Sigma = (U * F[None, :]) @ U.T
    return 0.5 * (Sigma + Sigma.T)

def second_differences_from_residual(residual: np.ndarray) -> np.ndarray:
    """Convert the interior residual coefficients into the class's d2 vector."""
    residual = np.asarray(residual, dtype=float).reshape(-1)
    expected_J = (
        ADAPTIVE_PSPLINE_KWARGS["n_internal_knots"]
        + ADAPTIVE_PSPLINE_KWARGS["degree"]
        + 1
    )
    if residual.size != expected_J:
        raise ValueError(
            f"Expected {expected_J} residual coefficients; got {residual.size}."
        )
    if not np.allclose(residual[[0, -1]], 0.0, atol=1e-14):
        raise ValueError("The endpoint residual coefficients must be zero.")
    interior = residual[1:-1]
    K = interior.size
    A = np.zeros((K, K), dtype=float)
    for j in range(K):
        A[j, j] = -2.0
        if j > 0:
            A[j, j - 1] = 1.0
        if j + 1 < K:
            A[j, j + 1] = 1.0
    return A @ interior

def best_leroux_approximation(
    lam: np.ndarray,
    F_true: np.ndarray,
    *,
    rho_eps: float = LEROUX_RHO_EPS,
    grid_size: int = 20001,
) -> Dict[str, np.ndarray | float]:
    """
    Find the Leroux spectrum that best approximates F_true in log-spectrum RMSE.
    For fixed rho,
        F_L(lambda) = tau2 / [(1-rho) + rho*lambda].
    The optimal log(tau2) under squared log-spectrum error is available in
    closed form, so only rho is searched numerically.
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    F_true = np.asarray(F_true, dtype=float).reshape(-1)
    if lam.shape != F_true.shape:
        raise ValueError("lam and F_true must have the same shape.")
    if np.any(F_true <= 0.0):
        raise ValueError("F_true must be strictly positive.")
    rho_grid = np.linspace(
        1e-8,
        1.0 - rho_eps,
        int(grid_size),
        dtype=float,
    )
    a = (
        (1.0 - rho_grid[:, None])
        + rho_grid[:, None] * lam[None, :]
    )
    log_F_true = np.log(F_true)
    log_tau2 = np.mean(
        log_F_true[None, :] + np.log(a),
        axis=1,
    )
    fitted_log_F = log_tau2[:, None] - np.log(a)
    mse = np.mean(
        (fitted_log_F - log_F_true[None, :]) ** 2,
        axis=1,
    )
    best = int(np.argmin(mse))
    rho_best = float(rho_grid[best])
    tau2_best = float(np.exp(log_tau2[best]))
    F_best = tau2_best / (
        (1.0 - rho_best) + rho_best * lam
    )
    return {
        "rho": rho_best,
        "tau2": tau2_best,
        "F": F_best,
        "log_rmse": float(np.sqrt(mse[best])),
        "relative_l2": float(
            np.linalg.norm(F_best - F_true)
            / np.linalg.norm(F_true)
        ),
    }

@torch.no_grad()

def build_true_sdm_spectrum(
    lam_true: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Construct the Experiment-2 truth using the exact fitted SDM-CAR family.
    The affine baseline is the reference Leroux precision
        q_ref(lambda) = [(1-rho_ref) + rho_ref*lambda] / tau2_ref,
    but nonzero d2 values introduce smooth interior curvature.  Therefore the
    resulting precision spectrum is in the Adaptive P-spline SDM-CAR family
    but is not in the affine Leroux precision family.
    """
    lam_true = np.asarray(lam_true, dtype=float).reshape(-1)
    lam_max = float(np.max(lam_true))
    q_left = (1.0 - REFERENCE_RHO) / REFERENCE_TAU2
    q_right = (
        (1.0 - REFERENCE_RHO)
        + REFERENCE_RHO * lam_max
    ) / REFERENCE_TAU2
    d2_true = second_differences_from_residual(
        TRUE_RESIDUAL_COEFFICIENTS
    )
    kwargs = dict(ADAPTIVE_PSPLINE_KWARGS)
    kwargs["init_d2"] = d2_true.tolist()
    truth_filter = AdaptivePrecisionPSplineFullVI(
        lam_max=lam_max,
        mu_log_q_left=math.log(q_left),
        mu_log_q_right=math.log(q_right),
        **kwargs,
    ).to(DEVICE)
    if truth_filter.K != d2_true.size:
        raise RuntimeError(
            "Configured truth d2 dimension does not match the SDM filter."
        )
    lam_t = torch.tensor(
        lam_true,
        dtype=torch.double,
        device=DEVICE,
    )
    theta_true = truth_filter.mean_unconstrained()
    q_true = (
        truth_filter.precision_from_unconstrained(
            lam_t,
            theta_true,
        )
        .detach()
        .cpu()
        .numpy()
    )
    F_true = (
        truth_filter.spectrum_from_unconstrained(
            lam_t,
            theta_true,
        )
        .detach()
        .cpu()
        .numpy()
    )
    components = truth_filter._coefficient_components(theta_true)
    residual_realized = (
        components["residual"]
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(
        residual_realized,
        TRUE_RESIDUAL_COEFFICIENTS,
        rtol=0.0,
        atol=1e-12,
    )
    if not np.all(np.isfinite(q_true)) or np.any(q_true <= 0.0):
        raise RuntimeError("True SDM precision spectrum is not strictly positive.")
    # Directly diagnose non-affinity of the true precision spectrum.
    linear_coef = np.polyfit(lam_true, q_true, deg=1)
    q_linear = np.polyval(linear_coef, lam_true)
    precision_nonaffinity_rel_l2 = float(
        np.linalg.norm(q_true - q_linear)
        / np.linalg.norm(q_true)
    )
    if precision_nonaffinity_rel_l2 <= 1e-8:
        raise RuntimeError(
            "Configured SDM truth is numerically affine and therefore too close "
            "to the Leroux precision family."
        )
    best_leroux = best_leroux_approximation(
        lam_true,
        F_true,
    )
    return {
        "q_left": float(q_left),
        "q_right": float(q_right),
        "d2_true": d2_true,
        "residual_true": residual_realized,
        "q_true": q_true,
        "F_true": F_true,
        "precision_nonaffinity_rel_l2": precision_nonaffinity_rel_l2,
        "best_leroux_rho": float(best_leroux["rho"]),
        "best_leroux_tau2": float(best_leroux["tau2"]),
        "best_leroux_F": np.asarray(best_leroux["F"], dtype=float),
        "best_leroux_log_rmse": float(best_leroux["log_rmse"]),
        "best_leroux_relative_l2": float(best_leroux["relative_l2"]),
    }

def spectrum_recovery_metrics(
    F_hat: np.ndarray,
    F_true: np.ndarray,
    U_true: np.ndarray,
) -> Dict[str, float]:
    """Recovery metrics for the marginal latent spatial covariance."""
    F_hat = np.asarray(F_hat, dtype=float).reshape(-1)
    F_true = np.asarray(F_true, dtype=float).reshape(-1)
    if np.any(F_hat <= 0.0) or np.any(F_true <= 0.0):
        raise ValueError("Spectra must be strictly positive.")
    log_rmse = float(
        np.sqrt(
            np.mean(
                (np.log(F_hat) - np.log(F_true)) ** 2
            )
        )
    )
    relative_l2 = float(
        np.linalg.norm(F_hat - F_true)
        / np.linalg.norm(F_true)
    )
    Sigma_hat = covariance_from_spectrum(U_true, F_hat)
    Sigma_true = covariance_from_spectrum(U_true, F_true)
    covariance_relative_frobenius = float(
        np.linalg.norm(
            Sigma_hat - Sigma_true,
            ord="fro",
        )
        / np.linalg.norm(
            Sigma_true,
            ord="fro",
        )
    )
    return {
        "spectrum_log_RMSE": log_rmse,
        "spectrum_relative_L2": relative_l2,
        "latent_cov_relative_frobenius": covariance_relative_frobenius,
    }

def generate_data(
    seed: int,
    F_true: np.ndarray,
    U_true: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Generate from the fixed non-Leroux SDM-CAR modal variance spectrum."""
    rng = np.random.default_rng(seed)
    X = np.ones((N, 1), dtype=float)
    F_true = np.asarray(F_true, dtype=float).reshape(-1)
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

def rho_to_raw(
    rho: float,
    rho_eps: float = LEROUX_RHO_EPS,
) -> float:
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
    lam_t: torch.Tensor,
    U_t: torch.Tensor,
    is_holdout_t: torch.Tensor,
    prior_V0: torch.Tensor,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SpectralCAR_HoldoutVI(
        X=X_t,
        y=y_fit_t,
        lam=lam_t,
        U=U_t,
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
# LINEAR ALGEBRA

# =============================================================================

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
    raise np.linalg.LinAlgError(
        "stable_cholesky_np failed"
    ) from last_error

def solve_spd(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    L = stable_cholesky_np(A)
    y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, y)

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
    scale = (
        torch.diagonal(matrix)
        .abs()
        .mean()
        .detach()
        .clamp_min(1.0)
    )
    last_error = None
    for attempt in range(max_tries):
        jitter = base_jitter * (10.0 ** attempt) * scale
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        "stable_cholesky_torch failed."
    ) from last_error

# =============================================================================
# ORACLE PREDICTIVE DISTRIBUTION

# =============================================================================

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
    Exact Gaussian posterior predictive distribution for y_H | y_O
    under the true response covariance, using the same beta prior
    as the fitted models.
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
    V_beta = np.linalg.inv(
        V0_inv + X_O.T @ Cinv_X
    )
    m_beta = V_beta @ (
        V0_inv @ prior_m0
        + X_O.T @ Cinv_y
    )
    A_beta = X_H - gain @ X_O
    mean = gain @ y_O + A_beta @ m_beta
    conditional_cov = C_HH - gain @ C_OH
    conditional_cov = 0.5 * (
        conditional_cov + conditional_cov.T
    )
    covariance = (
        conditional_cov
        + A_beta @ V_beta @ A_beta.T
    )
    covariance = 0.5 * (
        covariance + covariance.T
    )
    return {
        "mean": mean,
        "cov": covariance,
        "conditional_cov_given_beta": conditional_cov,
        "beta_mean": m_beta,
        "beta_cov": V_beta,
    }

# =============================================================================
# FITTED JOINT PREDICTIVE DISTRIBUTION

# =============================================================================

@torch.no_grad()

def posterior_joint_response_prediction(
    model,
    *,
    num_mc: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Posterior predictive mixture for the full held-out response vector.
    Each variational hyperparameter draw gives
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
            model.U_test
            * response_precision_modes.unsqueeze(0)
        ) @ model.U_test.T
        P_HO = (
            model.U_test
            * response_precision_modes.unsqueeze(0)
        ) @ model.U_train.T
        chol_PHH = stable_cholesky_torch(
            P_HH,
            base_jitter=max(model.jitter, 1e-10),
        )
        # B = P_HH^{-1} P_HO;
        # covariance-form conditional gain is -B.
        B = torch.cholesky_solve(P_HO, chol_PHH)
        conditional_cov = torch.cholesky_inverse(chol_PHH)
        conditional_cov = 0.5 * (
            conditional_cov + conditional_cov.T
        )
        A_beta = model.X_test + B @ model.X_train
        pred_mean = (
            -B @ model.y_train
            + A_beta @ m_beta
        )
        pred_cov = (
            conditional_cov
            + A_beta @ V_beta @ A_beta.T
        )
        pred_cov = 0.5 * (
            pred_cov + pred_cov.T
        )
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
        covs
        + means.unsqueeze(2) * means.unsqueeze(1)
    ).mean(dim=0)
    covariance = second - torch.outer(mean, mean)
    covariance = 0.5 * (
        covariance + covariance.T
    )
    return {
        "mean": mean,
        "cov": covariance,
        "component_means": means,
        "component_covs": covs,
        "draws": draws,
        "test_idx": model.test_idx,
    }

# =============================================================================
# PREDICTIVE SCORES

# =============================================================================

def mixture_logpdf_torch(
    samples: torch.Tensor,
    component_means: torch.Tensor,
    component_covs: torch.Tensor,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    K, h = component_means.shape
    eye = torch.eye(
        h,
        dtype=component_covs.dtype,
        device=component_covs.device,
    )
    scale = (
        torch.diagonal(
            component_covs,
            dim1=-2,
            dim2=-1,
        )
        .abs()
        .mean(dim=1)
        .clamp_min(1.0)
    )
    base_covs = 0.5 * (
        component_covs
        + component_covs.transpose(-1, -2)
    )
    last_error = None
    chol = None
    for attempt in range(8):
        jitter = (
            1e-10
            * (10.0 ** attempt)
            * scale
        )[:, None, None]
        try:
            chol = torch.linalg.cholesky(
                base_covs + jitter * eye
            )
            break
        except RuntimeError as error:
            last_error = error
    if chol is None:
        raise RuntimeError(
            "Batched Cholesky failed for predictive mixture covariances."
        ) from last_error
    precision = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(
        torch.diagonal(
            chol,
            dim1=-2,
            dim2=-1,
        )
    ).sum(dim=1)
    constant = h * math.log(2.0 * math.pi)
    outputs = []
    for start in range(
        0,
        samples.shape[0],
        batch_size,
    ):
        x = samples[start:start + batch_size]
        residual = (
            x[:, None, :]
            - component_means[None, :, :]
        )
        quadratic = torch.einsum(
            "bki,kij,bkj->bk",
            residual,
            precision,
            residual,
        )
        component_logpdf = -0.5 * (
            constant
            + logdet[None, :]
            + quadratic
        )
        outputs.append(
            torch.logsumexp(
                component_logpdf,
                dim=1,
            ) - math.log(K)
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
        chol,
        residual.T,
        upper=False,
    )
    quadratic = solved.square().sum(dim=0)
    logdet = 2.0 * torch.log(
        torch.diagonal(chol)
    ).sum()
    return -0.5 * (
        h * math.log(2.0 * math.pi)
        + logdet
        + quadratic
    )

def energy_score(
    target: torch.Tensor,
    draws: torch.Tensor,
) -> float:
    target = target.reshape(-1)
    first = torch.linalg.vector_norm(
        draws - target.unsqueeze(0),
        dim=1,
    ).mean()
    pairwise = torch.cdist(
        draws,
        draws,
        p=2,
    )
    second = 0.5 * pairwise.mean()
    return float(
        (first - second).detach().cpu()
    )

def draw_oracle_samples(
    oracle_mean: np.ndarray,
    oracle_cov: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    mean = torch.tensor(
        oracle_mean,
        dtype=torch.double,
        device=DEVICE,
    )
    cov = torch.tensor(
        oracle_cov,
        dtype=torch.double,
        device=DEVICE,
    )
    chol = stable_cholesky_torch(cov)
    z = torch.randn(
        draws,
        mean.numel(),
        dtype=torch.double,
        device=DEVICE,
    )
    return mean.unsqueeze(0) + z @ chol.T

def predictive_metrics(
    *,
    target: np.ndarray,
    prediction: Dict[str, torch.Tensor],
    oracle: Dict[str, np.ndarray],
    oracle_samples: torch.Tensor,
) -> Dict[str, float]:
    target_t = torch.as_tensor(
        target,
        dtype=prediction["mean"].dtype,
        device=prediction["mean"].device,
    ).reshape(-1)
    h = target_t.numel()
    error = prediction["mean"] - target_t
    rmse = torch.sqrt(
        torch.mean(error.square())
    )
    mae = torch.mean(
        torch.abs(error)
    )
    log_density = mixture_logpdf_torch(
        target_t,
        prediction["component_means"],
        prediction["component_covs"],
    )[0]
    joint_nlpd_per_site = (
        -log_density / float(h)
    )
    es = energy_score(
        target_t,
        prediction["draws"],
    )
    # KL from the true oracle predictive distribution
    # to the fitted posterior predictive mixture.
    oracle_mean_t = torch.as_tensor(
        oracle["mean"],
        dtype=target_t.dtype,
        device=target_t.device,
    )
    oracle_cov_t = torch.as_tensor(
        oracle["cov"],
        dtype=target_t.dtype,
        device=target_t.device,
    )
    log_p0 = gaussian_logpdf_torch(
        oracle_samples,
        oracle_mean_t,
        oracle_cov_t,
    )
    log_pm = mixture_logpdf_torch(
        oracle_samples,
        prediction["component_means"],
        prediction["component_covs"],
    )
    log_ratio = log_p0 - log_pm
    kl_per_site = (
        log_ratio.mean() / float(h)
    )
    kl_mcse_per_site = (
        log_ratio.std(unbiased=True)
        / math.sqrt(log_ratio.numel())
        / float(h)
    )
    oracle_cov_np = np.asarray(
        oracle["cov"],
        dtype=float,
    )
    pred_cov_np = (
        prediction["cov"]
        .detach()
        .cpu()
        .numpy()
    )
    predictive_cov_rel_frob = float(
        np.linalg.norm(
            pred_cov_np - oracle_cov_np,
            ord="fro",
        )
        / np.linalg.norm(
            oracle_cov_np,
            ord="fro",
        )
    )
    return {
        "response_RMSE": float(
            rmse.detach().cpu()
        ),
        "response_MAE": float(
            mae.detach().cpu()
        ),
        "joint_response_NLPD_per_site": float(
            joint_nlpd_per_site.detach().cpu()
        ),
        "energy_score": es,
        "oracle_conditional_KL_per_site": float(
            kl_per_site.detach().cpu()
        ),
        "oracle_conditional_KL_MCSE_per_site": float(
            kl_mcse_per_site.detach().cpu()
        ),
        "joint_predictive_cov_relative_frobenius": (
            predictive_cov_rel_frob
        ),
    }

# =============================================================================
# POSTERIOR-MEAN SPECTRUM

# =============================================================================

@torch.no_grad()

def posterior_mean_spectrum(
    result,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    model = result["model"]
    torch.manual_seed(seed)
    F_sum = np.zeros(
        model.lam.numel(),
        dtype=float,
    )
    for _ in range(int(draws)):
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(
            model.lam,
            theta,
        ).clamp_min(1e-12)
        F_sum += (
            F.detach()
            .cpu()
            .numpy()
        )
    return F_sum / float(draws)

# =============================================================================
# ONE SIMULATION REPLICATE

# =============================================================================

def run_prefix(seed: int) -> str:
    return f"seed_{seed:05d}"

def save_elbo_diagnostics(
    *,
    leroux_result,
    sdm_result,
    run_dir: Path,
    prefix: str,
    dpi: int = 180,
    rolling_window: int = 100,
):
    leroux_history = np.asarray(
        leroux_result["history"],
        dtype=float,
    )

    sdm_history = np.asarray(
        sdm_result["history"],
        dtype=float,
    )

    max_len = max(
        len(leroux_history),
        len(sdm_history),
    )

    history_df = pd.DataFrame(
        {
            "iteration": np.arange(1, max_len + 1),
            "leroux_elbo": np.pad(
                leroux_history,
                (0, max_len - len(leroux_history)),
                constant_values=np.nan,
            ),
            "sdm_elbo": np.pad(
                sdm_history,
                (0, max_len - len(sdm_history)),
                constant_values=np.nan,
            ),
        }
    )

    history_df.to_csv(
        run_dir / f"{prefix}__elbo_history.csv",
        index=False,
    )

    leroux_roll = (
        pd.Series(leroux_history)
        .rolling(
            rolling_window,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    sdm_roll = (
        pd.Series(sdm_history)
        .rolling(
            rolling_window,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(
        np.arange(1, len(leroux_roll) + 1),
        leroux_roll,
        linewidth=1.8,
        label="Leroux CAR",
    )

    ax.plot(
        np.arange(1, len(sdm_roll) + 1),
        sdm_roll,
        linewidth=1.8,
        label="Adaptive P-spline SDM-CAR",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        f"ELBO ({rolling_window}-iteration rolling mean)"
    )
    ax.set_title("Variational inference convergence")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        run_dir / f"{prefix}__elbo_convergence.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


@torch.no_grad()
def posterior_mean_precision(
    result,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    model = result["model"]
    torch.manual_seed(seed)

    q_sum = np.zeros(
        model.lam.numel(),
        dtype=float,
    )

    for _ in range(int(draws)):
        theta = model.filter.sample_unconstrained()

        F = model.filter.spectrum(
            model.lam,
            theta,
        ).clamp_min(1e-12)

        q = 1.0 / F

        q_sum += (
            q.detach()
            .cpu()
            .numpy()
        )

    return q_sum / float(draws)


@torch.no_grad()
def save_sdm_parameter_diagnostics(
    *,
    sdm_result,
    common,
    run_dir: Path,
    prefix: str,
):
    filt = sdm_result["model"].filter
    summary = filt.shrinkage_summary()

    d2_hat = (
        summary["d2"]
        .detach()
        .cpu()
        .numpy()
    )

    residual_hat = (
        summary["residual"]
        .detach()
        .cpu()
        .numpy()
    )

    local_scales = (
        summary["local_scales"]
        .detach()
        .cpu()
        .numpy()
    )

    d2_true = np.asarray(
        common["d2_true"],
        dtype=float,
    )

    residual_true = np.asarray(
        common["residual_true"],
        dtype=float,
    )

    d2_df = pd.DataFrame(
        {
            "index": np.arange(d2_true.size),
            "d2_true": d2_true,
            "d2_posterior_mean": d2_hat,
            "local_scale_posterior_mean": local_scales,
        }
    )

    d2_df.to_csv(
        run_dir / f"{prefix}__sdm_d2_diagnostics.csv",
        index=False,
    )

    residual_df = pd.DataFrame(
        {
            "index": np.arange(residual_true.size),
            "residual_true": residual_true,
            "residual_posterior_mean": residual_hat,
        }
    )

    residual_df.to_csv(
        run_dir / f"{prefix}__sdm_residual_diagnostics.csv",
        index=False,
    )

    endpoint_df = pd.DataFrame(
        [
            {
                "q_left_true": float(common["q_left"]),
                "q_right_true": float(common["q_right"]),
                "q_left_posterior_mean": float(
                    summary["q_left"].detach().cpu()
                ),
                "q_right_posterior_mean": float(
                    summary["q_right"].detach().cpu()
                ),
                "d2_l2_posterior_mean": float(
                    summary["d2_l2"].detach().cpu()
                ),
                "max_abs_d2_posterior_mean": float(
                    summary["max_abs_d2"].detach().cpu()
                ),
                "max_abs_residual_posterior_mean": float(
                    summary["max_abs_residual"].detach().cpu()
                ),
            }
        ]
    )

    endpoint_df.to_csv(
        run_dir / f"{prefix}__sdm_endpoint_diagnostics.csv",
        index=False,
    )


def save_precision_recovery_plot(
    *,
    lam: np.ndarray,
    q_true: np.ndarray,
    q_best_leroux: np.ndarray,
    q_leroux_mean: np.ndarray,
    q_sdm_mean: np.ndarray,
    run_dir: Path,
    prefix: str,
    dpi: int = 180,
):
    precision_df = pd.DataFrame(
        {
            "lambda": lam,
            "q_true": q_true,
            "q_best_leroux": q_best_leroux,
            "q_leroux_mean": q_leroux_mean,
            "q_sdm_mean": q_sdm_mean,
        }
    )

    precision_df.to_csv(
        run_dir / f"{prefix}__precision_recovery.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        lam,
        q_true,
        linewidth=2.2,
        label="True non-Leroux SDM precision",
    )

    ax.plot(
        lam,
        q_best_leroux,
        linewidth=1.6,
        linestyle="--",
        label="Best Leroux approximation",
    )

    ax.plot(
        lam,
        q_leroux_mean,
        linewidth=1.8,
        label="Fitted Leroux",
    )

    ax.plot(
        lam,
        q_sdm_mean,
        linewidth=1.8,
        label="Fitted SDM-CAR",
    )

    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Modal precision")
    ax.set_title(
        "Non-Leroux SDM truth: precision recovery"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        run_dir / f"{prefix}__precision_recovery.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_one(
    *,
    seed: int,
    common: Dict[str, np.ndarray],
    output_dir: Path,
):
    prefix = run_prefix(seed)
    run_dir = (
        output_dir
        / "per_run"
        / prefix
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    lam_true = common["lam_true"]
    U_true = common["U_true"]
    test_mask = common["test_mask"]
    train_mask = ~test_mask
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    data = generate_data(
        seed,
        common["F_true"],
        U_true,
    )
    X = data["X"]
    y = data["y"]
    C_true = data["C_true"]
    # ---------------------------------------------------------
    # IMPORTANT:
    # There is NO warping and NO graph misspecification.
    # Both fitted models receive exactly the true graph basis.
    # ---------------------------------------------------------
    lam_fit = lam_true.copy()
    U_fit = U_true.copy()
    np.testing.assert_allclose(
        lam_fit,
        lam_true,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        U_fit,
        U_true,
        rtol=0.0,
        atol=0.0,
    )
    X_t = torch.tensor(
        X,
        dtype=torch.double,
        device=DEVICE,
    )
    y_all_t = torch.tensor(
        y,
        dtype=torch.double,
        device=DEVICE,
    )
    is_holdout_t = torch.tensor(
        test_mask,
        dtype=torch.bool,
        device=DEVICE,
    )
    y_fit_t = y_all_t.clone()
    y_fit_t[is_holdout_t] = torch.nan
    lam_t = torch.tensor(
        lam_fit,
        dtype=torch.double,
        device=DEVICE,
    )
    U_t = torch.tensor(
        U_fit,
        dtype=torch.double,
        device=DEVICE,
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
        lam_t=lam_t,
        U_t=U_t,
        is_holdout_t=is_holdout_t,
        prior_V0=prior_V0,
    )
    print(
        "    fitting Adaptive P-spline SDM-CAR",
        flush=True,
    )
    sdm_result = fit_spectral_vi(
        label="Adaptive P-spline SDM-CAR",
        filter_module=make_pspline_filter(lam_t),
        iterations=PSPLINE_ITERS,
        learning_rate=PSPLINE_LR,
        seed=seed,
        X_t=X_t,
        y_fit_t=y_fit_t,
        lam_t=lam_t,
        U_t=U_t,
        is_holdout_t=is_holdout_t,
        prior_V0=prior_V0,
    )
    
    save_elbo_diagnostics(
        leroux_result=leroux_result,
        sdm_result=sdm_result,
        run_dir=run_dir,
        prefix=prefix,
    )
    # ---------------------------------------------------------
    # Exact oracle predictive distribution under the non-Leroux SDM truth
    # ---------------------------------------------------------
    prior_V0_np = (
        BETA_PRIOR_VAR
        * np.eye(X.shape[1])
    )
    oracle_predictive = oracle_response_predictive(
        C_true=C_true,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        prior_V0=prior_V0_np,
    )
    # Use the SAME oracle Monte Carlo sample for both fitted models.
    oracle_samples = draw_oracle_samples(
        oracle_predictive["mean"],
        oracle_predictive["cov"],
        draws=ORACLE_KL_MC,
        seed=seed + 40_000,
    )
    # ---------------------------------------------------------
    # Posterior predictive distributions
    # ---------------------------------------------------------
    leroux_prediction = (
        posterior_joint_response_prediction(
            leroux_result["model"],
            num_mc=PREDICTIVE_DRAWS,
            seed=seed + 41_001,
        )
    )
    sdm_prediction = (
        posterior_joint_response_prediction(
            sdm_result["model"],
            num_mc=PREDICTIVE_DRAWS,
            seed=seed + 41_002,
        )
    )
    y_test = y[test_idx]
    leroux_metrics = predictive_metrics(
        target=y_test,
        prediction=leroux_prediction,
        oracle=oracle_predictive,
        oracle_samples=oracle_samples,
    )
    sdm_metrics = predictive_metrics(
        target=y_test,
        prediction=sdm_prediction,
        oracle=oracle_predictive,
        oracle_samples=oracle_samples,
    )
    # ---------------------------------------------------------
    # Posterior-mean spectra
    # ---------------------------------------------------------
    F_leroux_mean = posterior_mean_spectrum(
        leroux_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 50_001,
    )
    F_sdm_mean = posterior_mean_spectrum(
        sdm_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 50_002,
    )
    leroux_metrics.update(
        spectrum_recovery_metrics(
            F_leroux_mean,
            data["F_true"],
            U_true,
        )
    )
    q_leroux_mean = posterior_mean_precision(
        leroux_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 60_001,
    )

    q_sdm_mean = posterior_mean_precision(
        sdm_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 60_002,
    )

    q_best_leroux = (
        1.0
        / np.asarray(
            common["best_leroux_F"],
            dtype=float,
        )
    )

    save_precision_recovery_plot(
        lam=lam_true,
        q_true=np.asarray(
            common["q_true"],
            dtype=float,
        ),
        q_best_leroux=q_best_leroux,
        q_leroux_mean=q_leroux_mean,
        q_sdm_mean=q_sdm_mean,
        run_dir=run_dir,
        prefix=prefix,
    )

    save_sdm_parameter_diagnostics(
        sdm_result=sdm_result,
        common=common,
        run_dir=run_dir,
        prefix=prefix,
    )
    sdm_metrics.update(
        spectrum_recovery_metrics(
            F_sdm_mean,
            data["F_true"],
            U_true,
        )
    )
    metrics = pd.DataFrame(
        [
            {
                "seed": seed,
                "model": "Leroux CAR",
                **leroux_metrics,
            },
            {
                "seed": seed,
                "model": "Adaptive P-spline SDM-CAR",
                **sdm_metrics,
            },
        ]
    )
    metrics.to_csv(
        run_dir
        / f"{prefix}__predictive_metrics.csv",
        index=False,
    )
    spectrum = pd.DataFrame(
        {
            "seed": seed,
            "mode": np.arange(N),
            "lambda": lam_true,
            "F_true": data["F_true"],
            "F_best_leroux_family": common["best_leroux_F"],
            "F_leroux_mean": F_leroux_mean,
            "F_sdm_mean": F_sdm_mean,
        }
    )
    spectrum.to_csv(
        run_dir
        / f"{prefix}__spectral_recovery.csv",
        index=False,
    )
    # ---------------------------------------------------------
    # Held-out predictive means
    # ---------------------------------------------------------
    pred_means = pd.DataFrame(
        {
            "node": test_idx,
            "y_observed": y_test,
            "oracle_mean": oracle_predictive["mean"],
            "leroux_mean": (
                leroux_prediction["mean"]
                .detach()
                .cpu()
                .numpy()
            ),
            "sdm_mean": (
                sdm_prediction["mean"]
                .detach()
                .cpu()
                .numpy()
            ),
        }
    )
    pred_means.to_csv(
        run_dir
        / f"{prefix}__predictive_means.csv",
        index=False,
    )
    metadata = {
        "seed": int(seed),
        "n": N,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "n_rows": N_ROWS,
        "n_cols": N_COLS,
        "holdout_side": HOLDOUT_SIDE,
        "reference_tau2": REFERENCE_TAU2,
        "reference_rho": REFERENCE_RHO,
        "truth_family": "AdaptivePrecisionPSplineFullVI",
        "truth_is_leroux": False,
        "truth_q_left": float(common["q_left"]),
        "truth_q_right": float(common["q_right"]),
        "truth_residual_coefficients": common["residual_true"].tolist(),
        "truth_d2": common["d2_true"].tolist(),
        "precision_nonaffinity_relative_l2": float(
            common["precision_nonaffinity_rel_l2"]
        ),
        "best_leroux_rho_for_truth": float(common["best_leroux_rho"]),
        "best_leroux_tau2_for_truth": float(common["best_leroux_tau2"]),
        "best_leroux_log_spectrum_rmse": float(
            common["best_leroux_log_rmse"]
        ),
        "best_leroux_spectrum_relative_l2": float(
            common["best_leroux_relative_l2"]
        ),
        "sigma2_true": SIGMA2_TRUE,
        "beta_true": BETA_TRUE,
        "same_graph_for_truth_and_fit": True,
        "same_eigenvectors_for_truth_and_fit": True,
        "same_eigenvalues_for_truth_and_fit": True,
        "power_warp": False,
        "graph_misspecification": False,
        "fixed_sigma2": FIX_SIGMA2,
        "predictive_draws": PREDICTIVE_DRAWS,
        "oracle_kl_mc_draws": ORACLE_KL_MC,
        "spectrum_draws": SPECTRUM_DRAWS,
        "leroux_iterations": LEROUX_ITERS,
        "sdm_iterations": PSPLINE_ITERS,
    }
    (
        run_dir
        / f"{prefix}__metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
    (
        run_dir
        / f"{prefix}__COMPLETE.txt"
    ).write_text(
        "Non-Leroux SDM-truth benchmark complete.\n",
        encoding="utf-8",
    )
    del leroux_result, sdm_result

# =============================================================================
# CROSS-SEED SUMMARIES

# =============================================================================

def collect_predictive_metrics(
    output_dir: Path,
) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__predictive_metrics.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

def summarize_predictive_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"seed", "model"}
        and pd.api.types.is_numeric_dtype(
            metrics[c]
        )
    ]
    grouped = metrics.groupby(
        "model",
        sort=False,
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

def paired_predictive_differences(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in metrics.columns
        if c not in {"seed", "model"}
        and pd.api.types.is_numeric_dtype(
            metrics[c]
        )
    ]
    leroux = (
        metrics[
            metrics["model"] == "Leroux CAR"
        ]
        .set_index("seed")[numeric_cols]
    )
    sdm = (
        metrics[
            metrics["model"]
            == "Adaptive P-spline SDM-CAR"
        ]
        .set_index("seed")[numeric_cols]
    )
    common = leroux.index.intersection(
        sdm.index
    )
    diff = (
        sdm.loc[common]
        - leroux.loc[common]
    )
    diff.columns = [
        f"sdm_minus_leroux__{c}"
        for c in diff.columns
    ]
    return diff.reset_index()

def summarize_paired_differences(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in paired.columns:
        if col == "seed":
            continue
        x = paired[col].to_numpy(
            dtype=float
        )
        rows.append(
            {
                "metric": col,
                "mean_difference": float(
                    np.mean(x)
                ),
                "sd_difference": float(
                    np.std(x, ddof=1)
                ) if len(x) > 1 else np.nan,
                "median_difference": float(
                    np.median(x)
                ),
                "q025_difference": float(
                    np.quantile(x, 0.025)
                ),
                "q975_difference": float(
                    np.quantile(x, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)

def collect_spectra(
    output_dir: Path,
) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__spectral_recovery.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

def build_spectral_summary(
    spectra: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        spectra.groupby(
            ["mode", "lambda"],
            as_index=False,
        )[
            [
                "F_true",
                "F_best_leroux_family",
                "F_leroux_mean",
                "F_sdm_mean",
            ]
        ]
        .mean()
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    return summary

def plot_spectral_summary(
    spectral_summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )
    ax.plot(
        spectral_summary["lambda"],
        spectral_summary["F_true"],
        linewidth=2.2,
        label="True non-Leroux SDM spectrum",
    )
    ax.plot(
        spectral_summary["lambda"],
        spectral_summary["F_best_leroux_family"],
        linewidth=1.6,
        linestyle="--",
        label="Best Leroux approximation to truth",
    )
    ax.plot(
        spectral_summary["lambda"],
        spectral_summary["F_leroux_mean"],
        linewidth=1.8,
        label="Fitted Leroux",
    )
    ax.plot(
        spectral_summary["lambda"],
        spectral_summary["F_sdm_mean"],
        linewidth=1.8,
        label="Fitted SDM-CAR",
    )
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Modal variance")
    ax.set_title(
        "Non-Leroux SDM truth: spectral recovery"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

def build_cross_seed_outputs(
    output_dir: Path,
    dpi: int,
):
    master_dir = (
        output_dir / "master_tables"
    )
    plot_dir = (
        output_dir / "summary_plots"
    )
    master_dir.mkdir(
        exist_ok=True
    )
    plot_dir.mkdir(
        exist_ok=True
    )
    metrics = collect_predictive_metrics(
        output_dir
    )
    if metrics.empty:
        print(
            "No completed predictive metric files found."
        )
        return
    metrics = metrics.sort_values(
        ["seed", "model"]
    ).reset_index(drop=True)
    metrics.to_csv(
        master_dir
        / "all_runs__predictive_metrics.csv",
        index=False,
    )
    summary = summarize_predictive_metrics(
        metrics
    )
    summary.to_csv(
        master_dir
        / "summary_across_seeds__predictive_metrics.csv",
        index=False,
    )
    paired = paired_predictive_differences(
        metrics
    )
    paired.to_csv(
        master_dir
        / "paired_sdm_minus_leroux__predictive_metrics.csv",
        index=False,
    )
    paired_summary = (
        summarize_paired_differences(
            paired
        )
    )
    paired_summary.to_csv(
        master_dir
        / "summary__paired_sdm_minus_leroux.csv",
        index=False,
    )
    spectra = collect_spectra(
        output_dir
    )
    if not spectra.empty:
        spectra.to_csv(
            master_dir
            / "all_runs__spectral_recovery.csv",
            index=False,
        )
        spectral_summary = (
            build_spectral_summary(
                spectra
            )
        )
        spectral_summary.to_csv(
            master_dir
            / "summary_across_seeds__spectral_recovery.csv",
            index=False,
        )
        plot_spectral_summary(
            spectral_summary,
            plot_dir
            / "nonleroux_sdm_truth__spectral_recovery.png",
            dpi=dpi,
        )

# =============================================================================
# CLI

# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Non-Leroux SDM-truth benchmark: "
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
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "nonleroux_sdm_truth_prediction_results"
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
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )
    from sdmcar.filters import (
        LerouxCARFilterFullVI
        as _LerouxCARFilterFullVI,
        AdaptivePrecisionPSplineFullVI
        as _AdaptivePrecisionPSplineFullVI,
    )
    from sdmcar.models_holdout import (
        SpectralCAR_HoldoutVI
        as _SpectralCAR_HoldoutVI,
    )
    LerouxCARFilterFullVI = (
        _LerouxCARFilterFullVI
    )
    AdaptivePrecisionPSplineFullVI = (
        _AdaptivePrecisionPSplineFullVI
    )
    SpectralCAR_HoldoutVI = (
        _SpectralCAR_HoldoutVI
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    # ---------------------------------------------------------
    # One fixed rook graph for:
    #   1. data generation,
    #   2. fitted Leroux,
    #   3. fitted SDM-CAR.
    # ---------------------------------------------------------
    W_true = build_rook_adjacency(
        N_ROWS,
        N_COLS,
    )
    L_true = graph_laplacian(
        W_true
    )
    lam_true, U_true = np.linalg.eigh(
        L_true
    )
    lam_true[
        np.abs(lam_true) < ZERO_TOL
    ] = 0.0
    lam_true = np.clip(
        lam_true,
        0.0,
        None,
    )
    n_zero_modes = int(
        np.sum(lam_true <= ZERO_TOL)
    )
    if n_zero_modes != 1:
        raise RuntimeError(
            "Expected exactly one zero Laplacian mode "
            f"for the connected rook graph; found {n_zero_modes}."
        )
    test_mask = holdout_block_mask(
        N_ROWS,
        N_COLS,
        HOLDOUT_SIDE,
        HOLDOUT_PATTERN,
    )
    truth = build_true_sdm_spectrum(
        lam_true
    )
    # Save a graph-level truth definition once.  It is identical for all seeds.
    truth_definition = pd.DataFrame(
        {
            "mode": np.arange(N),
            "lambda": lam_true,
            "q_true": truth["q_true"],
            "F_true": truth["F_true"],
            "F_best_leroux_family": truth["best_leroux_F"],
        }
    )
    truth_definition.to_csv(
        output_dir / "truth_spectrum_definition.csv",
        index=False,
    )
    common = {
        "W_true": W_true,
        "L_true": L_true,
        "lam_true": lam_true,
        "U_true": U_true,
        "test_mask": test_mask,
        **truth,
    }
    print("Project root:", project_root)
    print("Output directory:", output_dir)
    print("Device:", DEVICE)
    print("Seeds:", args.seeds)
    print("reference_rho:", REFERENCE_RHO)
    print("reference_tau2:", REFERENCE_TAU2)
    print("sigma2_true:", SIGMA2_TRUE)
    print(
        "Truth and both fitted models use the same "
        "rook graph and graph eigenbasis."
    )
    print("truth d2:", np.asarray(truth["d2_true"]))
    print(
        "precision non-affinity relative L2:",
        f"{truth['precision_nonaffinity_rel_l2']:.6f}",
    )
    print(
        "best Leroux approximation to truth: ",
        f"rho={truth['best_leroux_rho']:.6f}, ",
        f"tau2={truth['best_leroux_tau2']:.6f}, ",
        f"log-spectrum RMSE={truth['best_leroux_log_rmse']:.6f}",
        sep="",
    )
    for seed in args.seeds:
        prefix = run_prefix(
            int(seed)
        )
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
            f"\n=== seed={seed} ===",
            flush=True,
        )
        if args.resume and complete.exists():
            print(
                "    SKIP: completed run already exists.",
                flush=True,
            )
            continue
        if run_dir.exists():
            shutil.rmtree(
                run_dir
            )
        run_one(
            seed=int(seed),
            common=common,
            output_dir=output_dir,
        )
    print(
        "\nBuilding cross-seed summaries...",
        flush=True,
    )
    build_cross_seed_outputs(
        output_dir,
        dpi=args.dpi,
    )
    print(
        "\n=== NON-LEROUX SDM-TRUTH EXPERIMENT COMPLETE ==="
    )
    print(
        "Per-run results:",
        output_dir / "per_run",
    )
    print(
        "Master tables:",
        output_dir / "master_tables",
    )
    print(
        "Summary plots:",
        output_dir / "summary_plots",
    )

if __name__ == "__main__":
    main()
