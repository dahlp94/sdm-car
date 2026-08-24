"""
Full artifact sweep for the controlled eigenvalue-warp experiment.

Source experiment
-----------------
This script follows the reordered notebook
"controlled_power_warp_leroux_vs_adaptive_pspline_reordered(1).ipynb".
It repeats the same Leroux CAR vs Adaptive Precision P-spline SDM-CAR
experiment across multiple SEED and POWER_GAMMA values while keeping the
remaining configuration locked.

For a fixed SEED, the same generated data and the same holdout are reused for
all POWER_GAMMA values. Only the eigenvalue coordinate supplied to the fitted
models is warped. This preserves the controlled experiment and makes gamma
comparisons paired within seed.

Artifacts
---------
For every (seed, gamma) pair the script saves:
  * every displayed notebook table as CSV;
  * all 10 notebook figures as PNG;
  * CSV data underlying several plots (ELBO, spectral curves, predictions,
    and precision operators);
  * an artifact manifest and a completion marker.

At the end it also combines run-level tables across all seeds/gammas and
creates across-seed trend plots versus POWER_GAMMA.

Example
-------
python power_warp_full_artifact_sweep.py \
    --seeds 111 222 333 444 555 \
    --gammas 1.0 0.90 0.75 0.50 0.35 \
    --resume

Windows PowerShell
------------------
python power_warp_full_artifact_sweep.py `
    --seeds 111 222 333 444 555 `
    --gammas 1.0 0.90 0.75 0.50 0.35 `
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
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Preserved from the notebook's Windows setup.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# LOCKED EXPERIMENT CONFIGURATION -- copied from the reordered notebook
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
HOLDOUT_OPTIONS = (
    "middle",
    "left",
    "right",
    "top",
    "down",
)

FIX_SIGMA2 = True
NUM_MC = 8
PRED_MC = 512
POSTERIOR_SPECTRUM_DRAWS = 512
POSTERIOR_PARAMETER_DRAWS = 2048
GRAD_CLIP = 10.0
JITTER = 1e-8
BETA_PRIOR_VAR = 10.0

LEROUX_ITERS = 3000
LEROUX_LR = 1e-3

PSPLINE_ITERS = 6000
PSPLINE_LR = 3e-4

# Direct capacity optimization
DIRECT_LEROUX_ITERS = 3000
DIRECT_LEROUX_LR = 1e-2
DIRECT_PSPLINE_ITERS = 4000
DIRECT_PSPLINE_LR = 2e-2

# Leroux initialization
INIT_LEROUX_TAU2 = 0.50
INIT_LEROUX_RHO = 0.80
LEROUX_RHO_EPS = 1e-4
LEROUX_LOG_STD_INIT = -3.0

# Adaptive precision P-spline initialization and locked prior
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

# Set by main() after importing the project package.
LerouxCARFilterFullVI = None
AdaptivePrecisionPSplineFullVI = None
SpectralCAR_HoldoutVI = None


# =============================================================================
# GENERAL ARTIFACT HELPERS
# =============================================================================

def gamma_token(gamma: float) -> str:
    """Stable filename token, e.g. 0.35 -> 0p350."""
    return f"{float(gamma):.3f}".replace("-", "m").replace(".", "p")


def run_prefix(seed: int, gamma: float) -> str:
    return f"seed_{int(seed):05d}__power_gamma_{gamma_token(gamma)}"


def ensure_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns/index while preserving all table values."""
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join(
                str(part)
                for part in col
                if str(part) not in {"", "None"}
            )
            for col in out.columns.to_flat_index()
        ]

    if not isinstance(out.index, pd.RangeIndex) or out.index.name is not None:
        out = out.reset_index()

    return out


def with_run_ids(df: pd.DataFrame, seed: int, gamma: float) -> pd.DataFrame:
    out = ensure_dataframe_for_csv(df)
    if "seed" in out.columns:
        out = out.drop(columns=["seed"])
    if "power_gamma" in out.columns:
        out = out.drop(columns=["power_gamma"])
    out.insert(0, "power_gamma", float(gamma))
    out.insert(0, "seed", int(seed))
    return out


def save_table(
    df: pd.DataFrame,
    *,
    seed: int,
    gamma: float,
    tables_dir: Path,
    stem: str,
    manifest: List[dict],
) -> Path:
    prefix = run_prefix(seed, gamma)
    path = tables_dir / f"{prefix}__table__{stem}.csv"
    out = with_run_ids(df, seed, gamma)
    out.to_csv(path, index=False)
    manifest.append(
        {
            "seed": seed,
            "power_gamma": gamma,
            "artifact_type": "table_csv",
            "artifact_name": stem,
            "path": str(path),
        }
    )
    return path


def save_plot_data(
    df: pd.DataFrame,
    *,
    seed: int,
    gamma: float,
    data_dir: Path,
    stem: str,
    manifest: List[dict],
) -> Path:
    prefix = run_prefix(seed, gamma)
    path = data_dir / f"{prefix}__plot_data__{stem}.csv"
    out = with_run_ids(df, seed, gamma)
    out.to_csv(path, index=False)
    manifest.append(
        {
            "seed": seed,
            "power_gamma": gamma,
            "artifact_type": "plot_data_csv",
            "artifact_name": stem,
            "path": str(path),
        }
    )
    return path


def save_figure(
    fig,
    *,
    seed: int,
    gamma: float,
    plots_dir: Path,
    stem: str,
    number: int,
    dpi: int,
    manifest: List[dict],
) -> Path:
    prefix = run_prefix(seed, gamma)
    path = plots_dir / f"{prefix}__plot_{number:02d}__{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "seed": seed,
            "power_gamma": gamma,
            "artifact_type": "plot_png",
            "artifact_name": stem,
            "path": str(path),
        }
    )
    return path


def write_manifest(
    manifest: List[dict],
    *,
    seed: int,
    gamma: float,
    run_dir: Path,
) -> Path:
    prefix = run_prefix(seed, gamma)
    path = run_dir / f"{prefix}__artifact_manifest.csv"
    pd.DataFrame(manifest).to_csv(path, index=False)
    return path


# =============================================================================
# GRAPH / DATA / WARP HELPERS
# =============================================================================

def build_rook_adjacency(n_rows, n_cols):
    n = n_rows * n_cols
    W = np.zeros((n, n), dtype=float)

    def idx(row, col):
        return row * n_cols + col

    for row in range(n_rows):
        for col in range(n_cols):
            i = idx(row, col)

            if col + 1 < n_cols:
                j = idx(row, col + 1)
                W[i, j] = 1.0
                W[j, i] = 1.0

            if row + 1 < n_rows:
                j = idx(row + 1, col)
                W[i, j] = 1.0
                W[j, i] = 1.0

    return W


def graph_laplacian(W):
    degree = W.sum(axis=1)
    L = np.diag(degree) - W
    return 0.5 * (L + L.T)


def power_warp(lam, gamma):
    lam = np.asarray(lam, dtype=float)
    lam_max = float(lam.max())

    warped = lam_max * (
        np.clip(lam, 0.0, None) / lam_max
    ) ** gamma
    warped[lam <= ZERO_TOL] = 0.0
    return warped


def holdout_block_mask(n_rows, n_cols, side, pattern):
    if side <= 0:
        raise ValueError("side must be positive.")
    if side > n_rows or side > n_cols:
        raise ValueError("side must not exceed either grid dimension.")
    if pattern not in HOLDOUT_OPTIONS:
        raise ValueError(f"Unknown holdout pattern: {pattern}.")

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    row_center = (n_rows - side) // 2
    col_center = (n_cols - side) // 2

    if pattern == "middle":
        mask[row_center:row_center + side, col_center:col_center + side] = True
    elif pattern == "left":
        mask[row_center:row_center + side, :side] = True
    elif pattern == "right":
        mask[row_center:row_center + side, n_cols - side:] = True
    elif pattern == "top":
        mask[:side, col_center:col_center + side] = True
    elif pattern == "down":
        mask[n_rows - side:, col_center:col_center + side] = True

    return mask.ravel()


def generate_data(seed: int, lam_true: np.ndarray, U_true: np.ndarray):
    """Exactly the notebook's unwarped Leroux data-generating mechanism."""
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

    return {
        "X": X,
        "F_true": F_true,
        "phi_true": phi_true,
        "eta_true": eta_true,
        "y": y,
    }


# =============================================================================
# FILTER / MODEL CONSTRUCTORS AND VI
# =============================================================================

def rho_to_raw(rho, rho_eps=LEROUX_RHO_EPS):
    probability = rho / (1.0 - rho_eps)
    if not 0.0 < probability < 1.0:
        raise ValueError("rho is incompatible with rho_eps.")
    return math.log(probability / (1.0 - probability))


def make_leroux_filter():
    return LerouxCARFilterFullVI(
        mu_log_tau2=math.log(INIT_LEROUX_TAU2),
        log_std_log_tau2=LEROUX_LOG_STD_INIT,
        mu_rho_raw=rho_to_raw(INIT_LEROUX_RHO),
        log_std_rho_raw=LEROUX_LOG_STD_INIT,
        fixed_rho=None,
        rho_eps=LEROUX_RHO_EPS,
    ).to(DEVICE)


def make_pspline_filter(lam_t):
    return AdaptivePrecisionPSplineFullVI(
        lam_max=float(lam_t.max().item()),
        mu_log_q_left=math.log(INIT_Q_LEFT),
        mu_log_q_right=math.log(INIT_Q_RIGHT),
        **ADAPTIVE_PSPLINE_KWARGS,
    ).to(DEVICE)


def fit_spectral_vi(
    *,
    label,
    filter_module,
    iterations,
    learning_rate,
    seed,
    X_t,
    y_fit_t,
    lam_warp_t,
    U_warp_t,
    is_holdout_t,
    prior_V0,
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

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        elbo, _ = model.elbo()
        loss = -elbo

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{label}: non-finite loss at iteration {iteration}."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()

        history.append(float(elbo.detach().cpu()))

        if iteration % 500 == 0:
            print(
                f"      {label}: {iteration}/{iterations}, "
                f"mean last 100 ELBO={np.mean(history[-100:]):.3f}",
                flush=True,
            )

    return {
        "label": label,
        "model": model,
        "history": np.asarray(history, dtype=float),
    }


# =============================================================================
# PART I -- DIRECT REPRESENTATIONAL CAPACITY
# =============================================================================

def log_spectrum_rmse(estimated, target, mask=None):
    estimated = np.asarray(estimated, dtype=float)
    target = np.asarray(target, dtype=float)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        estimated = estimated[mask]
        target = target[mask]

    estimated = np.clip(estimated, 1e-12, None)
    target = np.clip(target, 1e-12, None)

    return float(
        np.sqrt(
            np.mean((np.log(estimated) - np.log(target)) ** 2)
        )
    )


def direct_leroux_theta(filter_module):
    return {
        "log_tau2": filter_module.mu_log_tau2.reshape(1),
        "rho_raw": filter_module.mu_rho_raw.reshape(1),
    }


def direct_pspline_theta(filter_module):
    theta = {
        "log_q_left": filter_module.mu_log_q_endpoints[0:1],
        "log_q_right": filter_module.mu_log_q_endpoints[1:2],
    }
    for j in range(filter_module.K):
        theta[f"d2_{j}_raw"] = filter_module.mu_d2[j:j + 1]
    return theta


def direct_capacity_fit(
    *,
    lam_warp_t: torch.Tensor,
    q_target_t: torch.Tensor,
    q_target: np.ndarray,
    positive_mode_mask: np.ndarray,
    seed: int,
):
    """Notebook capacity test: no likelihood and no statistical estimation."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    direct_leroux = make_leroux_filter()
    optimizer = torch.optim.Adam(
        [direct_leroux.mu_log_tau2, direct_leroux.mu_rho_raw],
        lr=DIRECT_LEROUX_LR,
    )

    for _ in range(1, DIRECT_LEROUX_ITERS + 1):
        optimizer.zero_grad(set_to_none=True)
        F_fit = direct_leroux.spectrum(
            lam_warp_t,
            direct_leroux_theta(direct_leroux),
        ).clamp_min(1e-12)
        q_fit = 1.0 / F_fit
        loss = torch.mean((torch.log(q_fit) - torch.log(q_target_t)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [direct_leroux.mu_log_tau2, direct_leroux.mu_rho_raw],
            GRAD_CLIP,
        )
        optimizer.step()

    with torch.no_grad():
        F_direct_leroux = (
            direct_leroux.spectrum(
                lam_warp_t,
                direct_leroux_theta(direct_leroux),
            )
            .detach()
            .cpu()
            .numpy()
        )
    q_direct_leroux = 1.0 / np.clip(F_direct_leroux, 1e-12, None)

    direct_pspline = make_pspline_filter(lam_warp_t)
    optimizer = torch.optim.Adam(
        [direct_pspline.mu_log_q_endpoints, direct_pspline.mu_d2],
        lr=DIRECT_PSPLINE_LR,
    )

    for _ in range(1, DIRECT_PSPLINE_ITERS + 1):
        optimizer.zero_grad(set_to_none=True)
        q_fit = direct_pspline.precision_from_unconstrained(
            lam_warp_t,
            direct_pspline_theta(direct_pspline),
        )
        loss = torch.mean((torch.log(q_fit) - torch.log(q_target_t)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [direct_pspline.mu_log_q_endpoints, direct_pspline.mu_d2],
            GRAD_CLIP,
        )
        optimizer.step()

    with torch.no_grad():
        q_direct_pspline = (
            direct_pspline.precision_from_unconstrained(
                lam_warp_t,
                direct_pspline_theta(direct_pspline),
            )
            .detach()
            .cpu()
            .numpy()
        )

    direct_d2_l2 = float(
        direct_pspline.shrinkage_summary()["d2_l2"].detach().cpu()
    )

    table = pd.DataFrame(
        [
            {
                "model": "Leroux CAR",
                "log_precision_RMSE": log_spectrum_rmse(
                    q_direct_leroux, q_target
                ),
                "positive_mode_log_precision_RMSE": log_spectrum_rmse(
                    q_direct_leroux, q_target, positive_mode_mask
                ),
                "d2_l2": np.nan,
            },
            {
                "model": "Adaptive P-spline",
                "log_precision_RMSE": log_spectrum_rmse(
                    q_direct_pspline, q_target
                ),
                "positive_mode_log_precision_RMSE": log_spectrum_rmse(
                    q_direct_pspline, q_target, positive_mode_mask
                ),
                "d2_l2": direct_d2_l2,
            },
        ]
    )

    return {
        "table": table,
        "q_direct_leroux": q_direct_leroux,
        "q_direct_pspline": q_direct_pspline,
    }


# =============================================================================
# POSTERIOR SPECTRAL RECOVERY
# =============================================================================

def summarize_draw_matrix(draws, credible_level=0.95):
    tail = (1.0 - credible_level) / 2.0
    return {
        "mean": draws.mean(dim=0),
        "lower": torch.quantile(draws, tail, dim=0),
        "upper": torch.quantile(draws, 1.0 - tail, dim=0),
    }


@torch.no_grad()
def posterior_spectrum_summary(result, draws, seed):
    torch.manual_seed(seed)
    model = result["model"]
    F_draws = []

    for _ in range(draws):
        theta = model.filter.sample_unconstrained()
        F_draws.append(
            model.filter.spectrum(model.lam, theta).clamp_min(1e-12)
        )

    F_draws = torch.stack(F_draws, dim=0)
    q_draws = 1.0 / F_draws

    F_summary = summarize_draw_matrix(F_draws)
    q_summary = summarize_draw_matrix(q_draws)

    def np_(x):
        return x.detach().cpu().numpy()

    return {
        "F_mean": np_(F_summary["mean"]),
        "F_lower": np_(F_summary["lower"]),
        "F_upper": np_(F_summary["upper"]),
        "q_mean": np_(q_summary["mean"]),
        "q_lower": np_(q_summary["lower"]),
        "q_upper": np_(q_summary["upper"]),
    }


@torch.no_grad()
def posterior_pspline_component_summary(result, draws, seed):
    torch.manual_seed(seed)
    filt = result["model"].filter
    lam_t = result["model"].lam

    q_full_draws = []
    q_affine_draws = []
    log_correction_draws = []

    for _ in range(draws):
        theta = filt.sample_unconstrained()
        q_full = filt.precision_from_unconstrained(lam_t, theta)
        q_affine = filt.affine_precision_from_unconstrained(lam_t, theta)
        q_full_draws.append(q_full)
        q_affine_draws.append(q_affine)
        log_correction_draws.append(
            torch.log(
                q_full.clamp_min(1e-12) / q_affine.clamp_min(1e-12)
            )
        )

    summaries = {
        "q_full": summarize_draw_matrix(torch.stack(q_full_draws, dim=0)),
        "q_affine": summarize_draw_matrix(torch.stack(q_affine_draws, dim=0)),
        "log_correction": summarize_draw_matrix(
            torch.stack(log_correction_draws, dim=0)
        ),
    }

    def np_(x):
        return x.detach().cpu().numpy()

    out = {}
    for name, summary in summaries.items():
        for key, value in summary.items():
            out[f"{name}_{key}"] = np_(value)
    return out


@torch.no_grad()
def posterior_leroux_parameter_summary(result, draws, seed):
    torch.manual_seed(seed)
    filt = result["model"].filter

    tau2_draws = []
    rho_draws = []

    for _ in range(draws):
        constrained = filt._constrain(filt.sample_unconstrained())
        tau2_draws.append(constrained["tau2"].reshape(()))
        rho_draws.append(constrained["rho"].reshape(()))

    tau2_draws = torch.stack(tau2_draws)
    rho_draws = torch.stack(rho_draws)

    def scalar_summary(draws_t):
        return {
            "mean": float(draws_t.mean().cpu()),
            "lower_95": float(torch.quantile(draws_t, 0.025).cpu()),
            "upper_95": float(torch.quantile(draws_t, 0.975).cpu()),
        }

    return {
        "tau2": scalar_summary(tau2_draws),
        "rho": scalar_summary(rho_draws),
    }


@torch.no_grad()
def posterior_pspline_shrinkage_summary(result, draws, seed):
    torch.manual_seed(seed)
    filt = result["model"].filter

    d2_l2 = []
    max_abs_d2 = []
    mean_local_scale = []
    max_local_scale = []

    for _ in range(draws):
        theta = filt.sample_unconstrained()
        constrained = filt._constrain(theta)
        d2 = constrained["d2"]
        local_scales = torch.exp(
            torch.stack(
                [theta[f"log_lambda_{j}"].reshape(()) for j in range(filt.K)]
            )
        )

        d2_l2.append(torch.linalg.vector_norm(d2))
        max_abs_d2.append(torch.max(torch.abs(d2)))
        mean_local_scale.append(local_scales.mean())
        max_local_scale.append(local_scales.max())

    metrics = {
        "d2_l2": torch.stack(d2_l2),
        "max_abs_d2": torch.stack(max_abs_d2),
        "mean_local_scale": torch.stack(mean_local_scale),
        "max_local_scale": torch.stack(max_local_scale),
    }

    rows = []
    for name, draws_t in metrics.items():
        rows.append(
            {
                "quantity": name,
                "posterior_mean": float(draws_t.mean().cpu()),
                "lower_95": float(torch.quantile(draws_t, 0.025).cpu()),
                "upper_95": float(torch.quantile(draws_t, 0.975).cpu()),
            }
        )
    return pd.DataFrame(rows)


def variance_weighted_log_spectrum_rmse(estimated, target, mask=None):
    estimated = np.asarray(estimated, dtype=float)
    target = np.asarray(target, dtype=float)

    if mask is None:
        mask = np.ones(target.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    estimated = np.clip(estimated[mask], 1e-12, None)
    target = np.clip(target[mask], 1e-12, None)
    weights = target / target.sum()
    squared_log_error = (np.log(estimated) - np.log(target)) ** 2
    return float(np.sqrt(np.sum(weights * squared_log_error)))


def frequency_band_masks(lam, positive_mask):
    positive_idx = np.flatnonzero(np.asarray(positive_mask, dtype=bool))
    positive_idx = positive_idx[np.argsort(np.asarray(lam)[positive_idx])]
    low_idx, mid_idx, high_idx = np.array_split(positive_idx, 3)

    out = {}
    for name, idx in [("low", low_idx), ("mid", mid_idx), ("high", high_idx)]:
        mask = np.zeros(len(lam), dtype=bool)
        mask[idx] = True
        out[name] = mask
    return out


def spectral_metric_row(model_name, F_estimated, F_target, lam_warp, positive_mode_mask):
    band_masks = frequency_band_masks(lam_warp, positive_mode_mask)
    return {
        "model": model_name,
        "log_RMSE": log_spectrum_rmse(F_estimated, F_target),
        "positive_mode_log_RMSE": log_spectrum_rmse(
            F_estimated, F_target, positive_mode_mask
        ),
        "variance_weighted_log_RMSE": variance_weighted_log_spectrum_rmse(
            F_estimated, F_target
        ),
        "low_frequency_log_RMSE": log_spectrum_rmse(
            F_estimated, F_target, band_masks["low"]
        ),
        "mid_frequency_log_RMSE": log_spectrum_rmse(
            F_estimated, F_target, band_masks["mid"]
        ),
        "high_frequency_log_RMSE": log_spectrum_rmse(
            F_estimated, F_target, band_masks["high"]
        ),
    }


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# =============================================================================
# POSTERIOR PREDICTION / CALIBRATION
# =============================================================================

def stable_cholesky(matrix, base_jitter=1e-10, max_tries=8):
    matrix = 0.5 * (matrix + matrix.T)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    scale = torch.diagonal(matrix).abs().mean().detach().clamp_min(1.0)
    last_error = None

    for attempt in range(max_tries):
        jitter = base_jitter * (10.0 ** attempt) * scale
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
        except RuntimeError as error:
            last_error = error

    raise RuntimeError("stable_cholesky failed.") from last_error


@torch.no_grad()
def predict_eta_vi_mc(model, num_mc, seed):
    K = int(num_mc)
    if K < 2:
        raise ValueError("num_mc must be at least 2.")

    torch.manual_seed(seed)
    component_means = []
    component_vars = []
    draws = []

    second_moment_sum = torch.zeros(
        (model.n_test, model.n_test),
        dtype=model.y_train.dtype,
        device=model.y_train.device,
    )
    I_train = torch.eye(
        model.n_train,
        dtype=model.y_train.dtype,
        device=model.y_train.device,
    )

    for _ in range(K):
        sigma2 = model._sample_sigma2()
        theta = model.filter.sample_unconstrained()
        F_lam = model.filter.spectrum(model.lam, theta).clamp_min(model.min_variance)

        terms = model._observed_terms(F_lam, sigma2)
        m_beta, V_beta = model._beta_update_from_terms(terms)

        S_HO = (model.U_test * F_lam.unsqueeze(0)) @ model.U_train.T
        S_HH = (model.U_test * F_lam.unsqueeze(0)) @ model.U_test.T
        S_OO = (model.U_train * F_lam.unsqueeze(0)) @ model.U_train.T

        C_OO = S_OO + sigma2 * I_train
        chol_OO = stable_cholesky(C_OO, base_jitter=max(model.jitter, 1e-10))
        gain = torch.cholesky_solve(S_HO.T, chol_OO).T
        A_beta = model.X_test - gain @ model.X_train
        pred_mean = gain @ model.y_train + A_beta @ m_beta

        conditional_cov = S_HH - gain @ S_HO.T
        conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)

        pred_cov = conditional_cov + A_beta @ V_beta @ A_beta.T
        pred_cov = 0.5 * (pred_cov + pred_cov.T)
        pred_var = torch.diagonal(pred_cov).clamp_min(0.0)

        component_means.append(pred_mean)
        component_vars.append(pred_var)
        second_moment_sum += pred_cov + torch.outer(pred_mean, pred_mean)

        chol_beta = torch.linalg.cholesky(V_beta + model.jitter * model.I_beta)
        beta_draw = m_beta + chol_beta @ torch.randn(
            model.p,
            dtype=m_beta.dtype,
            device=m_beta.device,
        )

        chol_eta = stable_cholesky(
            conditional_cov,
            base_jitter=max(model.jitter, 1e-10),
        )
        eta_draw = (
            gain @ model.y_train
            + A_beta @ beta_draw
            + chol_eta @ torch.randn(
                model.n_test,
                dtype=m_beta.dtype,
                device=m_beta.device,
            )
        )
        draws.append(eta_draw)

    means = torch.stack(component_means, dim=0)
    variances = torch.stack(component_vars, dim=0)
    draws = torch.stack(draws, dim=0)
    mean = means.mean(dim=0)

    covariance = second_moment_sum / float(K) - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    variance = torch.diagonal(covariance).clamp_min(0.0)

    return {
        "mean": mean,
        "var": variance,
        "sd": torch.sqrt(variance),
        "cov": covariance,
        "q025": torch.quantile(draws, 0.025, dim=0),
        "q975": torch.quantile(draws, 0.975, dim=0),
        "draws": draws,
        "component_means": means,
        "component_vars": variances,
        "test_idx": model.test_idx,
    }


def marginal_mixture_nlpd(target, component_means, component_vars):
    target = torch.as_tensor(
        target,
        dtype=component_means.dtype,
        device=component_means.device,
    ).reshape(-1)
    variances = component_vars.clamp_min(1e-12)

    log_prob = -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variances)
        + (target.unsqueeze(0) - component_means).square() / variances
    )
    log_density = torch.logsumexp(log_prob, dim=0) - math.log(component_means.shape[0])
    return float((-log_density.mean()).detach().cpu())


def crps_from_draws(target, draws):
    target = torch.as_tensor(target, dtype=draws.dtype, device=draws.device).reshape(-1)
    M = draws.shape[0]
    first_term = torch.mean(torch.abs(draws - target.unsqueeze(0)), dim=0)
    sorted_draws, _ = torch.sort(draws, dim=0)
    ranks = torch.arange(1, M + 1, dtype=draws.dtype, device=draws.device).reshape(-1, 1)
    half_pairwise_term = (
        ((2.0 * ranks - M - 1.0) * sorted_draws).sum(dim=0) / float(M * M)
    )
    return float((first_term - half_pairwise_term).mean().detach().cpu())


def predictive_metrics(target, prediction):
    target_t = torch.as_tensor(
        target,
        dtype=prediction["mean"].dtype,
        device=prediction["mean"].device,
    ).reshape(-1)
    error = prediction["mean"] - target_t
    coverage = (
        (target_t >= prediction["q025"]) & (target_t <= prediction["q975"])
    ).double().mean()

    return {
        "rmse": float(torch.sqrt(torch.mean(error.square())).cpu()),
        "nlpd": marginal_mixture_nlpd(
            target_t,
            prediction["component_means"],
            prediction["component_vars"],
        ),
        "crps": crps_from_draws(target_t, prediction["draws"]),
        "coverage_95": float(coverage.cpu()),
        "interval_width_95": float(
            (prediction["q975"] - prediction["q025"]).mean().cpu()
        ),
    }


def latent_variance_calibration(target, prediction):
    target_t = torch.as_tensor(
        target,
        dtype=prediction["mean"].dtype,
        device=prediction["mean"].device,
    ).reshape(-1)

    residual = target_t - prediction["mean"]
    sd = prediction["sd"].clamp_min(1e-12)
    z = residual / sd
    msse = torch.mean(z.square())

    chol = stable_cholesky(prediction["cov"], base_jitter=1e-10)
    solved = torch.cholesky_solve(residual.unsqueeze(1), chol).squeeze(1)
    normalized_mahalanobis = residual @ solved / float(target_t.numel())

    return {
        "MSSE": float(msse.cpu()),
        "normalized_Mahalanobis": float(normalized_mahalanobis.cpu()),
    }


# =============================================================================
# FULL PRECISION-OPERATOR RECOVERY
# =============================================================================

def precision_matrix_from_modes(U, q):
    U = np.asarray(U, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    if U.shape[1] != q.size:
        raise ValueError("U and q have incompatible dimensions.")
    Q = (U * q[None, :]) @ U.T
    return 0.5 * (Q + Q.T)


@torch.no_grad()
def posterior_modal_precision_draws(result, *, draws, seed):
    model = result["model"]
    rng_state = torch.random.get_rng_state()

    try:
        torch.manual_seed(seed)
        q_draws = []
        for _ in range(draws):
            theta = model.filter.sample_unconstrained()
            F = model.filter.spectrum(model.lam, theta).clamp_min(1e-12)
            q_draws.append(1.0 / F)
        q_draws = torch.stack(q_draws, dim=0)
    finally:
        torch.random.set_rng_state(rng_state)

    return q_draws.detach().cpu().numpy()


def precision_operator_metric_draws(q_draws, q_target_local, U_local):
    q_draws = np.asarray(q_draws, dtype=float)
    q_target_local = np.asarray(q_target_local, dtype=float).reshape(-1)
    U_local = np.asarray(U_local, dtype=float)

    if q_draws.ndim == 1:
        q_draws = q_draws[None, :]
    if q_draws.shape[1] != q_target_local.size:
        raise ValueError("q_draws and q_target_local have incompatible sizes.")
    if U_local.shape[1] != q_target_local.size:
        raise ValueError("U_local and q_target_local have incompatible sizes.")

    n = U_local.shape[0]
    n_offdiag = n * (n - 1)
    eps = np.finfo(float).eps

    delta_q = q_draws - q_target_local[None, :]
    frobenius_error = np.linalg.norm(delta_q, axis=1)
    target_frobenius = np.linalg.norm(q_target_local)
    relative_frobenius = frobenius_error / max(target_frobenius, eps)
    entrywise_frobenius_rmse = frobenius_error / n

    operator_norm_error = np.max(np.abs(delta_q), axis=1)
    target_operator_norm = np.max(np.abs(q_target_local))
    relative_operator_norm = operator_norm_error / max(target_operator_norm, eps)

    U_squared = U_local ** 2
    diag_draws = q_draws @ U_squared.T
    diag_target = U_squared @ q_target_local
    diag_error = diag_draws - diag_target[None, :]
    diag_error_norm_sq = np.sum(diag_error ** 2, axis=1)
    diagonal_rmse = np.sqrt(np.mean(diag_error ** 2, axis=1))

    offdiag_error_norm_sq = np.maximum(
        frobenius_error ** 2 - diag_error_norm_sq,
        0.0,
    )
    offdiagonal_rmse = np.sqrt(offdiag_error_norm_sq / n_offdiag)

    target_diag_norm_sq = np.sum(diag_target ** 2)
    target_offdiag_norm_sq = max(
        target_frobenius ** 2 - target_diag_norm_sq,
        0.0,
    )
    relative_offdiagonal_frobenius = (
        np.sqrt(offdiag_error_norm_sq)
        / max(np.sqrt(target_offdiag_norm_sq), eps)
    )

    column_sums = U_local.sum(axis=0)
    sum_all_draws = q_draws @ (column_sums ** 2)
    trace_draws = q_draws.sum(axis=1)
    sum_offdiag_draws = sum_all_draws - trace_draws

    sum_all_target = np.sum(q_target_local * (column_sums ** 2))
    trace_target = np.sum(q_target_local)
    sum_offdiag_target = sum_all_target - trace_target

    sq_offdiag_draws = np.maximum(
        np.sum(q_draws ** 2, axis=1) - np.sum(diag_draws ** 2, axis=1),
        0.0,
    )
    sq_offdiag_target = max(
        np.sum(q_target_local ** 2) - np.sum(diag_target ** 2),
        0.0,
    )
    cross_offdiag = q_draws @ q_target_local - diag_draws @ diag_target

    covariance_numerator = (
        cross_offdiag
        - sum_offdiag_draws * sum_offdiag_target / n_offdiag
    )
    variance_draws = np.maximum(
        sq_offdiag_draws - sum_offdiag_draws ** 2 / n_offdiag,
        0.0,
    )
    variance_target = max(
        sq_offdiag_target - sum_offdiag_target ** 2 / n_offdiag,
        0.0,
    )
    correlation_denominator = np.sqrt(variance_draws * variance_target)

    offdiagonal_correlation = np.full(q_draws.shape[0], np.nan, dtype=float)
    valid = correlation_denominator > 1e-14
    offdiagonal_correlation[valid] = (
        covariance_numerator[valid] / correlation_denominator[valid]
    )
    offdiagonal_correlation = np.clip(offdiagonal_correlation, -1.0, 1.0)

    return {
        "relative_frobenius": relative_frobenius,
        "entrywise_frobenius_rmse": entrywise_frobenius_rmse,
        "diagonal_rmse": diagonal_rmse,
        "offdiagonal_rmse": offdiagonal_rmse,
        "relative_offdiagonal_frobenius": relative_offdiagonal_frobenius,
        "operator_norm_error": operator_norm_error,
        "relative_operator_norm": relative_operator_norm,
        "offdiagonal_correlation": offdiagonal_correlation,
    }


def posterior_metric_summary_table(metrics_by_model):
    rows = {}
    for model_name, metric_dict in metrics_by_model.items():
        row = {}
        for metric_name, values in metric_dict.items():
            values = np.asarray(values, dtype=float)
            row[(metric_name, "mean")] = np.nanmean(values)
            row[(metric_name, "median")] = np.nanmedian(values)
            row[(metric_name, "q025")] = np.nanquantile(values, 0.025)
            row[(metric_name, "q975")] = np.nanquantile(values, 0.975)
        rows[model_name] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(
        table.columns,
        names=["metric", "posterior_summary"],
    )
    table.index.name = "model"
    return table


def posterior_mean_operator_metric_row(
    model_name,
    scope,
    q_mean,
    q_target_local,
    U_local,
):
    metrics = precision_operator_metric_draws(
        np.asarray(q_mean)[None, :],
        q_target_local,
        U_local,
    )
    return {
        "model": model_name,
        "operator": scope,
        **{name: float(values[0]) for name, values in metrics.items()},
    }


def matrix_metric_guide_table():
    return pd.DataFrame(
        [
            {
                "metric": "relative_frobenius",
                "interpretation": "Global precision-operator error relative to target magnitude.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "entrywise_frobenius_rmse",
                "interpretation": "RMS error of a typical Q matrix entry.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "diagonal_rmse",
                "interpretation": "Recovery of diagonal/self-precision terms.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "offdiagonal_rmse",
                "interpretation": "RMS error in spatial interaction terms.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "relative_offdiagonal_frobenius",
                "interpretation": "Total interaction-pattern error relative to target off-diagonal magnitude.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "operator_norm_error",
                "interpretation": "Largest precision error over any unit-norm latent direction.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "relative_operator_norm",
                "interpretation": "Worst-direction error relative to the target operator norm.",
                "better": "lower; 0 is ideal",
            },
            {
                "metric": "offdiagonal_correlation",
                "interpretation": "Agreement in the spatial interaction pattern, after removing scale/mean effects.",
                "better": "higher; 1 is ideal",
            },
        ]
    )


# =============================================================================
# PLOT HELPERS -- same figures/order as the notebook
# =============================================================================

def plot_01_eigenvalue_warp(lam_true, lam_warp, gamma):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        lam_true,
        lam_warp,
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label=fr"power warp, $\gamma={gamma}$",
    )
    ax.plot(
        [0.0, lam_true.max()],
        [0.0, lam_true.max()],
        linestyle="--",
        linewidth=1.2,
        label=r"$\mu=\lambda$",
    )
    ax.set_xlabel(r"True eigenvalue $\lambda$")
    ax.set_ylabel(r"Warped eigenvalue $\mu$")
    ax.set_title("Eigenvalue-coordinate warp")
    ax.legend()
    return fig


def plot_02_holdout_region(test_mask):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.imshow(test_mask.reshape(N_ROWS, N_COLS), interpolation="nearest")
    ax.set_title(f"Holdout pattern: {HOLDOUT_PATTERN}, side={HOLDOUT_SIDE}")
    ax.axis("off")
    return fig


def plot_03_direct_capacity(lam_warp, q_target, q_direct_leroux, q_direct_pspline):
    order = np.argsort(lam_warp)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(lam_warp[order], q_target[order], linewidth=2.8, label="Target precision")
    ax.plot(
        lam_warp[order],
        q_direct_leroux[order],
        linestyle="--",
        linewidth=2.0,
        label="Best direct Leroux",
    )
    ax.plot(
        lam_warp[order],
        q_direct_pspline[order],
        linestyle=":",
        linewidth=2.4,
        label="Best direct adaptive P-spline",
    )
    ax.set_xlabel(r"Warped eigenvalue $\mu$")
    ax.set_ylabel("Precision eigenvalue")
    ax.set_title("Direct representational capacity")
    ax.legend()
    return fig


def plot_04_elbo(leroux_result, pspline_result):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in (leroux_result, pspline_result):
        smoothed = moving_average(result["history"], window=100)
        ax.plot(np.arange(smoothed.size) + 1, smoothed, label=result["label"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("100-iteration mean ELBO")
    ax.set_title("Likelihood-based VI optimization")
    ax.legend()
    return fig


def plot_05_posterior_spectral_recovery(
    lam_warp,
    F_target,
    q_target,
    leroux_post,
    pspline_post,
):
    order = np.argsort(lam_warp)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    axes[0].plot(
        lam_warp[order],
        F_target[order],
        linewidth=2.8,
        label="Target covariance",
    )
    for post, label in [
        (leroux_post, "Leroux"),
        (pspline_post, "Adaptive P-spline"),
    ]:
        axes[0].plot(
            lam_warp[order],
            post["F_mean"][order],
            linewidth=1.8,
            label=label,
        )
        axes[0].fill_between(
            lam_warp[order],
            post["F_lower"][order],
            post["F_upper"][order],
            alpha=0.12,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Warped eigenvalue $\mu$")
    axes[0].set_ylabel(r"Covariance spectrum $F$")
    axes[0].set_title("Posterior covariance-spectrum recovery")
    axes[0].legend()

    axes[1].plot(
        lam_warp[order],
        q_target[order],
        linewidth=2.8,
        label="Target precision",
    )
    for post, label in [
        (leroux_post, "Leroux"),
        (pspline_post, "Adaptive P-spline"),
    ]:
        axes[1].plot(
            lam_warp[order],
            post["q_mean"][order],
            linewidth=1.8,
            label=label,
        )
        axes[1].fill_between(
            lam_warp[order],
            post["q_lower"][order],
            post["q_upper"][order],
            alpha=0.12,
        )
    axes[1].set_xlabel(r"Warped eigenvalue $\mu$")
    axes[1].set_ylabel(r"Precision spectrum $q$")
    axes[1].set_title("Posterior precision-spectrum recovery")
    axes[1].legend()
    return fig


def plot_06_pspline_decomposition(lam_warp, q_target, pspline_components):
    order = np.argsort(lam_warp)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    axes[0].plot(
        lam_warp[order], q_target[order], linewidth=2.8, label="Target precision"
    )
    axes[0].plot(
        lam_warp[order],
        pspline_components["q_full_mean"][order],
        linewidth=2.0,
        label="P-spline full precision",
    )
    axes[0].plot(
        lam_warp[order],
        pspline_components["q_affine_mean"][order],
        linestyle="--",
        linewidth=2.0,
        label="P-spline affine baseline",
    )
    axes[0].set_xlabel(r"Warped eigenvalue $\mu$")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Adaptive P-spline precision decomposition")
    axes[0].legend()

    axes[1].plot(
        lam_warp[order],
        pspline_components["log_correction_mean"][order],
        linewidth=2.0,
        label=r"$E[\log(q/q_{\mathrm{affine}})]$",
    )
    axes[1].fill_between(
        lam_warp[order],
        pspline_components["log_correction_lower"][order],
        pspline_components["log_correction_upper"][order],
        alpha=0.12,
    )
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"Warped eigenvalue $\mu$")
    axes[1].set_ylabel("Log-precision correction")
    axes[1].set_title("Nonlinear correction beyond affine precision")
    axes[1].legend()
    return fig


def plot_07_latent_prediction(truth, leroux_prediction, pspline_prediction):
    truth = np.asarray(truth, dtype=float)
    sort_idx = np.argsort(truth)
    x = np.arange(truth.size)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, truth[sort_idx], linewidth=2.5, label="True latent field")

    for prediction, label in [
        (leroux_prediction, "Leroux"),
        (pspline_prediction, "Adaptive P-spline"),
    ]:
        mean = prediction["mean"].detach().cpu().numpy()[sort_idx]
        lower = prediction["q025"].detach().cpu().numpy()[sort_idx]
        upper = prediction["q975"].detach().cpu().numpy()[sort_idx]
        ax.plot(x, mean, linewidth=1.6, label=label)
        ax.fill_between(x, lower, upper, alpha=0.12)

    ax.set_xlabel("Held-out sites ordered by true latent value")
    ax.set_ylabel(r"Latent field $\eta_H$")
    ax.set_title("Held-out latent posterior prediction")
    ax.legend()
    return fig


def plot_08_operator_heatmaps(Q_target, Q_leroux_mean, Q_pspline_mean):
    Q_common_limit = max(
        np.max(np.abs(Q_target)),
        np.max(np.abs(Q_leroux_mean)),
        np.max(np.abs(Q_pspline_mean)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    matrices = [
        (Q_target, r"Target $Q_{\mathrm{target}}$"),
        (Q_leroux_mean, r"Leroux $E[Q\mid y_O]$"),
        (Q_pspline_mean, r"Adaptive P-spline $E[Q\mid y_O]$"),
    ]
    images = []
    for ax, (matrix, title) in zip(axes, matrices):
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-Q_common_limit,
            vmax=Q_common_limit,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
    fig.colorbar(images[0], ax=axes, shrink=0.80, label="Precision-matrix entry")
    return fig


def plot_09_operator_differences(Q_target, Q_leroux_mean, Q_pspline_mean):
    Q_leroux_difference = Q_leroux_mean - Q_target
    Q_pspline_difference = Q_pspline_mean - Q_target
    difference_limit = max(
        np.max(np.abs(Q_leroux_difference)),
        np.max(np.abs(Q_pspline_difference)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    difference_matrices = [
        (Q_leroux_difference, r"Leroux: $E[Q\mid y_O]-Q_{\mathrm{target}}$"),
        (Q_pspline_difference, r"P-spline: $E[Q\mid y_O]-Q_{\mathrm{target}}$"),
    ]
    images = []
    for ax, (matrix, title) in zip(axes, difference_matrices):
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
    fig.colorbar(images[0], ax=axes, shrink=0.80, label="Precision error")
    return fig


def add_identity_line(ax, arrays):
    values = np.concatenate([np.asarray(x).reshape(-1) for x in arrays])
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = 0.03 * max(upper - lower, 1e-12)
    lower -= padding
    upper += padding
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.2,
        label="45-degree line",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)


def plot_10_operator_scatter(Q_target, Q_leroux_mean, Q_pspline_mean):
    diagonal_mask_matrix = np.eye(N, dtype=bool)
    offdiagonal_mask_matrix = ~diagonal_mask_matrix

    target_diag = np.diag(Q_target)
    leroux_diag = np.diag(Q_leroux_mean)
    pspline_diag = np.diag(Q_pspline_mean)

    target_offdiag = Q_target[offdiagonal_mask_matrix]
    leroux_offdiag = Q_leroux_mean[offdiagonal_mask_matrix]
    pspline_offdiag = Q_pspline_mean[offdiagonal_mask_matrix]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].scatter(target_diag, leroux_diag, s=22, alpha=0.65, label="Leroux")
    axes[0].scatter(
        target_diag,
        pspline_diag,
        s=22,
        alpha=0.65,
        label="Adaptive P-spline",
    )
    add_identity_line(axes[0], [target_diag, leroux_diag, pspline_diag])
    axes[0].set_xlabel("Target diagonal precision")
    axes[0].set_ylabel("Posterior mean diagonal precision")
    axes[0].set_title("Diagonal precision recovery")
    axes[0].legend()

    axes[1].scatter(
        target_offdiag, leroux_offdiag, s=10, alpha=0.25, label="Leroux"
    )
    axes[1].scatter(
        target_offdiag,
        pspline_offdiag,
        s=10,
        alpha=0.25,
        label="Adaptive P-spline",
    )
    add_identity_line(
        axes[1], [target_offdiag, leroux_offdiag, pspline_offdiag]
    )
    axes[1].set_xlabel("Target off-diagonal precision")
    axes[1].set_ylabel("Posterior mean off-diagonal precision")
    axes[1].set_title("Spatial interaction recovery")
    axes[1].legend()
    return fig


# =============================================================================
# ONE COMPLETE NOTEBOOK RUN FOR ONE (SEED, POWER_GAMMA)
# =============================================================================

def run_one(
    *,
    seed: int,
    gamma: float,
    common: Mapping[str, np.ndarray],
    output_dir: Path,
    dpi: int,
    direct_capacity_cache: dict,
):
    prefix = run_prefix(seed, gamma)
    run_dir = output_dir / "per_run" / prefix
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    data_dir = run_dir / "plot_data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []

    lam_true = common["lam_true"]
    U_true = common["U_true"]
    L_true = common["L_true"]
    test_mask = common["test_mask"]
    train_mask = ~test_mask
    positive_mode_mask = common["positive_mode_mask"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    data = generate_data(seed, lam_true, U_true)
    X = data["X"]
    F_true = data["F_true"]
    eta_true = data["eta_true"]
    y = data["y"]

    lam_warp = power_warp(lam_true, gamma=gamma)
    U_warp = U_true.copy()
    np.testing.assert_allclose(U_warp, U_true, rtol=0.0, atol=0.0)

    lam_warp_t = torch.tensor(lam_warp, dtype=torch.double, device=DEVICE)
    U_warp_t = torch.tensor(U_warp, dtype=torch.double, device=DEVICE)

    q_true = ((1.0 - RHO_TRUE) + RHO_TRUE * lam_true) / TAU2_TRUE
    F_target = F_true.copy()
    q_target = q_true.copy()
    np.testing.assert_allclose(q_target, 1.0 / F_target, rtol=1e-12, atol=1e-12)
    q_target_t = torch.tensor(q_target, dtype=torch.double, device=DEVICE)

    X_t = torch.tensor(X, dtype=torch.double, device=DEVICE)
    y_all_t = torch.tensor(y, dtype=torch.double, device=DEVICE)
    is_holdout_t = torch.tensor(test_mask, dtype=torch.bool, device=DEVICE)
    y_test_t = y_all_t[is_holdout_t].clone()
    y_fit_t = y_all_t.clone()
    y_fit_t[is_holdout_t] = torch.nan
    eta_test_t = torch.tensor(
        eta_true[test_mask], dtype=torch.double, device=DEVICE
    )
    prior_V0 = BETA_PRIOR_VAR * torch.eye(
        X_t.shape[1], dtype=torch.double, device=DEVICE
    )

    # ------------------------------------------------------------------
    # Plot 1: eigenvalue warp + data
    # ------------------------------------------------------------------
    fig = plot_01_eigenvalue_warp(lam_true, lam_warp, gamma)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="eigenvalue_warp",
        number=1,
        dpi=dpi,
        manifest=manifest,
    )
    save_plot_data(
        pd.DataFrame({"lambda_true": lam_true, "mu_warped": lam_warp}),
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="eigenvalue_warp",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Plot 2: holdout region
    # ------------------------------------------------------------------
    fig = plot_02_holdout_region(test_mask)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="holdout_region",
        number=2,
        dpi=dpi,
        manifest=manifest,
    )
    save_plot_data(
        pd.DataFrame(
            {
                "node": np.arange(N),
                "row": np.arange(N) // N_COLS,
                "col": np.arange(N) % N_COLS,
                "is_holdout": test_mask,
            }
        ),
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="holdout_region",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Direct capacity test. It is seed-invariant by design; cache by gamma.
    # We still save a seed+gamma-labeled copy for artifact tracking.
    # ------------------------------------------------------------------
    cache_key = float(gamma)
    if cache_key not in direct_capacity_cache:
        direct_capacity_cache[cache_key] = direct_capacity_fit(
            lam_warp_t=lam_warp_t,
            q_target_t=q_target_t,
            q_target=q_target,
            positive_mode_mask=positive_mode_mask,
            seed=seed,
        )
    direct = direct_capacity_cache[cache_key]

    direct_table = direct["table"].copy()
    direct_table["seed_invariant_by_design"] = True
    save_table(
        direct_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="direct_capacity",
        manifest=manifest,
    )

    fig = plot_03_direct_capacity(
        lam_warp,
        q_target,
        direct["q_direct_leroux"],
        direct["q_direct_pspline"],
    )
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="direct_representational_capacity",
        number=3,
        dpi=dpi,
        manifest=manifest,
    )
    save_plot_data(
        pd.DataFrame(
            {
                "mu_warped": lam_warp,
                "q_target": q_target,
                "q_direct_leroux": direct["q_direct_leroux"],
                "q_direct_pspline": direct["q_direct_pspline"],
            }
        ),
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="direct_capacity_curves",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Likelihood-based VI fits -- exactly the notebook's settings.
    # ------------------------------------------------------------------
    print(f"    fitting Leroux CAR", flush=True)
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

    print(f"    fitting Adaptive P-spline SDM-CAR", flush=True)
    pspline_result = fit_spectral_vi(
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

    # ------------------------------------------------------------------
    # Plot 4: ELBO histories + CSV
    # ------------------------------------------------------------------
    fig = plot_04_elbo(leroux_result, pspline_result)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="vi_elbo_histories",
        number=4,
        dpi=dpi,
        manifest=manifest,
    )

    max_iter = max(len(leroux_result["history"]), len(pspline_result["history"]))
    elbo_df = pd.DataFrame({"iteration": np.arange(1, max_iter + 1)})
    elbo_df["leroux_elbo"] = pd.Series(leroux_result["history"])
    elbo_df["pspline_elbo"] = pd.Series(pspline_result["history"])
    save_plot_data(
        elbo_df,
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="vi_elbo_histories",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Posterior spectral summaries and displayed parameter tables.
    # ------------------------------------------------------------------
    leroux_post = posterior_spectrum_summary(
        leroux_result,
        draws=POSTERIOR_SPECTRUM_DRAWS,
        seed=seed + 20_001,
    )
    pspline_post = posterior_spectrum_summary(
        pspline_result,
        draws=POSTERIOR_SPECTRUM_DRAWS,
        seed=seed + 20_002,
    )
    pspline_components = posterior_pspline_component_summary(
        pspline_result,
        draws=POSTERIOR_SPECTRUM_DRAWS,
        seed=seed + 20_002,
    )

    leroux_parameters = posterior_leroux_parameter_summary(
        leroux_result,
        draws=POSTERIOR_PARAMETER_DRAWS,
        seed=seed + 20_101,
    )
    leroux_parameter_table = pd.DataFrame(
        [
            {
                "parameter": "tau2",
                "true_operator_value": TAU2_TRUE,
                **leroux_parameters["tau2"],
            },
            {
                "parameter": "rho",
                "true_operator_value": RHO_TRUE,
                **leroux_parameters["rho"],
            },
        ]
    )
    save_table(
        leroux_parameter_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="leroux_parameter_summary",
        manifest=manifest,
    )

    pspline_shrinkage_table = posterior_pspline_shrinkage_summary(
        pspline_result,
        draws=POSTERIOR_PARAMETER_DRAWS,
        seed=seed + 20_202,
    )
    save_table(
        pspline_shrinkage_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="pspline_shrinkage_curvature_summary",
        manifest=manifest,
    )

    spectral_curve_df = pd.DataFrame(
        {
            "lambda_true": lam_true,
            "mu_warped": lam_warp,
            "F_target": F_target,
            "q_target": q_target,
            "leroux_F_mean": leroux_post["F_mean"],
            "leroux_F_lower": leroux_post["F_lower"],
            "leroux_F_upper": leroux_post["F_upper"],
            "leroux_q_mean": leroux_post["q_mean"],
            "leroux_q_lower": leroux_post["q_lower"],
            "leroux_q_upper": leroux_post["q_upper"],
            "pspline_F_mean": pspline_post["F_mean"],
            "pspline_F_lower": pspline_post["F_lower"],
            "pspline_F_upper": pspline_post["F_upper"],
            "pspline_q_mean": pspline_post["q_mean"],
            "pspline_q_lower": pspline_post["q_lower"],
            "pspline_q_upper": pspline_post["q_upper"],
        }
    )
    save_plot_data(
        spectral_curve_df,
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="posterior_spectral_recovery",
        manifest=manifest,
    )

    # Plot 5
    fig = plot_05_posterior_spectral_recovery(
        lam_warp,
        F_target,
        q_target,
        leroux_post,
        pspline_post,
    )
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="posterior_spectral_recovery",
        number=5,
        dpi=dpi,
        manifest=manifest,
    )

    # Plot 6
    fig = plot_06_pspline_decomposition(lam_warp, q_target, pspline_components)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="pspline_affine_vs_nonlinear_correction",
        number=6,
        dpi=dpi,
        manifest=manifest,
    )
    save_plot_data(
        pd.DataFrame(
            {
                "mu_warped": lam_warp,
                "q_target": q_target,
                **{key: value for key, value in pspline_components.items()},
            }
        ),
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="pspline_affine_vs_nonlinear_correction",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Spectral recovery metrics table.
    # ------------------------------------------------------------------
    spectral_metrics = pd.DataFrame(
        [
            spectral_metric_row(
                "Leroux CAR",
                leroux_post["F_mean"],
                F_target,
                lam_warp,
                positive_mode_mask,
            ),
            spectral_metric_row(
                "Adaptive P-spline SDM-CAR",
                pspline_post["F_mean"],
                F_target,
                lam_warp,
                positive_mode_mask,
            ),
        ]
    )
    save_table(
        spectral_metrics,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="spectral_recovery_metrics",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Held-out posterior prediction and predictive table.
    # ------------------------------------------------------------------
    PRED_SEED = seed + 10_000

    torch.manual_seed(PRED_SEED)
    leroux_y_prediction = leroux_result["model"].predict_vi_mc(num_mc=PRED_MC)
    torch.manual_seed(PRED_SEED)
    pspline_y_prediction = pspline_result["model"].predict_vi_mc(num_mc=PRED_MC)

    leroux_eta_prediction = predict_eta_vi_mc(
        leroux_result["model"], num_mc=PRED_MC, seed=PRED_SEED
    )
    pspline_eta_prediction = predict_eta_vi_mc(
        pspline_result["model"], num_mc=PRED_MC, seed=PRED_SEED
    )

    leroux_y_metrics = predictive_metrics(y_test_t, leroux_y_prediction)
    pspline_y_metrics = predictive_metrics(y_test_t, pspline_y_prediction)
    leroux_eta_metrics = predictive_metrics(eta_test_t, leroux_eta_prediction)
    pspline_eta_metrics = predictive_metrics(eta_test_t, pspline_eta_prediction)

    predictive_table = pd.DataFrame(
        [
            {
                "model": "Leroux CAR",
                "latent_RMSE": leroux_eta_metrics["rmse"],
                "response_RMSE": leroux_y_metrics["rmse"],
                "latent_NLPD": leroux_eta_metrics["nlpd"],
                "response_NLPD": leroux_y_metrics["nlpd"],
                "latent_CRPS": leroux_eta_metrics["crps"],
                "response_CRPS": leroux_y_metrics["crps"],
                "latent_coverage95": leroux_eta_metrics["coverage_95"],
                "response_coverage95": leroux_y_metrics["coverage_95"],
                "latent_width95": leroux_eta_metrics["interval_width_95"],
                "response_width95": leroux_y_metrics["interval_width_95"],
            },
            {
                "model": "Adaptive P-spline SDM-CAR",
                "latent_RMSE": pspline_eta_metrics["rmse"],
                "response_RMSE": pspline_y_metrics["rmse"],
                "latent_NLPD": pspline_eta_metrics["nlpd"],
                "response_NLPD": pspline_y_metrics["nlpd"],
                "latent_CRPS": pspline_eta_metrics["crps"],
                "response_CRPS": pspline_y_metrics["crps"],
                "latent_coverage95": pspline_eta_metrics["coverage_95"],
                "response_coverage95": pspline_y_metrics["coverage_95"],
                "latent_width95": pspline_eta_metrics["interval_width_95"],
                "response_width95": pspline_y_metrics["interval_width_95"],
            },
        ]
    )
    save_table(
        predictive_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="heldout_predictive_metrics",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Variance calibration table.
    # ------------------------------------------------------------------
    leroux_calibration = latent_variance_calibration(
        eta_test_t, leroux_eta_prediction
    )
    pspline_calibration = latent_variance_calibration(
        eta_test_t, pspline_eta_prediction
    )
    calibration_table = pd.DataFrame(
        [
            {"model": "Leroux CAR", **leroux_calibration},
            {"model": "Adaptive P-spline SDM-CAR", **pspline_calibration},
        ]
    )
    save_table(
        calibration_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="latent_variance_calibration",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Final predictive comparison table.
    # ------------------------------------------------------------------
    spectral_lookup = spectral_metrics.set_index("model").to_dict(orient="index")
    final_comparison = pd.DataFrame(
        [
            {
                "model": "Leroux CAR",
                "latent_RMSE": leroux_eta_metrics["rmse"],
                "response_RMSE": leroux_y_metrics["rmse"],
                "latent_NLPD": leroux_eta_metrics["nlpd"],
                "response_NLPD": leroux_y_metrics["nlpd"],
                "latent_CRPS": leroux_eta_metrics["crps"],
                "response_CRPS": leroux_y_metrics["crps"],
                "latent_coverage95": leroux_eta_metrics["coverage_95"],
                "response_coverage95": leroux_y_metrics["coverage_95"],
                "latent_width95": leroux_eta_metrics["interval_width_95"],
                "response_width95": leroux_y_metrics["interval_width_95"],
                "latent_MSSE": leroux_calibration["MSSE"],
                "latent_Mahalanobis_per_site": leroux_calibration[
                    "normalized_Mahalanobis"
                ],
                "spectral_log_RMSE": spectral_lookup["Leroux CAR"]["log_RMSE"],
                "variance_weighted_spectral_log_RMSE": spectral_lookup[
                    "Leroux CAR"
                ]["variance_weighted_log_RMSE"],
            },
            {
                "model": "Adaptive P-spline SDM-CAR",
                "latent_RMSE": pspline_eta_metrics["rmse"],
                "response_RMSE": pspline_y_metrics["rmse"],
                "latent_NLPD": pspline_eta_metrics["nlpd"],
                "response_NLPD": pspline_y_metrics["nlpd"],
                "latent_CRPS": pspline_eta_metrics["crps"],
                "response_CRPS": pspline_y_metrics["crps"],
                "latent_coverage95": pspline_eta_metrics["coverage_95"],
                "response_coverage95": pspline_y_metrics["coverage_95"],
                "latent_width95": pspline_eta_metrics["interval_width_95"],
                "response_width95": pspline_y_metrics["interval_width_95"],
                "latent_MSSE": pspline_calibration["MSSE"],
                "latent_Mahalanobis_per_site": pspline_calibration[
                    "normalized_Mahalanobis"
                ],
                "spectral_log_RMSE": spectral_lookup[
                    "Adaptive P-spline SDM-CAR"
                ]["log_RMSE"],
                "variance_weighted_spectral_log_RMSE": spectral_lookup[
                    "Adaptive P-spline SDM-CAR"
                ]["variance_weighted_log_RMSE"],
            },
        ]
    )
    save_table(
        final_comparison,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="final_predictive_comparison",
        manifest=manifest,
    )

    # Plot 7 + prediction data
    fig = plot_07_latent_prediction(
        eta_true[test_mask], leroux_eta_prediction, pspline_eta_prediction
    )
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="heldout_latent_prediction",
        number=7,
        dpi=dpi,
        manifest=manifest,
    )

    heldout_nodes = np.flatnonzero(test_mask)
    prediction_df = pd.DataFrame(
        {
            "node": heldout_nodes,
            "eta_true": eta_true[test_mask],
            "y_observed": y[test_mask],
            "leroux_eta_mean": leroux_eta_prediction["mean"].cpu().numpy(),
            "leroux_eta_q025": leroux_eta_prediction["q025"].cpu().numpy(),
            "leroux_eta_q975": leroux_eta_prediction["q975"].cpu().numpy(),
            "pspline_eta_mean": pspline_eta_prediction["mean"].cpu().numpy(),
            "pspline_eta_q025": pspline_eta_prediction["q025"].cpu().numpy(),
            "pspline_eta_q975": pspline_eta_prediction["q975"].cpu().numpy(),
            "leroux_y_mean": leroux_y_prediction["mean"].cpu().numpy(),
            "leroux_y_q025": leroux_y_prediction["q025"].cpu().numpy(),
            "leroux_y_q975": leroux_y_prediction["q975"].cpu().numpy(),
            "pspline_y_mean": pspline_y_prediction["mean"].cpu().numpy(),
            "pspline_y_q025": pspline_y_prediction["q025"].cpu().numpy(),
            "pspline_y_q975": pspline_y_prediction["q975"].cpu().numpy(),
        }
    )
    save_plot_data(
        prediction_df,
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="heldout_predictions",
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Full precision-operator recovery.
    # ------------------------------------------------------------------
    np.testing.assert_allclose(U_warp, U_true, rtol=1e-10, atol=1e-10)
    Q_target = precision_matrix_from_modes(U_true, q_target)
    Q_target_closed_form = (
        ((1.0 - RHO_TRUE) / TAU2_TRUE) * np.eye(N)
        + (RHO_TRUE / TAU2_TRUE) * L_true
    )
    np.testing.assert_allclose(
        Q_target, Q_target_closed_form, rtol=1e-9, atol=1e-9
    )

    q_leroux_mean = np.asarray(leroux_post["q_mean"], dtype=float)
    q_pspline_mean = np.asarray(pspline_post["q_mean"], dtype=float)
    Q_leroux_mean = precision_matrix_from_modes(U_warp, q_leroux_mean)
    Q_pspline_mean = precision_matrix_from_modes(U_warp, q_pspline_mean)

    leroux_q_draws = posterior_modal_precision_draws(
        leroux_result,
        draws=POSTERIOR_SPECTRUM_DRAWS,
        seed=seed + 20_001,
    )
    pspline_q_draws = posterior_modal_precision_draws(
        pspline_result,
        draws=POSTERIOR_SPECTRUM_DRAWS,
        seed=seed + 20_002,
    )

    leroux_mean_check = float(
        np.max(np.abs(leroux_q_draws.mean(axis=0) - q_leroux_mean))
    )
    pspline_mean_check = float(
        np.max(np.abs(pspline_q_draws.mean(axis=0) - q_pspline_mean))
    )
    reproduction_check = pd.DataFrame(
        [
            {
                "model": "Leroux CAR",
                "max_abs_draw_mean_minus_existing_q_mean": leroux_mean_check,
            },
            {
                "model": "Adaptive P-spline SDM-CAR",
                "max_abs_draw_mean_minus_existing_q_mean": pspline_mean_check,
            },
        ]
    )
    save_plot_data(
        reproduction_check,
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="posterior_modal_precision_reproduction_check",
        manifest=manifest,
    )

    leroux_matrix_metrics_all = precision_operator_metric_draws(
        leroux_q_draws, q_target, U_warp
    )
    pspline_matrix_metrics_all = precision_operator_metric_draws(
        pspline_q_draws, q_target, U_warp
    )
    leroux_matrix_metrics_positive = precision_operator_metric_draws(
        leroux_q_draws[:, positive_mode_mask],
        q_target[positive_mode_mask],
        U_warp[:, positive_mode_mask],
    )
    pspline_matrix_metrics_positive = precision_operator_metric_draws(
        pspline_q_draws[:, positive_mode_mask],
        q_target[positive_mode_mask],
        U_warp[:, positive_mode_mask],
    )

    matrix_metrics_all_table = posterior_metric_summary_table(
        {
            "Leroux CAR": leroux_matrix_metrics_all,
            "Adaptive P-spline SDM-CAR": pspline_matrix_metrics_all,
        }
    )
    matrix_metrics_positive_table = posterior_metric_summary_table(
        {
            "Leroux CAR": leroux_matrix_metrics_positive,
            "Adaptive P-spline SDM-CAR": pspline_matrix_metrics_positive,
        }
    )
    save_table(
        matrix_metrics_all_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="operator_posterior_metrics_all_modes",
        manifest=manifest,
    )
    save_table(
        matrix_metrics_positive_table,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="operator_posterior_metrics_positive_modes",
        manifest=manifest,
    )

    posterior_mean_operator_metrics = pd.DataFrame(
        [
            posterior_mean_operator_metric_row(
                "Leroux CAR", "all modes", q_leroux_mean, q_target, U_warp
            ),
            posterior_mean_operator_metric_row(
                "Adaptive P-spline SDM-CAR",
                "all modes",
                q_pspline_mean,
                q_target,
                U_warp,
            ),
            posterior_mean_operator_metric_row(
                "Leroux CAR",
                "positive modes",
                q_leroux_mean[positive_mode_mask],
                q_target[positive_mode_mask],
                U_warp[:, positive_mode_mask],
            ),
            posterior_mean_operator_metric_row(
                "Adaptive P-spline SDM-CAR",
                "positive modes",
                q_pspline_mean[positive_mode_mask],
                q_target[positive_mode_mask],
                U_warp[:, positive_mode_mask],
            ),
        ]
    )
    save_table(
        posterior_mean_operator_metrics,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="posterior_mean_operator_metrics",
        manifest=manifest,
    )

    matrix_metric_guide = matrix_metric_guide_table()
    save_table(
        matrix_metric_guide,
        seed=seed,
        gamma=gamma,
        tables_dir=tables_dir,
        stem="matrix_metric_guide",
        manifest=manifest,
    )

    # Operator matrices underlying plots.
    for stem, matrix in [
        ("Q_target", Q_target),
        ("Q_leroux_posterior_mean", Q_leroux_mean),
        ("Q_pspline_posterior_mean", Q_pspline_mean),
        ("Q_leroux_minus_target", Q_leroux_mean - Q_target),
        ("Q_pspline_minus_target", Q_pspline_mean - Q_target),
    ]:
        matrix_df = pd.DataFrame(matrix)
        save_plot_data(
            matrix_df,
            seed=seed,
            gamma=gamma,
            data_dir=data_dir,
            stem=stem,
            manifest=manifest,
        )

    # Plot 8
    fig = plot_08_operator_heatmaps(Q_target, Q_leroux_mean, Q_pspline_mean)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="precision_operator_heatmaps",
        number=8,
        dpi=dpi,
        manifest=manifest,
    )

    # Plot 9
    fig = plot_09_operator_differences(Q_target, Q_leroux_mean, Q_pspline_mean)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="precision_operator_difference_heatmaps",
        number=9,
        dpi=dpi,
        manifest=manifest,
    )

    # Plot 10
    fig = plot_10_operator_scatter(Q_target, Q_leroux_mean, Q_pspline_mean)
    save_figure(
        fig,
        seed=seed,
        gamma=gamma,
        plots_dir=plots_dir,
        stem="diagonal_offdiagonal_precision_scatter",
        number=10,
        dpi=dpi,
        manifest=manifest,
    )

    offdiag_mask = ~np.eye(N, dtype=bool)
    operator_scatter_df = pd.DataFrame(
        {
            "entry_type": np.concatenate(
                [
                    np.repeat("diagonal", N),
                    np.repeat("offdiagonal", N * (N - 1)),
                ]
            ),
            "target": np.concatenate(
                [np.diag(Q_target), Q_target[offdiag_mask]]
            ),
            "leroux": np.concatenate(
                [np.diag(Q_leroux_mean), Q_leroux_mean[offdiag_mask]]
            ),
            "pspline": np.concatenate(
                [np.diag(Q_pspline_mean), Q_pspline_mean[offdiag_mask]]
            ),
        }
    )
    save_plot_data(
        operator_scatter_df,
        seed=seed,
        gamma=gamma,
        data_dir=data_dir,
        stem="precision_operator_scatter",
        manifest=manifest,
    )

    # Experiment metadata/sanity checks.
    metadata = {
        "seed": int(seed),
        "power_gamma": float(gamma),
        "N_ROWS": N_ROWS,
        "N_COLS": N_COLS,
        "N": N,
        "TAU2_TRUE": TAU2_TRUE,
        "RHO_TRUE": RHO_TRUE,
        "BETA_TRUE": BETA_TRUE,
        "SIGMA2_TRUE": SIGMA2_TRUE,
        "HOLDOUT_SIDE": HOLDOUT_SIDE,
        "HOLDOUT_PATTERN": HOLDOUT_PATTERN,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "LEROUX_ITERS": LEROUX_ITERS,
        "LEROUX_LR": LEROUX_LR,
        "PSPLINE_ITERS": PSPLINE_ITERS,
        "PSPLINE_LR": PSPLINE_LR,
        "DIRECT_LEROUX_ITERS": DIRECT_LEROUX_ITERS,
        "DIRECT_PSPLINE_ITERS": DIRECT_PSPLINE_ITERS,
        "NUM_MC": NUM_MC,
        "PRED_MC": PRED_MC,
        "POSTERIOR_SPECTRUM_DRAWS": POSTERIOR_SPECTRUM_DRAWS,
        "POSTERIOR_PARAMETER_DRAWS": POSTERIOR_PARAMETER_DRAWS,
        "GRAD_CLIP": GRAD_CLIP,
        "leroux_modal_mean_reproduction_max_abs_error": leroux_mean_check,
        "pspline_modal_mean_reproduction_max_abs_error": pspline_mean_check,
    }
    metadata_path = run_dir / f"{prefix}__run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest.append(
        {
            "seed": seed,
            "power_gamma": gamma,
            "artifact_type": "metadata_json",
            "artifact_name": "run_metadata",
            "path": str(metadata_path),
        }
    )

    manifest_path = write_manifest(
        manifest, seed=seed, gamma=gamma, run_dir=run_dir
    )

    complete_path = run_dir / f"{prefix}__COMPLETE.txt"
    complete_path.write_text(
        "Complete: all notebook tables and all 10 notebook plots were saved.\n",
        encoding="utf-8",
    )

    print(
        f"    saved {len(manifest)} artifacts | manifest={manifest_path.name}",
        flush=True,
    )

    # Release the fitted models before the next run.
    del leroux_result, pspline_result


# =============================================================================
# CROSS-SEED / CROSS-GAMMA MASTER TABLES AND TREND PLOTS
# =============================================================================

def collect_named_run_tables(output_dir: Path, table_stem: str) -> pd.DataFrame:
    # Recursive search lets completed runs be combined regardless of ordering.
    paths = sorted(
        (output_dir / "per_run").rglob(f"*__table__{table_stem}.csv")
    )
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_numeric_by_gamma(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    id_cols = set(group_cols)
    numeric_cols = [
        c
        for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Do not summarize seed as an outcome.
    numeric_cols = [c for c in numeric_cols if c != "seed"]
    if not numeric_cols:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, sort=True)[numeric_cols]
    mean = grouped.mean().add_suffix("__mean")
    sd = grouped.std(ddof=1).add_suffix("__sd")
    median = grouped.median().add_suffix("__median")
    q025 = grouped.quantile(0.025).add_suffix("__q025")
    q975 = grouped.quantile(0.975).add_suffix("__q975")
    return pd.concat([mean, sd, median, q025, q975], axis=1).reset_index()


def paired_model_differences(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    leroux_name: str = "Leroux CAR",
    pspline_name: str = "Adaptive P-spline SDM-CAR",
    extra_id_cols: Iterable[str] = (),
) -> pd.DataFrame:
    if df.empty or model_col not in df.columns:
        return pd.DataFrame()

    id_cols = ["seed", "power_gamma", *extra_id_cols]
    numeric_cols = [
        c
        for c in df.columns
        if c not in set(id_cols + [model_col])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    leroux = df[df[model_col] == leroux_name].set_index(id_cols)[numeric_cols]
    pspline = df[df[model_col] == pspline_name].set_index(id_cols)[numeric_cols]
    common = leroux.index.intersection(pspline.index)
    if common.empty:
        return pd.DataFrame()

    diff = pspline.loc[common] - leroux.loc[common]
    diff.columns = [f"pspline_minus_leroux__{c}" for c in diff.columns]
    return diff.reset_index()


def save_metric_trend_plots(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    source_name: str,
    model_col: str = "model",
    extra_filter: Mapping[str, str] | None = None,
    dpi: int = 180,
):
    if df.empty or model_col not in df.columns:
        return

    work = df.copy()
    suffix_bits = []
    if extra_filter:
        for col, value in extra_filter.items():
            if col not in work.columns:
                return
            work = work[work[col] == value]
            suffix_bits.append(f"{col}_{str(value).replace(' ', '_')}")

    if work.empty:
        return

    trend_dir = output_dir / "summary_plots"
    trend_dir.mkdir(parents=True, exist_ok=True)

    id_cols = {"seed", "power_gamma", model_col}
    if extra_filter:
        id_cols.update(extra_filter.keys())
    metric_cols = [
        c
        for c in work.columns
        if c not in id_cols
        and pd.api.types.is_numeric_dtype(work[c])
        and not pd.api.types.is_bool_dtype(work[c])
    ]

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for model_name, model_df in work.groupby(model_col, sort=False):
            summary = (
                model_df.groupby("power_gamma", sort=True)[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("power_gamma")
            )
            ax.plot(
                summary["power_gamma"],
                summary["mean"],
                marker="o",
                linewidth=1.8,
                label=str(model_name),
            )
            if summary["std"].notna().any():
                lower = summary["mean"] - summary["std"].fillna(0.0)
                upper = summary["mean"] + summary["std"].fillna(0.0)
                ax.fill_between(
                    summary["power_gamma"], lower, upper, alpha=0.12
                )
        ax.set_xlabel("POWER_GAMMA")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs POWER_GAMMA across seeds")
        ax.legend()
        ax.grid(alpha=0.2)
        safe_metric = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in metric)
        extra = "__" + "__".join(suffix_bits) if suffix_bits else ""
        path = trend_dir / (
            f"all_seeds__{source_name}{extra}__{safe_metric}__vs_power_gamma.png"
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def build_cross_run_outputs(output_dir: Path, dpi: int):
    master_dir = output_dir / "master_tables"
    master_dir.mkdir(parents=True, exist_ok=True)

    table_stems = [
        "direct_capacity",
        "leroux_parameter_summary",
        "pspline_shrinkage_curvature_summary",
        "spectral_recovery_metrics",
        "heldout_predictive_metrics",
        "latent_variance_calibration",
        "final_predictive_comparison",
        "operator_posterior_metrics_all_modes",
        "operator_posterior_metrics_positive_modes",
        "posterior_mean_operator_metrics",
        "matrix_metric_guide",
    ]

    collected = {}
    for stem in table_stems:
        df = collect_named_run_tables(output_dir, stem)
        collected[stem] = df
        if not df.empty:
            df = df.sort_values(
                [c for c in ["power_gamma", "seed", "model"] if c in df.columns]
            ).reset_index(drop=True)
            df.to_csv(master_dir / f"all_runs__{stem}.csv", index=False)

    # Across-seed summaries for the model-comparison tables.
    summary_specs = {
        "spectral_recovery_metrics": ["power_gamma", "model"],
        "heldout_predictive_metrics": ["power_gamma", "model"],
        "latent_variance_calibration": ["power_gamma", "model"],
        "final_predictive_comparison": ["power_gamma", "model"],
        "posterior_mean_operator_metrics": ["power_gamma", "model", "operator"],
    }

    for stem, groups in summary_specs.items():
        df = collected.get(stem, pd.DataFrame())
        if df.empty:
            continue
        summary = summarize_numeric_by_gamma(df, groups)
        summary.to_csv(
            master_dir / f"summary_across_seeds__{stem}.csv", index=False
        )

    # Paired P-spline minus Leroux differences.
    final_df = collected.get("final_predictive_comparison", pd.DataFrame())
    if not final_df.empty:
        diff = paired_model_differences(final_df)
        if not diff.empty:
            diff.to_csv(
                master_dir / "paired_pspline_minus_leroux__final_predictive_comparison.csv",
                index=False,
            )
            diff_summary = summarize_numeric_by_gamma(diff, ["power_gamma"])
            diff_summary.to_csv(
                master_dir / "paired_summary_across_seeds__final_predictive_comparison.csv",
                index=False,
            )

    operator_df = collected.get("posterior_mean_operator_metrics", pd.DataFrame())
    if not operator_df.empty:
        all_mode_df = operator_df[operator_df["operator"] == "all modes"].copy()
        diff = paired_model_differences(all_mode_df, extra_id_cols=["operator"])
        if not diff.empty:
            diff.to_csv(
                master_dir / "paired_pspline_minus_leroux__operator_all_modes.csv",
                index=False,
            )
            diff_summary = summarize_numeric_by_gamma(
                diff, ["power_gamma", "operator"]
            )
            diff_summary.to_csv(
                master_dir / "paired_summary_across_seeds__operator_all_modes.csv",
                index=False,
            )

    # Trend plots for every numeric metric in the main tables.
    save_metric_trend_plots(
        collected.get("spectral_recovery_metrics", pd.DataFrame()),
        output_dir=output_dir,
        source_name="spectral_metrics",
        dpi=dpi,
    )
    save_metric_trend_plots(
        collected.get("heldout_predictive_metrics", pd.DataFrame()),
        output_dir=output_dir,
        source_name="predictive_metrics",
        dpi=dpi,
    )
    save_metric_trend_plots(
        collected.get("latent_variance_calibration", pd.DataFrame()),
        output_dir=output_dir,
        source_name="calibration_metrics",
        dpi=dpi,
    )
    save_metric_trend_plots(
        collected.get("final_predictive_comparison", pd.DataFrame()),
        output_dir=output_dir,
        source_name="final_comparison",
        dpi=dpi,
    )
    save_metric_trend_plots(
        collected.get("posterior_mean_operator_metrics", pd.DataFrame()),
        output_dir=output_dir,
        source_name="operator_metrics",
        extra_filter={"operator": "all modes"},
        dpi=dpi,
    )

    # Direct capacity is seed-invariant; deduplicate seed copies before plotting.
    direct_df = collected.get("direct_capacity", pd.DataFrame())
    if not direct_df.empty:
        direct_unique = direct_df.drop_duplicates(
            subset=["power_gamma", "model"], keep="first"
        ).copy()
        direct_unique.to_csv(
            master_dir / "direct_capacity_unique_by_gamma.csv", index=False
        )
        save_metric_trend_plots(
            direct_unique,
            output_dir=output_dir,
            source_name="direct_capacity",
            dpi=dpi,
        )

    # Global artifact manifest across completed runs.
    manifest_paths = sorted(
        (output_dir / "per_run").rglob("*__artifact_manifest.csv")
    )
    manifests = [pd.read_csv(p) for p in manifest_paths if p.exists()]
    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(
            output_dir / "all_runs__artifact_manifest.csv", index=False
        )


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the full reordered notebook experiment over SEED and "
            "POWER_GAMMA, saving every table and all plots for each run."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path containing the sdmcar package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("power_warp_full_artifact_sweep_results"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution for all saved plots.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seed/gamma runs with a COMPLETE marker.",
    )
    return parser.parse_args()


def main():
    global LerouxCARFilterFullVI
    global AdaptivePrecisionPSplineFullVI
    global SpectralCAR_HoldoutVI

    args = parse_args()

    if not FIX_SIGMA2:
        raise ValueError("This controlled experiment keeps sigma^2 fixed.")
    if HOLDOUT_PATTERN not in HOLDOUT_OPTIONS:
        raise ValueError(f"HOLDOUT_PATTERN must be one of {HOLDOUT_OPTIONS}.")
    for gamma in args.gammas:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"Each POWER_GAMMA must lie in (0, 1]; got {gamma}.")

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sdmcar.filters import (
        LerouxCARFilterFullVI as _LerouxCARFilterFullVI,
        AdaptivePrecisionPSplineFullVI as _AdaptivePrecisionPSplineFullVI,
    )
    from sdmcar.models_holdout import SpectralCAR_HoldoutVI as _SpectralCAR_HoldoutVI

    LerouxCARFilterFullVI = _LerouxCARFilterFullVI
    AdaptivePrecisionPSplineFullVI = _AdaptivePrecisionPSplineFullVI
    SpectralCAR_HoldoutVI = _SpectralCAR_HoldoutVI

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common graph/eigendecomposition and holdout.
    W_true = build_rook_adjacency(N_ROWS, N_COLS)
    L_true = graph_laplacian(W_true)
    lam_true, U_true = np.linalg.eigh(L_true)
    lam_true[np.abs(lam_true) < ZERO_TOL] = 0.0
    lam_true = np.clip(lam_true, 0.0, None)
    positive_mode_mask = lam_true > ZERO_TOL

    test_mask = holdout_block_mask(
        N_ROWS, N_COLS, HOLDOUT_SIDE, HOLDOUT_PATTERN
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
    print("Default dtype:", torch.get_default_dtype())
    print("Seeds:", args.seeds)
    print("POWER_GAMMA values:", args.gammas)
    print("Per run: 11 notebook tables + 10 notebook figures")
    print("Total likelihood fits:", len(args.seeds) * len(args.gammas) * 2)
    print(
        "Direct capacity fits are cached by gamma because capacity is "
        "seed-invariant by design."
    )

    direct_capacity_cache = {}

    for seed_idx, seed in enumerate(args.seeds, start=1):
        for gamma_idx, gamma in enumerate(args.gammas, start=1):
            gamma = float(gamma)
            prefix = run_prefix(seed, gamma)
            run_dir = output_dir / "per_run" / prefix
            complete_path = run_dir / f"{prefix}__COMPLETE.txt"

            print(
                f"\n=== seed={seed} ({seed_idx}/{len(args.seeds)}), "
                f"POWER_GAMMA={gamma:g} ({gamma_idx}/{len(args.gammas)}) ===",
                flush=True,
            )

            if args.resume and complete_path.exists():
                print("    SKIP: complete artifact set already exists.", flush=True)
                continue

            # Remove a stale/partial run directory so files cannot be mixed.
            if run_dir.exists():
                shutil.rmtree(run_dir)

            run_one(
                seed=int(seed),
                gamma=gamma,
                common=common,
                output_dir=output_dir,
                dpi=args.dpi,
                direct_capacity_cache=direct_capacity_cache,
            )

    print("\nBuilding cross-seed/cross-gamma master tables and trend plots...", flush=True)
    build_cross_run_outputs(output_dir, dpi=args.dpi)

    print("\n=== SWEEP COMPLETE ===")
    print("Per-run artifacts:", output_dir / "per_run")
    print("Master CSVs:", output_dir / "master_tables")
    print("Across-seed trend plots:", output_dir / "summary_plots")
    print("Global manifest:", output_dir / "all_runs__artifact_manifest.csv")


if __name__ == "__main__":
    main()
