"""Experiment 1 — Leroux-truth benchmark (12x12 rook, 4x4 block holdout).

One implementation for a five-replication N=144 validation pilot and a
50-replication final experiment. The fitted regression intercept is fixed
at the generating value beta=0, sigma2=0.15 is known and fixed, and the
same unwarped graph eigenvectors/eigenvalues are used for truth and fit.

Both fitted models use a two-stage Adam schedule (first-stage Adam state
is retained when reducing the learning rate). We record stochastic
convergence diagnostics without allowing diagnostic draws to advance the
optimizer's RNG stream. Incomplete runs can resume from optimizer-state
snapshots; completed runs are verified before being skipped.

Candidate schedule — validate at N=144 before treating the 50-seed run as
scientific inference. See --help for all CLI options.
"""

from __future__ import annotations
import argparse
import json
import hashlib
import copy
import math
import os
import shutil
import sys
import time
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
# One Leroux truth
TAU2_TRUE = 0.70
RHO_TRUE = 0.92
BETA_TRUE = 0.0
SIGMA2_TRUE = 0.15
ZERO_TOL = 1e-10
# Spatial block holdout
HOLDOUT_SIDE = 4
HOLDOUT_PATTERN = "middle"
# Inference
FIX_SIGMA2 = True
NUM_MC = 4
GRAD_CLIP = 10.0
JITTER = 1e-8
# The regression coefficient beta is known (zero): no beta prior.
LEROUX_ITERS = 4000       # Stage 1
LEROUX_LR = 1e-3
LEROUX_REFINE_ITERS = 2000  # Stage 2: resume same Adam state, reduce LR
LEROUX_REFINE_LR = 2e-4
PSPLINE_ITERS = 12000     # Stage 1
PSPLINE_LR = 3e-4
PSPLINE_REFINE_ITERS = 2000  # Stage 2
PSPLINE_REFINE_LR = 6e-5
KEEP_ALL_CHECKPOINTS = False
# Posterior / predictive Monte Carlo
SPECTRUM_DRAWS = 512
PREDICTIVE_DRAWS = 512
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
    "global_scale_d2": 2.0, #0.50,
    "prior_std_log_lambda": 2.0,
    "mu_log_lambda_init": 0.0,
    "log_std_log_lambda": -3.0,
    "log_std_log_q": -3.0,
    "log_std_d2": -3.0,
    "init_d2": 0.0,
    "log_q_min": -20.0,
    "log_q_max": 20.0,
}
EXPERIMENT_VERSION = "leroux_truth_experiment1_N144_fixed_beta_two_stage_v1"
# Only two replications for development, not scientific summaries.
PILOT_SEEDS = [111, 222, 333, 444, 555]
FINAL_SEEDS = list(range(1001, 1051))
DEFAULT_SEEDS = PILOT_SEEDS
DEFAULT_PROJECT_ROOT = Path(
    r"C:/Users/pd006/Desktop/internship_search/sdm-car"
)
DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.double)
LerouxCARFilterFullVI = None
AdaptivePrecisionPSplineFullVI = None
SpectralCAR_HoldoutVI = None

# Pilot convergence diagnostics (not validated dissertation stopping thresholds).
# Each model.elbo() evaluation uses the model's existing NUM_MC draws.
DIAG_EVERY = 250
DIAG_ELBO_REPEATS = 32
DIAG_SPECTRUM_DRAWS = 256
DIAG_BOOTSTRAP_REPS = 250
SAVE_CHECKPOINT_EVERY = 1000
LONG_WINDOW_STEPS = (1000, 2000)
# Retain 2*max(long windows) for comparing consecutive displacement directions.
DIAG_CACHE_HORIZON = 2 * max(LONG_WINDOW_STEPS)
DIAG_ELBO_TOL_PER_SITE = 0.01         # absolute change + 2 MCSEs, per train site
DIAG_PRECISION_TOL = 0.005            # global relative L2 change of mean precision
# Zero-mode safeguards: the full-spectrum L2 norm can mask substantial
# movement at lambda=0 because the true precision there is very small.
# Pilot tolerances only; calibrate before final N=144 experiment.
DIAG_ZERO_PRECISION_TOL = 0.02       # relative change in E_q[1/F(0)]
DIAG_ZERO_VARIANCE_TOL = 0.02        # relative change in E_q[F(0)]
DIAG_CONSECUTIVE_CHECKPOINTS = 3
DIAG_MCSE_MULTIPLIER = 2.0          # conservative approximate uncertainty allowance

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

def generate_data(
    seed: int,
    lam_true: np.ndarray,
    U_true: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Generate exactly from the Leroux spectral covariance
        F_true(lambda_k)
            = tau2_true / [(1-rho_true) + rho_true * lambda_k].
    """
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

@torch.no_grad()
def exact_pspline_zero_mode(filter_module):
    """Analytic untruncated endpoint moments; valid if log-q clamps are inactive.

    SDM filter uses q(0)=exp(Z_L), F(0)=exp(-Z_L),
    Z_L ~ Normal(mu, s**2). The filter clamps log q to [min,max].
    For diagnostics, report whether a conservative mu +/- 8*s range
    lies inside that clamp interval. Exact values should not be used if
    the clamp is materially active.
    """
    if not isinstance(filter_module, AdaptivePrecisionPSplineFullVI):
        return {}
    mu = float(filter_module.mu_log_q_endpoints[0].item())
    log_std = float(filter_module.log_std_log_q_endpoints[0].item())
    std = math.exp(log_std)
    var = std * std
    safe = bool(mu - 8 * std > filter_module.log_q_min and
                mu + 8 * std < filter_module.log_q_max)
    return {
        'zero_log_precision_mu': mu,
        'zero_log_precision_std': std,
        'zero_log_precision_log_std': log_std,
        'zero_precision_mean_exact_unclipped': math.exp(mu + 0.5 * var),
        'zero_variance_mean_exact_unclipped': math.exp(-mu + 0.5 * var),
        'zero_precision_sd_exact_unclipped': math.sqrt(math.expm1(var) * math.exp(2 * mu + var)),
        'zero_variance_sd_exact_unclipped': math.sqrt(math.expm1(var) * math.exp(-2 * mu + var)),
        'zero_exact_clip_safe_8sd': safe,
    }


@torch.no_grad()
def evaluate_convergence(model, seed):
    """Independent checkpoint evaluation with common Torch base random draws.

    fork_rng ensures evaluation has no impact on the optimizer draw stream.
    Keeping full draw arrays allows *paired* uncertainty diagnostics.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 100_000)
        elbo_repeats = np.empty(DIAG_ELBO_REPEATS, dtype=float)
        for r in range(DIAG_ELBO_REPEATS):
            elbo, _ = model.elbo()
            elbo_repeats[r] = float(elbo.detach().item())

        torch.manual_seed(int(seed) + 200_000)
        q_draws = np.empty((DIAG_SPECTRUM_DRAWS, model.lam.numel()), dtype=float)
        zero_variance_draws = np.empty(DIAG_SPECTRUM_DRAWS, dtype=float)
        for j in range(DIAG_SPECTRUM_DRAWS):
            theta = model.filter.sample_unconstrained()
            F = model.filter.spectrum(model.lam, theta).clamp_min(1e-12)
            q_draws[j] = (1.0 / F).detach().cpu().numpy()
            zero_variance_draws[j] = float(F[0].item())

    if (not np.isfinite(elbo_repeats).all() or
            not np.isfinite(q_draws).all() or
            not np.isfinite(zero_variance_draws).all()):
        raise RuntimeError('Nonfinite checkpoint ELBO or spectral draw.')
    return {
        'elbo_repeats': elbo_repeats,
        'precision_draws': q_draws,
        'zero_variance_draws': zero_variance_draws,
        'precision_mean': q_draws.mean(axis=0),
        'precision_sd': q_draws.std(axis=0, ddof=1),
        'zero_variance_mean': float(zero_variance_draws.mean()),
        'zero_variance_sd': float(zero_variance_draws.std(ddof=1)),
        'zero_exact': exact_pspline_zero_mode(model.filter),
    }


def paired_change_diagnostics(new, old, seed, iteration, window):
    """Matched-index bootstrap of CRN relative changes (MC uncertainty only).

    The paired bootstrap targets the Monte Carlo evaluation uncertainty
    *conditional on the two fitted variational distributions*, NOT
    optimization randomness, posterior uncertainty, or sampling across data.
    """
    qnew, qold = new['precision_draws'], old['precision_draws']
    fnew, fold = new['zero_variance_draws'], old['zero_variance_draws']
    if qnew.shape != qold.shape or fnew.shape != fold.shape:
        raise ValueError('Paired checkpoints must use the same draw count.')
    m = qnew.shape[0]
    rng = np.random.default_rng(int(seed) + 300_000 + int(iteration) * 19 + int(window))
    boot_q = np.empty(DIAG_BOOTSTRAP_REPS)
    boot_q0 = np.empty(DIAG_BOOTSTRAP_REPS)
    boot_f0 = np.empty(DIAG_BOOTSTRAP_REPS)
    for k in range(DIAG_BOOTSTRAP_REPS):
        idx = rng.integers(0, m, size=m)
        a = qnew[idx].mean(axis=0)
        b = qold[idx].mean(axis=0)
        boot_q[k] = _relative_change(a, b)
        boot_q0[k] = abs(a[0] - b[0]) / max(abs(b[0]), 1e-12)
        boot_f0[k] = abs(fnew[idx].mean() - fold[idx].mean()) / max(abs(fold[idx].mean()), 1e-12)
    return {
        'precision_change_boot_q025': float(np.quantile(boot_q, 0.025)),
        'precision_change_boot_q975': float(np.quantile(boot_q, 0.975)),
        'zero_precision_change_boot_q025': float(np.quantile(boot_q0, 0.025)),
        'zero_precision_change_boot_q975': float(np.quantile(boot_q0, 0.975)),
        'zero_variance_change_boot_q025': float(np.quantile(boot_f0, 0.025)),
        'zero_variance_change_boot_q975': float(np.quantile(boot_f0, 0.975)),
        'zero_variance_paired_delta_mcse': float((fnew - fold).std(ddof=1) / math.sqrt(m)),
    }


def training_data_digest(y_fit_t, lam_t):
    """Hash observed training data and eigenvalues for checkpoint validation."""
    y = y_fit_t.detach().cpu().numpy()
    lam = lam_t.detach().cpu().numpy()
    h = hashlib.sha256()
    h.update(np.asarray(y, dtype='<f8').tobytes())
    h.update(np.asarray(lam, dtype='<f8').tobytes())
    return h.hexdigest()


def atomic_save_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_trusted_checkpoint(path):
    # Use only local checkpoint files produced by this script; torch.load
    # unpickles the saved NumPy and Python RNG/history objects.
    return torch.load(Path(path), map_location='cpu', weights_only=False)


def _relative_change(new, old):
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))


def convergence_summary_row(label: str, history: pd.DataFrame) -> dict:
    """Pilot diagnostic classification; does not establish global convergence."""
    passed = history.loc[history['pilot_persistent_pass'], 'iteration']
    final = history.iloc[-1]
    return {
        'model': label,
        'first_pilot_pass_iteration': int(passed.iloc[0]) if len(passed) else np.nan,
        'pilot_pass_at_budget': bool(final['pilot_persistent_pass']),
        'final_iteration': int(final['iteration']),
        'final_elbo_change_per_site': float(final['ELBO_improvement_per_site']),
        'final_elbo_change_MCSE_per_site': float(final['ELBO_improvement_MCSE_per_site']),
        'final_precision_relative_change': float(final['precision_relative_change']),
        'final_zero_precision_relative_change': float(final['zero_mode_precision_relative_change']),
        'final_zero_precision_pass': bool(final['pilot_zero_precision_pass']),
        'final_zero_variance_pass': bool(final['pilot_zero_variance_pass']),
        'final_precision_sd_relative_change': float(final['precision_sd_relative_change']),
        'final_zero_variance_relative_change': float(final['zero_mode_variance_relative_change']),
        'optimizer_seconds': float(final['optimizer_seconds']),
        'diagnostic_seconds': float(final['diagnostic_seconds']),
        'fit_elapsed_seconds': float(final['elapsed_seconds']),
    }


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
    fixed_beta: torch.Tensor,
    checkpoint_dir: Path | None = None,
    run_prefix_for_checkpoints: str = '',
    saved_checkpoint: dict | None = None,
    extra_iterations: int = 0,
    continuation_lr: float | None = None,
    continuation_rng_seed: int | None = None,
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
        fixed_beta=fixed_beta,
        mu_log_sigma2=math.log(SIGMA2_TRUE),
        log_std_log_sigma2=-2.3,
        num_mc=NUM_MC,
        fixed_sigma2=SIGMA2_TRUE,
        jitter=JITTER,
    ).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    history = []
    convergence_history = []
    precision_checkpoint_rows = []
    parameter_checkpoint_rows = []
    diagnostic_cache = {}
    previous = None
    consecutive_passes = 0
    start_iteration = 0
    if saved_checkpoint is not None:
        if saved_checkpoint.get('experiment_version') != EXPERIMENT_VERSION:
            raise ValueError('Checkpoint experiment version does not match this script.')
        if saved_checkpoint['seed'] != int(seed) or saved_checkpoint['label'] != label:
            raise ValueError('Checkpoint model or simulation seed does not match.')
        if saved_checkpoint.get('diag_elbo_repeats') != DIAG_ELBO_REPEATS:
            raise ValueError('Use the same --diag-elbo-repeats as the source checkpoint.')
        if saved_checkpoint.get('diag_spectrum_draws') != DIAG_SPECTRUM_DRAWS:
            raise ValueError('Use the same --diag-spectrum-draws as the source checkpoint.')
        if label.startswith('Adaptive') and saved_checkpoint.get('adaptive_pspline_kwargs') != ADAPTIVE_PSPLINE_KWARGS:
            raise ValueError('Adaptive P-spline prior configuration differs from checkpoint.')
        if saved_checkpoint.get('num_mc') != NUM_MC:
            raise ValueError('Checkpoint num_mc differs from this script.')
        if saved_checkpoint['data_digest'] != training_data_digest(y_fit_t, lam_t):
            raise ValueError('Checkpoint training data or graph eigenvalues do not match.')
        model.load_state_dict(saved_checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(saved_checkpoint['optimizer_state'])
        start_iteration = int(saved_checkpoint['iteration'])
        history = list(saved_checkpoint['history'])
        convergence_history = list(saved_checkpoint['convergence_history'])
        precision_checkpoint_rows = list(saved_checkpoint['precision_checkpoint_rows'])
        parameter_checkpoint_rows = list(saved_checkpoint['parameter_checkpoint_rows'])
        diagnostic_cache = saved_checkpoint['diagnostic_cache']
        previous = diagnostic_cache.get(start_iteration)
        if previous is None:
            raise ValueError('Checkpoint missing current diagnostic draw bank.')
        consecutive_passes = int(saved_checkpoint['consecutive_passes'])
        if continuation_lr is not None:
            if continuation_lr <= 0:
                raise ValueError('Continuation learning rate must be positive.')
            for group in optimizer.param_groups:
                group['lr'] = continuation_lr
        if continuation_rng_seed is None:
            torch.set_rng_state(saved_checkpoint['torch_rng_state'].clone())
            np.random.set_state(saved_checkpoint['numpy_rng_state'])
        else:
            torch.manual_seed(continuation_rng_seed)
            np.random.seed(continuation_rng_seed)
        iterations = start_iteration + int(extra_iterations)
    elif extra_iterations or continuation_lr is not None or continuation_rng_seed is not None:
        raise ValueError('Continuation options require a saved checkpoint.')
    n_observed = int((~is_holdout_t).sum().item())
    optimizer_seconds = 0.0
    diagnostic_seconds = 0.0
    fit_start = time.perf_counter()

    def record_checkpoint(it: int):
        nonlocal previous, consecutive_passes, diagnostic_seconds
        diag_start = time.perf_counter()
        current = evaluate_convergence(model, seed)
        elbo_values = current['elbo_repeats']
        validation_elbo = float(elbo_values.mean())
        elbo_mcse = float(elbo_values.std(ddof=1) / np.sqrt(len(elbo_values)))
        qmean = current['precision_mean']
        qsd = current['precision_sd']
        fzero = current['zero_variance_mean']
        exact = current['zero_exact']
        exact_old = previous['zero_exact'] if previous is not None else {}
        exact_ok = bool(exact.get('zero_exact_clip_safe_8sd', False) and
                        exact_old.get('zero_exact_clip_safe_8sd', False))
        bootstrap = {}
        long_window = {}
        if previous is None:
            elbo_change = elbo_change_mcse = elbo_upper = np.nan
            precision_change = precision_sd_change = np.nan
            qzero_change = fzero_change = np.nan
            elbo_pass = precision_pass = zero_precision_pass = zero_variance_pass = False
            qzero_change_exact = fzero_change_exact = np.nan
        else:
            # Pair repeat r with repeat r using the identical evaluation RNG bank.
            paired_differences = (elbo_values - previous['elbo_repeats']) / n_observed
            elbo_change = float(paired_differences.mean())
            elbo_change_mcse = float(
                paired_differences.std(ddof=1) / np.sqrt(len(paired_differences))
            )
            elbo_upper = abs(elbo_change) + DIAG_MCSE_MULTIPLIER * elbo_change_mcse
            precision_change = _relative_change(qmean, previous['precision_mean'])
            precision_sd_change = _relative_change(qsd, previous['precision_sd'])
            qzero_change = float(
                abs(qmean[0] - previous['precision_mean'][0])
                / max(abs(previous['precision_mean'][0]), 1e-12)
            )
            fzero_change = float(
                abs(fzero - previous['zero_variance_mean'])
                / max(abs(previous['zero_variance_mean']), 1e-12)
            )
            bootstrap = paired_change_diagnostics(current, previous, seed, it, DIAG_EVERY)
            if exact_ok:
                qzero_change_exact = abs(
                    exact['zero_precision_mean_exact_unclipped'] -
                    exact_old['zero_precision_mean_exact_unclipped']
                ) / exact_old['zero_precision_mean_exact_unclipped']
                fzero_change_exact = abs(
                    exact['zero_variance_mean_exact_unclipped'] -
                    exact_old['zero_variance_mean_exact_unclipped']
                ) / exact_old['zero_variance_mean_exact_unclipped']
            else:
                qzero_change_exact = fzero_change_exact = np.nan
            elbo_pass = bool(elbo_upper < DIAG_ELBO_TOL_PER_SITE)
            precision_pass = bool(precision_change < DIAG_PRECISION_TOL)
            # For the SDM endpoint, use analytic *unclipped* moments only
            # when the log-precision clamp is far outside the variational mass.
            # Otherwise use the original Monte Carlo rule, flagged in output.
            zero_precision_pass = bool((qzero_change_exact if exact_ok else qzero_change) < DIAG_ZERO_PRECISION_TOL)
            zero_variance_pass = bool((fzero_change_exact if exact_ok else fzero_change) < DIAG_ZERO_VARIANCE_TOL)

        # Long windows are computed from the actual mode-level vectors,
        # and matched draw-bank uncertainty is recorded separately.
        for window in LONG_WINDOW_STEPS:
            old_window = diagnostic_cache.get(it - window)
            tag = f'w{window}'
            if old_window is not None:
                old_q = old_window['precision_mean']
                delta = qmean - old_q
                long_window[f'{tag}_precision_relative_change'] = _relative_change(qmean, old_q)
                long_window[f'{tag}_zero_variance_relative_change'] = (
                    abs(fzero - old_window['zero_variance_mean']) /
                    max(abs(old_window['zero_variance_mean']), 1e-12))
                long_window[f'{tag}_zero_variance_paired_delta_mcse'] = float(
                    (current['zero_variance_draws'] - old_window['zero_variance_draws']).std(ddof=1)
                    / math.sqrt(DIAG_SPECTRUM_DRAWS))
                exact_window_old = old_window['zero_exact']
                if exact.get('zero_exact_clip_safe_8sd') and exact_window_old.get('zero_exact_clip_safe_8sd'):
                    long_window[f'{tag}_zero_variance_relative_change_exact'] = (
                        abs(exact['zero_variance_mean_exact_unclipped'] -
                            exact_window_old['zero_variance_mean_exact_unclipped']) /
                        exact_window_old['zero_variance_mean_exact_unclipped'])
                older_window = diagnostic_cache.get(it - 2 * window)
                if older_window is not None:
                    prev_delta = old_q - older_window['precision_mean']
                    denom = np.linalg.norm(delta) * np.linalg.norm(prev_delta)
                    long_window[f'{tag}_direction_cosine'] = (
                        float(delta @ prev_delta / denom) if denom > 1e-14 else np.nan)
            else:
                long_window[f'{tag}_precision_relative_change'] = np.nan

        interval_pass = bool(
            elbo_pass and precision_pass and zero_precision_pass and zero_variance_pass
        )
        consecutive_passes = consecutive_passes + 1 if interval_pass else 0
        diagnostic_seconds += time.perf_counter() - diag_start
        convergence_history.append({
            'iteration': it,
            'n_train': n_observed,
            'evaluation_ELBO': validation_elbo,
            'evaluation_ELBO_MCSE': elbo_mcse,
            'ELBO_improvement_per_site': elbo_change,
            'ELBO_improvement_MCSE_per_site': elbo_change_mcse,
            'ELBO_abs_upper_2MCSE_per_site': elbo_upper,
            'precision_relative_change': precision_change,
            'precision_sd_relative_change': precision_sd_change,
            'zero_mode_precision_mean': float(qmean[0]),
            'zero_mode_precision_sd': float(qsd[0]),
            'zero_mode_precision_relative_change': qzero_change,
            'zero_mode_variance_mean': fzero,
            'zero_mode_variance_sd': current['zero_variance_sd'],
            'zero_mode_variance_MCSE': current['zero_variance_sd'] / np.sqrt(DIAG_SPECTRUM_DRAWS),
            'zero_mode_variance_relative_change': fzero_change,
            'zero_mode_precision_relative_change_exact': qzero_change_exact,
            'zero_mode_variance_relative_change_exact': fzero_change_exact,
            'zero_mode_guard_uses_exact': exact_ok,
            **exact,
            **bootstrap,
            **long_window,
            'pilot_ELBO_pass': elbo_pass,
            'pilot_precision_pass': precision_pass,
            'pilot_zero_precision_pass': zero_precision_pass,
            'pilot_zero_variance_pass': zero_variance_pass,
            'pilot_checkpoint_pass': interval_pass,
            'pilot_consecutive_passes': consecutive_passes,
            'pilot_persistent_pass': bool(
                consecutive_passes >= DIAG_CONSECUTIVE_CHECKPOINTS
            ),
            'optimizer_seconds': optimizer_seconds,
            'diagnostic_seconds': diagnostic_seconds,
            'elapsed_seconds': time.perf_counter() - fit_start,
        })
        # Save all variational means and log standard deviations individually.
        # These are deterministic states, not Monte Carlo estimates.
        for name, parameter in model.filter.named_parameters():
            values = parameter.detach().cpu().numpy().reshape(-1)
            for component, value in enumerate(values):
                parameter_checkpoint_rows.append({
                    'iteration': int(it), 'parameter': name, 'component': int(component),
                    'value': float(value),
                    'scale_if_log_std': (float(np.exp(value)) if 'log_std' in name else np.nan),
                })
        # Save the mode-level trajectory for later audit of low-frequency effects.
        lam_np = model.lam.detach().cpu().numpy()
        precision_checkpoint_rows.extend(
            {
                'iteration': it, 'mode': int(k), 'lambda': float(lam_np[k]),
                'precision_mean': float(qmean[k]), 'precision_sd': float(qsd[k]),
            }
            for k in range(len(lam_np))
        )
        print(
            f'{label} | Iteration {it} | ELBO delta/site={elbo_change:.6f} '
            f'(MCSE={elbo_change_mcse:.6f}) | '
            f'Precision change={precision_change:.6f} | '
            f'Zero q change={qzero_change:.6f} | '
            f'Zero F change={fzero_change:.6f} | '
            f'Pilot consecutive passes={consecutive_passes} | '
            f'Elapsed={time.perf_counter() - fit_start:.1f}s',
            flush=True,
        )
        previous = current
        diagnostic_cache[it] = current
        for old_it in list(diagnostic_cache):
            if old_it < it - DIAG_CACHE_HORIZON:
                del diagnostic_cache[old_it]

    def save_state(it: int):
        if checkpoint_dir is None:
            return
        prefix = run_prefix_for_checkpoints or f'seed_{seed:05d}'
        short_model = 'sdm' if label.startswith('Adaptive') else 'leroux'
        path = Path(checkpoint_dir) / f'{prefix}__{short_model}_iter_{it:06d}.pt'
        payload = {
            'experiment_version': EXPERIMENT_VERSION,
            'seed': int(seed), 'label': label, 'iteration': int(it),
            'data_digest': training_data_digest(y_fit_t, lam_t),
            'model_state': copy.deepcopy(model.state_dict()),
            'optimizer_state': copy.deepcopy(optimizer.state_dict()),
            'torch_rng_state': torch.get_rng_state().clone(),
            'numpy_rng_state': np.random.get_state(),
            'history': history.copy(),
            'convergence_history': copy.deepcopy(convergence_history),
            'precision_checkpoint_rows': copy.deepcopy(precision_checkpoint_rows),
            'parameter_checkpoint_rows': copy.deepcopy(parameter_checkpoint_rows),
            'diagnostic_cache': copy.deepcopy(diagnostic_cache),
            'consecutive_passes': int(consecutive_passes),
            'source_learning_rate': float(learning_rate),
            'current_learning_rate': float(optimizer.param_groups[0]['lr']),
            'num_mc': NUM_MC,
            'diag_spectrum_draws': DIAG_SPECTRUM_DRAWS,
            'diag_elbo_repeats': DIAG_ELBO_REPEATS,
            'n_train': n_observed,
            'adaptive_pspline_kwargs': ADAPTIVE_PSPLINE_KWARGS if label.startswith('Adaptive') else None,
        }
        atomic_save_checkpoint(payload, path)
        # Keep the phase-1 boundary and the most recent snapshot per model.
        # This bounds checkpoint disk usage across 50 replications while
        # preserving a restart point for an interrupted optimization.
        if not KEEP_ALL_CHECKPOINTS:
            stage_boundary = LEROUX_ITERS if short_model == 'leroux' else PSPLINE_ITERS
            for old in path.parent.glob(f'{prefix}__{short_model}_iter_*.pt'):
                if old == path:
                    continue
                try:
                    old_it = int(old.stem.rsplit('_', 1)[-1])
                except ValueError:
                    continue
                if old_it != stage_boundary:
                    old.unlink()
        print(f'      Saved state: {path}', flush=True)

    # Baseline lets the first comparison be at iteration 250 rather than 500.
    if saved_checkpoint is None:
        record_checkpoint(0)
    for it in range(start_iteration + 1, iterations + 1):
        optimization_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        elbo, _ = model.elbo()
        loss = -elbo
        if not torch.isfinite(loss):
            raise RuntimeError(f'{label}: non-finite loss at iteration {it}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        history.append(float(elbo.detach().cpu()))
        optimizer_seconds += time.perf_counter() - optimization_start

        if it % DIAG_EVERY == 0 or it == iterations:
            record_checkpoint(it)
        if it % SAVE_CHECKPOINT_EVERY == 0 or it == iterations:
            # Always save a diagnostic at a checkpointable iteration.
            if it % DIAG_EVERY != 0 and it != iterations:
                record_checkpoint(it)
            save_state(it)
        if it % 500 == 0:
            print(
                f'      {label}: {it}/{iterations}, '
                f'mean last 100 training ELBO={np.mean(history[-100:]):.3f}',
                flush=True,
            )

    return {
        'label': label,
        'model': model,
        'history': np.asarray(history, dtype=float),
        'convergence_history': pd.DataFrame(convergence_history),
        'precision_checkpoint_history': pd.DataFrame(precision_checkpoint_rows),
        'parameter_checkpoint_history': pd.DataFrame(parameter_checkpoint_rows),
        'start_iteration': start_iteration,
        'final_iteration': iterations,
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
    prior_V0: np.ndarray | None = None,
    prior_m0: np.ndarray | None = None,
    fixed_beta: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Oracle for known beta, or prior-integrated oracle when beta is unknown."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    p = X.shape[1]
    X_O = X[train_idx]
    X_H = X[test_idx]
    y_O = y[train_idx]
    C_OO = C_true[np.ix_(train_idx, train_idx)]
    C_HO = C_true[np.ix_(test_idx, train_idx)]
    C_OH = C_HO.T
    C_HH = C_true[np.ix_(test_idx, test_idx)]
    gain = solve_spd(C_OO, C_OH).T
    conditional_cov = C_HH - gain @ C_OH
    conditional_cov = 0.5 * (
        conditional_cov + conditional_cov.T
    )
    A_beta = X_H - gain @ X_O
    if fixed_beta is not None:
        if prior_V0 is not None or prior_m0 is not None:
            raise ValueError("Known-beta oracle must not receive a beta prior.")
        m_beta = np.asarray(fixed_beta, dtype=float).reshape(-1)
        if m_beta.shape != (p,):
            raise ValueError("fixed_beta must have shape [p].")
        V_beta = np.zeros((p, p), dtype=float)
    else:
        if prior_V0 is None:
            raise ValueError("An estimated-beta oracle requires prior_V0.")
        prior_V0 = np.asarray(prior_V0, dtype=float)
        if prior_m0 is None:
            prior_m0 = np.zeros(p, dtype=float)
        prior_m0 = np.asarray(prior_m0, dtype=float).reshape(-1)
        Cinv_X = solve_spd(C_OO, X_O)
        Cinv_y = solve_spd(C_OO, y_O)
        V0_inv = np.linalg.inv(prior_V0)
        V_beta = np.linalg.inv(V0_inv + X_O.T @ Cinv_X)
        m_beta = V_beta @ (V0_inv @ prior_m0 + X_O.T @ Cinv_y)
    mean = gain @ y_O + A_beta @ m_beta
    covariance = conditional_cov + A_beta @ V_beta @ A_beta.T
    covariance = 0.5 * (covariance + covariance.T)
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
    with known beta=0 (so V_beta=0 and no beta prior is integrated).
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
        m_beta, V_beta = model._beta_moments_from_terms(terms)
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
    predictive_draws = prediction["draws"]
    lower_95 = torch.quantile(
        predictive_draws,
        0.025,
        dim=0,
    )
    upper_95 = torch.quantile(
        predictive_draws,
        0.975,
        dim=0,
    )
    predictive_95_coverage = (
        ((target_t >= lower_95) & (target_t <= upper_95))
        .double()
        .mean()
    )
    predictive_95_avg_width = (
        (upper_95 - lower_95).mean()
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
        predictive_draws,
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
        "predictive_95_coverage": float(
            predictive_95_coverage.detach().cpu()
        ),
        "predictive_95_avg_width": float(
            predictive_95_avg_width.detach().cpu()
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

@torch.no_grad()

def posterior_latent_spatial_mean(
    model,
    *,
    num_mc: int,
    seed: int,
) -> torch.Tensor:
    """Posterior mean of held-out phi after averaging VI hyperparameter draws."""
    K = int(num_mc)
    if K < 1:
        raise ValueError("num_mc must be at least 1.")
    torch.manual_seed(seed)
    phi_means = []
    n_train = model.y_train.numel()
    eye_train = torch.eye(
        n_train,
        dtype=model.y_train.dtype,
        device=model.y_train.device,
    )
    for _ in range(K):
        sigma2 = model._sample_sigma2()
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(
            model.lam,
            theta,
        ).clamp_min(model.min_variance)
        terms = model._observed_terms(F, sigma2)
        m_beta, _ = model._beta_moments_from_terms(terms)
        Sigma_OO = (
            model.U_train
            * F.unsqueeze(0)
        ) @ model.U_train.T
        Sigma_HO = (
            model.U_test
            * F.unsqueeze(0)
        ) @ model.U_train.T
        C_OO = Sigma_OO + sigma2 * eye_train
        chol_COO = stable_cholesky_torch(
            C_OO,
            base_jitter=max(model.jitter, 1e-10),
        )
        residual = (
            model.y_train
            - model.X_train @ m_beta
        )
        alpha = torch.cholesky_solve(
            residual.unsqueeze(1),
            chol_COO,
        ).squeeze(1)
        phi_means.append(Sigma_HO @ alpha)
    return torch.stack(phi_means, dim=0).mean(dim=0)

# =============================================================================
# POSTERIOR SPECTRAL SUMMARIES
# =============================================================================

@torch.no_grad()

def posterior_spectrum_draws(
    result,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    model = result["model"]
    torch.manual_seed(seed)
    spectra = []
    for _ in range(int(draws)):
        theta = model.filter.sample_unconstrained()
        F = model.filter.spectrum(
            model.lam,
            theta,
        ).clamp_min(1e-12)
        spectra.append(
            F.detach().cpu().numpy()
        )
    return np.stack(spectra, axis=0)

def summarize_spectrum_draws(
    draws: np.ndarray,
) -> Dict[str, np.ndarray]:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("Spectrum draws must have shape (draws, modes).")
    return {
        "mean": np.mean(draws, axis=0),
        "median": np.quantile(draws, 0.50, axis=0),
        "q025": np.quantile(draws, 0.025, axis=0),
        "q975": np.quantile(draws, 0.975, axis=0),
    }

def affine_precision_projection(
    lam: np.ndarray,
    precision: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Project modal precision onto span{1, lambda} using all eigenmodes.

    All graph modes have equal weight, including repeated eigenvalues and
    the zero mode. The residual norm therefore equals the Frobenius norm
    of Q - Q_affine in the shared orthonormal graph basis.

    Accepts precision shape (n_modes,) or (n_draws, n_modes).
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    q = np.asarray(precision, dtype=float)
    one_curve = (q.ndim == 1)
    if one_curve:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != lam.size:
        raise ValueError("Precision must have shape (n_modes,) or (draws, n_modes).")
    if not np.isfinite(lam).all() or not np.isfinite(q).all():
        raise ValueError("Affine precision projection requires finite data.")
    design = np.column_stack((np.ones(lam.size), lam))
    # One unconstrained affine least-squares line per posterior draw.
    coefficients = np.linalg.lstsq(design, q.T, rcond=None)[0].T
    affine = coefficients @ design.T
    residual = q - affine
    relative_l2 = (
        np.linalg.norm(residual, axis=1)
        / np.linalg.norm(q, axis=1)
    )
    if one_curve:
        return {
            "intercept": float(coefficients[0, 0]),
            "slope": float(coefficients[0, 1]),
            "affine": affine[0],
            "residual": residual[0],
            "relative_l2": float(relative_l2[0]),
        }
    return {
        "intercept": coefficients[:, 0],
        "slope": coefficients[:, 1],
        "affine": affine,
        "residual": residual,
        "relative_l2": relative_l2,
    }


def posterior_affine_precision_diagnostics(
    lam: np.ndarray,
    F_draws: np.ndarray,
    q_true: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Measure non-affinity draw-by-draw, not from pointwise medians.

    A pointwise posterior median of Leroux curves need not itself be
    perfectly affine, even though *every Leroux draw* is affine.
    """
    q_draws = 1.0 / np.clip(np.asarray(F_draws, dtype=float), 1e-12, None)
    draw_fit = affine_precision_projection(lam, q_draws)
    q_mean = q_draws.mean(axis=0)
    mean_fit = affine_precision_projection(lam, q_mean)
    draw_distance = draw_fit["relative_l2"]
    q_true = np.asarray(q_true, dtype=float)
    return {
        "q_mean": q_mean,
        "q_q025": np.quantile(q_draws, 0.025, axis=0),
        "q_q975": np.quantile(q_draws, 0.975, axis=0),
        "q_affine_mean": mean_fit["affine"],
        "q_affine_residual_mean": mean_fit["residual"],
        "affine_intercept_mean_curve": mean_fit["intercept"],
        "affine_slope_mean_curve": mean_fit["slope"],
        "affine_precision_relative_L2_mean_curve": mean_fit["relative_l2"],
        "affine_precision_relative_L2_draw_mean": float(np.mean(draw_distance)),
        "affine_precision_relative_L2_draw_median": float(np.median(draw_distance)),
        "affine_precision_relative_L2_draw_q025": float(np.quantile(draw_distance, 0.025)),
        "affine_precision_relative_L2_draw_q975": float(np.quantile(draw_distance, 0.975)),
        "affine_precision_relative_L2_draw_max": float(np.max(draw_distance)),
        "precision_relative_L2_to_truth_mean_curve": float(
            np.linalg.norm(q_mean - q_true) / np.linalg.norm(q_true)
        ),
        "draw_distances": draw_distance,
    }


def spectral_recovery_metrics(
    F_true: np.ndarray,
    F_estimate: np.ndarray,
) -> Dict[str, float]:
    F_true = np.asarray(F_true, dtype=float)
    F_estimate = np.asarray(F_estimate, dtype=float)
    relative_l2 = float(
        np.linalg.norm(F_estimate - F_true)
        / np.linalg.norm(F_true)
    )
    log_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.log(np.clip(F_estimate, 1e-12, None))
                    - np.log(np.clip(F_true, 1e-12, None))
                ) ** 2
            )
        )
    )
    return {
        "spectrum_relative_L2": relative_l2,
        "log_spectrum_RMSE": log_rmse,
    }

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

def save_checkpoint_convergence_plots(
    *, leroux_result, sdm_result, run_dir: Path, prefix: str, dpi: int = 180,
):
    """Independent-MC estimates of the observed-data training ELBO; no held-out y."""
    histories = [
        ('Leroux CAR', leroux_result['convergence_history']),
        ('Adaptive P-spline SDM-CAR', sdm_result['convergence_history']),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, frame in histories:
        x = frame['iteration'].to_numpy()
        y = (frame['evaluation_ELBO'] / frame['n_train']).to_numpy()
        mcse = (frame['evaluation_ELBO_MCSE'] / frame['n_train']).to_numpy()
        ax.plot(x, y, label=name)
        ax.fill_between(x, y - 2 * mcse, y + 2 * mcse, alpha=0.12)
    ax.set_xlabel('Optimization iteration')
    ax.set_ylabel('Independent ELBO evaluation per training site')
    ax.set_title('Checkpoint ELBO (bands: ±2 Monte Carlo SE)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(run_dir / f'{prefix}__checkpoint_ELBO.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, frame in histories:
        valid = frame['iteration'] > 0
        ax.plot(frame.loc[valid, 'iteration'],
                frame.loc[valid, 'precision_relative_change'], label=name)
    ax.axhline(DIAG_PRECISION_TOL, color='black', linestyle='--',
               label='Pilot precision tolerance')
    ax.set_xlabel('Optimization iteration')
    ax.set_ylabel('Relative L2 change in posterior mean precision')
    ax.set_title('250-iteration precision stability')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(run_dir / f'{prefix}__checkpoint_precision_stability.png',
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, frame in histories:
        ax.plot(frame['iteration'], frame['zero_mode_variance_mean'], label=name)
    ax.axhline(TAU2_TRUE / (1.0 - RHO_TRUE), color='black', linestyle='--',
               label='True zero-mode variance')
    ax.set_xlabel('Optimization iteration')
    ax.set_ylabel('Posterior mean F(0)')
    ax.set_title('Zero-mode variance during optimization')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(run_dir / f'{prefix}__checkpoint_zero_mode_variance.png',
                dpi=dpi, bbox_inches='tight')
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

def save_precision_recovery_plot(
    *,
    lam: np.ndarray,
    q_true: np.ndarray,
    q_leroux_mean: np.ndarray,
    q_sdm_mean: np.ndarray,
    run_dir: Path,
    prefix: str,
    dpi: int = 180,
):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        lam,
        q_true,
        linewidth=2.2,
        label="True Leroux precision",
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
        f"Leroux truth: precision recovery "
        f"(rho={RHO_TRUE:g})"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(
        run_dir / f"{prefix}__precision_recovery.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

def save_seed_spectral_recovery_plot(
    *,
    lam: np.ndarray,
    F_true: np.ndarray,
    F_sdm_median: np.ndarray,
    F_sdm_q025: np.ndarray,
    F_sdm_q975: np.ndarray,
    run_dir: Path,
    prefix: str,
    dpi: int = 180,
):
    order = np.argsort(lam)
    lam_plot = np.asarray(lam)[order]
    true_plot = np.asarray(F_true)[order]
    median_plot = np.asarray(F_sdm_median)[order]
    q025_plot = np.asarray(F_sdm_q025)[order]
    q975_plot = np.asarray(F_sdm_q975)[order]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        lam_plot,
        true_plot,
        linewidth=2.2,
        label="True Leroux spectrum",
    )
    ax.plot(
        lam_plot,
        median_plot,
        linewidth=1.8,
        label="SDM-CAR posterior median",
    )
    ax.fill_between(
        lam_plot,
        q025_plot,
        q975_plot,
        alpha=0.22,
        label="95% pointwise variational posterior credible band",
    )
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Modal variance")
    ax.set_title(
        f"Seed-specific SDM-CAR spectral recovery (rho={RHO_TRUE:g})"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(
        run_dir / f"{prefix}__sdm_spectral_posterior.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

# =============================================================================
# ONE SIMULATION REPLICATE
# =============================================================================

def run_prefix(seed: int) -> str:
    return f"seed_{seed:05d}"

def completed_run_is_current(
    run_dir: Path,
    prefix: str,
) -> bool:
    complete_path = run_dir / f"{prefix}__COMPLETE.txt"
    metadata_path = run_dir / f"{prefix}__metadata.json"
    if not complete_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "n": N,
        "n_rows": N_ROWS,
        "n_cols": N_COLS,
        "holdout_side": HOLDOUT_SIDE,
        "tau2_true": TAU2_TRUE,
        "rho_true": RHO_TRUE,
        "sigma2_true": SIGMA2_TRUE,
        "beta_true": BETA_TRUE,
        "beta_handling": "known_fixed_zero_generation_fit_oracle",
        "num_mc": NUM_MC,
        "leroux_iterations": LEROUX_ITERS + LEROUX_REFINE_ITERS,
        "sdm_iterations": PSPLINE_ITERS + PSPLINE_REFINE_ITERS,
        "leroux_stage1_iterations": LEROUX_ITERS,
        "sdm_stage1_iterations": PSPLINE_ITERS,
        "leroux_refine_iterations": LEROUX_REFINE_ITERS,
        "sdm_refine_iterations": PSPLINE_REFINE_ITERS,
        "leroux_refine_learning_rate": LEROUX_REFINE_LR,
        "sdm_refine_learning_rate": PSPLINE_REFINE_LR,
        "predictive_draws": PREDICTIVE_DRAWS,
        "oracle_kl_mc_draws": ORACLE_KL_MC,
        "spectrum_draws": SPECTRUM_DRAWS,
        "diag_every": DIAG_EVERY,
        "diag_elbo_repeats": DIAG_ELBO_REPEATS,
        "diag_spectrum_draws": DIAG_SPECTRUM_DRAWS,
        "diag_bootstrap_reps": DIAG_BOOTSTRAP_REPS,
        "save_checkpoint_every": SAVE_CHECKPOINT_EVERY,
        "diag_elbo_tol_per_site": DIAG_ELBO_TOL_PER_SITE,
        "diag_precision_tol": DIAG_PRECISION_TOL,
        "diag_zero_precision_tol": DIAG_ZERO_PRECISION_TOL,
        "diag_zero_variance_tol": DIAG_ZERO_VARIANCE_TOL,
        "diag_consecutive_checkpoints": DIAG_CONSECUTIVE_CHECKPOINTS,
        "diag_mcse_multiplier": DIAG_MCSE_MULTIPLIER,
        "leroux_learning_rate": LEROUX_LR,
        "sdm_learning_rate": PSPLINE_LR,
        "adaptive_pspline_kwargs": ADAPTIVE_PSPLINE_KWARGS,
    }
    return all(metadata.get(key) == value for key, value in expected.items())

def newest_model_checkpoint(run_dir: Path, prefix: str, model_tag: str):
    """Return latest persisted optimizer checkpoint, if any.

    Only filenames belonging to this seed and model are eligible.  The
    content, configuration and data digest are checked by fit_spectral_vi.
    Checkpoint files use pickle; never load a checkpoint from an untrusted
    source.
    """
    ckpt_dir = run_dir / 'checkpoints'
    found = []
    for ckpt in ckpt_dir.glob(f'{prefix}__{model_tag}_iter_*.pt'):
        try:
            iteration = int(ckpt.stem.rsplit('_', 1)[-1])
        except ValueError:
            continue
        found.append((iteration, ckpt))
    if not found:
        return None
    return max(found, key=lambda pair: pair[0])[1]


def fit_two_stage(
    *, label, model_tag, seed, lam_t, X_t, y_fit_t, U_t,
    is_holdout_t, fixed_beta_t, run_dir, prefix, resume_existing,
):
    """Fit both LR stages, or resume from the latest compatible saved state.

    Phase 2 loads *both* model and Adam optimizer states; it changes only
    Adam's learning rate and retains the saved Torch/NumPy RNG states.
    """
    if model_tag == 'leroux':
        first_iters, first_lr = LEROUX_ITERS, LEROUX_LR
        refine_iters, refine_lr = LEROUX_REFINE_ITERS, LEROUX_REFINE_LR
    elif model_tag == 'sdm':
        first_iters, first_lr = PSPLINE_ITERS, PSPLINE_LR
        refine_iters, refine_lr = PSPLINE_REFINE_ITERS, PSPLINE_REFINE_LR
    else:
        raise ValueError(f'Unexpected model tag: {model_tag}')
    total_iters = first_iters + refine_iters
    ckpt_dir = run_dir / 'checkpoints'
    checkpoint_path = newest_model_checkpoint(run_dir, prefix, model_tag) if resume_existing else None
    saved = load_trusted_checkpoint(checkpoint_path) if checkpoint_path else None
    if saved is not None and int(saved['iteration']) > total_iters:
        raise ValueError(f'{label} checkpoint exceeds the scheduled iteration budget.')
    state_it = int(saved['iteration']) if saved else 0
    print(f'    {label}: stage 1 {first_iters} at LR={first_lr}; stage 2 '
          f'{refine_iters} at LR={refine_lr}; starting from {state_it}.', flush=True)

    def fit_segment(saved_state, extra_iters, lr):
        np.random.seed(seed)
        torch.manual_seed(seed)
        filt = make_pspline_filter(lam_t) if model_tag == 'sdm' else make_leroux_filter()
        return fit_spectral_vi(
            label=label, filter_module=filt, iterations=first_iters,
            learning_rate=first_lr, seed=seed,
            X_t=X_t, y_fit_t=y_fit_t, lam_t=lam_t, U_t=U_t,
            is_holdout_t=is_holdout_t, fixed_beta=fixed_beta_t,
            checkpoint_dir=ckpt_dir, run_prefix_for_checkpoints=prefix,
            saved_checkpoint=saved_state,
            extra_iterations=extra_iters if saved_state is not None else 0,
            continuation_lr=lr if saved_state is not None else None,
        )

    if state_it < first_iters:
        first = fit_segment(saved, first_iters - state_it, first_lr)
        del first
        stage1_path = (ckpt_dir / f'{prefix}__{model_tag}_iter_{first_iters:06d}.pt')
        if not stage1_path.exists():
            raise RuntimeError(f'Missing phase-1 boundary checkpoint: {stage1_path}')
        saved = load_trusted_checkpoint(stage1_path)
        state_it = first_iters
    if state_it < total_iters:
        return fit_segment(saved, total_iters - state_it, refine_lr)
    if saved is None:
        raise RuntimeError('A complete model must have a saved final checkpoint.')
    # Resume a seed interrupted during evaluation after both models had fit.
    return fit_segment(saved, 0, refine_lr)


def run_one(
    *,
    seed: int,
    common: Dict[str, np.ndarray],
    output_dir: Path,
    resume_existing: bool = False,
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
        lam_true,
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
    fixed_beta_t = torch.zeros(
        X_t.shape[1],
        dtype=torch.double,
        device=DEVICE,
    )
    print("    fitting Leroux CAR", flush=True)

    # Constructors are called before fit_spectral_vi re-seeds its optimizer RNG.
    # Seed explicitly here too in case a filter initializer draws random values.
    np.random.seed(seed)
    torch.manual_seed(seed)
    leroux_start = time.perf_counter()

    leroux_result = fit_two_stage(
        label='Leroux CAR', model_tag='leroux', seed=seed,
        lam_t=lam_t, X_t=X_t, y_fit_t=y_fit_t, U_t=U_t,
        is_holdout_t=is_holdout_t, fixed_beta_t=fixed_beta_t,
        run_dir=run_dir, prefix=prefix, resume_existing=resume_existing,
    )

    leroux_elapsed = time.perf_counter() - leroux_start

    print(
        f"    Leroux-CAR total fit time (including diagnostics): {leroux_elapsed:.2f} seconds "
        f"({leroux_elapsed / 60:.2f} minutes)",
        flush=True,
    )

    print(
        "    fitting Adaptive P-spline SDM-CAR",
        flush=True,
    )
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    sdm_start = time.perf_counter()

    sdm_result = fit_two_stage(
        label='Adaptive P-spline SDM-CAR', model_tag='sdm', seed=seed,
        lam_t=lam_t, X_t=X_t, y_fit_t=y_fit_t, U_t=U_t,
        is_holdout_t=is_holdout_t, fixed_beta_t=fixed_beta_t,
        run_dir=run_dir, prefix=prefix, resume_existing=resume_existing,
    )

    sdm_elapsed = time.perf_counter() - sdm_start

    print(
        f"    SDM-CAR total fit time (including diagnostics): {sdm_elapsed:.2f} seconds "
        f"({sdm_elapsed / 60:.2f} minutes)",
        flush=True,
    )

    save_elbo_diagnostics(
        leroux_result=leroux_result,
        sdm_result=sdm_result,
        run_dir=run_dir,
        prefix=prefix,
    )

    leroux_result["convergence_history"].to_csv(
        run_dir / f"{prefix}__leroux_convergence.csv",
        index=False,
    )

    sdm_result["convergence_history"].to_csv(
        run_dir / f"{prefix}__sdm_convergence.csv",
        index=False,
    )

    # Additional long-form checkpoint spectra, diagnostic plots, and pilot summary.
    leroux_result['precision_checkpoint_history'].to_csv(
        run_dir / f'{prefix}__leroux_precision_checkpoints.csv', index=False,
    )
    sdm_result['precision_checkpoint_history'].to_csv(
        run_dir / f'{prefix}__sdm_precision_checkpoints.csv', index=False,
    )
    leroux_result['parameter_checkpoint_history'].to_csv(
        run_dir / f'{prefix}__leroux_variational_parameters.csv', index=False,
    )
    sdm_result['parameter_checkpoint_history'].to_csv(
        run_dir / f'{prefix}__sdm_variational_parameters.csv', index=False,
    )
    convergence_summary = pd.DataFrame([
        convergence_summary_row('Leroux CAR', leroux_result['convergence_history']),
        convergence_summary_row('Adaptive P-spline SDM-CAR',
                                sdm_result['convergence_history']),
    ])
    convergence_summary.insert(0, 'seed', int(seed))
    convergence_summary.to_csv(
        run_dir / f'{prefix}__convergence_summary.csv', index=False,
    )
    save_checkpoint_convergence_plots(
        leroux_result=leroux_result, sdm_result=sdm_result,
        run_dir=run_dir, prefix=prefix,
    )

    # ---------------------------------------------------------
    # Exact oracle predictive distribution under Leroux truth
    # ---------------------------------------------------------
    oracle_predictive = oracle_response_predictive(
        C_true=C_true,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        fixed_beta=np.zeros(X.shape[1], dtype=float),
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
    # Held-out latent spatial-effect posterior means and RMSE.
    leroux_phi_mean_t = posterior_latent_spatial_mean(
        leroux_result["model"],
        num_mc=PREDICTIVE_DRAWS,
        seed=seed + 42_001,
    )
    sdm_phi_mean_t = posterior_latent_spatial_mean(
        sdm_result["model"],
        num_mc=PREDICTIVE_DRAWS,
        seed=seed + 42_002,
    )
    phi_true_test = data["phi_true"][test_idx]
    phi_true_test_t = torch.as_tensor(
        phi_true_test,
        dtype=torch.double,
        device=DEVICE,
    )
    leroux_metrics["latent_phi_RMSE"] = float(
        torch.sqrt(
            torch.mean(
                (leroux_phi_mean_t - phi_true_test_t) ** 2
            )
        ).detach().cpu()
    )
    sdm_metrics["latent_phi_RMSE"] = float(
        torch.sqrt(
            torch.mean(
                (sdm_phi_mean_t - phi_true_test_t) ** 2
            )
        ).detach().cpu()
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
    # ---------------------------------------------------------
    # Posterior spectral summaries
    # ---------------------------------------------------------
    leroux_spectrum_draws = posterior_spectrum_draws(
        leroux_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 50_001,
    )
    sdm_spectrum_draws = posterior_spectrum_draws(
        sdm_result,
        draws=SPECTRUM_DRAWS,
        seed=seed + 50_002,
    )
    leroux_spectrum_summary = summarize_spectrum_draws(
        leroux_spectrum_draws
    )
    sdm_spectrum_summary = summarize_spectrum_draws(
        sdm_spectrum_draws
    )
    F_true = data["F_true"]
    leroux_covered = (
        (F_true >= leroux_spectrum_summary["q025"])
        & (F_true <= leroux_spectrum_summary["q975"])
    )
    sdm_covered = (
        (F_true >= sdm_spectrum_summary["q025"])
        & (F_true <= sdm_spectrum_summary["q975"])
    )
    q_true = (
        (1.0 - RHO_TRUE)
        + RHO_TRUE * lam_true
    ) / TAU2_TRUE
    leroux_affine = posterior_affine_precision_diagnostics(
        lam_true, leroux_spectrum_draws, q_true,
    )
    sdm_affine = posterior_affine_precision_diagnostics(
        lam_true, sdm_spectrum_draws, q_true,
    )
    if leroux_affine["affine_precision_relative_L2_draw_max"] > 1e-7:
        print(
            "    WARNING: Leroux posterior precision draws are not affine "
            "to numerical tolerance; inspect the Leroux filter.",
            flush=True,
        )
    affine_metrics = pd.DataFrame([
        {
            "seed": seed,
            "model": "Leroux CAR",
            **{
                k: v for k, v in leroux_affine.items()
                if np.isscalar(v) and k != "affine_precision_relative_L2_draw_max"
            },
        },
        {
            "seed": seed,
            "model": "Adaptive P-spline SDM-CAR",
            **{
                k: v for k, v in sdm_affine.items()
                if np.isscalar(v) and k != "affine_precision_relative_L2_draw_max"
            },
        },
    ])
    affine_metrics.to_csv(
        run_dir / f"{prefix}__affine_precision_metrics.csv", index=False,
    )
    precision = pd.DataFrame({
        "seed": seed,
        "mode": np.arange(N),
        "lambda": lam_true,
        "q_true": q_true,
        "q_leroux_mean": leroux_affine["q_mean"],
        "q_sdm_mean": sdm_affine["q_mean"],
        "q_sdm_best_affine": sdm_affine["q_affine_mean"],
        "q_sdm_affine_residual": sdm_affine["q_affine_residual_mean"],
    })
    precision.to_csv(
        run_dir / f"{prefix}__precision_recovery.csv", index=False,
    )
    save_precision_recovery_plot(
        lam=lam_true,
        q_true=q_true,
        q_leroux_mean=leroux_affine["q_mean"],
        q_sdm_mean=sdm_affine["q_mean"],
        run_dir=run_dir,
        prefix=prefix,
    )
    save_seed_spectral_recovery_plot(
        lam=lam_true,
        F_true=F_true,
        F_sdm_median=sdm_spectrum_summary["median"],
        F_sdm_q025=sdm_spectrum_summary["q025"],
        F_sdm_q975=sdm_spectrum_summary["q975"],
        run_dir=run_dir,
        prefix=prefix,
    )
    spectrum = pd.DataFrame(
        {
            "seed": seed,
            "mode": np.arange(N),
            "lambda": lam_true,
            "F_true": F_true,
            "F_leroux_mean": leroux_spectrum_summary["mean"],
            "F_leroux_median": leroux_spectrum_summary["median"],
            "F_leroux_q025": leroux_spectrum_summary["q025"],
            "F_leroux_q975": leroux_spectrum_summary["q975"],
            "F_leroux_covered": leroux_covered.astype(int),
            "F_sdm_mean": sdm_spectrum_summary["mean"],
            "F_sdm_median": sdm_spectrum_summary["median"],
            "F_sdm_q025": sdm_spectrum_summary["q025"],
            "F_sdm_q975": sdm_spectrum_summary["q975"],
            "F_sdm_covered": sdm_covered.astype(int),
        }
    )
    spectrum.to_csv(
        run_dir
        / f"{prefix}__spectral_recovery.csv",
        index=False,
    )
    leroux_spectral_metrics = spectral_recovery_metrics(
        F_true,
        leroux_spectrum_summary["median"],
    )
    leroux_spectral_metrics["spectral_pointwise_95_coverage"] = float(
        np.mean(leroux_covered)
    )
    sdm_spectral_metrics = spectral_recovery_metrics(
        F_true,
        sdm_spectrum_summary["median"],
    )
    sdm_spectral_metrics["spectral_pointwise_95_coverage"] = float(
        np.mean(sdm_covered)
    )
    spectral_metrics = pd.DataFrame(
        [
            {
                "seed": seed,
                "model": "Leroux CAR",
                **leroux_spectral_metrics,
            },
            {
                "seed": seed,
                "model": "Adaptive P-spline SDM-CAR",
                **sdm_spectral_metrics,
            },
        ]
    )
    spectral_metrics.to_csv(
        run_dir
        / f"{prefix}__spectral_metrics.csv",
        index=False,
    )
    # ---------------------------------------------------------
    # Held-out predictive means
    # ---------------------------------------------------------
    leroux_y_q025 = torch.quantile(
        leroux_prediction["draws"],
        0.025,
        dim=0,
    ).detach().cpu().numpy()
    leroux_y_q975 = torch.quantile(
        leroux_prediction["draws"],
        0.975,
        dim=0,
    ).detach().cpu().numpy()
    sdm_y_q025 = torch.quantile(
        sdm_prediction["draws"],
        0.025,
        dim=0,
    ).detach().cpu().numpy()
    sdm_y_q975 = torch.quantile(
        sdm_prediction["draws"],
        0.975,
        dim=0,
    ).detach().cpu().numpy()
    pred_means = pd.DataFrame(
        {
            "node": test_idx,
            "y_observed": y_test,
            "phi_true": phi_true_test,
            "oracle_mean": oracle_predictive["mean"],
            "leroux_mean": (
                leroux_prediction["mean"]
                .detach()
                .cpu()
                .numpy()
            ),
            "leroux_q025": leroux_y_q025,
            "leroux_q975": leroux_y_q975,
            "leroux_phi_mean": (
                leroux_phi_mean_t.detach().cpu().numpy()
            ),
            "sdm_mean": (
                sdm_prediction["mean"]
                .detach()
                .cpu()
                .numpy()
            ),
            "sdm_q025": sdm_y_q025,
            "sdm_q975": sdm_y_q975,
            "sdm_phi_mean": (
                sdm_phi_mean_t.detach().cpu().numpy()
            ),
        }
    )
    pred_means.to_csv(
        run_dir
        / f"{prefix}__predictive_means.csv",
        index=False,
    )
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "seed": int(seed),
        "n": N,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(test_mask.sum()),
        "n_rows": N_ROWS,
        "n_cols": N_COLS,
        "holdout_side": HOLDOUT_SIDE,
        "tau2_true": TAU2_TRUE,
        "rho_true": RHO_TRUE,
        "sigma2_true": SIGMA2_TRUE,
        "beta_true": BETA_TRUE,
        "beta_handling": "known_fixed_zero_generation_fit_oracle",
        "beta_prior": None,
        "zero_eigenmode_retained": True,
        "same_graph_for_truth_and_fit": True,
        "same_eigenvectors_for_truth_and_fit": True,
        "same_eigenvalues_for_truth_and_fit": True,
        "power_warp": False,
        "fixed_sigma2": FIX_SIGMA2,
        "num_mc": NUM_MC,
        "predictive_draws": PREDICTIVE_DRAWS,
        "oracle_kl_mc_draws": ORACLE_KL_MC,
        "spectrum_draws": SPECTRUM_DRAWS,
        "affine_precision_metric": "modal_unweighted_relative_L2_to_best_affine",
        "affine_precision_evaluation": "per_variational_hyperparameter_draw",
        "leroux_iterations": LEROUX_ITERS + LEROUX_REFINE_ITERS,
        "sdm_iterations": PSPLINE_ITERS + PSPLINE_REFINE_ITERS,
        "leroux_stage1_iterations": LEROUX_ITERS,
        "sdm_stage1_iterations": PSPLINE_ITERS,
        "leroux_refine_iterations": LEROUX_REFINE_ITERS,
        "sdm_refine_iterations": PSPLINE_REFINE_ITERS,
        "leroux_refine_learning_rate": LEROUX_REFINE_LR,
        "sdm_refine_learning_rate": PSPLINE_REFINE_LR,
        "diag_every": DIAG_EVERY,
        "diag_elbo_repeats": DIAG_ELBO_REPEATS,
        "diag_spectrum_draws": DIAG_SPECTRUM_DRAWS,
        "diag_bootstrap_reps": DIAG_BOOTSTRAP_REPS,
        "save_checkpoint_every": SAVE_CHECKPOINT_EVERY,
        "diag_elbo_tol_per_site": DIAG_ELBO_TOL_PER_SITE,
        "diag_precision_tol": DIAG_PRECISION_TOL,
        "diag_zero_precision_tol": DIAG_ZERO_PRECISION_TOL,
        "diag_zero_variance_tol": DIAG_ZERO_VARIANCE_TOL,
        "diag_consecutive_checkpoints": DIAG_CONSECUTIVE_CHECKPOINTS,
        "diag_mcse_multiplier": DIAG_MCSE_MULTIPLIER,
        "leroux_learning_rate": LEROUX_LR,
        "sdm_learning_rate": PSPLINE_LR,
        "adaptive_pspline_kwargs": ADAPTIVE_PSPLINE_KWARGS,
        "convergence_rule_status": "diagnostic_only_not_automatic_early_stopping",
        "two_stage_optimization": True,
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
        "Leroux-truth benchmark complete.\n",
        encoding="utf-8",
    )
    del leroux_result, sdm_result

# =============================================================================
# CROSS-SEED SUMMARIES
# =============================================================================

def _filter_requested_seeds(
    frame: pd.DataFrame,
    requested_seeds,
) -> pd.DataFrame:
    if frame.empty or requested_seeds is None:
        return frame
    requested = {int(seed) for seed in requested_seeds}
    return frame[
        frame["seed"].astype(int).isin(requested)
    ].copy()

def collect_predictive_metrics(
    output_dir: Path,
    requested_seeds=None,
) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__predictive_metrics.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    frame = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )
    return _filter_requested_seeds(
        frame,
        requested_seeds,
    )

def summarize_predictive_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize numeric metrics, including Boolean convergence indicators.

    Convert Boolean metrics to 0/1 *before* grouping: pandas GroupBy.quantile
    does not accept Boolean dtype in some versions. The mean of a Boolean
    convergence indicator is the fraction of seeds satisfying that criterion.
    """
    numeric_cols = [
        c for c in metrics.columns
        if c not in {"seed", "model"}
        and (
            pd.api.types.is_numeric_dtype(metrics[c])
            or pd.api.types.is_bool_dtype(metrics[c])
        )
    ]
    summary_data = metrics[["model", *numeric_cols]].copy()
    # Critical: convert ALL groupby inputs to float, not merely bool columns;
    # use summary_data (not the original metrics) for grouping below.
    summary_data[numeric_cols] = summary_data[numeric_cols].astype(float)
    grouped = summary_data.groupby("model", sort=False)[numeric_cols]
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
    requested_seeds=None,
) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__spectral_recovery.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    frame = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )
    return _filter_requested_seeds(
        frame,
        requested_seeds,
    )

def collect_spectral_metrics(
    output_dir: Path,
    requested_seeds=None,
) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "per_run").rglob(
            "*__spectral_metrics.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
    frame = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )
    return _filter_requested_seeds(
        frame,
        requested_seeds,
    )

def collect_precision_curves(
    output_dir: Path,
    requested_seeds=None,
) -> pd.DataFrame:
    paths = sorted((output_dir / "per_run").rglob("*__precision_recovery.csv"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return _filter_requested_seeds(frame, requested_seeds)


def collect_affine_precision_metrics(
    output_dir: Path,
    requested_seeds=None,
) -> pd.DataFrame:
    paths = sorted((output_dir / "per_run").rglob("*__affine_precision_metrics.csv"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return _filter_requested_seeds(frame, requested_seeds)


def plot_precision_summary(
    precision: pd.DataFrame,
    output_path: Path,
    dpi: int,
):
    summary_rows = []
    for mode, group in precision.groupby("mode", sort=True):
        vals = group["q_sdm_mean"].to_numpy(dtype=float)
        summary_rows.append({
            "mode": mode,
            "lambda": float(group["lambda"].iloc[0]),
            "truth": float(group["q_true"].iloc[0]),
            "leroux": float(group["q_leroux_mean"].mean()),
            "sdm": float(vals.mean()),
            "sdm_q025": float(np.quantile(vals, 0.025)),
            "sdm_q975": float(np.quantile(vals, 0.975)),
            "residual": float(group["q_sdm_affine_residual"].mean()),
            "residual_q025": float(np.quantile(group["q_sdm_affine_residual"], 0.025)),
            "residual_q975": float(np.quantile(group["q_sdm_affine_residual"], 0.975)),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["lambda", "mode"])
    summary.to_csv(
        output_path.parent.parent / "master_tables" /
        "summary_across_seeds__precision_recovery.csv",
        index=False,
    )
    x = summary["lambda"].to_numpy()
    projection = affine_precision_projection(x, summary["sdm"].to_numpy())
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, summary["truth"], label="True affine Leroux precision", linewidth=2.3)
    ax.plot(x, summary["leroux"], label="Leroux: average posterior mean precision", linewidth=1.8)
    ax.plot(x, summary["sdm"], label="SDM-CAR: average posterior mean precision", linewidth=1.8)
    ax.plot(x, projection["affine"], linestyle="--", linewidth=1.5,
            label="Best affine approximation to average SDM-CAR curve")
    ax.fill_between(x, summary["sdm_q025"], summary["sdm_q975"], alpha=0.17,
                    label="95% cross-seed SDM-CAR simulation envelope")
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Modal precision")
    ax.set_title("Leroux truth: fitted modal precision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.plot(x, summary["residual"], linewidth=2.0,
            label="Mean residual from each seed's best affine fit")
    ax.fill_between(x, summary["residual_q025"], summary["residual_q975"],
                    alpha=0.2, label="95% cross-seed residual envelope")
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("SDM-CAR modal precision − fitted affine precision")
    ax.set_title("Residual non-affinity of fitted SDM-CAR precision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(output_path.parent / "leroux_truth__precision_affine_residual.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_spectral_summary(
    spectra: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "seed",
        "mode",
        "lambda",
        "F_true",
        "F_sdm_median",
        "F_sdm_covered",
    }
    missing = required.difference(spectra.columns)
    if missing:
        raise ValueError(
            "Spectral files are from an older experiment version; "
            f"missing columns: {sorted(missing)}"
        )
    rows = []
    for (mode, lam), group in spectra.groupby(
        ["mode", "lambda"],
        sort=False,
    ):
        sdm_medians = group["F_sdm_median"].to_numpy(dtype=float)
        row = {
            "mode": int(mode),
            "lambda": float(lam),
            "F_true": float(group["F_true"].iloc[0]),
            "n_replications": int(group["seed"].nunique()),
            "F_sdm_median_mean_across_seeds": float(
                np.mean(sdm_medians)
            ),
            "F_sdm_median_q025_across_seeds": float(
                np.quantile(sdm_medians, 0.025)
            ),
            "F_sdm_median_q975_across_seeds": float(
                np.quantile(sdm_medians, 0.975)
            ),
            "F_sdm_pointwise_credible_band_coverage": float(
                group["F_sdm_covered"].mean()
            ),
        }
        if "F_leroux_median" in group:
            leroux_medians = group["F_leroux_median"].to_numpy(dtype=float)
            row.update(
                {
                    "F_leroux_median_mean_across_seeds": float(
                        np.mean(leroux_medians)
                    ),
                    "F_leroux_median_q025_across_seeds": float(
                        np.quantile(leroux_medians, 0.025)
                    ),
                    "F_leroux_median_q975_across_seeds": float(
                        np.quantile(leroux_medians, 0.975)
                    ),
                    "F_leroux_pointwise_credible_band_coverage": float(
                        group["F_leroux_covered"].mean()
                    ),
                }
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("lambda")
        .reset_index(drop=True)
    )

def plot_spectral_summary(
    spectral_summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
):
    x = spectral_summary["lambda"].to_numpy(dtype=float)
    truth = spectral_summary["F_true"].to_numpy(dtype=float)
    mean_median = spectral_summary[
        "F_sdm_median_mean_across_seeds"
    ].to_numpy(dtype=float)
    q025 = spectral_summary[
        "F_sdm_median_q025_across_seeds"
    ].to_numpy(dtype=float)
    q975 = spectral_summary[
        "F_sdm_median_q975_across_seeds"
    ].to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )
    ax.plot(
        x,
        truth,
        linewidth=2.2,
        label="True Leroux spectrum",
    )
    ax.plot(
        x,
        mean_median,
        linewidth=1.8,
        label="Average SDM-CAR posterior median across seeds",
    )
    ax.fill_between(
        x,
        q025,
        q975,
        alpha=0.22,
        label="95% simulation envelope across replications",
    )
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Modal variance")
    ax.set_title(
        f"Leroux truth: SDM-CAR spectral recovery "
        f"(rho={RHO_TRUE:g})"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_spectral_coverage(
    spectral_summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )
    ax.plot(
        spectral_summary["lambda"],
        spectral_summary[
            "F_sdm_pointwise_credible_band_coverage"
        ],
        linewidth=1.8,
        label="Empirical spectral coverage",
    )
    ax.axhline(
        0.95,
        linestyle="--",
        linewidth=1.3,
        label="Nominal 95%",
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Graph Laplacian eigenvalue")
    ax.set_ylabel("Coverage probability across replications")
    ax.set_title(
        "SDM-CAR 95% pointwise variational posterior spectral coverage"
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
    requested_seeds=None,
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
        output_dir,
        requested_seeds=requested_seeds,
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
    convergence_paths = sorted(
        (output_dir / 'per_run').rglob('*__convergence_summary.csv')
    )
    if convergence_paths:
        convergence_frame = pd.concat(
            [pd.read_csv(path) for path in convergence_paths], ignore_index=True,
        )
        convergence_frame = _filter_requested_seeds(
            convergence_frame, requested_seeds
        )
        if not convergence_frame.empty:
            convergence_frame.sort_values(['seed', 'model']).to_csv(
                master_dir / 'all_runs__convergence_summary.csv', index=False,
            )
            summarize_predictive_metrics(convergence_frame).to_csv(
                master_dir / 'summary_across_seeds__convergence.csv', index=False,
            )

    spectral_metrics = collect_spectral_metrics(
        output_dir,
        requested_seeds=requested_seeds,
    )
    if not spectral_metrics.empty:
        spectral_metrics = spectral_metrics.sort_values(
            ["seed", "model"]
        ).reset_index(drop=True)
        spectral_metrics.to_csv(
            master_dir / "all_runs__spectral_metrics.csv",
            index=False,
        )
        summarize_predictive_metrics(
            spectral_metrics
        ).to_csv(
            master_dir / "summary_across_seeds__spectral_metrics.csv",
            index=False,
        )
    affine_metrics = collect_affine_precision_metrics(
        output_dir, requested_seeds=requested_seeds,
    )
    if not affine_metrics.empty:
        affine_metrics.to_csv(
            master_dir / "all_runs__affine_precision_metrics.csv", index=False,
        )
        summarize_predictive_metrics(affine_metrics).to_csv(
            master_dir / "summary_across_seeds__affine_precision_metrics.csv",
            index=False,
        )
        paired_affine = paired_predictive_differences(affine_metrics)
        paired_affine.to_csv(
            master_dir / "paired_sdm_minus_leroux__affine_precision_metrics.csv",
            index=False,
        )
        summarize_paired_differences(paired_affine).to_csv(
            master_dir / "summary__paired_sdm_minus_leroux__affine_precision.csv",
            index=False,
        )
    precision_curves = collect_precision_curves(
        output_dir, requested_seeds=requested_seeds,
    )
    if not precision_curves.empty:
        precision_curves.to_csv(
            master_dir / "all_runs__precision_recovery.csv", index=False,
        )
        plot_precision_summary(
            precision_curves,
            plot_dir / "leroux_truth__precision_recovery.png",
            dpi=dpi,
        )
    spectra = collect_spectra(
        output_dir,
        requested_seeds=requested_seeds,
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
        overall_spectral_coverage = pd.DataFrame(
            [
                {
                    "model": "Adaptive P-spline SDM-CAR",
                    "overall_spectral_95_coverage": float(
                        spectra["F_sdm_covered"].mean()
                    ),
                    "n_replications": int(
                        spectra["seed"].nunique()
                    ),
                    "n_modes": int(
                        spectra["mode"].nunique()
                    ),
                }
            ]
        )
        overall_spectral_coverage.to_csv(
            master_dir
            / "overall__spectral_coverage.csv",
            index=False,
        )
        plot_spectral_summary(
            spectral_summary,
            plot_dir
            / "leroux_truth__spectral_recovery.png",
            dpi=dpi,
        )
        plot_spectral_coverage(
            spectral_summary,
            plot_dir
            / "leroux_truth__sdm_spectral_coverage.png",
            dpi=dpi,
        )

# =============================================================================
# CONTROLLED CONTINUATION (no predictive rescoring, no overwritten baseline)
# =============================================================================

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Experiment 1, N=144: fixed-beta Leroux truth, two-stage Leroux and SDM-CAR.'
    )
    parser.add_argument('--mode', choices=('pilot', 'final'), default='pilot',
                        help='pilot: seeds 111,222,333,444,555; final: seeds 1001..1050')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Override the seed list, e.g. --seeds 111 for a one-seed test.')
    parser.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--resume', action='store_true',
                        help='Skip compatible finished runs and resume incomplete fits from the newest checkpoint.')
    parser.add_argument('--overwrite-existing', action='store_true',
                        help='Delete existing per-seed results and start again; use with care.')
    parser.add_argument('--diag-elbo-repeats', type=int, default=DIAG_ELBO_REPEATS)
    parser.add_argument('--diag-spectrum-draws', type=int, default=DIAG_SPECTRUM_DRAWS)
    parser.add_argument('--diag-bootstrap-reps', type=int, default=DIAG_BOOTSTRAP_REPS)
    parser.add_argument('--save-checkpoint-every', type=int, default=SAVE_CHECKPOINT_EVERY)
    parser.add_argument('--keep-all-checkpoints', action='store_true',
                        help='Keep all periodic .pt files; default retains phase-1 and latest states.')
    parser.add_argument('--leroux-stage1-iters', type=int, default=LEROUX_ITERS)
    parser.add_argument('--leroux-stage2-iters', type=int, default=LEROUX_REFINE_ITERS)
    parser.add_argument('--sdm-stage1-iters', type=int, default=PSPLINE_ITERS)
    parser.add_argument('--sdm-stage2-iters', type=int, default=PSPLINE_REFINE_ITERS)
    parser.add_argument('--leroux-stage1-lr', type=float, default=LEROUX_LR)
    parser.add_argument('--leroux-stage2-lr', type=float, default=LEROUX_REFINE_LR)
    parser.add_argument('--sdm-stage1-lr', type=float, default=PSPLINE_LR)
    parser.add_argument('--sdm-stage2-lr', type=float, default=PSPLINE_REFINE_LR)
    parser.add_argument('--predictive-draws', type=int, default=PREDICTIVE_DRAWS)
    parser.add_argument('--oracle-draws', type=int, default=ORACLE_KL_MC)
    parser.add_argument('--spectrum-draws', type=int, default=SPECTRUM_DRAWS)
    return parser.parse_args()


def main():
    global DIAG_SPECTRUM_DRAWS, DIAG_ELBO_REPEATS, DIAG_BOOTSTRAP_REPS
    global SAVE_CHECKPOINT_EVERY, KEEP_ALL_CHECKPOINTS
    global LEROUX_ITERS, LEROUX_LR, LEROUX_REFINE_ITERS, LEROUX_REFINE_LR
    global PSPLINE_ITERS, PSPLINE_LR, PSPLINE_REFINE_ITERS, PSPLINE_REFINE_LR
    global SPECTRUM_DRAWS, PREDICTIVE_DRAWS, ORACLE_KL_MC
    global LerouxCARFilterFullVI, AdaptivePrecisionPSplineFullVI, SpectralCAR_HoldoutVI
    args = parse_args()
    if args.diag_elbo_repeats < 2:
        raise ValueError('--diag-elbo-repeats must be at least 2.')
    if args.diag_spectrum_draws < 2 or args.diag_bootstrap_reps < 20:
        raise ValueError('Use at least 2 spectral draws and 20 bootstrap replicates.')
    if args.save_checkpoint_every <= 0 or args.save_checkpoint_every % DIAG_EVERY:
        raise ValueError('--save-checkpoint-every must be a positive multiple of 250.')
    for name in ('leroux_stage1_iters', 'leroux_stage2_iters',
                 'sdm_stage1_iters', 'sdm_stage2_iters'):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    for name in ('leroux_stage1_lr', 'leroux_stage2_lr',
                 'sdm_stage1_lr', 'sdm_stage2_lr'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive and finite.')
    for name in ('predictive_draws', 'oracle_draws', 'spectrum_draws'):
        if getattr(args, name) < 2:
            raise ValueError(f'--{name.replace("_", "-")} must be at least 2.')
    if args.resume and args.overwrite_existing:
        raise ValueError('Choose --resume OR --overwrite-existing, not both.')
    DIAG_ELBO_REPEATS = args.diag_elbo_repeats
    DIAG_SPECTRUM_DRAWS = args.diag_spectrum_draws
    DIAG_BOOTSTRAP_REPS = args.diag_bootstrap_reps
    SAVE_CHECKPOINT_EVERY = args.save_checkpoint_every
    KEEP_ALL_CHECKPOINTS = args.keep_all_checkpoints
    LEROUX_ITERS, LEROUX_REFINE_ITERS = args.leroux_stage1_iters, args.leroux_stage2_iters
    PSPLINE_ITERS, PSPLINE_REFINE_ITERS = args.sdm_stage1_iters, args.sdm_stage2_iters
    LEROUX_LR, LEROUX_REFINE_LR = args.leroux_stage1_lr, args.leroux_stage2_lr
    PSPLINE_LR, PSPLINE_REFINE_LR = args.sdm_stage1_lr, args.sdm_stage2_lr
    PREDICTIVE_DRAWS, ORACLE_KL_MC, SPECTRUM_DRAWS = (
        args.predictive_draws, args.oracle_draws, args.spectrum_draws)
    if not FIX_SIGMA2 or BETA_TRUE != 0.0:
        raise RuntimeError('The fixed-beta, fixed-sigma2 design must not change.')
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from sdmcar.filters import (
        LerouxCARFilterFullVI as _LerouxCARFilterFullVI,
        AdaptivePrecisionPSplineFullVI as _AdaptivePrecisionPSplineFullVI,
    )
    from sdmcar.models_holdout_fixed_beta import SpectralCAR_HoldoutVI as _SpectralCAR_HoldoutVI
    LerouxCARFilterFullVI = _LerouxCARFilterFullVI
    AdaptivePrecisionPSplineFullVI = _AdaptivePrecisionPSplineFullVI
    SpectralCAR_HoldoutVI = _SpectralCAR_HoldoutVI

    seeds = args.seeds if args.seeds is not None else (PILOT_SEEDS if args.mode == 'pilot' else FINAL_SEEDS)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Supply a nonempty seed list with no duplicates.')
    output_dir = (args.output_dir if args.output_dir is not None else
                  Path(f'experiment1_N144_{args.mode}_two_stage')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    W_true = build_rook_adjacency(N_ROWS, N_COLS)
    L_true = graph_laplacian(W_true)
    lam_true, U_true = np.linalg.eigh(L_true)
    lam_true[np.abs(lam_true) < ZERO_TOL] = 0.0
    lam_true = np.clip(lam_true, 0.0, None)
    if int(np.sum(lam_true <= ZERO_TOL)) != 1:
        raise RuntimeError('Connected rook graph must have exactly one zero eigenmode.')
    test_mask = holdout_block_mask(N_ROWS, N_COLS, HOLDOUT_SIDE, HOLDOUT_PATTERN)
    if int(test_mask.sum()) != 16 or int((~test_mask).sum()) != 128:
        raise RuntimeError('Expected a 16-site holdout and 128 training sites.')
    common = dict(W_true=W_true, L_true=L_true, lam_true=lam_true,
                  U_true=U_true, test_mask=test_mask)
    print(f'Experiment 1 — {args.mode.upper()} MODE, N={N}, train=128, test=16', flush=True)
    print(f'Seeds: {seeds}; experiment version: {EXPERIMENT_VERSION}', flush=True)
    print(f'Fixed beta=0, sigma2={SIGMA2_TRUE}; Leroux rho={RHO_TRUE}, tau2={TAU2_TRUE}', flush=True)
    print('Unmodified rook graph eigensystem for truth and both fits.', flush=True)
    print(f'Output directory: {output_dir}', flush=True)
    print(f'Leroux: {LEROUX_ITERS} @ {LEROUX_LR}, then {LEROUX_REFINE_ITERS} @ {LEROUX_REFINE_LR}', flush=True)
    print(f'SDM: {PSPLINE_ITERS} @ {PSPLINE_LR}, then {PSPLINE_REFINE_ITERS} @ {PSPLINE_REFINE_LR}', flush=True)
    for seed in seeds:
        prefix = run_prefix(int(seed))
        run_dir = output_dir / 'per_run' / prefix
        print(f'\n=== seed={seed} ===', flush=True)
        if args.resume and completed_run_is_current(run_dir, prefix):
            print('    SKIP: compatible complete run.', flush=True)
            continue
        resume_existing = False
        if run_dir.exists() and any(run_dir.iterdir()):
            if args.overwrite_existing:
                shutil.rmtree(run_dir)
            elif args.resume:
                # Only incomplete fits with optimizer states can be resumed safely.
                if (run_dir / f'{prefix}__COMPLETE.txt').exists():
                    raise RuntimeError(f'Completed run has incompatible metadata: {run_dir}')
                if not any((run_dir / 'checkpoints').glob('*.pt')):
                    raise RuntimeError(f'Incomplete run without a usable checkpoint: {run_dir}')
                resume_existing = True
            else:
                raise FileExistsError(f'Existing output at {run_dir}; use --resume or a new output directory.')
        seed_start = time.perf_counter()
        run_one(seed=int(seed), common=common, output_dir=output_dir,
                resume_existing=resume_existing)
        elapsed = time.perf_counter() - seed_start
        print(f'=== SEED {seed} COMPLETE; elapsed {elapsed:.1f} seconds ===', flush=True)
    print('Building cross-seed summaries...', flush=True)
    build_cross_seed_outputs(output_dir, dpi=args.dpi, requested_seeds=seeds)
    print('=== EXPERIMENT 1 COMPLETE ===', flush=True)
    print('Per-run results:', output_dir / 'per_run', flush=True)
    print('Master tables:', output_dir / 'master_tables', flush=True)
    print('Summary plots:', output_dir / 'summary_plots', flush=True)


if __name__ == '__main__':
    main()
