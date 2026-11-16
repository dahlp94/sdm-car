"""Experiment 3: controlled Laplacian eigenvalue power-warp robustness.

True data: Leroux CAR on a 12 x 12 rook graph, beta=0 and sigma2=0.15 fixed.
Fitted data: SAME eigenvectors, eigenvalues mu=lambda_max*(lambda/lambda_max)**gamma.
Gamma grid: 1.0, 0.90, 0.75, 0.50, 0.35.

Self-contained workflow based on the user's validated Experiment 2 two-start
pipeline and its fixed-beta model. It imports ONLY the project's existing
sdmcar.filters and sdmcar.models_holdout_fixed_beta modules.

Final/pilot modes both enforce the locked full fitting protocol. Smoke mode
uses deliberately tiny optimization budgets for functional testing ONLY.

Outputs: frozen per-seed observations shared over all gamma values; all
three restartable VI trajectories per seed/gamma; selected-start metadata;
training-only convergence diagnostics; predictive, spectral, covariance,
and aggregate metrics; paired and degradation summaries; plots.

Scientific scope: eigenvalue-spectrum misspecification ONLY. Neither a change
of eigenvectors nor robustness to arbitrary adjacency errors is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

N_ROWS=N_COLS=12
N=N_ROWS*N_COLS
TAU2_TRUE=0.70
RHO_TRUE=0.92
BETA_TRUE=0.0
SIGMA2_TRUE=0.15
REFERENCE_TAU2=TAU2_TRUE
REFERENCE_RHO=RHO_TRUE
ZERO_TOL=1e-10
HOLDOUT_SIDE=4
HOLDOUT_PATTERN='middle'
FIX_SIGMA2=True
NUM_MC=4
GRAD_CLIP=10.0
JITTER=1e-8
LEROUX_ITERS=4000
LEROUX_LR=1e-3
LEROUX_REFINE_ITERS=2000
LEROUX_REFINE_LR=2e-4
PSPLINE_ITERS=12000
PSPLINE_LR=3e-4
PSPLINE_REFINE_ITERS=2000
PSPLINE_REFINE_LR=6e-5
KEEP_ALL_CHECKPOINTS=False
SPECTRUM_DRAWS=2048
PREDICTIVE_DRAWS=2048
ORACLE_KL_MC=4096
INIT_LEROUX_TAU2=0.50
INIT_LEROUX_RHO=0.80
LEROUX_RHO_EPS=1e-4
LEROUX_LOG_STD_INIT=-3.0
INIT_Q_LEFT=0.20
INIT_Q_RIGHT=20.0
ADAPTIVE_PSPLINE_KWARGS={
  'degree':3,'n_internal_knots':4,'prior_std_log_q':3.0,
  'global_scale_d2':2.0,'prior_std_log_lambda':2.0,
  'mu_log_lambda_init':0.0,'log_std_log_lambda':-3.0,
  'log_std_log_q':-3.0,'log_std_d2':-3.0,'init_d2':0.0,
  'log_q_min':-20.0,'log_q_max':20.0,
}
EXPERIMENT_VERSION='experiment3_N144_power_warp_fixed_beta_two_start_v1'
SELECTION_ELBO_REPEATS=128
SELECTION_MCSE_MULTIPLIER=2.0
CURVED_INITIAL_D2=[0.12,-0.12,0.12,-0.12,0.12,-0.12]
CONTINUATION_BLOCK=2000
CONTINUATION_MAX_BLOCKS=2
ADDITIONAL_CONTINUATION_BLOCKS=0
CONTINUATION_STOP_RULE='block-terminal'
PROTOCOL_IMPLEMENTATION='checkpoint_stop_v2'
RUN_MODE='pilot'
PILOT_SEEDS=[111,222,333,444,555]
FINAL_SEEDS=list(range(1001,1051))
DEFAULT_GAMMAS=[1.0,0.90,0.75,0.50,0.35]
DEFAULT_PROJECT_ROOT=Path(r'C:/Users/pd006/Desktop/internship_search/sdm-car')
DEVICE=torch.device('cpu')
torch.set_default_dtype(torch.double)
LerouxCARFilterFullVI=None
AdaptivePrecisionPSplineFullVI=None
SpectralCAR_HoldoutVI=None

DIAG_EVERY=250
DIAG_ELBO_REPEATS=32
DIAG_SPECTRUM_DRAWS=256
DIAG_BOOTSTRAP_REPS=250
SAVE_CHECKPOINT_EVERY=1000
LONG_WINDOW_STEPS=(1000,2000)
DIAG_CACHE_HORIZON=2*max(LONG_WINDOW_STEPS)
DIAG_ELBO_TOL_PER_SITE=0.01
DIAG_PRECISION_TOL=0.005
DIAG_ZERO_PRECISION_TOL=0.02
DIAG_ZERO_VARIANCE_TOL=0.02
DIAG_CONSECUTIVE_CHECKPOINTS=3
DIAG_MCSE_MULTIPLIER=2.0



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


def make_pspline_filter(lam_t: torch.Tensor, start: str = "original"):
    filt = AdaptivePrecisionPSplineFullVI(
        lam_max=float(lam_t.max().item()),
        mu_log_q_left=math.log(INIT_Q_LEFT),
        mu_log_q_right=math.log(INIT_Q_RIGHT),
        **ADAPTIVE_PSPLINE_KWARGS,
    ).to(DEVICE)
    if start not in ("original", "curve"):
        raise ValueError(f"Unknown SDM initial state {start!r}")
    if start == "curve":
        if filt.K != len(CURVED_INITIAL_D2):
            raise RuntimeError("Curved start dimension changed")
        with torch.no_grad():
            filt.mu_d2.copy_(torch.tensor(CURVED_INITIAL_D2, dtype=torch.double))
    return filt


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


def training_data_digest(y_fit_t, lam_t, U_t=None, is_holdout_t=None, fixed_beta=None):
    """Hash observed training data and eigenvalues for checkpoint validation."""
    y = y_fit_t.detach().cpu().numpy()
    lam = lam_t.detach().cpu().numpy()
    h = hashlib.sha256()
    h.update(np.asarray(y, dtype='<f8').tobytes())
    h.update(np.asarray(lam, dtype='<f8').tobytes())
    for optional in (U_t, is_holdout_t, fixed_beta):
        if optional is not None:
            arr = optional.detach().cpu().numpy() if isinstance(optional, torch.Tensor) else np.asarray(optional)
            h.update(np.asarray(arr, dtype='<f8').tobytes())
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
    stop_on_persistent_at_or_after: int | None = None,
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
        if saved_checkpoint['data_digest'] != training_data_digest(y_fit_t, lam_t, U_t, is_holdout_t, fixed_beta):
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
    actual_final_iteration = start_iteration
    if stop_on_persistent_at_or_after is not None:
        if saved_checkpoint is None:
            raise ValueError('In-block stopping is only valid during continuation from a saved checkpoint.')
        if int(stop_on_persistent_at_or_after) < 0:
            raise ValueError('Invalid continuation stopping boundary.')

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
            'data_digest': training_data_digest(y_fit_t, lam_t, U_t, is_holdout_t, fixed_beta),
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
            stage_boundaries = ({LEROUX_ITERS, LEROUX_ITERS + LEROUX_REFINE_ITERS}
                                if short_model == 'leroux' else
                                {PSPLINE_ITERS, PSPLINE_ITERS + PSPLINE_REFINE_ITERS})
            for old in path.parent.glob(f'{prefix}__{short_model}_iter_*.pt'):
                if old == path:
                    continue
                try:
                    old_it = int(old.stem.rsplit('_', 1)[-1])
                except ValueError:
                    continue
                # Retain reached continuation-block boundaries to document
                # any failed numerical caps across protocol amendments.
                is_continuation_boundary = (
                    old_it > max(stage_boundaries)
                    and (old_it - max(stage_boundaries)) % CONTINUATION_BLOCK == 0
                )
                if old_it not in stage_boundaries and not is_continuation_boundary:
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

        # Continuation can terminate at the FIRST 250-step diagnostic
        # checkpoint that satisfies three consecutive passing checks. The
        # two-stage (training and two-start selection) budgets are unchanged.
        diagnostic_due = (it % DIAG_EVERY == 0 or it == iterations)
        if diagnostic_due:
            record_checkpoint(it)
        accepted_here = bool(
            diagnostic_due
            and stop_on_persistent_at_or_after is not None
            and it > stop_on_persistent_at_or_after
            and convergence_history[-1]['pilot_persistent_pass']
        )
        if it % SAVE_CHECKPOINT_EVERY == 0 or it == iterations or accepted_here:
            save_state(it)  # save the EXACT evaluated optimizer state
        actual_final_iteration = it
        if accepted_here:
            print(f'      {label}: in-block stopping criterion reached at {it}; '
                  'state saved for evaluation.', flush=True)
            break
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
        'final_iteration': actual_final_iteration,
    }


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
    """Experiment 1 convention: V-statistic including k=l diagonal pairs."""
    target = target.reshape(-1)
    first = torch.linalg.vector_norm(
        draws - target.unsqueeze(0), dim=1,
    ).mean()
    pairwise = torch.cdist(draws, draws, p=2)
    second = 0.5 * pairwise.mean()
    return float((first - second).detach().cpu())


def energy_score_unbiased(
    target: torch.Tensor, draws: torch.Tensor,
) -> float:
    """Off-diagonal U-statistic for finite-MC comparison (supplemental)."""
    target = target.reshape(-1)
    k = draws.shape[0]
    if k < 2:
        raise ValueError('At least two predictive draws required.')
    first = torch.linalg.vector_norm(draws - target.unsqueeze(0), dim=1).mean()
    # E[||X-X'||] uses i != j; diagonal entries are zero by definition.
    second = 0.5 * torch.cdist(draws, draws, p=2).sum() / (k * (k-1))
    return float((first-second).detach().cpu())


def variogram_score_equal_pairs_p05(target: torch.Tensor, draws: torch.Tensor) -> float:
    """p=0.5, uniform weights, mean over unordered distinct holdout pairs."""
    h = int(target.numel())
    i, j = torch.triu_indices(h, h, offset=1, device=target.device)
    true_increments = torch.abs(target[i] - target[j]).pow(0.5)
    pred_increments = torch.abs(draws[:, i] - draws[:, j]).pow(0.5).mean(dim=0)
    return float(torch.mean((true_increments-pred_increments).square()).detach().cpu())


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
    es_u = energy_score_unbiased(target_t, predictive_draws)
    vs = variogram_score_equal_pairs_p05(target_t, predictive_draws)
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
        "energy_score_unbiased_MC": es_u,
        "variogram_score_p05_mean_pairs": vs,
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
    variance_diffs = F_estimate - F_true
    squared_total = float(np.dot(variance_diffs, variance_diffs))
    squared_positive = float(np.dot(variance_diffs[1:], variance_diffs[1:]))
    zero_fraction = float(variance_diffs[0]**2 / squared_total) if squared_total > 0 else 0.0
    positive_err = float(np.linalg.norm(variance_diffs[1:]) / np.linalg.norm(F_true[1:]))
    return {
        "variance_zero_mode_rel_abs_error": float(abs(variance_diffs[0])/F_true[0]),
        "variance_positive_mode_relative_L2": positive_err,
        "variance_sq_error_fraction_zero_mode": zero_fraction,
        "variance_sq_error_fraction_positive_modes": 1.0-zero_fraction,
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
    ax.axhline(REFERENCE_TAU2 / (1.0 - REFERENCE_RHO), color='black', linestyle='--',
               label='True Leroux zero-mode variance')
    ax.set_xlabel('Optimization iteration')
    ax.set_ylabel('Posterior mean F(0)')
    ax.set_title('Zero-mode variance during optimization')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(run_dir / f'{prefix}__checkpoint_zero_mode_variance.png',
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fit_two_stage(
    *, label, model_tag, seed, lam_t, X_t, y_fit_t, U_t,
    is_holdout_t, fixed_beta_t, run_dir, prefix, resume_existing,
    sdm_start="original", optimizer_seed=None,
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
    optimizer_seed = int(seed if optimizer_seed is None else optimizer_seed)
    ckpt_dir = run_dir / 'checkpoints'
    # A continuation may have progressed beyond the two-stage selection horizon.
    # Always evaluate the prespecified candidate at its EXACT original budget.
    eligible = []
    if resume_existing:
        for candidate in ckpt_dir.glob(f"{prefix}__{model_tag}_iter_*.pt"):
            try:
                it_candidate = int(candidate.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if it_candidate <= total_iters:
                eligible.append((it_candidate, candidate))
    checkpoint_path = max(eligible)[1] if eligible else None
    saved = load_trusted_checkpoint(checkpoint_path) if checkpoint_path else None
    if saved is not None and int(saved['iteration']) > total_iters:
        raise ValueError(f'{label} checkpoint exceeds the scheduled iteration budget.')
    state_it = int(saved['iteration']) if saved else 0
    print(f'    {label}: stage 1 {first_iters} at LR={first_lr}; stage 2 '
          f'{refine_iters} at LR={refine_lr}; starting from {state_it}.', flush=True)

    def fit_segment(saved_state, extra_iters, lr):
        np.random.seed(optimizer_seed)
        torch.manual_seed(optimizer_seed)
        filt = make_pspline_filter(lam_t, start=sdm_start) if model_tag == 'sdm' else make_leroux_filter()
        return fit_spectral_vi(
            label=label, filter_module=filt, iterations=first_iters,
            learning_rate=first_lr, seed=optimizer_seed,
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


def curved_optimizer_seed(seed: int) -> int:
    """Exactly the independent Curved stream used in the Step B pilot."""
    return 1_000_000 + int(seed) * 10 + 3


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def score_two_start_elbo(original_model, curve_model, seed: int) -> dict:
    """Pair independent TRAINING-ONLY MC banks; never use truth or holdout."""
    def bank(model):
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(900_000 + int(seed))
            out = np.asarray([float(model.elbo()[0].detach().cpu().item())
                              for _ in range(SELECTION_ELBO_REPEATS)], dtype=float)
        if not np.isfinite(out).all():
            raise RuntimeError('Non-finite independent ELBO evaluation')
        return out
    a = bank(original_model)
    b = bank(curve_model)
    ntrain = int(original_model.y_train.numel())
    if ntrain != 128 or int(curve_model.y_train.numel()) != ntrain:
        raise RuntimeError('Training sample sizes disagree')
    delta = (b - a) / ntrain
    mu = float(delta.mean())
    se = float(delta.std(ddof=1)/np.sqrt(delta.size))
    chosen = 'curve' if mu > SELECTION_MCSE_MULTIPLIER*se else 'original'
    return dict(seed=seed, selected_start=chosen, elbo_original_per_site=float(a.mean()/ntrain),
                elbo_curve_per_site=float(b.mean()/ntrain), difference_per_site=mu,
                mcse_per_site=se, difference_lower_95=mu-1.96*se,
                difference_upper_95=mu+1.96*se, n_evaluations=SELECTION_ELBO_REPEATS,
                mcse_multiplier=SELECTION_MCSE_MULTIPLIER,
                selection_criterion='curve if difference_per_site > 2*paired_MCSE; otherwise original',
                original_optimizer_seed=seed, curved_optimizer_seed=curved_optimizer_seed(seed))


def enforce_selection_file(run_dir: Path, selection: dict) -> None:
    path = run_dir / 'sdm_two_start_selection.json'
    if path.exists():
        prior = json.loads(path.read_text(encoding='utf-8'))
        if prior != selection:
            raise RuntimeError('Previously frozen selection differs from rescore; stop rather than change decision')
    else:
        atomic_json(path, selection)


def extend_if_not_converged(result, *, model_tag: str, start: str, seed: int,
                            X_t, y_fit_t, lam_t, U_t, is_holdout_t, fixed_beta_t,
                            fit_dir: Path, prefix: str):
    """After selecting, continue only the chosen path in fixed 2000-step blocks.

    Selection remains based on the original two-stage checkpoints. If the
    prespecified maximum cannot pass, fail closed before writing COMPLETE.
    """
    if RUN_MODE == 'smoke':
        return result, 0
    label = 'Leroux CAR' if model_tag == 'leroux' else 'Adaptive P-spline SDM-CAR'
    optimizer_seed = seed if model_tag == 'leroux' or start == 'original' else curved_optimizer_seed(seed)
    base_it = (LEROUX_ITERS+LEROUX_REFINE_ITERS if model_tag == 'leroux' else
               PSPLINE_ITERS+PSPLINE_REFINE_ITERS)
    cap = base_it + CONTINUATION_BLOCK * (CONTINUATION_MAX_BLOCKS + ADDITIONAL_CONTINUATION_BLOCKS)
    ckpt_dir = fit_dir/'checkpoints'
    def checkpoints():
        pairs=[]
        for p in ckpt_dir.glob(f'{prefix}__{model_tag}_iter_*.pt'):
            try: it=int(p.stem.rsplit('_', 1)[-1])
            except ValueError: continue
            if it >= base_it and it <= cap:
                pairs.append((it,p))
        return sorted(pairs)
    while True:
        available = checkpoints()
        if not available:
            raise RuntimeError('Missing final checkpoint for continuation audit')
        it,path = available[-1]
        # If continuation already exists, rehydrate its actual state and history.
        if it != result['final_iteration']:
            saved=load_trusted_checkpoint(path)
            filt = (make_leroux_filter() if model_tag == 'leroux' else
                    make_pspline_filter(lam_t, start=start))
            result=fit_spectral_vi(label=label, filter_module=filt,
                iterations=base_it, learning_rate=(LEROUX_LR if model_tag=='leroux' else PSPLINE_LR),
                seed=optimizer_seed, X_t=X_t, y_fit_t=y_fit_t, lam_t=lam_t,
                U_t=U_t, is_holdout_t=is_holdout_t, fixed_beta=fixed_beta_t,
                checkpoint_dir=ckpt_dir, run_prefix_for_checkpoints=prefix,
                saved_checkpoint=saved, extra_iterations=0,
                continuation_lr=(LEROUX_REFINE_LR if model_tag=='leroux' else PSPLINE_REFINE_LR))
        passed=bool(result['convergence_history'].iloc[-1]['pilot_persistent_pass'])
        if passed:
            # Retain an earlier cap-failure record, and explicitly record successful
            # recovery if a later amended-budget continuation passed.
            prior_failure = fit_dir/'convergence_flag.json'
            if prior_failure.exists():
                atomic_json(fit_dir/'convergence_recovered.json', {
                    'seed':seed, 'model':model_tag, 'start':start,
                    'pass':True, 'recovered_iteration':int(it),
                    'prior_failure_record':prior_failure.name,
                    'additional_continuation_blocks_authorized':ADDITIONAL_CONTINUATION_BLOCKS,
                })
            return result, int(result['final_iteration'])-base_it
        if it >= cap:
            status={'seed':seed,'model':model_tag,'start':start,'final_iteration':it,
                    'pass':False,'reason':'convergence criterion still failed after budget cap',
                    'initial_max_continuation_blocks':CONTINUATION_MAX_BLOCKS,
                    'additional_continuation_blocks':ADDITIONAL_CONTINUATION_BLOCKS,
                    'continuation_learning_rate':(LEROUX_REFINE_LR if model_tag=='leroux' else PSPLINE_REFINE_LR)}
            atomic_json(fit_dir/'convergence_flag.json',status)
            raise RuntimeError(f'{label} seed {seed} ({start}): failed convergence at {it}. '
                               f'Check {fit_dir}; did not mark seed complete.')
        next_target = base_it + (1+(it-base_it)//CONTINUATION_BLOCK)*CONTINUATION_BLOCK
        next_target = min(next_target,cap)
        saved=load_trusted_checkpoint(path)
        filt=(make_leroux_filter() if model_tag=='leroux' else
              make_pspline_filter(lam_t, start=start))
        print(f'    {label} {start} failed persistent criterion: continuing {it} -> {next_target}', flush=True)
        result=fit_spectral_vi(label=label, filter_module=filt,
            iterations=base_it, learning_rate=(LEROUX_LR if model_tag=='leroux' else PSPLINE_LR),
            seed=optimizer_seed, X_t=X_t, y_fit_t=y_fit_t, lam_t=lam_t,
            U_t=U_t, is_holdout_t=is_holdout_t, fixed_beta=fixed_beta_t,
            checkpoint_dir=ckpt_dir, run_prefix_for_checkpoints=prefix,
            saved_checkpoint=saved, extra_iterations=next_target-it,
            continuation_lr=(LEROUX_REFINE_LR if model_tag=='leroux' else PSPLINE_REFINE_LR),
            stop_on_persistent_at_or_after=(base_it if CONTINUATION_STOP_RULE == 'first-persistent' else None))


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
        # Treat each independent generated dataset as one paired replicate.
        # These are CIs for the MEAN, not empirical quantiles of differences.
        from scipy.stats import t as _student_t
        n = len(x)
        se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        tcrit = float(_student_t.ppf(0.975, df=n-1)) if n > 1 else np.nan
        rows.append(
            {
                "metric": col,
                "n_datasets": n,
                "mean_diff_SE_across_datasets": se,
                "mean_diff_CI95_lower": float(np.mean(x) - tcrit * se) if n > 1 else np.nan,
                "mean_diff_CI95_upper": float(np.mean(x) + tcrit * se) if n > 1 else np.nan,
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

# =============================================================================
# EXPERIMENT 3 SPECIFIC GRAPH WARP, FROZEN DATA AND EVALUATION
# =============================================================================

def gamma_token(gamma: float) -> str:
    return f'{gamma:.3f}'.replace('.', 'p')


def run_prefix(seed: int, gamma: float) -> str:
    return f'seed_{seed:05d}__gamma_{gamma_token(gamma)}'


def power_warp(lam: np.ndarray, gamma: float) -> np.ndarray:
    """Warp only eigenvalue magnitudes, preserving zero and max endpoints."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError('The warp exponent must lie in (0, 1].')
    a = np.asarray(lam, dtype=float).reshape(-1)
    if np.min(a) < -ZERO_TOL:
        raise ValueError('Significantly negative Laplacian eigenvalues.')
    a = np.maximum(a, 0.0)
    top = float(a.max())
    if top <= 0:
        raise ValueError('Graph must have positive nonzero eigenvalues.')
    if gamma == 1.0:
        return a.copy()  # avoid rounding even in the unwarped baseline
    result = top * np.power(a / top, gamma)
    result[a <= ZERO_TOL] = 0.0
    result[np.argmax(a)] = top
    return result


def best_affine_precision_discrepancy(mu: np.ndarray, q_true: np.ndarray) -> dict:
    """Unconstrained affine projection, equal weight per graph eigenmode."""
    proj = affine_precision_projection(mu, q_true)
    return dict(
        best_affine_precision_relative_L2=float(proj['relative_l2']),
        best_affine_intercept=float(proj['intercept']),
        best_affine_slope=float(proj['slope']),
        affine_intercept_positive=bool(proj['intercept'] > 0),
        affine_slope_positive=bool(proj['slope'] > 0),
    )


def seed_data(seed: int, output_dir: Path, common: dict) -> dict:
    """Frozen once per seed, shared byte-for-byte across ALL warp levels.

    Importantly U (including repeated eigenspaces) is frozen rather than
    re-diagonalized in each gamma configuration or on resume.
    """
    folder = output_dir/'frozen_datasets'
    folder.mkdir(parents=True, exist_ok=True)
    file = folder/f'seed_{seed:05d}.npz'
    if file.exists():
        with np.load(file, allow_pickle=False) as src:
            data = {key: src[key].copy() for key in src.files}
        if int(data['seed']) != int(seed):
            raise RuntimeError('Seed does not match frozen dataset.')
        if data['y'].shape != (N,) or data['phi_true'].shape != (N,):
            raise RuntimeError('Frozen y/phi dimensions are incorrect.')
        np.testing.assert_array_equal(data['test_mask'], common['test_mask'])
        np.testing.assert_allclose(data['lam_true'], common['lam_true'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(data['U_true']@np.diag(data['lam_true'])@data['U_true'].T,
                                   common['L_true'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(data['F_true'], common['F_true'], atol=1e-10, rtol=0)
        return data
    rng = np.random.default_rng(seed)
    X = np.ones((N,1),dtype=float)
    phi = common['U_true'] @ (np.sqrt(common['F_true'])*rng.normal(size=N))
    y = phi + BETA_TRUE + rng.normal(scale=np.sqrt(SIGMA2_TRUE),size=N)
    data = dict(seed=np.asarray(seed), lam_true=common['lam_true'], U_true=common['U_true'],
                X=X, phi_true=phi, y=y, F_true=common['F_true'],
                test_mask=common['test_mask'])
    temp = file.with_suffix('.npz.tmp')
    with open(temp,'wb') as fp:
        np.savez_compressed(fp,**data)
    os.replace(temp,file)
    return data



def gaussian_kl_covariance_per_dim(truth_cov: np.ndarray, fitted_cov: np.ndarray) -> float:
    '''Zero-mean Gaussian covariance KL(p_true || p_fitted), per coordinate.'''
    d=int(truth_cov.shape[0])
    a=0.5*(truth_cov+truth_cov.T)
    b=0.5*(fitted_cov+fitted_cov.T)
    trace_term=float(np.trace(solve_spd(b,a)))
    chol_a=stable_cholesky_np(a)
    chol_b=stable_cholesky_np(b)
    logdet_a=2*np.log(np.diag(chol_a)).sum()
    logdet_b=2*np.log(np.diag(chol_b)).sum()
    return float(max(0.,.5*(trace_term-d+logdet_b-logdet_a)/d))

def corr_np(covariance: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(covariance),1e-14))
    return covariance/d[:,None]/d[None,:]


def covariance_metrics_from_spectra(
        F_draws: np.ndarray, F_true: np.ndarray, U: np.ndarray,
        oracle_cov: np.ndarray, pred: dict, positive_mask: np.ndarray) -> dict:
    """Posterior-MEAN covariance recovery, with a strict zero-mode separation.

    Means are taken over posterior draws of F before forming matrix metrics.
    Orthogonal invariance makes latent covariance Frobenius error exactly a
    modal Euclidean error; there is no need to reconstruct 144x144 matrices
    for every single draw. The predictive mixture covariance is assembled
    from drawwise conditional means/covariances before comparison to oracle.
    """
    mean_F = np.asarray(F_draws,dtype=float).mean(axis=0)
    F_true = np.asarray(F_true,dtype=float)
    differences = mean_F-F_true
    pos = np.asarray(positive_mask,dtype=bool)
    if pos.sum()!=N-1 or np.flatnonzero(~pos).tolist()!=[0]:
        raise RuntimeError('Expected exactly one constant eigenmode at index 0.')
    err2 = float(differences@differences)
    fz_true = float(F_true[0])
    full_rel = float(np.linalg.norm(differences)/np.linalg.norm(F_true))
    positive_rel = float(np.linalg.norm(differences[pos])/np.linalg.norm(F_true[pos]))
    zero_fraction = float(differences[0]**2/err2) if err2>0 else 0.0
    sigma_est = covariance_from_spectrum(U,mean_F)
    sigma_true = covariance_from_spectrum(U,F_true)
    sigma_pos_est = covariance_from_spectrum(U[:,pos],mean_F[pos])
    sigma_pos_true = covariance_from_spectrum(U[:,pos],F_true[pos])
    np.testing.assert_allclose(
        np.linalg.norm(sigma_est-sigma_true,'fro')/np.linalg.norm(sigma_true,'fro'),
        full_rel, atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(
        np.linalg.norm(sigma_pos_est-sigma_pos_true,'fro')/np.linalg.norm(sigma_pos_true,'fro'),
        positive_rel, atol=1e-9,rtol=1e-9)
    pred_cov = pred['cov'].detach().cpu().numpy()
    conditional_avg = pred['component_covs'].mean(0).detach().cpu().numpy()
    resp_est=sigma_est+SIGMA2_TRUE*np.eye(N)
    resp_true=sigma_true+SIGMA2_TRUE*np.eye(N)
    return dict(
        full_latent_cov_relative_frobenius=full_rel,
        positive_latent_cov_relative_frobenius=positive_rel,
        zero_mode_variance_mean=float(mean_F[0]),
        zero_mode_variance_true=fz_true,
        zero_mode_variance_absolute_relative_error=float(abs(differences[0])/fz_true),
        zero_mode_variance_signed_relative_error=float(differences[0]/fz_true),
        zero_mode_fraction_of_full_squared_error=zero_fraction,
        response_cov_relative_frobenius=float(np.linalg.norm(resp_est-resp_true,'fro')/
                                              np.linalg.norm(resp_true,'fro')),
        latent_correlation_relative_frobenius=float(np.linalg.norm(corr_np(sigma_est)-corr_np(sigma_true),'fro')/
                                                   np.linalg.norm(corr_np(sigma_true),'fro')),
        positive_latent_correlation_relative_frobenius=float(np.linalg.norm(corr_np(sigma_pos_est)-corr_np(sigma_pos_true),'fro')/
                                                            np.linalg.norm(corr_np(sigma_pos_true),'fro')),
        joint_predictive_cov_relative_frobenius=float(np.linalg.norm(pred_cov-oracle_cov,'fro')/
                                                     np.linalg.norm(oracle_cov,'fro')),
        mean_draw_conditional_cov_relative_frobenius=float(np.linalg.norm(conditional_avg-oracle_cov,'fro')/
                                                          np.linalg.norm(oracle_cov,'fro')),
        mean_draw_conditional_cov_gaussian_KL_per_site=gaussian_kl_covariance_per_dim(
            oracle_cov,conditional_avg),
        predictive_moment_cov_gaussian_KL_per_site=gaussian_kl_covariance_per_dim(
            oracle_cov,pred_cov),
        latent_cov_gaussian_KL_per_mode=float(np.mean(0.5*(F_true/mean_F - 1 + np.log(mean_F/F_true)))),
        positive_latent_cov_gaussian_KL_per_mode=float(np.mean(0.5*(F_true[pos]/mean_F[pos]-1+
                                                               np.log(mean_F[pos]/F_true[pos])))),
    )


def aggregate_weights() -> list[tuple[str,str,np.ndarray]]:
    """Five linear aggregates in the 4x4 held-out block's row-major ordering."""
    result=[]
    w=np.ones(16,dtype=float)/16
    result.append(('full_4x4','full',w))
    for row0 in (0,2):
        for col0 in (0,2):
            w=np.zeros(16,dtype=float)
            for r in range(row0,row0+2):
                for c in range(col0,col0+2):
                    w[4*r+c]=0.25
            result.append(('subblock_2x2',f'r{row0}_c{col0}',w))
    return result


def aggregate_uncertainty_rows(
        seed:int, gamma:float, model_label:str, y_test:np.ndarray,
        oracle:dict, prediction:dict) -> list[dict]:
    """Each subblock's weight vector is evaluated jointly, not sitewise."""
    mean=prediction['mean'].detach().cpu().numpy()
    cov=prediction['cov'].detach().cpu().numpy()
    draws=prediction['draws'].detach().cpu().numpy()
    oracle_mean=oracle['mean']
    oracle_cov=oracle['cov']
    result=[]
    for scale,group,w in aggregate_weights():
        true_var=float(w@oracle_cov@w)
        fit_var=float(w@cov@w)
        observed=float(w@y_test)
        predictive_draws=draws@w
        lo,hi=np.quantile(predictive_draws,[.025,.975])
        signed=(fit_var-true_var)/true_var
        result.append(dict(seed=int(seed),gamma=float(gamma),model=model_label,
            scale=scale,group=group,observed_aggregate=observed,
            oracle_aggregate_mean=float(w@oracle_mean),fitted_aggregate_mean=float(w@mean),
            oracle_aggregate_var=true_var,fitted_aggregate_var=fit_var,
            signed_relative_variance_error=float(signed),
            absolute_relative_variance_error=float(abs(signed)),
            coverage_95=float(lo<=observed<=hi),interval_width_95=float(hi-lo)))
    return result


def is_complete(run_dir:Path, prefix:str, seed:int, gamma:float) -> bool:
    metadata_file=run_dir/f'{prefix}__metadata.json'
    if not metadata_file.exists() or not (run_dir/f'{prefix}__COMPLETE.txt').exists():
        return False
    try:
        metadata=json.loads(metadata_file.read_text())
        expected=dict(experiment_version=EXPERIMENT_VERSION,seed=int(seed),gamma=float(gamma),
                      run_mode=RUN_MODE,beta_handling='known_fixed_zero_generation_fit_oracle',
                      leroux_iterations=LEROUX_ITERS+LEROUX_REFINE_ITERS,
                      sdm_iterations=PSPLINE_ITERS+PSPLINE_REFINE_ITERS,
                      predictive_draws=PREDICTIVE_DRAWS,oracle_draws=ORACLE_KL_MC,
                      spectrum_draws=SPECTRUM_DRAWS,
                      selection_elbo_repeats=SELECTION_ELBO_REPEATS,
                      diag_elbo_repeats=DIAG_ELBO_REPEATS,
                      diag_spectrum_draws=DIAG_SPECTRUM_DRAWS,
                      diag_bootstrap_reps=DIAG_BOOTSTRAP_REPS,
                      adaptive_pspline_kwargs=ADAPTIVE_PSPLINE_KWARGS)
        files=[run_dir/f'{prefix}__predictive_metrics.csv',
               run_dir/f'{prefix}__covariance_metrics.csv',
               run_dir/f'{prefix}__spectral_metrics.csv',
               run_dir/f'{prefix}__aggregate_uncertainty.csv',
               run_dir/f'{prefix}__spectral_recovery.csv',
               run_dir/'sdm_two_start_selection.json',
               run_dir/'sdm_two_start_final_status.json',
               run_dir/'frozen_seed_input_sha256.txt']
        frozen_file=run_dir.parent.parent.parent/'frozen_datasets'/f'seed_{seed:05d}.npz'
        frozen_ok=(frozen_file.exists() and
                   hashlib.sha256(frozen_file.read_bytes()).hexdigest()==metadata.get('frozen_input_sha256'))
        budget_ok=(int(metadata.get('additional_continuation_blocks',-1))<=ADDITIONAL_CONTINUATION_BLOCKS)
        return (all(metadata.get(k)==v for k,v in expected.items()) and
                all(p.exists() for p in files) and frozen_ok and budget_ok)
    except (OSError,ValueError,TypeError,KeyError):
        return False


def _save_spectrum_figure(lam_fit,q_true,q_leroux,q_sdm,output:Path):
    fig,ax=plt.subplots(figsize=(8.0,5.0))
    ax.plot(lam_fit,q_true,label='True Leroux precision, indexed by warped eigenvalue',linewidth=2)
    ax.plot(lam_fit,q_leroux,label='Leroux fitted posterior mean',linewidth=1.7)
    ax.plot(lam_fit,q_sdm,label='SDM-CAR fitted posterior mean',linewidth=1.7)
    ax.set_xlabel('Fitted eigenvalue μ(γ)');ax.set_ylabel('Modal precision')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)


def run_one(seed:int,gamma:float,common:dict,output_dir:Path,resume_existing:bool=False):
    prefix=run_prefix(seed,gamma)
    run_dir=output_dir/'per_run'/f'seed_{seed:05d}'/f'gamma_{gamma_token(gamma)}'
    run_dir.mkdir(parents=True,exist_ok=True)
    frozen=seed_data(seed,output_dir,common)
    lam_true=frozen['lam_true'];U_true=frozen['U_true']
    lam_fit=power_warp(lam_true,gamma)
    if gamma==1.0:
        np.testing.assert_array_equal(lam_fit,lam_true)
    assert np.array_equal(lam_fit<=ZERO_TOL,lam_true<=ZERO_TOL)
    assert np.all(np.diff(lam_fit)>=-1e-10)
    np.testing.assert_allclose(lam_fit.max(),lam_true.max(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(U_true.T@U_true,np.eye(N),atol=1e-9)
    # y_test is never stored on the fitted model. The holdout mask is frozen.
    X=frozen['X']; y=frozen['y'];phi_true=frozen['phi_true']
    test_mask=frozen['test_mask'];train_idx=np.flatnonzero(~test_mask)
    test_idx=np.flatnonzero(test_mask)
    F_true=frozen['F_true']
    Sigma_true=covariance_from_spectrum(U_true,F_true)
    C_true=Sigma_true+SIGMA2_TRUE*np.eye(N)
    X_t=torch.as_tensor(X,dtype=torch.double)
    y_fit_t=torch.as_tensor(y,dtype=torch.double).clone()
    is_holdout_t=torch.as_tensor(test_mask,dtype=torch.bool)
    y_fit_t[is_holdout_t]=float('nan')
    U_t=torch.as_tensor(U_true,dtype=torch.double)
    lam_t=torch.as_tensor(lam_fit,dtype=torch.double)
    fixed_beta_t=torch.zeros(X_t.shape[1],dtype=torch.double)
    # One independent Leroux optimization and TWO independently initialized SDM fits.
    fit_leroux_dir=run_dir/'fits'/'leroux'
    fit_original_dir=run_dir/'fits'/'sdm_original'
    fit_curve_dir=run_dir/'fits'/'sdm_curve'
    print('    fitting Leroux-CAR',flush=True)
    lr=fit_two_stage(label='Leroux CAR',model_tag='leroux',seed=seed,
        lam_t=lam_t,X_t=X_t,y_fit_t=y_fit_t,U_t=U_t,is_holdout_t=is_holdout_t,
        fixed_beta_t=fixed_beta_t,run_dir=fit_leroux_dir,prefix=prefix,
        resume_existing=resume_existing,optimizer_seed=seed)
    print('    fitting SDM ORIGINAL start',flush=True)
    o=fit_two_stage(label='Adaptive P-spline SDM-CAR',model_tag='sdm',seed=seed,
        lam_t=lam_t,X_t=X_t,y_fit_t=y_fit_t,U_t=U_t,is_holdout_t=is_holdout_t,
        fixed_beta_t=fixed_beta_t,run_dir=fit_original_dir,prefix=prefix,
        resume_existing=resume_existing,sdm_start='original',optimizer_seed=seed)
    print('    fitting SDM CURVED start',flush=True)
    c=fit_two_stage(label='Adaptive P-spline SDM-CAR',model_tag='sdm',seed=seed,
        lam_t=lam_t,X_t=X_t,y_fit_t=y_fit_t,U_t=U_t,is_holdout_t=is_holdout_t,
        fixed_beta_t=fixed_beta_t,run_dir=fit_curve_dir,prefix=prefix,
        resume_existing=resume_existing,sdm_start='curve',optimizer_seed=curved_optimizer_seed(seed))
    selection=score_two_start_elbo(o['model'],c['model'],seed)
    enforce_selection_file(run_dir,selection)
    choice=selection['selected_start']
    sd=o if choice=='original' else c
    chosen_dir=fit_original_dir if choice=='original' else fit_curve_dir
    del o,c
    print(f"    selected SDM start={choice}, ELBO difference/site={selection['difference_per_site']:.8f}",flush=True)
    sd,sd_added=extend_if_not_converged(sd,model_tag='sdm',start=choice,seed=seed,
        X_t=X_t,y_fit_t=y_fit_t,lam_t=lam_t,U_t=U_t,is_holdout_t=is_holdout_t,
        fixed_beta_t=fixed_beta_t,fit_dir=chosen_dir,prefix=prefix)
    lr,lr_added=extend_if_not_converged(lr,model_tag='leroux',start='original',seed=seed,
        X_t=X_t,y_fit_t=y_fit_t,lam_t=lam_t,U_t=U_t,is_holdout_t=is_holdout_t,
        fixed_beta_t=fixed_beta_t,fit_dir=fit_leroux_dir,prefix=prefix)
    selection.update(selected_final_iteration=int(sd['final_iteration']),
        selected_convergence_pass=bool(sd['convergence_history'].iloc[-1]['pilot_persistent_pass']),
        selected_continuation_iterations=int(sd_added),
        leroux_final_iteration=int(lr['final_iteration']),
        leroux_convergence_pass=bool(lr['convergence_history'].iloc[-1]['pilot_persistent_pass']),
        leroux_continuation_iterations=int(lr_added))
    atomic_json(run_dir/'sdm_two_start_final_status.json',selection)
    for short,result in [('leroux',lr),('sdm',sd)]:
        result['convergence_history'].to_csv(run_dir/f'{prefix}__{short}_convergence.csv',index=False)
        result['precision_checkpoint_history'].to_csv(run_dir/f'{prefix}__{short}_precision_checkpoints.csv',index=False)
        result['parameter_checkpoint_history'].to_csv(run_dir/f'{prefix}__{short}_variational_parameters.csv',index=False)
    convergence_summary=pd.DataFrame([convergence_summary_row('Leroux CAR',lr['convergence_history']),
        convergence_summary_row('Adaptive P-spline SDM-CAR',sd['convergence_history'])])
    convergence_summary.insert(0,'seed',seed)
    convergence_summary.insert(1,'gamma',gamma)
    convergence_summary.to_csv(run_dir/f'{prefix}__convergence_summary.csv',index=False)
    save_elbo_diagnostics(leroux_result=lr,sdm_result=sd,run_dir=run_dir,prefix=prefix)
    save_checkpoint_convergence_plots(leroux_result=lr,sdm_result=sd,run_dir=run_dir,prefix=prefix)
    oracle=oracle_response_predictive(C_true=C_true,X=X,y=y,
        train_idx=train_idx,test_idx=test_idx,fixed_beta=np.zeros(X.shape[1]))
    oracle_samples=draw_oracle_samples(oracle['mean'],oracle['cov'],
        draws=ORACLE_KL_MC,seed=seed+40000)
    # Common fixed evaluation seeds across models AND gamma values.
    lr_pred=posterior_joint_response_prediction(lr['model'],num_mc=PREDICTIVE_DRAWS,seed=seed+41000)
    sd_pred=posterior_joint_response_prediction(sd['model'],num_mc=PREDICTIVE_DRAWS,seed=seed+41000)
    y_test=y[test_idx]
    lr_metrics=predictive_metrics(target=y_test,prediction=lr_pred,oracle=oracle,oracle_samples=oracle_samples)
    sd_metrics=predictive_metrics(target=y_test,prediction=sd_pred,oracle=oracle,oracle_samples=oracle_samples)
    true_phi_test=phi_true[test_idx]
    for metrics,result in [(lr_metrics,lr),(sd_metrics,sd)]:
        pred_phi=posterior_latent_spatial_mean(result['model'],num_mc=PREDICTIVE_DRAWS,seed=seed+42000)
        metrics['latent_phi_RMSE']=float(np.sqrt(np.mean((pred_phi.detach().cpu().numpy()-true_phi_test)**2)))
    models=[('Leroux CAR',lr,lr_pred,lr_metrics),
            ('Adaptive P-spline SDM-CAR',sd,sd_pred,sd_metrics)]
    pd.DataFrame([dict(seed=seed,gamma=gamma,model=label,**metrics)
                  for label,_,_,metrics in models]).to_csv(
        run_dir/f'{prefix}__predictive_metrics.csv',index=False)
    q_true=((1-RHO_TRUE)+RHO_TRUE*lam_true)/TAU2_TRUE
    warp=best_affine_precision_discrepancy(lam_fit,q_true)
    pd.DataFrame([dict(seed=seed,gamma=gamma,**warp)]).to_csv(run_dir/f'{prefix}__warp_diagnostics.csv',index=False)
    # Posterior spectra share spatial eigenvectors with truth for every gamma.
    spectra={};sigma_rows=[];spectral_rows=[];aggregate_rows=[]
    for label,result,pred,metrics in models:
        draw_bank=posterior_spectrum_draws(result,draws=SPECTRUM_DRAWS,seed=seed+50000)
        summ=summarize_spectrum_draws(draw_bank)
        spectra[label]=summ
        this_cov=covariance_metrics_from_spectra(draw_bank,F_true,U_true,
            oracle['cov'],pred,lam_true>ZERO_TOL)
        sigma_rows.append(dict(seed=seed,gamma=gamma,model=label,**this_cov))
        s_metrics=spectral_recovery_metrics(F_true,summ['median'])
        q_draws=1.0/np.clip(draw_bank,1e-12,None)
        q_mean=q_draws.mean(axis=0)
        s_metrics['precision_posterior_mean_relative_L2']=float(np.linalg.norm(q_mean-q_true)/
                                                                np.linalg.norm(q_true))
        s_metrics['affine_nonaffinity_of_posterior_mean_precision']=float(
            affine_precision_projection(lam_fit,q_mean)['relative_l2'])
        covered=(F_true>=summ['q025'])&(F_true<=summ['q975'])
        s_metrics['spectral_pointwise_95_coverage']=float(covered.mean())
        spectral_rows.append(dict(seed=seed,gamma=gamma,model=label,**s_metrics))
        aggregate_rows.extend(aggregate_uncertainty_rows(seed,gamma,label,y_test,oracle,pred))
        spectra[label]['q_mean']=q_mean
    pd.DataFrame(sigma_rows).to_csv(run_dir/f'{prefix}__covariance_metrics.csv',index=False)
    pd.DataFrame(spectral_rows).to_csv(run_dir/f'{prefix}__spectral_metrics.csv',index=False)
    pd.DataFrame(aggregate_rows).to_csv(run_dir/f'{prefix}__aggregate_uncertainty.csv',index=False)
    lr_spec=spectra['Leroux CAR']; sd_spec=spectra['Adaptive P-spline SDM-CAR']
    pd.DataFrame(dict(seed=seed,gamma=gamma,mode=np.arange(N),lambda_true=lam_true,
        mu_fit=lam_fit,q_true=q_true,F_true=F_true,
        q_leroux_mean=lr_spec['q_mean'],q_sdm_mean=sd_spec['q_mean'],
        F_leroux_mean=lr_spec['mean'],F_sdm_mean=sd_spec['mean'],
        F_leroux_median=lr_spec['median'],F_sdm_median=sd_spec['median'],
        F_leroux_q025=lr_spec['q025'],F_leroux_q975=lr_spec['q975'],
        F_sdm_q025=sd_spec['q025'],F_sdm_q975=sd_spec['q975'])).to_csv(
        run_dir/f'{prefix}__spectral_recovery.csv',index=False)
    _save_spectrum_figure(lam_fit,q_true,lr_spec['q_mean'],sd_spec['q_mean'],
                          run_dir/f'{prefix}__precision_recovery.png')
    pd.DataFrame(dict(seed=seed,gamma=gamma,node=test_idx,y_true=y_test,phi_true=true_phi_test,
        oracle_mean=oracle['mean'],leroux_mean=lr_pred['mean'].detach().cpu().numpy(),
        sdm_mean=sd_pred['mean'].detach().cpu().numpy())).to_csv(
            run_dir/f'{prefix}__predictive_means.csv',index=False)
    # Lightweight covariance matrices for representative analyses. Matrices are
    # reconstructed from posterior-mean modal variances, not from a plugin draw.
    if gamma in (1.0,0.35) and seed in (PILOT_SEEDS[0],FINAL_SEEDS[0]):
        for label,spec,short,pred in [('Leroux CAR',lr_spec,'leroux',lr_pred),
                                      ('Adaptive P-spline SDM-CAR',sd_spec,'sdm',sd_pred)]:
            np.savez_compressed(run_dir/f'{prefix}__{short}_covariance_matrices.npz',
                Sigma_mean=covariance_from_spectrum(U_true,spec['mean']),
                predictive_cov=pred['cov'].detach().cpu().numpy(),
                oracle_predictive_cov=oracle['cov'],Sigma_true=Sigma_true)
    # Validate same exact frozen y/phi is used across all gamma conditions.
    frozen_path=output_dir/'frozen_datasets'/f'seed_{seed:05d}.npz'
    frozen_hash=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    (run_dir/'frozen_seed_input_sha256.txt').write_text(frozen_hash+'\n',encoding='utf-8')
    metadata=dict(experiment_version=EXPERIMENT_VERSION,seed=int(seed),gamma=float(gamma),
        run_mode=RUN_MODE,beta_handling='known_fixed_zero_generation_fit_oracle',
        beta_true=BETA_TRUE,sigma2_true=SIGMA2_TRUE,rho_true=RHO_TRUE,tau2_true=TAU2_TRUE,
        n=N,n_train=128,n_holdout=16,graph_type='12x12 combinatorial rook Laplacian',
        same_eigenvectors=True,same_frozen_data_across_gammas=True,
        changed_coordinate_only='lambda_fit=lambda_max*(lambda_true/lambda_max)**gamma',
        full_latent_zero_mode_retained=True,
        leroux_iterations=LEROUX_ITERS+LEROUX_REFINE_ITERS,
        sdm_iterations=PSPLINE_ITERS+PSPLINE_REFINE_ITERS,
        leroux_final_iteration=lr['final_iteration'],
        sdm_selected_final_iteration=sd['final_iteration'],
        leroux_convergence_pass=bool(selection['leroux_convergence_pass']),
        sdm_convergence_pass=bool(selection['selected_convergence_pass']),
        selected_sdm_start=choice,predictive_draws=PREDICTIVE_DRAWS,
        oracle_draws=ORACLE_KL_MC,spectrum_draws=SPECTRUM_DRAWS,
        selection_elbo_repeats=SELECTION_ELBO_REPEATS,
        diag_elbo_repeats=DIAG_ELBO_REPEATS,
        diag_spectrum_draws=DIAG_SPECTRUM_DRAWS,
        diag_bootstrap_reps=DIAG_BOOTSTRAP_REPS,
        additional_continuation_blocks=ADDITIONAL_CONTINUATION_BLOCKS,
        continuation_stop_rule=CONTINUATION_STOP_RULE,
        protocol_implementation=PROTOCOL_IMPLEMENTATION,
        frozen_input_sha256=frozen_hash,
        warp_best_affine_precision_relative_L2=warp['best_affine_precision_relative_L2'],
        adaptive_pspline_kwargs=ADAPTIVE_PSPLINE_KWARGS)
    atomic_json(run_dir/f'{prefix}__metadata.json',metadata)
    (run_dir/f'{prefix}__COMPLETE.txt').write_text('Experiment 3 completed with selected model fit.\n',encoding='utf-8')
    del lr,sd


def paired_ci(values:np.ndarray) -> dict:
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    n=len(a)
    mean=float(a.mean()) if n else float('nan')
    sd=float(a.std(ddof=1)) if n>1 else float('nan')
    crit=float(student_t.ppf(.975,n-1)) if n>1 else float('nan')
    half=crit*sd/np.sqrt(n) if n>1 else float('nan')
    return dict(n_replications=n,mean=mean,sd=sd,CI95_lower=mean-half,CI95_upper=mean+half)


def aggregate_for_seed(aggs:pd.DataFrame) -> pd.DataFrame:
    """Average four subblock scores WITHIN seed; do not pseudoreplicate."""
    keys=['seed','gamma','model','scale']
    metrics=['signed_relative_variance_error','absolute_relative_variance_error',
             'coverage_95','interval_width_95']
    return aggs.groupby(keys,as_index=False)[metrics].mean()


def summarize_numeric_by_group(df:pd.DataFrame,group_cols:list[str]) -> pd.DataFrame:
    skip=set(group_cols)|{'seed','group','selected_sdm_start'}
    num=[c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    out=[]
    for grouping,part in df.groupby(group_cols,sort=True):
        if not isinstance(grouping,tuple):grouping=(grouping,)
        base=dict(zip(group_cols,grouping))
        for col in num:
            vals=part[col].dropna().to_numpy(dtype=float)
            if vals.size:
                colname='metric' if 'metric' not in base else 'value_column'
                out.append(dict(**base,**{colname:col},**paired_ci(vals)))
    return pd.DataFrame(out)


def paired_model_comparison(df:pd.DataFrame,group_cols:list[str]) -> pd.DataFrame:
    """SDM - Leroux per matched seed and gamma; CIs are across datasets."""
    exclude=set(group_cols)|{'seed','model','group'}
    numeric=[c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    rows=[]
    for group,part in df.groupby(group_cols,sort=True):
        if not isinstance(group,tuple):group=(group,)
        base=dict(zip(group_cols,group))
        l=part.loc[part.model=='Leroux CAR'].set_index('seed')
        s=part.loc[part.model=='Adaptive P-spline SDM-CAR'].set_index('seed')
        common=l.index.intersection(s.index)
        for col in numeric:
            diffs=(s.loc[common,col].astype(float)-l.loc[common,col].astype(float)).to_numpy(dtype=float)
            rows.append(dict(**base,metric=col,comparison='SDM_minus_Leroux',**paired_ci(diffs)))
    return pd.DataFrame(rows)


def paired_degradation(df:pd.DataFrame) -> pd.DataFrame:
    """Within-seed, same model: M(gamma)-M(1); then SDM-Leroux contrast."""
    numeric=['oracle_conditional_KL_per_site','joint_response_NLPD_per_site',
             'response_RMSE','joint_predictive_cov_relative_frobenius',
             'positive_latent_cov_relative_frobenius','full_latent_cov_relative_frobenius']
    if df.empty:return pd.DataFrame()
    rows=[]
    for (seed,model),part in df.groupby(['seed','model']):
        baseline=part[part.gamma==1.0]
        if len(baseline)!=1:continue
        baseline=baseline.iloc[0]
        for _,record in part.iterrows():
            for metric in numeric:
                if metric in part:
                    rows.append(dict(seed=seed,model=model,gamma=record.gamma,
                          metric=metric,degradation=float(record[metric]-baseline[metric])))
    return pd.DataFrame(rows)


def plot_metric_by_gamma(df:pd.DataFrame,metric:str,output:Path,ylabel:str|None=None):
    if df.empty or metric not in df:return
    fig,ax=plt.subplots(figsize=(7.0,4.6))
    for name,part in df.groupby('model',sort=False):
        summ=part.groupby('gamma')[metric].agg(['mean','std']).sort_index()
        x=summ.index.to_numpy(dtype=float);mean=summ['mean'].to_numpy(dtype=float)
        sd=summ['std'].fillna(0).to_numpy(dtype=float)
        ax.plot(x,mean,marker='o',label=name,linewidth=1.8)
        ax.fill_between(x,mean-sd,mean+sd,alpha=.12)
    ax.set_xlabel('Warp exponent γ');ax.set_ylabel(ylabel or metric)
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)


def build_cross_run_outputs(output_dir:Path,requested_seeds:list[int],gammas:list[float]):
    table_dir=output_dir/'master_tables';plot_dir=output_dir/'summary_plots'
    table_dir.mkdir(exist_ok=True);plot_dir.mkdir(exist_ok=True)
    files={'predictive':'__predictive_metrics.csv','covariance':'__covariance_metrics.csv',
           'spectral':'__spectral_metrics.csv','aggregates':'__aggregate_uncertainty.csv',
           'warp':'__warp_diagnostics.csv','convergence':'__convergence_summary.csv'}
    frames={}
    allowed=set(requested_seeds)
    for tag,ending in files.items():
        paths=sorted((output_dir/'per_run').rglob('*'+ending))
        if not paths:continue
        f=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
        f=f.loc[f.seed.astype(int).isin(allowed)&f.gamma.round(6).isin(np.round(gammas,6))].copy()
        if f.empty:continue
        if f.duplicated(subset=['seed','gamma']+(['model'] if 'model' in f else [])+
                         (['scale','group'] if tag=='aggregates' else [])).any():
            raise RuntimeError(f'Duplicate data rows in {tag}; investigate before summaries.')
        f.to_csv(table_dir/f'all_runs__{tag}.csv',index=False)
        if tag=='aggregates':
            f=aggregate_for_seed(f)
            f.to_csv(table_dir/'per_seed__aggregates.csv',index=False)
        frames[tag]=f
        groups=(['gamma']+(['model'] if 'model' in f else [])+
                (['scale'] if tag=='aggregates' else []))
        summarize_numeric_by_group(f,groups).to_csv(
            table_dir/f'summary__{tag}.csv',index=False)
        if 'model' in f:
            paired_model_comparison(f,[x for x in groups if x!='model']).to_csv(
                table_dir/f'paired_SDM_minus_Leroux__{tag}.csv',index=False)
    if 'predictive' not in frames or 'covariance' not in frames:return
    combo=frames['predictive'].merge(frames['covariance'],on=['seed','gamma','model'],validate='one_to_one')
    deg=paired_degradation(combo)
    if not deg.empty:
        deg.to_csv(table_dir/'per_seed__degradation_from_gamma1.csv',index=False)
        summarize_numeric_by_group(deg,['gamma','model','metric']).to_csv(
            table_dir/'summary__degradation_from_gamma1.csv',index=False)
        by_seed=deg.pivot_table(index=['seed','gamma','metric'],columns='model',values='degradation')
        if {'Adaptive P-spline SDM-CAR','Leroux CAR'}.issubset(by_seed.columns):
            by_seed['difference_in_degradation_SDM_minus_Leroux']=(
                by_seed['Adaptive P-spline SDM-CAR']-by_seed['Leroux CAR'])
            contrast=by_seed.reset_index()[['seed','gamma','metric','difference_in_degradation_SDM_minus_Leroux']]
            contrast.to_csv(table_dir/'per_seed__difference_in_degradation.csv',index=False)
            summarize_numeric_by_group(contrast,['gamma','metric']).to_csv(
                table_dir/'summary__difference_in_degradation.csv',index=False)
    targets=[('response_RMSE',frames['predictive']),
             ('oracle_conditional_KL_per_site',frames['predictive']),
             ('joint_response_NLPD_per_site',frames['predictive']),
             ('positive_latent_cov_relative_frobenius',frames['covariance']),
             ('full_latent_cov_relative_frobenius',frames['covariance']),
             ('joint_predictive_cov_relative_frobenius',frames['covariance'])]
    for metric,frame in targets:
        plot_metric_by_gamma(frame,metric,plot_dir/f'{metric}__vs_gamma.png')
    if 'aggregates' in frames:
        for scale in ['full_4x4','subblock_2x2']:
            frame=frames['aggregates'][frames['aggregates']['scale']==scale]
            plot_metric_by_gamma(frame,'absolute_relative_variance_error',
                 plot_dir/f'aggregate_{scale}_variance_error__vs_gamma.png',
                 ylabel='Mean absolute relative predictive-variance error')
    print('    Summary tables:',table_dir,'plots:',plot_dir,flush=True)


def parse_args():
    p=argparse.ArgumentParser(description='Experiment 3: fixed-eigenvector spectral warp, fixed beta, two-start VI.')
    p.add_argument('--mode',choices=('smoke','pilot','final'),default='smoke',
                   help='smoke = short functional test; pilot/final use full locked VI budget.')
    p.add_argument('--project-root',type=Path,default=DEFAULT_PROJECT_ROOT)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--seeds',type=int,nargs='+',default=None)
    p.add_argument('--gammas',type=float,nargs='+',default=None)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--overwrite-existing',action='store_true')
    p.add_argument('--keep-all-checkpoints',action='store_true')
    p.add_argument('--extra-continuation-blocks',type=int,default=0,
                   help='Additional 2000-step stability continuations; document any amendment.')
    p.add_argument('--continuation-stop-rule',choices=('block-terminal','first-persistent'),
                   default='block-terminal',
                   help=('block-terminal retains v1 behavior. first-persistent saves and stops '
                         'at the first qualifying 250-step checkpoint DURING continuation.'))
    return p.parse_args()


def main():
    global RUN_MODE, KEEP_ALL_CHECKPOINTS, SELECTION_ELBO_REPEATS
    global LEROUX_ITERS,LEROUX_REFINE_ITERS,PSPLINE_ITERS,PSPLINE_REFINE_ITERS
    global SPECTRUM_DRAWS,PREDICTIVE_DRAWS,ORACLE_KL_MC
    global DIAG_ELBO_REPEATS,DIAG_SPECTRUM_DRAWS,DIAG_BOOTSTRAP_REPS,SAVE_CHECKPOINT_EVERY
    global LerouxCARFilterFullVI,AdaptivePrecisionPSplineFullVI,SpectralCAR_HoldoutVI
    global ADDITIONAL_CONTINUATION_BLOCKS, CONTINUATION_STOP_RULE
    args=parse_args();RUN_MODE=args.mode
    if args.resume and args.overwrite_existing:
        raise ValueError('Cannot combine --resume and --overwrite-existing.')
    if not 0<=args.extra_continuation_blocks<=4:
        raise ValueError('--extra-continuation-blocks must be 0..4')
    ADDITIONAL_CONTINUATION_BLOCKS=args.extra_continuation_blocks
    CONTINUATION_STOP_RULE=args.continuation_stop_rule
    KEEP_ALL_CHECKPOINTS=args.keep_all_checkpoints
    if RUN_MODE=='smoke':
        LEROUX_ITERS=2;LEROUX_REFINE_ITERS=2
        PSPLINE_ITERS=2;PSPLINE_REFINE_ITERS=2
        SPECTRUM_DRAWS=16;PREDICTIVE_DRAWS=16;ORACLE_KL_MC=32
        DIAG_ELBO_REPEATS=4;DIAG_SPECTRUM_DRAWS=8;DIAG_BOOTSTRAP_REPS=20
        SAVE_CHECKPOINT_EVERY=250;SELECTION_ELBO_REPEATS=8
    seeds=(args.seeds if args.seeds is not None else
           ([9991] if RUN_MODE=='smoke' else PILOT_SEEDS if RUN_MODE=='pilot' else FINAL_SEEDS))
    gammas=(args.gammas if args.gammas is not None else
            ([1.0,0.35] if RUN_MODE=='smoke' else DEFAULT_GAMMAS))
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Seeds must be unique and nonempty.')
    if not gammas or len(set(gammas))!=len(gammas) or any(not 0<g<=1 for g in gammas):
        raise ValueError('Distinct power warp exponents must be in (0,1].')
    if min(abs(g-1.0) for g in gammas)>1e-10:
        print('WARNING: gamma=1 is absent; degradation-from-baseline results cannot be calculated.',flush=True)
    project_root=args.project_root.resolve()
    if str(project_root) not in sys.path:sys.path.insert(0,str(project_root))
    from sdmcar.filters import LerouxCARFilterFullVI as _Leroux
    from sdmcar.filters import AdaptivePrecisionPSplineFullVI as _SDM
    from sdmcar.models_holdout_fixed_beta import SpectralCAR_HoldoutVI as _VI
    LerouxCARFilterFullVI=_Leroux;AdaptivePrecisionPSplineFullVI=_SDM
    SpectralCAR_HoldoutVI=_VI
    if RUN_MODE in ('pilot','final'):
        if tuple(DEFAULT_GAMMAS)!=tuple(gammas):
            print('WARNING: non-default gamma grid; record this departure in methods.',flush=True)
        if not FIX_SIGMA2 or BETA_TRUE!=0 or NUM_MC!=4:
            raise RuntimeError('Locked beta, sigma2, and VI MC settings were altered.')
    output_dir=(args.output_dir if args.output_dir else
                Path(f'experiment3_N144_{RUN_MODE}_power_warp_two_start')).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    config_file=output_dir/'run_configuration.json'
    config=dict(experiment_version=EXPERIMENT_VERSION,run_mode=RUN_MODE,seeds=list(seeds),
        gammas=list(gammas),additional_continuation_blocks=ADDITIONAL_CONTINUATION_BLOCKS,
        vi_mc=NUM_MC,leroux_stage1_iterations=LEROUX_ITERS,leroux_refine_iterations=LEROUX_REFINE_ITERS,
        sdm_stage1_iterations=PSPLINE_ITERS,sdm_refine_iterations=PSPLINE_REFINE_ITERS,
        spectrum_draws=SPECTRUM_DRAWS,predictive_draws=PREDICTIVE_DRAWS,oracle_draws=ORACLE_KL_MC,
        project_root=str(project_root))
    if config_file.exists():
        old=json.loads(config_file.read_text())
        ignore={'seeds','gammas','additional_continuation_blocks'}
        comparable={k:v for k,v in old.items() if k not in ignore}
        new_comparable={k:v for k,v in config.items() if k not in ignore}
        if comparable!=new_comparable:
            raise RuntimeError('Output directory contains a different configuration; use another directory.')
        # Existing v1 directories keep their original run_configuration.json
        # and single legacy amendment unchanged. Track all later changes in
        # a separate append-only JSON history; repeated --resume is idempotent.
        initial_budget = int(old['additional_continuation_blocks'])
        latest_budget = initial_budget
        latest_rule = 'block-terminal'
        legacy_path = output_dir / 'numerical_budget_amendment.json'
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text())
            if int(legacy['original_additional_blocks']) != initial_budget:
                raise RuntimeError('Legacy budget amendment conflicts with initial configuration.')
            latest_budget = int(legacy['amended_additional_blocks'])
        protocol_path = output_dir / 'numerical_protocol_amendments.json'
        protocol_history = []
        if protocol_path.exists():
            protocol_history = json.loads(protocol_path.read_text())
            if not isinstance(protocol_history, list):
                raise RuntimeError('Protocol history must be a JSON list.')
            for rec in protocol_history:
                if (rec['previous_extra_blocks'] != latest_budget or
                        rec['previous_stop_rule'] != latest_rule):
                    raise RuntimeError('Protocol amendment history has an inconsistent chain.')
                latest_budget = int(rec['new_extra_blocks'])
                latest_rule = rec['new_stop_rule']
        if ADDITIONAL_CONTINUATION_BLOCKS < latest_budget:
            raise RuntimeError('Cannot reduce the existing authorized continuation cap.')
        if latest_rule == 'first-persistent' and CONTINUATION_STOP_RULE != latest_rule:
            raise RuntimeError('Cannot revert the amended stopping rule in this directory.')
        if (ADDITIONAL_CONTINUATION_BLOCKS > latest_budget or
                CONTINUATION_STOP_RULE != latest_rule):
            if CONTINUATION_STOP_RULE not in ('block-terminal', 'first-persistent'):
                raise ValueError('Unsupported continuation stopping rule.')
            rec = dict(previous_extra_blocks=latest_budget,
                       new_extra_blocks=ADDITIONAL_CONTINUATION_BLOCKS,
                       previous_stop_rule=latest_rule,
                       new_stop_rule=CONTINUATION_STOP_RULE,
                       continuation_block=CONTINUATION_BLOCK,
                       diagnostic_spacing=DIAG_EVERY,
                       unchanged=['data','eigenvalue_warp','prior','variational_family',
                                  'two_stage_budget','selection','learning_rates',
                                  'four_diagnostic_tolerances','evaluation_draws'],
                       rationale='Repeated block-terminal false negatives from intermittent '
                                 'precision changes; permit exact-state checkpoint stopping '
                                 'without changing the four pass criteria.',
                       disclosure='Post-pilot and mid-final numerical protocol amendment; '
                                  'report old and new stopping-rule fits and sensitivity.')
            protocol_history.append(rec)
            atomic_json(protocol_path, protocol_history)
            print('    Recorded protocol amendment:',rec,flush=True)
    else:
        atomic_json(config_file,config)
        if CONTINUATION_STOP_RULE != 'block-terminal':
            atomic_json(output_dir/'numerical_protocol_amendments.json', [dict(
                previous_extra_blocks=ADDITIONAL_CONTINUATION_BLOCKS,
                new_extra_blocks=ADDITIONAL_CONTINUATION_BLOCKS,
                previous_stop_rule='block-terminal',new_stop_rule=CONTINUATION_STOP_RULE,
                continuation_block=CONTINUATION_BLOCK,diagnostic_spacing=DIAG_EVERY,
                unchanged=['data','eigenvalue_warp','prior','variational_family',
                           'two_stage_budget','selection','learning_rates',
                           'four_diagnostic_tolerances','evaluation_draws'],
                rationale='Predeclared continuation stopping rule for this output directory.',
                disclosure='Document checkpoint stopping rule in simulation methods.')])
    W=build_rook_adjacency(N_ROWS,N_COLS);L=graph_laplacian(W)
    lam,U=np.linalg.eigh(L)
    lam[np.abs(lam)<ZERO_TOL]=0.0
    lam=np.clip(lam,0,None)
    if np.sum(lam<=ZERO_TOL)!=1:raise RuntimeError('Expected one constant eigenmode.')
    if np.sum(holdout_block_mask(N_ROWS,N_COLS,HOLDOUT_SIDE))!=16:
        raise RuntimeError('Expected 16 held-out sites.')
    F_true=TAU2_TRUE/((1-RHO_TRUE)+RHO_TRUE*lam)
    common=dict(L_true=L,lam_true=lam,U_true=U,F_true=F_true,
                test_mask=holdout_block_mask(N_ROWS,N_COLS,HOLDOUT_SIDE))
    # Precompute deterministic, data-independent warp-severity diagnostics.
    warp_rows=[]
    q_true=1.0/F_true
    for gamma in gammas:
        mu=power_warp(lam,gamma)
        severity=best_affine_precision_discrepancy(mu,q_true)
        warp_rows.append(dict(gamma=gamma,**severity))
        print('gamma',gamma,'affine nonaffinity',severity['best_affine_precision_relative_L2'],flush=True)
    (output_dir/'master_tables').mkdir(exist_ok=True)
    pd.DataFrame(warp_rows).to_csv(output_dir/'master_tables'/'warp_severity.csv',index=False)
    print('Experiment 3',RUN_MODE,'seeds:',seeds,'gammas:',gammas,flush=True)
    for seed in seeds:
        seed_data(seed,output_dir,common)
        for gamma in gammas:
            prefix=run_prefix(seed,gamma)
            run_dir=output_dir/'per_run'/f'seed_{seed:05d}'/f'gamma_{gamma_token(gamma)}'
            print(f'=== seed={seed} gamma={gamma} ===',flush=True)
            if args.resume and is_complete(run_dir,prefix,seed,gamma):
                print('    Validated COMPLETE: skipping',flush=True)
                continue
            if run_dir.exists() and args.overwrite_existing:
                shutil.rmtree(run_dir)
            elif run_dir.exists() and (run_dir/f'{prefix}__COMPLETE.txt').exists():
                raise RuntimeError('Existing COMPLETE marker is incompatible. Use a new output directory '
                                   'or explicitly --overwrite-existing (will delete run outputs).')
            elif run_dir.exists() and not args.resume and any(run_dir.iterdir()):
                raise RuntimeError('Partial run directory exists: supply --resume or --overwrite-existing.')
            start=time.perf_counter()
            run_one(int(seed),float(gamma),common,output_dir,resume_existing=args.resume)
            print('    complete, elapsed_seconds=',round(time.perf_counter()-start,1),flush=True)
    print('Building across-seed/gamma summaries...',flush=True)
    build_cross_run_outputs(output_dir,list(seeds),list(gammas))
    print('EXPERIMENT 3 FINISHED. Files:',output_dir,flush=True)


if __name__=='__main__':
    main()
