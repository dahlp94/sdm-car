# examples/run_deterministic_surface_benchmark_vi.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

try:
    from libpysal.weights import lat2W, KNN, DistanceBand, Kernel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This script requires libpysal. Install with: pip install libpysal"
    ) from e

from sdmcar.models import SpectralCAR_FullVI

# Importing examples.benchmarks populates the filter registry.
from examples.benchmarks.registry import get_filter_spec, available_filters
import examples.benchmarks  # noqa: F401


torch.set_default_dtype(torch.double)


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def seconds_to_str(x: float) -> str:
    if x < 60:
        return f"{x:.2f}s"
    if x < 3600:
        return f"{x / 60:.2f}min"
    return f"{x / 3600:.2f}hr"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_string_int(s: str) -> int:
    """Deterministic small integer hash independent of PYTHONHASHSEED."""
    out = 0
    for ch in s:
        out = (out * 131 + ord(ch)) % 1_000_000_007
    return out


def rmse_np(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


def mae_np(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.mean(np.abs(y_true - y_hat)))


def r2_np(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y_true - y_hat) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def resolve_fixed_tokens(fixed: dict, eps_car: float) -> dict:
    """Allow benchmark case specs to use the string token 'eps_car'."""
    out = {}
    for k, v in fixed.items():
        out[k] = eps_car if v == "eps_car" else v
    return out


@dataclass(frozen=True)
class ModelSpec:
    label: str
    filter_name: str
    case_id: str


def parse_model_specs(items: list[str]) -> list[ModelSpec]:
    """
    Parse model specs of the form:
        label=filter:case
    Example:
        leroux=leroux:learn_rho
        sdm_bspline=anchored_bspline:k8_medium
    """
    specs: list[ModelSpec] = []
    for item in items:
        if "=" not in item or ":" not in item:
            raise ValueError(
                "Each --model-specs entry must look like label=filter:case. "
                f"Got: {item!r}"
            )
        label, rhs = item.split("=", 1)
        filter_name, case_id = rhs.split(":", 1)
        specs.append(ModelSpec(label=label, filter_name=filter_name, case_id=case_id))
    return specs


# ---------------------------------------------------------------------
# Grid, deterministic surfaces, and mean design
# ---------------------------------------------------------------------
def make_grid(nx: int, ny: int, *, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    return coords, coords.detach().cpu().numpy()


def standardize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = float(v.std())
    if sd <= 1e-12:
        raise ValueError("Cannot standardize a constant vector.")
    return (v - v.mean()) / sd


def make_surface(coords_np: np.ndarray, kind: str) -> np.ndarray:
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    if kind == "smooth":
        f = np.sin(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y)

    elif kind == "bumps":
        bump1 = 2.0 * np.exp(-((x - 0.30) ** 2 + (y - 0.35) ** 2) / 0.010)
        bump2 = -1.5 * np.exp(-((x - 0.75) ** 2 + (y - 0.70) ** 2) / 0.005)
        f = np.sin(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y) + bump1 + bump2

    elif kind == "ridge":
        f = np.sin(2.0 * np.pi * x) + np.tanh(20.0 * (x - 0.55))

    elif kind == "multiscale":
        f = (
            np.sin(2.0 * np.pi * x)
            + 0.75 * np.sin(8.0 * np.pi * x) * np.sin(6.0 * np.pi * y)
            + 0.50 * np.exp(-((x - 0.60) ** 2 + (y - 0.40) ** 2) / 0.010)
        )

    elif kind == "checker_soft":
        f = np.tanh(8.0 * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y))

    elif kind == "anisotropic_wave":
        f = (
            np.sin(10.0 * np.pi * x + 2.0 * np.pi * y)
            + 0.5 * np.cos(3.0 * np.pi * x - 12.0 * np.pi * y)
        )

    elif kind == "localized_hotspots":
        f = (
            2.5 * np.exp(-((x - 0.20) ** 2 + (y - 0.25) ** 2) / 0.004)
            - 2.0 * np.exp(-((x - 0.75) ** 2 + (y - 0.65) ** 2) / 0.008)
            + 1.5 * np.exp(-((x - 0.55) ** 2 + (y - 0.20) ** 2) / 0.002)
            + 0.5 * np.sin(4.0 * np.pi * x)
        )

    elif kind == "sharp_boundary":
        boundary = np.tanh(40.0 * (y - 0.45 - 0.15 * np.sin(2.0 * np.pi * x)))
        f = boundary + 0.4 * np.sin(6.0 * np.pi * x) * np.sin(4.0 * np.pi * y)

    elif kind == "nonstationary_frequency":
        f = np.sin(2.0 * np.pi * x * (2.0 + 8.0 * y)) + 0.5 * np.cos(10.0 * np.pi * y)

    elif kind == "trend_plus_texture":
        trend = 2.0 * (x - 0.5) + 1.5 * (y - 0.5) ** 2
        texture = 0.4 * np.sin(12.0 * np.pi * x) * np.sin(10.0 * np.pi * y)
        bump = 1.2 * np.exp(-((x - 0.65) ** 2 + (y - 0.30) ** 2) / 0.006)
        f = trend + texture + bump

    elif kind == "rings":
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        f = np.sin(18.0 * np.pi * r) * np.exp(-3.0 * r)
    
    elif kind == "paper_function1":
        f = (
            10.0
            + 15.0 * np.log1p(x) / (1.0 + y**2)
            + 11.0 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
            + 5.0 * x**2 * (1.0 - y**2)
        )
    
    elif kind == "hard_nonstationary_boundary":
        # Large-scale nonlinear drift
        trend = (
            1.2 * (x - 0.5)
            - 0.8 * (y - 0.5)
            + 1.5 * (x - 0.5) * (y - 0.5)
            - 1.0 * (y - 0.5) ** 2
        )

        # Curved sharp boundary / regime change
        boundary_curve = 0.42 + 0.16 * np.sin(2.0 * np.pi * x + 0.7)
        boundary = 1.4 * np.tanh(45.0 * (y - boundary_curve))

        # Nonstationary oscillation: frequency increases with x and y
        chirp = (
            0.75
            * np.sin(2.0 * np.pi * (2.0 * x + 7.0 * x**2 + 2.5 * y))
            * (0.3 + 1.2 * y)
        )

        # Directional anisotropic waves
        anisotropic = (
            0.55 * np.sin(10.0 * np.pi * x + 3.0 * np.pi * y)
            + 0.35 * np.cos(4.0 * np.pi * x - 12.0 * np.pi * y)
        )

        # Localized positive and negative anomalies
        hotspots = (
            2.2 * np.exp(-((x - 0.22) ** 2 + (y - 0.72) ** 2) / 0.0035)
            - 2.0 * np.exp(-((x - 0.72) ** 2 + (y - 0.33) ** 2) / 0.0045)
            + 1.5 * np.exp(-((x - 0.58) ** 2 + (y - 0.62) ** 2) / 0.0025)
            - 1.2 * np.exp(-((x - 0.38) ** 2 + (y - 0.22) ** 2) / 0.0030)
        )

        # Ring-like feature not aligned with the lattice
        r = np.sqrt((x - 0.67) ** 2 + (y - 0.68) ** 2)
        rings = 0.65 * np.sin(24.0 * np.pi * r) * np.exp(-7.0 * r)

        f = trend + boundary + chirp + anisotropic + hotspots + rings

    else:
        raise ValueError(
            f"Unknown surface kind: {kind}. "
            "Available: smooth, bumps, ridge, multiscale, checker_soft, "
            "anisotropic_wave, localized_hotspots, sharp_boundary, "
            "nonstationary_frequency, trend_plus_texture, rings"
        )

    return standardize(f)


def make_design(coords: torch.Tensor, kind: str) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    ones = torch.ones_like(x)

    if kind == "intercept":
        cols = [ones]
    elif kind == "coord":
        cols = [ones, x, y]
    elif kind == "poly2":
        cols = [ones, x, y, x**2, y**2, x * y]
    else:
        raise ValueError("--mean-model must be one of: intercept, coord, poly2")

    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------
# PySAL graph construction and Laplacian eigendecomposition
# ---------------------------------------------------------------------
def build_pysal_weights(coords_np: np.ndarray, nx: int, ny: int, kind: str):
    if kind == "rook":
        w = lat2W(nrows=nx, ncols=ny, rook=True)
    elif kind == "queen":
        w = lat2W(nrows=nx, ncols=ny, rook=False)
    elif kind == "knn4":
        w = KNN.from_array(coords_np, k=4).symmetrize()
    elif kind == "knn8":
        w = KNN.from_array(coords_np, k=8).symmetrize()
    elif kind == "knn12":
        w = KNN.from_array(coords_np, k=12).symmetrize()
    elif kind == "distance":
        # For a 40x40 grid, nearest-neighbor spacing is about 1/39 ~= 0.026.
        # 0.06 connects local neighborhoods while remaining sparse.
        w = DistanceBand.from_array(
            coords_np,
            threshold=0.06,
            binary=True,
            silence_warnings=True,
        ).symmetrize()
    elif kind == "kernel":
        w = Kernel.from_array(
            coords_np,
            bandwidth=0.06,
            fixed=True,
            function="gaussian",
        ).symmetrize()
    else:
        raise ValueError(
            f"Unknown W kind: {kind}. "
            "Available: rook, queen, knn4, knn8, knn12, distance, kernel"
        )
    return w


def pysal_w_to_sparse_adjacency(w) -> sp.csr_matrix:
    try:
        A = w.to_sparse(fmt="csr")
    except Exception:
        A = w.sparse.tocsr()

    A = A.astype(float).tocsr()
    A = A.maximum(A.T)  # force symmetry for CAR
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def laplacian_from_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).reshape(-1)
    return sp.diags(d, format="csr") - A


def eigendecomp_laplacian(L: sp.csr_matrix, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    L_dense = torch.tensor(L.toarray(), dtype=torch.double, device=device)
    L_dense = 0.5 * (L_dense + L_dense.T)
    lam, U = torch.linalg.eigh(L_dense)
    # Clean small negative numerical noise.
    lam = torch.clamp(lam, min=0.0)
    return lam, U


def graph_summary(A: sp.csr_matrix, lam: torch.Tensor, surface: str, w_kind: str) -> dict:
    degree = np.asarray(A.sum(axis=1)).reshape(-1)
    n_zero = int(torch.sum(lam < 1e-10).item())
    return {
        "surface": surface,
        "w_kind": w_kind,
        "n": int(A.shape[0]),
        "nnz": int(A.nnz),
        "degree_min": float(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_max": float(degree.max()),
        "n_zero_eigenvalues": n_zero,
        "lambda_max": float(lam.max().item()),
    }


# ---------------------------------------------------------------------
# Spatial block CV
# ---------------------------------------------------------------------
def make_spatial_blocks(coords_np: np.ndarray, n_block_x: int, n_block_y: int) -> np.ndarray:
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    bx = np.floor(x * n_block_x).astype(int)
    by = np.floor(y * n_block_y).astype(int)
    bx = np.clip(bx, 0, n_block_x - 1)
    by = np.clip(by, 0, n_block_y - 1)
    return by * n_block_x + bx


def make_block_cv_folds(
    coords_np: np.ndarray,
    *,
    n_block_x: int,
    n_block_y: int,
    n_folds: int,
    seed: int,
) -> tuple[list[dict], np.ndarray]:
    block_id = make_spatial_blocks(coords_np, n_block_x=n_block_x, n_block_y=n_block_y)
    unique_blocks = np.unique(block_id)

    if n_folds > unique_blocks.size:
        raise ValueError("n_folds cannot exceed number of spatial blocks.")

    rng = np.random.default_rng(seed)
    shuffled = unique_blocks.copy()
    rng.shuffle(shuffled)
    fold_blocks = np.array_split(shuffled, n_folds)

    folds = []
    for k, blocks in enumerate(fold_blocks):
        blocks_set = set(blocks.tolist())
        test_mask = np.array([b in blocks_set for b in block_id], dtype=bool)
        train_mask = ~test_mask
        folds.append(
            {
                "fold": k,
                "test_blocks": sorted(blocks_set),
                "train_mask": train_mask,
                "test_mask": test_mask,
            }
        )
    return folds, block_id

def make_manual_block_fold(coords_np, *, n_block_x, n_block_y, heldout_blocks):
    block_id = make_spatial_blocks(
        coords_np,
        n_block_x=n_block_x,
        n_block_y=n_block_y,
    )

    heldout_blocks = set(heldout_blocks)
    test_mask = np.array([b in heldout_blocks for b in block_id], dtype=bool)
    train_mask = ~test_mask

    return [
        {
            "fold": 0,
            "test_blocks": sorted(heldout_blocks),
            "train_mask": train_mask,
            "test_mask": test_mask,
        }
    ], block_id

def make_random_cv_folds(n: int, *, n_folds: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, n_folds)
    folds = []
    for k, test_idx in enumerate(parts):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        folds.append(
            {
                "fold": k,
                "test_blocks": [],
                "train_mask": ~test_mask,
                "test_mask": test_mask,
            }
        )
    return folds


# ---------------------------------------------------------------------
# Filter/model construction and VI fitting
# ---------------------------------------------------------------------
def build_filter_from_model_spec(
    model_spec: ModelSpec,
    *,
    tau2_ref: float,
    eps_car: float,
    lam_max: float,
    device: torch.device,
):
    try:
        filter_spec = get_filter_spec(model_spec.filter_name)
    except Exception as e:
        raise ValueError(
            f"Unknown filter {model_spec.filter_name!r}. Available filters: {available_filters()}"
        ) from e

    if model_spec.case_id not in filter_spec.cases:
        raise ValueError(
            f"Unknown case {model_spec.case_id!r} for filter {model_spec.filter_name!r}. "
            f"Available cases: {list(filter_spec.cases.keys())}"
        )

    case_spec = filter_spec.cases[model_spec.case_id]
    fixed = resolve_fixed_tokens(case_spec.fixed, eps_car=eps_car)

    filter_module = case_spec.build_filter(
        tau2_true=tau2_ref,
        eps_car=eps_car,
        device=device,
        lam_max=lam_max,
        **fixed,
    )
    return filter_module, case_spec, fixed


@torch.no_grad()
def predict_from_vi_imputation(model: SpectralCAR_FullVI, test_mask_t: torch.Tensor) -> dict:
    """
    For the current holdout implementation, the most direct held-out prediction is
    the optimized missing response y_miss. We also report plugin eta = X beta + E[phi].
    """
    y_full_dynamic, _ = model.get_dynamic_y()
    y_pred_impute = y_full_dynamic[test_mask_t].detach().cpu().numpy()

    m_beta_plugin, _, _, _ = model.beta_posterior_plugin()
    mean_phi_plugin, _ = model.posterior_phi(mode="plugin")
    yhat_plugin_all = (model.X @ m_beta_plugin + mean_phi_plugin).detach().cpu().numpy()

    return {
        "y_pred_impute": y_pred_impute,
        "yhat_plugin_all": yhat_plugin_all,
        "beta_plugin": m_beta_plugin.detach().cpu().numpy(),
    }


def fit_one_vi_fold(
    *,
    model_spec: ModelSpec,
    X: torch.Tensor,
    y_true_t: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    train_mask_np: np.ndarray,
    test_mask_np: np.ndarray,
    sigma2_obs: float,
    fix_sigma2: bool,
    eps_car: float,
    tau2_ref: float,
    prior_beta_var: float,
    vi_iters: int,
    vi_mc: int,
    vi_lr: float,
    grad_clip: float,
    seed: int,
    verbose: bool,
) -> tuple[dict, SpectralCAR_FullVI]:
    set_all_seeds(seed)
    device = y_true_t.device

    train_mask_t = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask_np, dtype=torch.bool, device=device)

    # Never pass true held-out values to the training objective.
    # Fill them with the training mean; SpectralCAR_FullVI will replace them by y_miss.
    y_model = y_true_t.clone()
    y_model[test_mask_t] = y_true_t[train_mask_t].mean()

    filter_module, case_spec, fixed = build_filter_from_model_spec(
        model_spec,
        tau2_ref=tau2_ref,
        eps_car=eps_car,
        lam_max=float(lam.max().item()),
        device=device,
    )

    prior_V0 = prior_beta_var * torch.eye(X.shape[1], dtype=torch.double, device=device)
    fixed_sigma2 = sigma2_obs if fix_sigma2 else None

    model = SpectralCAR_FullVI(
        X=X,
        y=y_model,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(max(sigma2_obs, 1e-8)),
        log_std_log_sigma2=-2.3,
        num_mc=vi_mc,
        fixed_sigma2=fixed_sigma2,
        is_holdout=test_mask_t,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)
    t0 = time.perf_counter()
    elbo_hist = []

    for it in range(vi_iters):
        opt.zero_grad(set_to_none=True)
        elbo, stats = model.elbo()
        loss = -elbo
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at iter {it + 1}: {loss.item()} "
                f"for model={model_spec.label}"
            )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        elbo_hist.append(float(elbo.detach().cpu().item()))

        if verbose and ((it + 1) % 100 == 0 or it == 0 or it + 1 == vi_iters):
            print(
                f"      [VI {it+1:04d}/{vi_iters}] "
                f"ELBO={elbo.item():.2f} "
                f"loglik={stats['mc_loglik'].item():.2f} "
                f"KLbeta={stats['mc_kl_beta'].item():.2f} "
                f"KLfilt={stats['kl_filter'].item():.2f}"
            )

    seconds = time.perf_counter() - t0
    pred = predict_from_vi_imputation(model, test_mask_t)

    y_true_np = y_true_t.detach().cpu().numpy()
    test_mask = test_mask_np
    train_mask = train_mask_np

    y_pred_test = pred["y_pred_impute"]
    yhat_plugin_all = pred["yhat_plugin_all"]

    metrics = {
        "model_label": model_spec.label,
        "filter": model_spec.filter_name,
        "case": model_spec.case_id,
        "fixed": json.dumps(fixed),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_rmse": rmse_np(y_true_np[test_mask], y_pred_test),
        "test_mae": mae_np(y_true_np[test_mask], y_pred_test),
        "test_r2": r2_np(y_true_np[test_mask], y_pred_test),
        "test_rmse_plugin_eta": rmse_np(y_true_np[test_mask], yhat_plugin_all[test_mask]),
        "test_mae_plugin_eta": mae_np(y_true_np[test_mask], yhat_plugin_all[test_mask]),
        "train_rmse_plugin_eta": rmse_np(y_true_np[train_mask], yhat_plugin_all[train_mask]),
        "train_mae_plugin_eta": mae_np(y_true_np[train_mask], yhat_plugin_all[train_mask]),
        "vi_time_seconds": float(seconds),
        "vi_final_elbo": float(elbo_hist[-1]),
        "vi_initial_elbo": float(elbo_hist[0]),
        "vi_elbo_gain": float(elbo_hist[-1] - elbo_hist[0]),
        "beta_plugin": json.dumps(pred["beta_plugin"].tolist()),
    }
    return metrics, model


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def save_surface_image(values: np.ndarray, nx: int, ny: int, path: Path, title: str) -> None:
    arr = values.reshape(nx, ny)
    plt.figure(figsize=(5.0, 4.2))
    plt.imshow(arr.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
    plt.colorbar(shrink=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_mask_image(mask: np.ndarray, nx: int, ny: int, path: Path, title: str) -> None:
    save_surface_image(mask.astype(float), nx, ny, path, title)


def save_empirical_modal_energy(
    *,
    f_clean_t: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    path: Path,
    title: str,
) -> None:
    with torch.no_grad():
        z = U.T @ f_clean_t
        energy = z.pow(2).detach().cpu().numpy()
        lam_np = lam.detach().cpu().numpy()

    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(lam_np, np.clip(energy, 1e-12, None), s=8, alpha=0.7)
    plt.yscale("log")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"empirical modal energy $(u_j'f)^2$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_block_fill_comparison(
    *,
    original_surface: np.ndarray,
    test_mask: np.ndarray,
    filled_surfaces: dict[str, np.ndarray],
    nx: int,
    ny: int,
    path: Path,
    title: str,
) -> None:
    """
    Save a 1x4 panel showing:
      1) original observed surface
      2) observed surface with the held-out block removed
      3) Leroux-filled surface
      4) SDM-CAR-filled surface

    Expected model labels are 'leroux' and 'sdm_bspline'.
    If those names are not present, the function falls back to the first two
    model labels found in filled_surfaces.
    """
    order = []
    for name in ["leroux", "sdm_bspline"]:
        if name in filled_surfaces:
            order.append(name)
    for name in filled_surfaces:
        if name not in order:
            order.append(name)
    order = order[:2]

    if len(order) < 2:
        return

    original = np.asarray(original_surface, dtype=float)
    removed = original.copy()
    removed[test_mask] = np.nan

    panels = [
        ("Original surface", original),
        ("Surface with removed block", removed),
        (f"Filled block - {order[0]}", np.asarray(filled_surfaces[order[0]], dtype=float)),
        (f"Filled block - {order[1]}", np.asarray(filled_surfaces[order[1]], dtype=float)),
    ]

    vmin = float(np.nanmin(original))
    vmax = float(np.nanmax(original))

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), constrained_layout=True)

    for ax, (panel_title, vals) in zip(axes, panels):
        arr = vals.reshape(nx, ny).T
        arr_masked = np.ma.masked_invalid(arr)

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(
            arr_masked,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", default=str(Path("examples") / "figures" / "deterministic_surface_benchmark_vi"))
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--surfaces", nargs="+", default=["smooth", "bumps", "multiscale"])
    parser.add_argument("--w-kinds", nargs="+", default=["rook", "queen", "knn8"])
    parser.add_argument("--mean-model", choices=["intercept", "coord", "poly2"], default="intercept")

    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=[
            "leroux=leroux:learn_rho",
            "sdm_bspline=anchored_bspline:k8_medium",
        ],
        help=(
            "Model specs of the form label=filter:case. "
            "Example: leroux=leroux:learn_rho sdm_bspline=anchored_bspline:k8_medium"
        ),
    )

    parser.add_argument("--cv", choices=["block", "random"], default="block")
    parser.add_argument("--heldout-blocks", nargs="+", type=int, default=None)
    parser.add_argument("--n-block-x", type=int, default=5)
    parser.add_argument("--n-block-y", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)

    parser.add_argument("--noise-sigma", type=float, default=0.10)
    parser.add_argument("--fix-sigma2", action="store_true")
    parser.add_argument("--sigma2-floor", type=float, default=1e-4)
    parser.add_argument("--eps-car", type=float, default=1e-3)
    parser.add_argument("--tau2-ref", type=float, default=0.4)
    parser.add_argument("--prior-beta-var", type=float, default=10.0)

    parser.add_argument("--vi-iters", type=int, default=800)
    parser.add_argument("--vi-mc", type=int, default=5)
    parser.add_argument("--vi-lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-fold-plots", action="store_true")
    parser.add_argument("--plot-first-fold-only", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small smoke-test run.")

    args = parser.parse_args()

    if args.fast:
        args.nx = min(args.nx, 20)
        args.ny = min(args.ny, 20)
        args.surfaces = args.surfaces[:1]
        args.w_kinds = args.w_kinds[:1]
        args.n_repeats = 1
        args.n_folds = min(args.n_folds, 3)
        args.vi_iters = min(args.vi_iters, 120)
        args.vi_mc = min(args.vi_mc, 3)
        args.save_fold_plots = True
        args.plot_first_fold_only = True
        print("[FAST MODE] Reduced grid/folds/VI settings for smoke test.")

    set_all_seeds(args.seed)
    device = torch.device("cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_specs = parse_model_specs(args.model_specs)

    print("\n[DETERMINISTIC SURFACE BENCHMARK — VI]")
    print(f"  outdir       = {outdir}")
    print(f"  grid         = {args.nx} x {args.ny}")
    print(f"  surfaces     = {args.surfaces}")
    print(f"  W kinds      = {args.w_kinds}")
    print(f"  CV           = {args.cv}, folds={args.n_folds}, repeats={args.n_repeats}")
    print(f"  mean model   = {args.mean_model}")
    print(f"  model specs  = {args.model_specs}")
    print(f"  VI           = iters={args.vi_iters}, mc={args.vi_mc}, lr={args.vi_lr}")

    coords_t, coords_np = make_grid(args.nx, args.ny, device=device)
    X = make_design(coords_t, args.mean_model)
    n = coords_t.shape[0]

    all_rows: list[dict] = []
    graph_rows: list[dict] = []

    sigma2_obs = max(args.noise_sigma**2, args.sigma2_floor)

    for surface in args.surfaces:
        f_clean_np = make_surface(coords_np, surface)
        f_clean_t = torch.tensor(f_clean_np, dtype=torch.double, device=device)

        # One noisy response per surface/repeat seed. Reusing the same y across W/model
        # comparisons is important for paired comparisons.
        rng_surface = np.random.default_rng(args.seed + 1000 * (1 + stable_string_int(surface) % 10000))
        noise = args.noise_sigma * rng_surface.standard_normal(n)
        y_np = f_clean_np + noise
        y_t = torch.tensor(y_np, dtype=torch.double, device=device)

        surface_dir = outdir / surface
        surface_dir.mkdir(parents=True, exist_ok=True)
        save_surface_image(f_clean_np, args.nx, args.ny, surface_dir / "true_surface_clean.png", f"True clean surface: {surface}")
        save_surface_image(y_np, args.nx, args.ny, surface_dir / "observed_surface.png", f"Observed surface: {surface}")

        for w_kind in args.w_kinds:
            print("\n" + "=" * 90)
            print(f"SURFACE={surface} | W={w_kind}")
            print("=" * 90)

            w = build_pysal_weights(coords_np, args.nx, args.ny, w_kind)
            A = pysal_w_to_sparse_adjacency(w)
            L = laplacian_from_adjacency(A)
            lam, U = eigendecomp_laplacian(L, device=device)

            gsum = graph_summary(A, lam, surface, w_kind)
            graph_rows.append(gsum)
            print(
                f"  Graph: n={gsum['n']} nnz={gsum['nnz']} "
                f"degree mean={gsum['degree_mean']:.3f} "
                f"zero eigs={gsum['n_zero_eigenvalues']} "
                f"lambda_max={gsum['lambda_max']:.3f}"
            )

            w_dir = surface_dir / w_kind
            w_dir.mkdir(parents=True, exist_ok=True)
            save_empirical_modal_energy(
                f_clean_t=f_clean_t,
                lam=lam,
                U=U,
                path=w_dir / "empirical_modal_energy_clean_surface.png",
                title=f"Empirical modal energy — {surface}/{w_kind}",
            )

            for rep in range(args.n_repeats):
                cv_seed = args.seed + 10_000 * rep + 97
                if args.cv == "block":
                    if args.heldout_blocks is not None:
                        folds, block_id = make_manual_block_fold(
                            coords_np,
                            n_block_x=args.n_block_x,
                            n_block_y=args.n_block_y,
                            heldout_blocks=args.heldout_blocks,
                        )
                    else:
                        folds, block_id = make_block_cv_folds(
                            coords_np,
                            n_block_x=args.n_block_x,
                            n_block_y=args.n_block_y,
                            n_folds=args.n_folds,
                            seed=cv_seed,
                        )
                    if rep == 0:
                        save_surface_image(
                            block_id.astype(float),
                            args.nx,
                            args.ny,
                            w_dir / "spatial_block_ids.png",
                            f"Spatial block IDs — {surface}/{w_kind}",
                        )
                else:
                    folds = make_random_cv_folds(n, n_folds=args.n_folds, seed=cv_seed)

                for fold in folds:
                    fold_id = int(fold["fold"])
                    train_mask = fold["train_mask"]
                    test_mask = fold["test_mask"]

                    print(
                        f"\n  repeat={rep} fold={fold_id} "
                        f"n_train={int(train_mask.sum())} n_test={int(test_mask.sum())}"
                    )

                    do_plot_fold = args.save_fold_plots and (
                        (not args.plot_first_fold_only) or (rep == 0 and fold_id == 0)
                    )

                    if do_plot_fold:
                        fold_dir = w_dir / f"rep{rep:02d}_fold{fold_id:02d}"
                        fold_dir.mkdir(parents=True, exist_ok=True)
                        save_mask_image(test_mask, args.nx, args.ny, fold_dir / "test_mask.png", "Held-out locations")
                    else:
                        fold_dir = None
                    
                    filled_surfaces = {}

                    for mspec in model_specs:
                        print(f"    model={mspec.label} ({mspec.filter_name}:{mspec.case_id})")
                        fit_seed = args.seed + 1_000_000 * rep + 10_000 * fold_id + stable_string_int(mspec.label) % 9973

                        metrics, model = fit_one_vi_fold(
                            model_spec=mspec,
                            X=X,
                            y_true_t=y_t,
                            lam=lam,
                            U=U,
                            train_mask_np=train_mask,
                            test_mask_np=test_mask,
                            sigma2_obs=sigma2_obs,
                            fix_sigma2=args.fix_sigma2,
                            eps_car=args.eps_car,
                            tau2_ref=args.tau2_ref,
                            prior_beta_var=args.prior_beta_var,
                            vi_iters=args.vi_iters,
                            vi_mc=args.vi_mc,
                            vi_lr=args.vi_lr,
                            grad_clip=args.grad_clip,
                            seed=fit_seed,
                            verbose=args.verbose,
                        )

                        row = {
                            "surface": surface,
                            "w_kind": w_kind,
                            "cv": args.cv,
                            "repeat": rep,
                            "fold": fold_id,
                            "noise_sigma": args.noise_sigma,
                            "sigma2_obs": sigma2_obs,
                            "fix_sigma2": bool(args.fix_sigma2),
                            "mean_model": args.mean_model,
                            **metrics,
                        }
                        all_rows.append(row)

                        print(
                            f"      test RMSE={row['test_rmse']:.4f} "
                            f"MAE={row['test_mae']:.4f} "
                            f"plugin_eta_RMSE={row['test_rmse_plugin_eta']:.4f} "
                            f"time={seconds_to_str(row['vi_time_seconds'])}"
                        )

                        if do_plot_fold and fold_dir is not None:
                            with torch.no_grad():
                                test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)
                                pred = predict_from_vi_imputation(model, test_mask_t)
                                y_pred_full = y_np.copy()
                                y_pred_full[test_mask] = pred["y_pred_impute"]
                                yhat_plugin_all = pred["yhat_plugin_all"]
                                abs_err = np.zeros_like(y_np)
                                abs_err[test_mask] = np.abs(y_np[test_mask] - pred["y_pred_impute"])

                                filled_surfaces[mspec.label] = y_pred_full.copy()

                            model_plot_dir = fold_dir / mspec.label
                            model_plot_dir.mkdir(parents=True, exist_ok=True)
                            save_surface_image(
                                y_pred_full,
                                args.nx,
                                args.ny,
                                model_plot_dir / "imputed_response_surface.png",
                                f"Imputed response — {surface}/{w_kind}/{mspec.label}",
                            )
                            save_surface_image(
                                yhat_plugin_all,
                                args.nx,
                                args.ny,
                                model_plot_dir / "plugin_eta_surface.png",
                                f"Plugin eta — {surface}/{w_kind}/{mspec.label}",
                            )
                            save_surface_image(
                                abs_err,
                                args.nx,
                                args.ny,
                                model_plot_dir / "test_abs_error.png",
                                f"Test absolute error — {surface}/{w_kind}/{mspec.label}",
                            )
                    
                    if do_plot_fold and fold_dir is not None:
                        save_block_fill_comparison(
                            original_surface=y_np,
                            test_mask=test_mask,
                            filled_surfaces=filled_surfaces,
                            nx=args.nx,
                            ny=args.ny,
                            path=fold_dir / "block_fill_comparison.png",
                            title=f"{surface}/{w_kind} rep={rep} fold={fold_id}",
                        )

    # -----------------------------------------------------------------
    # Save summaries
    # -----------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    graph_df = pd.DataFrame(graph_rows)

    fold_path = outdir / "fold_metrics.csv"
    graph_path = outdir / "graph_summaries.csv"
    summary_path = outdir / "summary_by_model.csv"
    paired_path = outdir / "paired_model_differences.csv"

    df.to_csv(fold_path, index=False)
    graph_df.to_csv(graph_path, index=False)

    group_cols = ["surface", "w_kind", "cv", "mean_model", "model_label", "filter", "case"]
    summary = (
        df.groupby(group_cols)
        .agg(
            n_fits=("test_rmse", "size"),
            mean_test_rmse=("test_rmse", "mean"),
            sd_test_rmse=("test_rmse", "std"),
            mean_test_mae=("test_mae", "mean"),
            sd_test_mae=("test_mae", "std"),
            mean_test_r2=("test_r2", "mean"),
            mean_test_rmse_plugin_eta=("test_rmse_plugin_eta", "mean"),
            mean_train_rmse_plugin_eta=("train_rmse_plugin_eta", "mean"),
            mean_vi_time_seconds=("vi_time_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    # Paired differences for every pair of model labels within the same fold.
    paired_rows = []
    labels = [m.label for m in model_specs]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            key_cols = ["surface", "w_kind", "cv", "repeat", "fold", "mean_model"]
            da = df[df["model_label"] == a][key_cols + ["test_rmse", "test_mae", "test_rmse_plugin_eta"]]
            db = df[df["model_label"] == b][key_cols + ["test_rmse", "test_mae", "test_rmse_plugin_eta"]]
            merged = da.merge(db, on=key_cols, suffixes=(f"_{a}", f"_{b}"))
            if merged.empty:
                continue
            merged["model_a"] = a
            merged["model_b"] = b
            merged["delta_rmse_a_minus_b"] = merged[f"test_rmse_{a}"] - merged[f"test_rmse_{b}"]
            merged["delta_mae_a_minus_b"] = merged[f"test_mae_{a}"] - merged[f"test_mae_{b}"]
            merged["delta_plugin_eta_rmse_a_minus_b"] = (
                merged[f"test_rmse_plugin_eta_{a}"] - merged[f"test_rmse_plugin_eta_{b}"]
            )
            paired_rows.append(merged)

    if paired_rows:
        paired_long = pd.concat(paired_rows, ignore_index=True)
        paired_summary = (
            paired_long.groupby(["surface", "w_kind", "cv", "mean_model", "model_a", "model_b"])
            .agg(
                n_pairs=("delta_rmse_a_minus_b", "size"),
                mean_delta_rmse_a_minus_b=("delta_rmse_a_minus_b", "mean"),
                sd_delta_rmse_a_minus_b=("delta_rmse_a_minus_b", "std"),
                mean_delta_mae_a_minus_b=("delta_mae_a_minus_b", "mean"),
                mean_delta_plugin_eta_rmse_a_minus_b=("delta_plugin_eta_rmse_a_minus_b", "mean"),
                a_wins_rmse=("delta_rmse_a_minus_b", lambda x: int(np.sum(np.asarray(x) < 0))),
                b_wins_rmse=("delta_rmse_a_minus_b", lambda x: int(np.sum(np.asarray(x) > 0))),
            )
            .reset_index()
        )
        paired_summary.to_csv(paired_path, index=False)
    else:
        pd.DataFrame().to_csv(paired_path, index=False)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Fold metrics:       {fold_path}")
    print(f"Model summary:      {summary_path}")
    print(f"Paired differences: {paired_path}")
    print(f"Graph summaries:    {graph_path}")
    print("\nInterpretation of paired differences:")
    print("  delta_rmse_a_minus_b = RMSE(model_a) - RMSE(model_b)")
    print("  If model_a=leroux and model_b=sdm_bspline, positive means SDM is better.")


if __name__ == "__main__":
    main()
