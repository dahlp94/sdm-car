from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from sdmcar.models import SpectralCAR_FullVI
from examples.benchmarks.registry import get_filter_spec
import examples.benchmarks  # noqa: F401

from realdata.base import (
    MethodSpec,
    FitArtifacts,
    BenchmarkSummary,
    MethodResult,
)
from realdata.registry import RealDataSpec


torch.set_default_dtype(torch.double)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _load_graph_eigs(
    eig_file: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    eig = np.load(eig_file, allow_pickle=True)

    required = {"lam", "U", "fips"}
    missing = required - set(eig.keys())
    if missing:
        raise KeyError(
            f"Missing keys in {eig_file}: {missing}. "
            f"Available: {list(eig.keys())}"
        )

    lam = torch.tensor(eig["lam"], dtype=torch.double, device=device)
    U = torch.tensor(eig["U"], dtype=torch.double, device=device)
    fips = pd.Series(np.asarray(eig["fips"]).astype(str)).str.zfill(5).to_numpy()

    return lam, U, fips


def _load_model_data(
    model_data_file: Path,
    outcome_col: str,
) -> pd.DataFrame:
    if not model_data_file.exists():
        raise FileNotFoundError(f"Model data not found: {model_data_file}")

    df = pd.read_csv(model_data_file, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)

    if outcome_col not in df.columns:
        raise KeyError(
            f"Outcome column '{outcome_col}' not found in {model_data_file}. "
            f"Columns: {df.columns.tolist()}"
        )

    return df


def _check_alignment(df: pd.DataFrame, fips_graph: np.ndarray) -> None:
    if len(df) != len(fips_graph):
        raise ValueError(
            f"Row mismatch: model_data has {len(df)} rows but eig file has {len(fips_graph)} rows"
        )

    fips_df = df["fips"].to_numpy()
    if not np.all(fips_df == fips_graph):
        bad = np.where(fips_df != fips_graph)[0][:10]
        raise ValueError(
            "FIPS ordering mismatch between model_data and eig file. "
            f"First bad indices: {bad.tolist()}"
        )


def _build_design_matrix(
    df: pd.DataFrame,
    covariates: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if len(covariates) == 0:
        X_np = np.ones((len(df), 1), dtype=float)
        coef_names = ["intercept"]
        return X_np, coef_names

    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise KeyError(f"Missing covariate columns in model_data: {missing}")

    X_np = np.column_stack(
        [np.ones(len(df), dtype=float)] +
        [df[c].to_numpy(dtype=float) for c in covariates]
    )
    coef_names = ["intercept"] + list(covariates)
    return X_np, coef_names


@torch.no_grad()
def _spectrum_vi_mean(filter_module, lam: torch.Tensor) -> torch.Tensor:
    theta_mean = filter_module.mean_unconstrained()
    return filter_module.spectrum(lam, theta_mean)


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def fit_sdmcar_gaussian(
    *,
    dataset_spec: RealDataSpec,
    method_spec: MethodSpec,
    use_covariates: bool,
    vi_iters: int = 1500,
    vi_lr: float = 1e-2,
    vi_mc: int = 8,
    device: torch.device | None = None,
) -> MethodResult:
    """
    Fit a Gaussian SDM-CAR model for a registered real-data dataset.
    """
    if dataset_spec.family != "gaussian":
        raise ValueError(
            f"fit_sdmcar_gaussian only supports gaussian family; got {dataset_spec.family}"
        )

    if method_spec.family != "gaussian":
        raise ValueError(
            f"Method spec family must be gaussian; got {method_spec.family}"
        )

    if method_spec.filter_name is None or method_spec.case_id is None:
        raise ValueError(
            "MethodSpec for SDM-CAR must include filter_name and case_id."
        )

    if device is None:
        device = torch.device("cpu")

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------
    lam, U, fips_graph = _load_graph_eigs(dataset_spec.eig_file, device=device)
    df = _load_model_data(dataset_spec.model_data_file, dataset_spec.outcome_column)
    _check_alignment(df, fips_graph)

    covariates = dataset_spec.default_covariates if use_covariates else []
    X_np, coef_names = _build_design_matrix(df, covariates)

    y_np = df[dataset_spec.outcome_column].to_numpy(dtype=float)

    X = torch.tensor(X_np, dtype=torch.double, device=device)
    y = torch.tensor(y_np, dtype=torch.double, device=device)
    prior_V0 = 10.0 * torch.eye(X.shape[1], dtype=torch.double, device=device)

    # --------------------------------------------------
    # Build filter and model
    # --------------------------------------------------
    spec = get_filter_spec(method_spec.filter_name)
    if method_spec.case_id not in spec.cases:
        raise ValueError(
            f"Case '{method_spec.case_id}' not found for filter '{method_spec.filter_name}'. "
            f"Available: {list(spec.cases.keys())}"
        )

    case = spec.cases[method_spec.case_id]

    build_kwargs = dict(case.fixed)
    build_kwargs.update(method_spec.extra_args)

    filter_module = case.build_filter(
        tau2_true=0.5,
        eps_car=1e-3,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **build_kwargs,
    )

    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(1.0),
        log_std_log_sigma2=-2.3,
        num_mc=vi_mc,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    t0 = time.time()
    objective_trace: list[float] = []

    for it in range(vi_iters):
        opt.zero_grad()
        elbo, _ = model.elbo()
        (-elbo).backward()
        opt.step()

        objective_trace.append(float(elbo.detach().cpu()))

        if (it + 1) % 250 == 0 or it == 0:
            print(
                f"  [{method_spec.method_name}] iter {it+1:4d}/{vi_iters} "
                f"ELBO={objective_trace[-1]:.3f}"
            )

    fit_time_sec = time.time() - t0

    # --------------------------------------------------
    # Post-fit quantities
    # --------------------------------------------------
    with torch.no_grad():
        phi_mean, _ = model.posterior_phi(mode="plugin")
        F_vi = _spectrum_vi_mean(model.filter, lam).detach().cpu().numpy()

        m_beta_plugin, _, sigma2_plugin, _ = model.beta_posterior_plugin()
        beta_mean = m_beta_plugin.detach().cpu().numpy()

        y_cpu = y.detach().cpu()
        X_cpu = X.detach().cpu()
        phi_cpu = phi_mean.detach().cpu()

        beta_t = torch.tensor(beta_mean, dtype=torch.double)
        fitted_mean = X_cpu @ beta_t + phi_cpu
        residual = y_cpu - fitted_mean

        residual_mean = float(residual.mean().item())
        residual_sd = float(residual.std().item())
        residual_mse = float(torch.mean(residual ** 2).item())
        residual_rmse = float(torch.sqrt(torch.mean(residual ** 2)).item())
        spatial_effect_sd = float(phi_cpu.std().item())

        sigma2_mean = float(sigma2_plugin.detach().cpu())

    # --------------------------------------------------
    # Standardized outputs
    # --------------------------------------------------
    summary = BenchmarkSummary(
        dataset_name=dataset_spec.dataset_name,
        method_name=method_spec.method_name,
        display_name=method_spec.display_name,
        family=dataset_spec.family,
        final_objective=float(objective_trace[-1]),
        objective_name="ELBO",
        residual_mean=residual_mean,
        residual_sd=residual_sd,
        residual_mse=residual_mse,
        spatial_effect_sd=spatial_effect_sd,
        fit_time_sec=fit_time_sec,
        n_obs=len(df),
        num_parameters=len(beta_mean),
        notes=f"filter={method_spec.filter_name}, case={method_spec.case_id}",
    )

    extras = {
        "sigma2_mean": sigma2_mean,
        "residual_rmse": residual_rmse,
        "use_covariates": use_covariates,
        "covariates": list(covariates),
        "outcome_column": dataset_spec.outcome_column,
        "graph_name": dataset_spec.graph_name,
        "filter_name": method_spec.filter_name,
        "case_id": method_spec.case_id,
    }

    artifacts = FitArtifacts(
        fips=df["fips"].astype(str).tolist(),
        graph_index=df["graph_index"].astype(int).tolist() if "graph_index" in df.columns else None,
        county_name=df["county_name"].astype(str).tolist() if "county_name" in df.columns else None,
        state_abbr=df["state_abbr"].astype(str).tolist() if "state_abbr" in df.columns else None,
        y=y_np.tolist(),
        fitted_mean=fitted_mean.numpy().tolist(),
        residual=residual.numpy().tolist(),
        spatial_effect_mean=phi_cpu.numpy().tolist(),
        coefficient_names=coef_names,
        coefficient_estimates=beta_mean.tolist(),
        objective_trace=objective_trace,
        objective_name="ELBO",
        learned_spectrum=F_vi.tolist(),
        lambda_grid=lam.detach().cpu().numpy().tolist(),
        extras=extras,
    )

    return MethodResult(summary=summary, artifacts=artifacts)