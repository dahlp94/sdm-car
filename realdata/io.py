from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from realdata.base import MethodResult, BenchmarkSummary, FitArtifacts, to_dict


# ---------------------------------------------------------------------
# Basic directory helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Summary saving
# ---------------------------------------------------------------------

def save_summary(summary: BenchmarkSummary, path: Path) -> None:
    """
    Save benchmark summary as JSON.
    """
    save_json(to_dict(summary), path)


# ---------------------------------------------------------------------
# Artifact table builders
# ---------------------------------------------------------------------

def artifacts_to_predictions_df(artifacts: FitArtifacts) -> pd.DataFrame:
    """
    Convert core fit artifacts into a prediction-level DataFrame.

    Required:
    - fips

    Optional columns are only added if present.
    """
    data = {
        "fips": artifacts.fips,
    }

    if artifacts.graph_index is not None:
        data["graph_index"] = artifacts.graph_index

    if artifacts.county_name is not None:
        data["county_name"] = artifacts.county_name

    if artifacts.state_abbr is not None:
        data["state_abbr"] = artifacts.state_abbr

    if artifacts.y is not None:
        data["y"] = artifacts.y

    if artifacts.fitted_mean is not None:
        data["fitted_mean"] = artifacts.fitted_mean

    if artifacts.residual is not None:
        data["residual"] = artifacts.residual

    if artifacts.spatial_effect_mean is not None:
        data["spatial_effect_mean"] = artifacts.spatial_effect_mean

    if artifacts.fitted_sd is not None:
        data["fitted_sd"] = artifacts.fitted_sd

    if artifacts.spatial_effect_sd_vec is not None:
        data["spatial_effect_sd_vec"] = artifacts.spatial_effect_sd_vec

    return pd.DataFrame(data)


def artifacts_to_coefficients_df(artifacts: FitArtifacts) -> pd.DataFrame:
    """
    Convert coefficient artifacts into a tidy coefficient table.
    """
    if artifacts.coefficient_names is None or artifacts.coefficient_estimates is None:
        return pd.DataFrame(columns=["term", "estimate", "is_intercept"])

    df = pd.DataFrame({
        "term": artifacts.coefficient_names,
        "estimate": artifacts.coefficient_estimates,
    })
    df["is_intercept"] = df["term"].eq("intercept")
    return df


def artifacts_to_trace_df(artifacts: FitArtifacts) -> pd.DataFrame:
    """
    Convert objective trace into a DataFrame.
    """
    if artifacts.objective_trace is None:
        return pd.DataFrame(columns=["iteration", "objective"])

    return pd.DataFrame({
        "iteration": list(range(1, len(artifacts.objective_trace) + 1)),
        "objective": artifacts.objective_trace,
    })


def artifacts_to_spectrum_df(artifacts: FitArtifacts) -> pd.DataFrame:
    """
    Convert learned spectrum into a DataFrame.
    """
    if artifacts.learned_spectrum is None:
        return pd.DataFrame(columns=["index", "lambda", "spectrum", "normalized_spectrum"])

    spec = np.asarray(artifacts.learned_spectrum, dtype=float)

    if artifacts.lambda_grid is not None:
        lam = np.asarray(artifacts.lambda_grid, dtype=float)
    else:
        lam = np.full(len(spec), np.nan)

    if len(lam) != len(spec):
        lam = np.full(len(spec), np.nan)

    denom = np.max(np.abs(spec)) if len(spec) > 0 else np.nan
    if np.isfinite(denom) and denom > 0:
        spec_norm = spec / denom
    else:
        spec_norm = np.full_like(spec, np.nan)

    return pd.DataFrame({
        "index": list(range(len(spec))),
        "lambda": lam,
        "spectrum": spec,
        "normalized_spectrum": spec_norm,
    })


# ---------------------------------------------------------------------
# CSV saving helpers
# ---------------------------------------------------------------------

def save_predictions(artifacts: FitArtifacts, path: Path) -> None:
    """
    Save predictions / residual / spatial effect table.
    """
    ensure_dir(path.parent)
    df = artifacts_to_predictions_df(artifacts)
    df.to_csv(path, index=False)


def save_coefficients(artifacts: FitArtifacts, path: Path) -> None:
    """
    Save coefficient table.
    """
    ensure_dir(path.parent)
    df = artifacts_to_coefficients_df(artifacts)
    df.to_csv(path, index=False)


def save_trace(artifacts: FitArtifacts, path: Path) -> None:
    """
    Save objective trace table.
    """
    ensure_dir(path.parent)
    df = artifacts_to_trace_df(artifacts)
    df.to_csv(path, index=False)


def save_spectrum(artifacts: FitArtifacts, path: Path) -> None:
    """
    Save learned spectrum table.
    """
    ensure_dir(path.parent)
    df = artifacts_to_spectrum_df(artifacts)
    df.to_csv(path, index=False)


def save_extras(artifacts: FitArtifacts, path: Path) -> None:
    """
    Save extras dictionary as JSON.
    """
    ensure_dir(path.parent)
    save_json(artifacts.extras or {}, path)


# ---------------------------------------------------------------------
# Full result saver
# ---------------------------------------------------------------------

def save_method_result(result: MethodResult, out_dir: Path) -> None:
    """
    Save a full method result using the standardized output layout.

    Files written:
    - summary.json
    - predictions.csv
    - coefficients.csv
    - trace.csv
    - spectrum.csv
    - extras.json
    """
    ensure_dir(out_dir)

    save_summary(result.summary, out_dir / "summary.json")
    save_predictions(result.artifacts, out_dir / "predictions.csv")
    save_coefficients(result.artifacts, out_dir / "coefficients.csv")
    save_trace(result.artifacts, out_dir / "trace.csv")
    save_spectrum(result.artifacts, out_dir / "spectrum.csv")
    save_extras(result.artifacts, out_dir / "extras.json")


# ---------------------------------------------------------------------
# Utility readers for later evaluation scripts
# ---------------------------------------------------------------------

def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_coefficients(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_spectrum(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_extras(path: Path) -> Dict[str, Any]:
    return load_json(path)