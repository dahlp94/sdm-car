from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Core metadata objects
# ---------------------------------------------------------------------

@dataclass
class MethodSpec:
    """
    Standardized method specification used by real-data runners.

    Examples:
    - SDM-CAR classic CAR
    - SDM-CAR Leroux
    - INLA BYM2
    - Graph GP
    """
    method_name: str
    runner_name: str
    family: str
    display_name: str

    # Optional method-specific identifiers / parameters
    filter_name: Optional[str] = None
    case_id: Optional[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitArtifacts:
    """
    Standardized fit outputs that can be written to disk or passed to
    downstream evaluation code.

    All methods do not need to populate every field, but they should
    use the same names when they do.
    """
    fips: List[str]

    # Optional indexing / join helpers
    graph_index: Optional[List[int]] = None
    county_name: Optional[List[str]] = None
    state_abbr: Optional[List[str]] = None

    # Core fitted quantities
    y: Optional[List[float]] = None
    fitted_mean: Optional[List[float]] = None
    residual: Optional[List[float]] = None
    spatial_effect_mean: Optional[List[float]] = None

    # Optional uncertainty summaries for future expansion
    fitted_sd: Optional[List[float]] = None
    spatial_effect_sd_vec: Optional[List[float]] = None

    # Regression coefficients
    coefficient_names: Optional[List[str]] = None
    coefficient_estimates: Optional[List[float]] = None

    # Method-specific diagnostics / traces
    objective_trace: Optional[List[float]] = None
    objective_name: Optional[str] = None

    # Optional spectral object for SDM-CAR-like methods
    learned_spectrum: Optional[List[float]] = None
    lambda_grid: Optional[List[float]] = None

    # Free-form method-specific extras
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Benchmark summary used across all methods
# ---------------------------------------------------------------------

@dataclass
class BenchmarkSummary:
    """
    Method-neutral summary for ranking and comparison.

    'final_objective' can represent:
    - ELBO for variational methods
    - approximate marginal log posterior
    - negative deviance surrogate
    - or another method-specific objective

    The corresponding label should be stored in objective_name.
    """
    dataset_name: str
    method_name: str
    display_name: str
    family: str

    final_objective: float
    objective_name: str

    residual_mean: float
    residual_sd: float
    residual_mse: float
    spatial_effect_sd: float

    fit_time_sec: float

    # Optional extras for later expansion
    n_obs: Optional[int] = None
    num_parameters: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class MethodResult:
    """
    Full standardized result returned by a method adapter.

    This is the main object that realdata/methods/*.py should return.
    """
    summary: BenchmarkSummary
    artifacts: FitArtifacts


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass instance to a dictionary.
    """
    return asdict(obj)


def summary_to_dict(summary: BenchmarkSummary) -> Dict[str, Any]:
    return asdict(summary)


def artifacts_to_dict(artifacts: FitArtifacts) -> Dict[str, Any]:
    return asdict(artifacts)


def result_to_dict(result: MethodResult) -> Dict[str, Any]:
    return asdict(result)