# examples/benchmarks/truncated_cubic_spline.py
from __future__ import annotations

import torch

from sdmcar.filters import TruncatedCubicSplineSpectrumFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_truncated_cubic_spline_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    K: int = 8,
    prior_std_theta: float = 2.0,
    prior_std_alpha: float = 0.5,
    log_std0: float = -2.3,
    init_theta: list[float] | None = None,
    init_alpha: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    logF_min: float = -30.0,
    logF_max: float = 30.0,
    **kwargs,
):
    """
    Build truncated cubic spline spectral filter.

    Notes:
        - tau2_true and eps_car are accepted for compatibility with
          the benchmark framework, but are not used by this filter.
        - The filter models:
              log F(lambda) = spline(lambda / lambda_max)
    """

    return TruncatedCubicSplineSpectrumFullVI(
        lam_max=float(lam_max),
        K=int(K),
        prior_std_theta=float(prior_std_theta),
        prior_std_alpha=float(prior_std_alpha),
        log_std0=float(log_std0),
        init_theta=init_theta,
        init_alpha=init_alpha,
        logF_min=float(logF_min),
        logF_max=float(logF_max),
    ).to(device)


def step_theta_truncated_cubic_spline(filter_module) -> dict[str, float]:
    """
    MH proposal step sizes for TruncatedCubicSplineSpectrumFullVI.
    """

    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        if nm.startswith("theta"):
            d[nm] = 0.14
        elif nm.startswith("alpha"):
            d[nm] = 0.2
        else:
            d[nm] = 0.10

    return d


register(
    FilterSpec(
        filter_name="truncated_cubic_spline",
        cases={
            # --------------------------------------------------
            # Primary/default spline model
            # --------------------------------------------------
            "k8_medium": CaseSpec(
                case_id="k8_medium",
                display_name="trunc_cubic_k8_alpha0p5",
                fixed=dict(
                    K=8,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.5,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),

            # --------------------------------------------------
            # Weaker shrinkage on spline deviations
            # --------------------------------------------------
            "k8_weak": CaseSpec(
                case_id="k8_weak",
                display_name="trunc_cubic_k8_alpha1p0",
                fixed=dict(
                    K=8,
                    prior_std_theta=2.0,
                    prior_std_alpha=1.0,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),

            # --------------------------------------------------
            # Stronger shrinkage on spline deviations
            # --------------------------------------------------
            "k8_strong": CaseSpec(
                case_id="k8_strong",
                display_name="trunc_cubic_k8_alpha0p25",
                fixed=dict(
                    K=8,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.25,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),

            # --------------------------------------------------
            # Denser spline basis for approximation experiments
            # --------------------------------------------------
            "k12_medium": CaseSpec(
                case_id="k12_medium",
                display_name="trunc_cubic_k12_alpha0p5",
                fixed=dict(
                    K=12,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.5,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k8_diffuse": CaseSpec(
                case_id="k8_diffuse",
                display_name="trunc_cubic_k8_alpha200p0",
                fixed=dict(
                    K=8,
                    prior_std_theta=50.0,
                    prior_std_alpha=200.0,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k8_relaxed_5": CaseSpec(
                case_id="k8_relaxed_5",
                display_name="trunc_cubic_k8_theta20_alpha5",
                fixed=dict(
                    K=8,
                    prior_std_theta=20.0,
                    prior_std_alpha=5.0,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k8_relaxed_10": CaseSpec(
                case_id="k8_relaxed_10",
                display_name="trunc_cubic_k8_theta20_alpha10",
                fixed=dict(
                    K=8,
                    prior_std_theta=20.0,
                    prior_std_alpha=10.0,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k8_relaxed_25": CaseSpec(
                case_id="k8_relaxed_25",
                display_name="trunc_cubic_k8_theta20_alpha25",
                fixed=dict(
                    K=8,
                    prior_std_theta=20.0,
                    prior_std_alpha=25.0,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0, 0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k8_oracle_init": CaseSpec(
                case_id="k8_oracle_init",
                display_name="trunc_cubic_k8_oracle_init",
                fixed=dict(
                    K=8,

                    # Use moderately broad priors for this diagnostic.
                    # If these are too tight, the KL penalty will immediately pull
                    # the coefficients away from the deterministic solution.
                    prior_std_theta=50.0,
                    prior_std_alpha=100.0,

                    log_std0=-2.3,

                    # Deterministic normalized-basis LS coefficients for exp_bump, K=8.
                    init_theta=[
                        -0.880587575,
                        -3.53463580,
                        19.2946267,
                        -69.5622866,
                    ],

                    init_alpha=[
                        106.011594,
                        -42.0974643,
                        -87.3054532,
                        99.1032097,
                        -20.1464618,
                        -4.79246212,
                        1.03690474,
                        -0.0468325777,
                    ],

                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_truncated_cubic_spline,
                transform_chain=None,
            ),
        },
    )
)