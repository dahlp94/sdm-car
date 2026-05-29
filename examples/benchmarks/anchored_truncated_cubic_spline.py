# examples/benchmarks/anchored_truncated_cubic_spline.py
from __future__ import annotations

import torch
import math

from sdmcar.filters import AnchoredTruncatedCubicSplineSpectrumFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_anchored_truncated_cubic_spline_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    K: int = 6,
    prior_std_theta: float = 2.0,
    prior_std_alpha: float = 0.25,
    log_std0: float = -2.3,
    init_theta: list[float] | None = None,
    init_alpha: float = 0.0,
    logF_min: float = -30.0,
    logF_max: float = 30.0,
    **kwargs,
):
    return AnchoredTruncatedCubicSplineSpectrumFullVI(
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


def step_theta_anchored_truncated_cubic_spline(filter_module) -> dict[str, float]:
    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        if nm.startswith("theta"):
            d[nm] = 0.14
        elif nm.startswith("alpha"):
            d[nm] = 0.10
        else:
            d[nm] = 0.10

    return d


register(
    FilterSpec(
        filter_name="anchored_truncated_cubic_spline",
        cases={
            "k6_medium": CaseSpec(
                case_id="k6_medium",
                display_name="anchored_trunc_cubic_k6_alpha0p25",
                fixed=dict(
                    K=6,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.25,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k6_weak": CaseSpec(
                case_id="k6_weak",
                display_name="anchored_trunc_cubic_k6_alpha0p75",
                fixed=dict(
                    K=6,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.75,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k6_strong": CaseSpec(
                case_id="k6_strong",
                display_name="anchored_trunc_cubic_k6_alpha0p10",
                fixed=dict(
                    K=6,
                    prior_std_theta=2.0,
                    prior_std_alpha=0.10,
                    log_std0=-2.3,
                    init_theta=[0.0, 0.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k6_bump": CaseSpec(
                case_id="k6_bump",
                display_name="anchored_trunc_cubic_k6_alpha1p00",
                fixed=dict(
                    K=6,
                    prior_std_theta=2.0,
                    prior_std_alpha=1.0,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_alpha=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_truncated_cubic_spline,
                transform_chain=None,
            ),
            "k6_bump_init": CaseSpec(
                case_id="k6_bump_init",
                display_name="anchored_trunc_cubic_k6_alpha1p00_bumpinit",
                fixed=dict(
                    K=6,
                    prior_std_theta=2.0,
                    prior_std_alpha=1.0,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_alpha=[0.0, 0.25, 0.75, 0.25, 0.0, 0.0],
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_truncated_cubic_spline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_truncated_cubic_spline,
                transform_chain=None,
            ),
        },
    )
)