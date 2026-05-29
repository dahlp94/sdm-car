# examples/benchmarks/anchored_bspline.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import AnchoredBSplineSpectrumFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_anchored_bspline_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    degree: int = 3,
    n_internal_knots: int = 8,
    prior_std_theta: float = 2.0,
    prior_std_w: float = 0.5,
    log_std0: float = -2.3,
    init_theta: list[float] | None = None,
    init_w: float = 0.0,
    logF_min: float = -30.0,
    logF_max: float = 30.0,
    **kwargs,
):
    return AnchoredBSplineSpectrumFullVI(
        lam_max=float(lam_max),
        degree=int(degree),
        n_internal_knots=int(n_internal_knots),
        prior_std_theta=float(prior_std_theta),
        prior_std_w=float(prior_std_w),
        log_std0=float(log_std0),
        init_theta=init_theta,
        init_w=init_w,
        logF_min=float(logF_min),
        logF_max=float(logF_max),
    ).to(device)


def step_theta_anchored_bspline(filter_module) -> dict[str, float]:
    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        if nm.startswith("theta"):
            d[nm] = 0.12
        elif nm.startswith("w"):
            d[nm] = 0.06
        else:
            d[nm] = 0.08

    return d


register(
    FilterSpec(
        filter_name="anchored_bspline",
        cases={
            "k8_medium": CaseSpec(
                case_id="k8_medium",
                display_name="anchored_bspline_deg3_k8_w0p50",
                fixed=dict(
                    degree=3,
                    n_internal_knots=8,
                    prior_std_theta=2.0,
                    prior_std_w=0.50,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_w=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_bspline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_bspline,
                transform_chain=None,
            ),
            "k12_medium": CaseSpec(
                case_id="k12_medium",
                display_name="anchored_bspline_deg3_k12_w0p50",
                fixed=dict(
                    degree=3,
                    n_internal_knots=12,
                    prior_std_theta=2.0,
                    prior_std_w=0.50,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_w=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_bspline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_bspline,
                transform_chain=None,
            ),
            "k12_weak": CaseSpec(
                case_id="k12_weak",
                display_name="anchored_bspline_deg3_k12_w1p00",
                fixed=dict(
                    degree=3,
                    n_internal_knots=12,
                    prior_std_theta=2.0,
                    prior_std_w=1.00,
                    log_std0=-2.3,
                    init_theta=[math.log(0.4), -3.0],
                    init_w=0.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_anchored_bspline_filter,
                step_s=0.14,
                step_theta=step_theta_anchored_bspline,
                transform_chain=None,
            ),
        },
    )
)