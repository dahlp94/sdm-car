# examples/benchmarks/logpoly_shrink.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import LogPolyShrinkFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_logpoly_shrink_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    degree: int = 10,
    floor: float = 1e-2,
    init_c0: float | None = None,
    init_c1: float = -1.0,
    init_other: float = 0.0,
    log_std0: float = -2.3,
    prior_mu_c0: float | None = None,
    prior_std_c0: float = 1.0,
    prior_mu_rest: float = 0.0,
    prior_std_rest: float = 0.5,
    logF_min: float = -30.0,
    logF_max: float = 30.0,
    **kwargs,
):
    """
    Build log-polynomial shrinkage filter.

    F(lam) = floor + exp(c0 + c1 x + ... + cK x^K),
    x = lam / lam_max.

    No separate tau2 is used. The scale is absorbed into c0.

    tau2_true is used only to initialize c0 and its prior center.
    """
    if init_c0 is None:
        init_c0 = math.log(max(float(tau2_true), 1e-12))

    if prior_mu_c0 is None:
        prior_mu_c0 = init_c0

    return LogPolyShrinkFilterFullVI(
        lam_max=float(lam_max),
        degree=int(degree),
        floor=float(floor),
        init_c0=float(init_c0),
        init_c1=float(init_c1),
        init_other=float(init_other),
        log_std0=float(log_std0),
        prior_mu_c0=float(prior_mu_c0),
        prior_std_c0=float(prior_std_c0),
        prior_mu_rest=float(prior_mu_rest),
        prior_std_rest=float(prior_std_rest),
        logF_min=float(logF_min),
        logF_max=float(logF_max),
    ).to(device)


def step_theta_logpoly_shrink(filter_module) -> dict[str, float]:
    """
    MH proposal step sizes for LogPolyShrinkFilterFullVI.
    """
    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        if nm == "c0_raw":
            d[nm] = 0.16
        elif nm == "c1_raw":
            d[nm] = 0.14
        elif nm.startswith("c") and nm.endswith("_raw"):
            d[nm] = 0.10
        else:
            d[nm] = 0.10

    return d


register(
    FilterSpec(
        filter_name="logpoly_shrink",
        cases={
            # Moderate shrinkage on inactive coefficients c2,...,cK.
            "deg10_medium": CaseSpec(
                case_id="deg10_medium",
                display_name="logpoly_deg10_medium",
                fixed=dict(
                    degree=10,
                    floor=1e-2,
                    init_c1=-1.0,
                    init_other=0.0,
                    log_std0=-2.3,
                    prior_std_c0=1.0,
                    prior_mu_rest=0.0,
                    prior_std_rest=0.5,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_logpoly_shrink_filter,
                step_s=0.14,
                step_theta=step_theta_logpoly_shrink,
                transform_chain=None,
            ),

            # Stronger shrinkage: unnecessary coefficients should be closer to zero.
            "deg10_strong": CaseSpec(
                case_id="deg10_strong",
                display_name="logpoly_deg10_strong",
                fixed=dict(
                    degree=10,
                    floor=1e-2,
                    init_c1=-1.0,
                    init_other=0.0,
                    log_std0=-2.3,
                    prior_std_c0=1.0,
                    prior_mu_rest=0.0,
                    prior_std_rest=0.25,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_logpoly_shrink_filter,
                step_s=0.14,
                step_theta=step_theta_logpoly_shrink,
                transform_chain=None,
            ),

            # Weak shrinkage: useful as a comparison to show overfitting/spread.
            "deg10_weak": CaseSpec(
                case_id="deg10_weak",
                display_name="logpoly_deg10_weak",
                fixed=dict(
                    degree=10,
                    floor=1e-2,
                    init_c1=-1.0,
                    init_other=0.0,
                    log_std0=-2.3,
                    prior_std_c0=1.0,
                    prior_mu_rest=0.0,
                    prior_std_rest=1.0,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_logpoly_shrink_filter,
                step_s=0.14,
                step_theta=step_theta_logpoly_shrink,
                transform_chain=None,
            ),

            # Lower-degree sanity check: correctly specified for log-linear truth.
            "deg1": CaseSpec(
                case_id="deg1",
                display_name="logpoly_deg1",
                fixed=dict(
                    degree=1,
                    floor=1e-2,
                    init_c1=-1.0,
                    init_other=0.0,
                    log_std0=-2.3,
                    prior_std_c0=1.0,
                    prior_mu_rest=0.0,
                    prior_std_rest=0.5,
                    logF_min=-30.0,
                    logF_max=30.0,
                ),
                build_filter=build_logpoly_shrink_filter,
                step_s=0.14,
                step_theta=step_theta_logpoly_shrink,
                transform_chain=None,
            ),
        },
    )
)