# examples/benchmarks/logspline.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import LogSplineFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_logspline_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    degree: int = 3,
    n_internal_knots: int = 8,
    learn_rho0: bool = False,
    prior_mu_w: float = 0.0,
    prior_std_w: float = 0.5,
    **kwargs,
):
    return LogSplineFilterFullVI(
        lam_max=float(lam_max),
        eps_car=float(eps_car),
        degree=int(degree),
        n_internal_knots=int(n_internal_knots),
        mu_log_tau2=math.log(tau2_true),
        log_std0=-2.3,
        learn_rho0=bool(learn_rho0),
        mu_rho0_raw=-6.0,
        log_std_rho0_raw=-2.3,
        prior_mu_w=float(prior_mu_w),
        prior_std_w=float(prior_std_w),
    ).to(device)

def step_theta_baseline(_filter=None) -> dict[str, float]:
    # log_tau2 moderate, spline weights a bit smaller to keep MH stable
    return {"log_tau2": 0.30, **{}}  # weights handled by default case filtering


def step_theta_learnrho(_filter=None) -> dict[str, float]:
    return {"log_tau2": 0.30, "rho0_raw": 0.20}


def step_theta_spline(filter_module) -> dict[str, float]:
    # set steps for w* based on unconstrained names
    d = {"log_tau2": 0.18}
    if "rho0_raw" in set(filter_module.unconstrained_names()):
        d["rho0_raw"] = 0.20
    for nm in filter_module.unconstrained_names():
        if nm.startswith("w"):
            d[nm] = 0.05
    return d


register(
    FilterSpec(
        filter_name="logspline",
        cases={
            "baseline": CaseSpec(
                case_id="baseline",
                display_name="logspline_deg3_k8_fixedrho",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=False),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
                transform_chain=None,
            ),
            "learnrho": CaseSpec(
                case_id="learnrho",
                display_name="logspline_deg3_k8_learnrho",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=True),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
                transform_chain=None,
            ),
            "rough": CaseSpec(
                case_id="rough",
                display_name="logspline_deg1_k12_fixedrho",
                fixed=dict(degree=1, n_internal_knots=12, learn_rho0=False),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
                transform_chain=None,
            ),
            "smooth": CaseSpec(
                case_id="smooth",
                display_name="logspline_deg3_k4_fixedrho",
                fixed=dict(degree=3, n_internal_knots=4, learn_rho0=False),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
                transform_chain=None,
            ),
            "tight_w": CaseSpec(
                case_id="tight_w",
                display_name="logspline_deg3_k8_fixedrho_wstd0p35",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=False, prior_std_w=0.35),
                build_filter=build_logspline_filter,
                step_s=0.22,
                step_theta=step_theta_spline,
            ),

            "loose_w": CaseSpec(
                case_id="loose_w",
                display_name="logspline_deg3_k8_fixedrho_wstd1p0",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=False, prior_std_w=1.0),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
            ),
            "wstd0p6": CaseSpec(
                case_id="wstd0p6",
                display_name="logspline_deg3_k8_fixedrho_wstd0p6",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=False, prior_std_w=0.6),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
            ),
            "wstd0p8": CaseSpec(
                case_id="wstd0p8",
                display_name="logspline_deg3_k8_fixedrho_wstd0p8",
                fixed=dict(degree=3, n_internal_knots=8, learn_rho0=False, prior_std_w=0.8),
                build_filter=build_logspline_filter,
                step_s=0.14,
                step_theta=step_theta_spline,
            ),
        },
    )
)