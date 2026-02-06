# examples/benchmarks/matern.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import MaternLikeFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_matern_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    fixed_nu=None,
    fixed_rho0=None,
    **kwargs,
):
    return MaternLikeFilterFullVI(
        mu_log_tau2=math.log(tau2_true),
        log_std_log_tau2=-2.0,
        mu_rho0_raw=-7.0 if fixed_rho0 is None else 0.0,
        log_std_rho0_raw=-2.5,
        mu_nu_raw=0.5,
        log_std_nu_raw=-2.0,
        fixed_nu=fixed_nu,
        fixed_rho0=fixed_rho0,
    ).to(device)


# -------------------------
# MCMC step sizes (packed-theta API)
# -------------------------
# Keys MUST match filter.unconstrained_names():
#   baseline matern: ["log_tau2", "rho0_raw", "nu_raw"]
#   B1 (fixed_nu):   ["log_tau2", "rho0_raw"]
#   B2 (fixed_rho0): ["log_tau2", "nu_raw"]

def step_theta_baseline(_filter_module=None) -> dict[str, float]:
    # (old StepSizes: t=0.35, rho_raw=0.90, nu_raw=0.10)
    # Note: keep rho0_raw step modest or you’ll get low acceptance; 0.90 is huge.
    return {"log_tau2": 0.35, "rho0_raw": 0.25, "nu_raw": 0.10}


def step_theta_B1(_filter_module=None) -> dict[str, float]:
    # old: t=0.35, rho_raw=0.60
    return {"log_tau2": 0.35, "rho0_raw": 0.20}


def step_theta_B2(_filter_module=None) -> dict[str, float]:
    # old: t=0.35, nu_raw=0.25
    return {"log_tau2": 0.35, "nu_raw": 0.25}


register(
    FilterSpec(
        filter_name="matern",
        cases={
            "baseline": CaseSpec(
                case_id="baseline",
                display_name="baseline",
                fixed=dict(fixed_nu=None, fixed_rho0=None),
                build_filter=build_matern_filter,
                # new fields (see base.py changes)
                step_s=0.16,
                step_theta=step_theta_baseline,
                transform_chain=None,  # decoding is now generic
            ),
            "B1": CaseSpec(
                case_id="B1",
                display_name="B1_fix_nu_1",
                fixed=dict(fixed_nu=1.0, fixed_rho0=None),
                build_filter=build_matern_filter,
                step_s=0.12,
                step_theta=step_theta_B1,
                transform_chain=None,
            ),
            "B2": CaseSpec(
                case_id="B2",
                display_name="B2_fix_rho0_eps",
                fixed=dict(fixed_nu=None, fixed_rho0="eps_car"),
                build_filter=build_matern_filter,
                step_s=0.12,
                step_theta=step_theta_B2,
                transform_chain=None,
            ),
        },
    )
)
