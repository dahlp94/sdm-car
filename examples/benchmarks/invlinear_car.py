# examples/benchmarks/invlinear_car.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import InverseLinearCARFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_invlinear_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    fixed_rho0=None,
    **kwargs,
):
    return InverseLinearCARFilterFullVI(
        mu_log_tau2=math.log(tau2_true),
        log_std_log_tau2=-2.0,
        mu_rho0_raw=-7.0,
        log_std_rho0_raw=-2.5,
        fixed_rho0=fixed_rho0,   # IMPORTANT: do not force eps here
    ).to(device)


register(
    FilterSpec(
        filter_name="invlinear_car",
        cases={
            "baseline": CaseSpec(
                case_id="baseline",
                display_name="baseline_learn_rho0",
                fixed=dict(fixed_rho0=None),
                build_filter=build_invlinear_filter,
                # MCMC steps:
                step_s=0.12,
                step_theta={
                    # names must match filter_module.unconstrained_names()
                    "log_tau2": 0.25,
                    "rho0_raw": 0.50,
                },
                # transform_chain not needed anymore; run_benchmark decodes rho0 via theta_constr
                transform_chain=None,
            ),
            "fix_rho0": CaseSpec(
                case_id="fix_rho0",
                display_name="fix_rho0_eps",
                fixed=dict(fixed_rho0="eps_car"),
                build_filter=build_invlinear_filter,
                step_s=0.12,
                # rho0 is fixed -> filter.unconstrained_names() should only include log_tau2
                step_theta={
                    "log_tau2": 0.25,
                    # "rho0_raw" will be filtered out automatically by get_step_theta()
                },
                transform_chain=None,
            ),
        },
    )
)
