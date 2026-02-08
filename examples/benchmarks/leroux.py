# examples/benchmarks/leroux.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import LerouxCARFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_leroux(
    *,
    tau2_true: float,
    eps_car: float,            # unused but kept for signature compatibility
    device: torch.device,
    fixed_rho: float | None = None,
    **kwargs,
):
    return LerouxCARFilterFullVI(
        mu_log_tau2=math.log(tau2_true),
        log_std_log_tau2=-2.3,
        mu_rho_raw=0.0,
        log_std_rho_raw=-2.3,
        fixed_rho=fixed_rho,
        rho_eps=1e-4,
    ).to(device)


register(FilterSpec(
    filter_name="leroux",
    cases={
        "learn_rho": CaseSpec(
            case_id="learn_rho",
            display_name="leroux_learn_rho",
            fixed={"fixed_rho": None},
            build_filter=build_leroux,
            step_s=0.12,
            step_theta={"log_tau2": 0.20, "rho_raw": 0.10},
        ),
        "fix_rho_095": CaseSpec(
            case_id="fix_rho_095",
            display_name="leroux_fix_rho_0p95",
            fixed={"fixed_rho": 0.95},
            build_filter=build_leroux,
            step_s=0.12,
            step_theta={"log_tau2": 0.20},
        ),
        "fix_rho_099": CaseSpec(
            case_id="fix_rho_099",
            display_name="leroux_fix_rho_0p99",
            fixed={"fixed_rho": 0.99},
            build_filter=build_leroux,
            step_s=0.12,
            step_theta={"log_tau2": 0.20},
        ),
        "fix_rho_000": CaseSpec(
            case_id="fix_rho_000",
            display_name="leroux_fix_rho_0p00",
            fixed={"fixed_rho": 0.0},
            build_filter=build_leroux,
            step_s=0.12,
            step_theta={"log_tau2": 0.20},
        ),
    }
))
