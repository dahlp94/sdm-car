# examples/benchmarks/classic_car.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import ClassicCARFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_classic_car(*, tau2_true: float, eps_car: float, device: torch.device, **kwargs):
    return ClassicCARFilterFullVI(
        eps_car=eps_car,
        mu_log_tau2=math.log(max(tau2_true, 1e-12)),
        log_std_log_tau2=-2.3,
    ).to(device)


register(
    FilterSpec(
        filter_name="classic_car",
        cases={
            "baseline": CaseSpec(
                case_id="baseline",
                display_name="car_baseline",
                fixed={},
                build_filter=build_classic_car,
                step_s=0.12,
                step_theta={"log_tau2": 0.20},
            ),
        },
    )
)
