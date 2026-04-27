# examples/benchmarks/diffusion_kernel.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import DiffusionKernelFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_diffusion_kernel_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    kappa0: float = 3.0,
    prior_mu: float = 0.0,
    prior_std: float = 1.0,
    log_std0: float = -2.3,
    **kwargs,
):
    return DiffusionKernelFilterFullVI(
        lam_max=float(lam_max),
        mu_log_tau2=math.log(max(float(tau2_true), 1e-12)),
        mu_log_kappa=math.log(max(float(kappa0), 1e-12)),
        log_std0=float(log_std0),
        prior_mu=float(prior_mu),
        prior_std=float(prior_std),
    ).to(device)


def step_theta_diffusion(filter_module) -> dict[str, float]:
    return {
        "log_tau2": 0.18,
        "log_kappa": 0.18,
    }


register(
    FilterSpec(
        filter_name="diffusion_kernel",
        cases={
            "baseline": CaseSpec(
                case_id="baseline",
                display_name="diffusion_baseline",
                fixed=dict(kappa0=3.0),
                build_filter=build_diffusion_kernel_filter,
                step_s=0.14,
                step_theta=step_theta_diffusion,
                transform_chain=None,
            ),
            "smooth": CaseSpec(
                case_id="smooth",
                display_name="diffusion_smooth",
                fixed=dict(kappa0=6.0),
                build_filter=build_diffusion_kernel_filter,
                step_s=0.14,
                step_theta=step_theta_diffusion,
                transform_chain=None,
            ),
            "weak": CaseSpec(
                case_id="weak",
                display_name="diffusion_weak",
                fixed=dict(kappa0=1.0),
                build_filter=build_diffusion_kernel_filter,
                step_s=0.14,
                step_theta=step_theta_diffusion,
                transform_chain=None,
            ),
        },
    )
)