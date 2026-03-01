# examples/benchmarks/multiscale_bump.py
from __future__ import annotations

import math
import torch

from sdmcar.filters import MultiScaleBumpFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_multiscale_bump_filter(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    K: int = 2,
    s_min: float = 0.05,
    prior_mu: float = 0.0,
    prior_std: float = 1.0,
    log_std0: float = -2.3,
    **kwargs,
):
    # init amplitudes so bumps start around the right scale
    # (tau2_true here is just a rough calibration hint, like logspline does)
    mu0_a = math.log(max(float(tau2_true), 1e-12))

    return MultiScaleBumpFilterFullVI(
        lam_max=float(lam_max),
        eps_car=float(eps_car),
        K=int(K),
        mu_log_tau2=math.log(max(float(tau2_true), 1e-12)),
        log_std0=-2.3,
    ).to(device)


def step_theta_multiscale(filter_module) -> dict[str, float]:
    """
    MH proposal step sizes in unconstrained space
    for MultiScaleBumpFilterFullVI.

    Unconstrained names now include:
      log_tau2
      a{k}_raw
      m{k}_raw
      log_s{k}_raw
      alpha{k}_raw
    """

    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        # Global scale (important — keep moderate)
        if nm == "log_tau2":
            d[nm] = 0.18
        # Log-amplitudes
        elif nm.startswith("a") and nm.endswith("_raw"):
            d[nm] = 0.18
        # Centers (constrained via sigmoid, so slightly larger ok)
        elif nm.startswith("m") and nm.endswith("_raw"):
            d[nm] = 0.22
        # Widths (softplus-constrained, can be sensitive)
        elif nm.startswith("log_s") and nm.endswith("_raw"):
            d[nm] = 0.18
        # Mixture logits (softmax downstream → moderate)
        elif nm.startswith("alpha") and nm.endswith("_raw"):
            d[nm] = 0.15
        # Fallback (should not trigger, but safe)
        else:
            d[nm] = 0.18
    return d

register(
    FilterSpec(
        filter_name="multiscale_bump",
        cases={
            # --- default: 2-band multiscale ---
            "k2": CaseSpec(
                case_id="k2",
                display_name="msbump_k2",
                fixed=dict(K=2, s_min=0.05),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
            # --- slightly more flexible: 3 bands ---
            "k3": CaseSpec(
                case_id="k3",
                display_name="msbump_k3",
                fixed=dict(K=3, s_min=0.05),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
            # --- wider bumps (more smooth / stable) ---
            "wide": CaseSpec(
                case_id="wide",
                display_name="msbump_k2_wide",
                fixed=dict(K=2, s_min=0.12),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
            # --- narrower bumps (more sharp / expressive) ---
            "narrow": CaseSpec(
                case_id="narrow",
                display_name="msbump_k2_narrow",
                fixed=dict(K=2, s_min=0.03),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
            # --- tighter prior (more regularization) ---
            "tight_prior": CaseSpec(
                case_id="tight_prior",
                display_name="msbump_k2_priorstd0p6",
                fixed=dict(K=2, s_min=0.05, prior_std=0.6),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
            # --- looser prior (more freedom) ---
            "loose_prior": CaseSpec(
                case_id="loose_prior",
                display_name="msbump_k2_priorstd1p5",
                fixed=dict(K=2, s_min=0.05, prior_std=1.5),
                build_filter=build_multiscale_bump_filter,
                step_s=0.14,
                step_theta=step_theta_multiscale,
                transform_chain=None,
            ),
        },
    )
)