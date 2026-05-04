from __future__ import annotations

import math
import torch

from sdmcar.filters import BernsteinLogSpectrumFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


def build_bernstein_log(
    *,
    tau2_true: float,
    eps_car: float,
    device: torch.device,
    lam_max: float,
    degree: int = 5,
    floor: float = 1e-6,
    prior_mu: float = 0.0,
    prior_std: float = 0.7,
    log_std0: float = -2.3,
    **kwargs,
):
    return BernsteinLogSpectrumFilterFullVI(
        lam_max=float(lam_max),
        degree=int(degree),
        floor=float(floor),
        mu_log_tau2=math.log(max(float(tau2_true), 1e-12)),
        prior_mu=float(prior_mu),
        prior_std=float(prior_std),
        log_std0=float(log_std0),
    ).to(device)


def step_theta_bernstein(filter_module) -> dict[str, float]:
    d: dict[str, float] = {}

    for nm in filter_module.unconstrained_names():
        if nm == "log_tau2":
            d[nm] = 0.16
        elif nm.startswith("c") and nm.endswith("_raw"):
            # Smaller steps for high-dimensional signed coefficients.
            d[nm] = 0.08
        else:
            d[nm] = 0.08

    return d


register(
    FilterSpec(
        filter_name="bernstein_log",
        cases={
            "deg3": CaseSpec(
                case_id="deg3",
                display_name="bernstein_log_deg3",
                fixed=dict(degree=3, floor=1e-6, prior_std=0.7),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg5": CaseSpec(
                case_id="deg5",
                display_name="bernstein_log_deg5",
                fixed=dict(degree=5, floor=1e-6, prior_std=0.7),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg10": CaseSpec(
                case_id="deg10",
                display_name="bernstein_log_deg10",
                fixed=dict(degree=10, floor=1e-6, prior_std=0.5),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg15": CaseSpec(
                case_id="deg15",
                display_name="bernstein_log_deg15",
                fixed=dict(degree=15, floor=1e-6, prior_std=0.4),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),

            "deg5_loose": CaseSpec(
                case_id="deg5_loose",
                display_name="bernstein_log_deg5_loose",
                fixed=dict(degree=5, floor=1e-6, prior_std=1.0),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg10_loose": CaseSpec(
                case_id="deg10_loose",
                display_name="bernstein_log_deg10_loose",
                fixed=dict(degree=10, floor=1e-6, prior_std=1.0),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg15_loose": CaseSpec(
                case_id="deg15_loose",
                display_name="bernstein_log_deg15_loose",
                fixed=dict(degree=15, floor=1e-6, prior_std=0.8),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg10_explore": CaseSpec(
                case_id="deg10_explore",
                display_name="bernstein_log_deg10_explore",
                fixed=dict(degree=10, floor=1e-6, prior_std=1.0, log_std0=-1.5),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
            "deg10_medium": CaseSpec(
                case_id="deg10_medium",
                display_name="bernstein_log_deg10_medium",
                fixed=dict(degree=10, floor=1e-6, prior_std=0.8, log_std0=-1.5),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),

            "deg10_loose_explore": CaseSpec(
                case_id="deg10_loose_explore",
                display_name="bernstein_log_deg10_loose_explore",
                fixed=dict(degree=10, floor=1e-6, prior_std=1.0, log_std0=-1.5),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),

            "deg10_very_loose": CaseSpec(
                case_id="deg10_very_loose",
                display_name="bernstein_log_deg10_very_loose",
                fixed=dict(degree=10, floor=1e-6, prior_std=1.25, log_std0=-1.5),
                build_filter=build_bernstein_log,
                step_s=0.12,
                step_theta=step_theta_bernstein,
                transform_chain=None,
            ),
        },
    )
)