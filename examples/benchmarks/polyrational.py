# examples/benchmarks/polyrational.py
from __future__ import annotations
import math
import torch

from sdmcar.filters import PolyPosCoeffFilterFullVI, RationalPosCoeffFilterFullVI
from .base import FilterSpec, CaseSpec
from .registry import register


# -------------------------
# POLY
# -------------------------
def build_poly(*, tau2_true: float, eps_car: float, device: torch.device, degree: int = 3, **kwargs):
    return PolyPosCoeffFilterFullVI(
        degree=degree,
        mu_log_tau2=math.log(tau2_true),
        log_std0=-2.3,
    ).to(device)


register(FilterSpec(
    filter_name="poly",
    cases={
        "deg1": CaseSpec(
            case_id="deg1",
            display_name="poly_deg1",
            fixed={"degree": 1},
            build_filter=build_poly,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.15 for nm in f.unconstrained_names() if nm.startswith("a")}},
        ),
        "deg3": CaseSpec(
            case_id="deg3",
            display_name="poly_deg3",
            fixed={"degree": 3},
            build_filter=build_poly,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.12 for nm in f.unconstrained_names() if nm.startswith("a")}},
        ),
        "deg5": CaseSpec(
            case_id="deg5",
            display_name="poly_deg5",
            fixed={"degree": 5},
            build_filter=build_poly,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.10 for nm in f.unconstrained_names() if nm.startswith("a")}},
        ),
    }
))


# -------------------------
# RATIONAL
# -------------------------
def build_rational(*, tau2_true: float, eps_car: float, device: torch.device, deg_num: int = 0, deg_den: int = 1,
                   joint_ab: bool = True, **kwargs):
    return RationalPosCoeffFilterFullVI(
        deg_num=deg_num,
        deg_den=deg_den,
        mu_log_tau2=math.log(tau2_true),
        log_std0=-2.3,
        eps_den=1e-12,
        joint_ab=joint_ab,
    ).to(device)


register(FilterSpec(
    filter_name="rational",
    cases={
        # CAR-like: numerator constant, denominator linear
        "car_like_01": CaseSpec(
            case_id="car_like_01",
            display_name="rat_num0_den1",
            fixed={"deg_num": 0, "deg_den": 1, "joint_ab": True},
            build_filter=build_rational,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.10 for nm in f.unconstrained_names() if nm != "log_tau2"}},
        ),
        # more flexible
        "flex_11": CaseSpec(
            case_id="flex_11",
            display_name="rat_num1_den1",
            fixed={"deg_num": 1, "deg_den": 1, "joint_ab": True},
            build_filter=build_rational,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.08 for nm in f.unconstrained_names() if nm != "log_tau2"}},
        ),
        "flex_21": CaseSpec(
            case_id="flex_21",
            display_name="rat_num2_den1",
            fixed={"deg_num": 2, "deg_den": 1, "joint_ab": True},
            build_filter=build_rational,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.07 for nm in f.unconstrained_names() if nm != "log_tau2"}},
        ),
        "flex_22": CaseSpec(
            case_id="flex_22",
            display_name="rat_num2_den2",
            fixed={"deg_num": 2, "deg_den": 2, "joint_ab": True},
            build_filter=build_rational,
            step_s=0.12,
            step_theta=lambda f: {"log_tau2": 0.20, **{nm: 0.06 for nm in f.unconstrained_names() if nm != "log_tau2"}},
        ),
    }
))
