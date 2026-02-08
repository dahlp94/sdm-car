# tests/test_filters_invariants.py
from __future__ import annotations

import pytest
import torch

import examples.benchmarks  # ensures FILTER_REGISTRY is populated
from examples.benchmarks.registry import available_filters, get_filter_spec

torch.set_default_dtype(torch.double)

FILTERS = list(available_filters())

def resolve_fixed_tokens(fixed: dict, eps_car: float) -> dict:
    """
    Allow case specs to include "eps_car" as a placeholder token.
    """
    out = {}
    for k, v in fixed.items():
        if v == "eps_car":
            out[k] = eps_car
        else:
            out[k] = v
    return out

def make_fake_lam(n=128, device="cpu"):
    return torch.linspace(0.0, 10.0, n, device=device, dtype=torch.double)


@pytest.mark.parametrize("filter_name", FILTERS)
def test_filter_builds_and_spectrum_has_right_shape_and_positive(filter_name):
    device = torch.device("cpu")
    lam = make_fake_lam(128, device=device)

    spec = get_filter_spec(filter_name)

    # keep tests fast: only first 2 cases
    case_ids = list(spec.cases.keys())[:2]

    for cid in case_ids:
        case = spec.cases[cid]

        tau2_true = 0.4
        eps_car = 1e-3

        fixed = resolve_fixed_tokens(dict(case.fixed), eps_car=eps_car)

        filt = case.build_filter(tau2_true=tau2_true, eps_car=eps_car, device=device, **fixed)

        assert hasattr(filt, "mean_unconstrained"), f"{filter_name}/{cid} missing mean_unconstrained"
        assert hasattr(filt, "spectrum"), f"{filter_name}/{cid} missing spectrum"

        theta = filt.mean_unconstrained()
        F = filt.spectrum(lam, theta)

        assert F.shape == lam.shape, f"{filter_name}/{cid} spectrum shape {F.shape} != {lam.shape}"
        assert torch.isfinite(F).all(), f"{filter_name}/{cid} spectrum has NaN/Inf"
        assert (F > 0).all(), f"{filter_name}/{cid} spectrum not strictly positive"


@pytest.mark.parametrize("filter_name", FILTERS)
def test_pack_unpack_roundtrip_unconstrained(filter_name):
    device = torch.device("cpu")
    spec = get_filter_spec(filter_name)
    case = spec.cases[list(spec.cases.keys())[0]]

    tau2_true = 0.4
    eps_car = 1e-3
    filt = case.build_filter(tau2_true=tau2_true, eps_car=eps_car, device=device, **case.fixed)

    assert hasattr(filt, "pack") and hasattr(filt, "unpack"), f"{filter_name} missing pack/unpack"

    theta = filt.mean_unconstrained()
    vec = filt.pack(theta)
    theta2 = filt.unpack(vec)

    assert set(theta.keys()) == set(theta2.keys())
    for k in theta.keys():
        assert theta[k].shape == theta2[k].shape


@pytest.mark.parametrize("filter_name", FILTERS)
def test_constrain_outputs_expected_types(filter_name):
    device = torch.device("cpu")
    spec = get_filter_spec(filter_name)
    case = spec.cases[list(spec.cases.keys())[0]]

    tau2_true = 0.4
    eps_car = 1e-3
    filt = case.build_filter(tau2_true=tau2_true, eps_car=eps_car, device=device, **case.fixed)

    if not hasattr(filt, "_constrain"):
        pytest.skip(f"{filter_name} has no _constrain")

    theta = filt.mean_unconstrained()
    c = filt._constrain(theta)

    assert "tau2" in c, f"{filter_name} constrain missing tau2"
    assert torch.isfinite(c["tau2"]).all()
    assert (c["tau2"] > 0).all()
