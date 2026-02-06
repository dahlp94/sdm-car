# examples/benchmarks/registry.py
from __future__ import annotations

from typing import Dict
from .base import FilterSpec

FILTER_REGISTRY: Dict[str, FilterSpec] = {}


def register(spec: FilterSpec) -> FilterSpec:
    if spec.filter_name in FILTER_REGISTRY:
        raise ValueError(f"Duplicate filter registration: {spec.filter_name}")
    FILTER_REGISTRY[spec.filter_name] = spec
    return spec


def get_filter_spec(name: str) -> FilterSpec:
    if name not in FILTER_REGISTRY:
        raise ValueError(f"Unknown filter '{name}'. Available: {list(FILTER_REGISTRY)}")
    return FILTER_REGISTRY[name]


def available_filters() -> list[str]:
    return sorted(FILTER_REGISTRY.keys())
