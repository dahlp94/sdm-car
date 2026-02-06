# examples/benchmarks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

import torch


@dataclass
class CaseSpec:
    """
    One runnable benchmark case (baseline/ablations) for a given filter family.
    """
    case_id: str                 # e.g. "baseline", "B1", "B2", "K2", ...
    display_name: str            # e.g. "B1_fix_nu_1"
    fixed: Dict[str, Any]        # filter-specific fixed params (may include tokens like "eps_car")

    build_filter: Callable[..., torch.nn.Module]
    #steps: Callable[[], StepSizes]
    # - step_s: float
    # - step_theta: dict or callable returning dict
    step_s: float = 0.15
    step_theta: Optional[Dict[str, float] | Callable[[Any], Dict[str, float]]] = None

    # Optional: map MCMC raw chain -> named parameter chains for printing/plots.
    # Signature: fn(out, fixed_resolved, eps_car) -> dict[str, np.ndarray]
    transform_chain: Optional[Callable[..., Dict[str, Any]]] = None

    def get_step_theta(self, filter_module) -> Dict[str, float]:
        """
        Return dict of RW step sizes keyed by filter unconstrained parameter names.
        Automatically filters out keys that aren't used by this filter (fixed params).
        """
        names = set(filter_module.unconstrained_names())

        if self.step_theta is None:
            # sensible default
            d = {name: 0.10 for name in names}
            if "log_tau2" in d:
                d["log_tau2"] = 0.15
            return d

        raw = self.step_theta(filter_module) if callable(self.step_theta) else dict(self.step_theta)
        # keep only relevant keys
        return {k: float(v) for k, v in raw.items() if k in names}



@dataclass
class FilterSpec:
    """
    Defines a filter family and the set of valid cases for it.
    """
    filter_name: str
    cases: Dict[str, CaseSpec]   # keys are case_id strings
