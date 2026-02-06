# sdmcar/filters.py

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence, Union, Tuple
from .utils import softplus, kl_normal_std


@dataclass(frozen=True)
class ParamBlock:
    name: str
    param_names: Tuple[str, ...]   # which theta keys are in this block

    @staticmethod
    def single(param: str) -> "ParamBlock":
        return ParamBlock(name=param, param_names=(param,))


class BaseSpectralFilter(nn.Module):
    """
    Unifies VI + MCMC interaction by standardizing:

      - unconstrained parameter naming
      - pack/unpack to flat vectors
      - generic log prior in unconstrained space
      - spectrum_from_unconstrained(lam, theta_dict)

    Concrete filters must implement:
      - unconstrained_names()
      - blocks()
      - _constrain(theta_dict) -> dict of constrained params
      - spectrum_from_unconstrained(lam, theta_dict) -> F(lam)
      - sample_unconstrained() for VI convenience (optional but recommended)
    """

    # ---------- Required: parameter bookkeeping ----------

    def unconstrained_names(self) -> list[str]:
        raise NotImplementedError

    def blocks(self) -> list[ParamBlock]:
        """
        Default: one block per unconstrained scalar.
        Override if you want grouped proposals (e.g., joint rho0_raw+nu_raw).
        """
        return [ParamBlock.single(n) for n in self.unconstrained_names()]

    def pack(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten theta dict into a 1D tensor following unconstrained_names order.
        Each entry must be shape [1] or scalar tensor.
        """
        parts = []
        for name in self.unconstrained_names():
            t = theta[name].reshape(-1)
            parts.append(t)

        if len(parts) > 0:
            return torch.cat(parts, dim=0)

        # No unconstrained parameters -> return an empty tensor with a sane dtype/device.
        p = next(self.parameters(), None)
        if p is None:
            if len(theta) > 0:
                any_t = next(iter(theta.values()))
                return torch.zeros(0, dtype=any_t.dtype, device=any_t.device)
            return torch.zeros(0, dtype=torch.double)
        return torch.zeros(0, dtype=p.dtype, device=p.device)


    def unpack(self, theta_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inverse of pack(): returns dict(name -> tensor([..])).
        """
        names = self.unconstrained_names()
        out = {}
        i = 0
        for name in names:
            out[name] = theta_vec[i:i+1]
            i += 1
        return out

    # ---------- Generic prior on unconstrained space ----------

    def log_prior(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard normal prior over all unconstrained scalars by default:
        sum_j log N(theta_j | 0, 1)

        This matches your VI KL assumptions.
        """
        if len(self.unconstrained_names()) == 0:
            p = next(self.parameters(), None)
            if p is not None:
                return torch.zeros((), dtype=p.dtype, device=p.device)
            if len(theta) > 0:
                any_t = next(iter(theta.values()))
                return torch.zeros((), dtype=any_t.dtype, device=any_t.device)
            return torch.tensor(0.0, dtype=torch.double)

        v = self.pack(theta)
        # log N(x|0,1) = -0.5 * (x^2 + log(2π))
        return (-0.5 * (v**2 + math.log(2.0 * math.pi))).sum()


    # ---------- Constrain + spectrum ----------

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Map unconstrained theta -> constrained params (tau2, rho0, nu, a, ...).
        Concrete filters implement.
        """
        raise NotImplementedError

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return F(lam) using unconstrained theta. Concrete filters implement.
        """
        raise NotImplementedError
    
    # ---------- VI + MCMC shared hooks ----------

    def kl_q_p(self) -> torch.Tensor:
        """
        Return KL(q(theta_unconstrained) || p(theta_unconstrained)).

        VI uses this term in the ELBO.

        Concrete filters should override this if they have variational parameters.
        For non-variational / fixed filters, returning 0 is acceptable.
        """
        p = next(self.parameters(), None)
        if p is not None:
            return torch.zeros((), dtype=p.dtype, device=p.device)
        return torch.tensor(0.0, dtype=torch.double)

    @torch.no_grad()
    def theta0(self) -> dict[str, torch.Tensor]:
        """
        Default initialization for MCMC in unconstrained space.

        By default we start at the variational mean (mean_unconstrained),
        which is typically a good, stable initializer.

        Concrete filters can override if they want custom init logic.
        """
        return self.mean_unconstrained()


    # ---------- VI convenience (optional but recommended) ----------

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        """
        VI reparameterized sample in unconstrained space.
        Concrete filters implement (using their variational params).
        """
        raise NotImplementedError

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        """
        Mean in unconstrained space under q (for reporting / initialization).
        Concrete filters implement if needed; default uses available parameters.
        """
        raise NotImplementedError
    
    def spectrum(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.spectrum_from_unconstrained(lam, theta)


class DiffusionFilterFullVI(BaseSpectralFilter):
    """
    Variational diffusion filter:

        F(λ) = τ^2 * exp(-a λ),   with a = softplus(a_raw) > 0.

    Variational posteriors (full VI):
        q(log τ^2) = N(μ_τ, s_τ^2)
        q(a_raw)   = N(μ_a, s_a^2)   (softplus transform for positivity of 'a')

    Priors (on unconstrained variables):
        log τ^2 ~ N(0, 1)
        a_raw   ~ N(0, 1)
    """
    def __init__(self,
                 mu_log_tau2: float = 0.0,
                 log_std_log_tau2: float = -2.3,
                 mu_a_raw: float = 0.4,
                 log_std_a_raw: float = -2.3):
        super().__init__()

        self.mu_log_tau2 = nn.Parameter(
            torch.tensor([mu_log_tau2], dtype=torch.double)
        )
        self.log_std_log_tau2 = nn.Parameter(
            torch.tensor([log_std_log_tau2], dtype=torch.double)
        )

        self.mu_a_raw = nn.Parameter(
            torch.tensor([mu_a_raw], dtype=torch.double)
        )
        self.log_std_a_raw = nn.Parameter(
            torch.tensor([log_std_a_raw], dtype=torch.double)
        )

    def sample_params(self):
        """
        Reparameterized samples of (τ^2, a) and the unconstrained variables.

        Returns:
            tau2: scalar τ²
            a:    scalar a > 0
            log_tau2: scalar log τ² sample
            a_raw:    scalar a_raw sample
        """
        eps1 = torch.randn_like(self.mu_log_tau2)
        eps2 = torch.randn_like(self.mu_a_raw)

        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps1
        a_raw    = self.mu_a_raw    + torch.exp(self.log_std_a_raw)    * eps2

        tau2 = torch.exp(log_tau2)
        a    = softplus(a_raw)
        return tau2, a, log_tau2, a_raw

    def F(self, lam, tau2, a):
        """
        Compute F(λ) elementwise for a given sample (τ², a).

        Args:
            lam: [n] eigenvalues.
            tau2: scalar τ².
            a: scalar a > 0.

        Returns:
            F_lam: [n] spectral variances.
        """
        return tau2 * torch.exp(-a * lam)

    def kl_q_p(self):
        """
        KL( q(log τ²)||N(0,1) ) + KL( q(a_raw)||N(0,1) ).

        Returns:
            scalar KL value.
        """
        kl = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2)
        kl += kl_normal_std(self.mu_a_raw,    self.log_std_a_raw)
        return kl.sum()

    @torch.no_grad()
    def mean_params(self):
        """
        Return mean parameters under q: τ²_mean, a_mean.

        Returns:
            tau2_mean: scalar E_q[τ²]
            a_mean:    scalar E_q[a]
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        a_mean    = softplus(self.mu_a_raw)
        return tau2_mean, a_mean

    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        return self.mu_a_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        return ["log_tau2", "a_raw"]

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        _, _, log_tau2, a_raw = self.sample_params()
        return {"log_tau2": log_tau2.reshape(1), "a_raw": a_raw.reshape(1)}

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        return {"log_tau2": self.mu_log_tau2.detach().reshape(1),
                "a_raw": self.mu_a_raw.detach().reshape(1)}

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        a_raw = theta["a_raw"]
        return {
            "tau2": torch.exp(log_tau2),
            "a": softplus(a_raw),
        }

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        return self.F(lam, c["tau2"], c["a"])


class MaternLikeFilterFullVI(BaseSpectralFilter):
    """
    Variational Matérn-like spectral filter:

        F(λ) = τ² * (λ + ρ₀)^(-ν),

    with ρ₀ > 0 and ν > 0 enforced via softplus transforms:

        ρ₀ = softplus(ρ0_raw),
        ν  = softplus(nu_raw).

    Variational posteriors (full VI):
        q(log τ²)   = N(μ_τ, s_τ²)
        q(a_raw)    = N(μ_a, diag(s_a²)), where
                      a_raw = (ρ0_raw, nu_raw),

    Priors (on unconstrained variables):
        log τ²  ~ N(0, 1)
        ρ0_raw  ~ N(0, 1)
        nu_raw  ~ N(0, 1)
    """
    def __init__(self,
                 mu_log_tau2: float = 0.0,
                 log_std_log_tau2: float = -2.3,
                 mu_rho0_raw: float = 0.0,
                 log_std_rho0_raw: float = -2.3,
                 mu_nu_raw: float = 0.0,
                 log_std_nu_raw: float = -2.3,
                 fixed_nu: float | None = None,
                 fixed_rho0: float | None = None,
    ):
        super().__init__()
        if fixed_nu is not None and fixed_rho0 is not None:
            raise ValueError("Choose at most one: fixed_nu or fixed_rho0 (not both)")
        
        self.fixed_nu = fixed_nu
        self.fixed_rho0 = fixed_rho0

        # q(log τ²)
        self.mu_log_tau2 = nn.Parameter(
            torch.tensor([mu_log_tau2], dtype=torch.double)
        )
        self.log_std_log_tau2 = nn.Parameter(
            torch.tensor([log_std_log_tau2], dtype=torch.double)
        )

        # Decide which components are free in a_raw
        # a_raw holds only the free unconstrained variables, but sample_params returns a=(rho0,nu) always.
        self.learn_rho0 = (fixed_rho0 is None)
        self.learn_nu = (fixed_nu is None)

        a_mu = []
        a_lstd = []
        if self.learn_rho0:
            a_mu.append(mu_rho0_raw)
            a_lstd.append(log_std_rho0_raw)
        if self.learn_nu:
            a_mu.append(mu_nu_raw)
            a_lstd.append(log_std_nu_raw)
        
        if len(a_mu) == 0:
            self.mu_a_raw = None
            self.log_std_a_raw = None
        else:
            self.mu_a_raw = nn.Parameter(torch.tensor(a_mu, dtype=torch.double))
            self.log_std_a_raw = nn.Parameter(torch.tensor(a_lstd, dtype=torch.double))
    
    def _assemble_a(self, a_raw_free: torch.Tensor) -> torch.Tensor:
        """
        Convert free raw vars into full a = (rho0, nu) with constraints applied.
        """
        device = self.mu_log_tau2.device
        dtype = self.mu_log_tau2.dtype

        idx = 0
        # rho0
        if self.learn_rho0:
            rho0 = softplus(a_raw_free[idx])
            idx += 1
        else:
            rho0 = torch.tensor(self.fixed_rho0, dtype=dtype, device=device)

        # nu
        if self.learn_nu:
            nu = softplus(a_raw_free[idx])
            idx += 1
        else:
            nu = torch.tensor(self.fixed_nu, dtype=dtype, device=device)

        return torch.stack([rho0, nu])

    def sample_params(self):
        """
        Reparameterized samples of (τ², a) and the unconstrained variables.

        Returns:
            tau2: scalar τ²
            a:    length-2 vector (ρ₀, ν), each > 0 (or fixed)
            log_tau2: scalar log τ² sample
            a_raw:    unconstrained FREE raw vector (len 1 or 2 depending on constraint)
        """
        eps1 = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps1
        tau2 = torch.exp(log_tau2)
        
        if self.mu_a_raw is None:
            a_raw = torch.zeros(0, dtype=log_tau2.dtype, device=log_tau2.device)
            a = self._assemble_a(a_raw)
            return tau2, a, log_tau2, a_raw

        eps_a = torch.randn_like(self.mu_a_raw)
        a_raw    = self.mu_a_raw    + torch.exp(self.log_std_a_raw)    * eps_a        
        a    = self._assemble_a(a_raw)
        return tau2, a, log_tau2, a_raw

    def F(self, lam, tau2, a):
        """
        Compute F(λ) elementwise for a given sample (τ², ρ₀, ν).

        Args:
            lam:  [n] eigenvalues (λ ≥ 0).
            tau2: scalar τ².
            a:    length-2 vector (ρ₀, ν), both > 0.

        Returns:
            F_lam: [n] spectral variances.
        """
        rho0, nu = a.unbind(-1)  # both scalars if a.shape == [2]
        return tau2 * (lam + rho0).pow(-nu)
    
    def kl_q_p(self):
        """
        KL( q(log τ²)||N(0,1) ) + KL( q(a_raw)||N(0,1) ).

        Returns:
            scalar KL value.
        """
        # Each kl_normal_std returns per-dimension KLs; sum them to scalars.
        kl_log_tau2 = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()
        if self.mu_a_raw is None:
            return kl_log_tau2
        kl_a = kl_normal_std(self.mu_a_raw, self.log_std_a_raw).sum()
        return kl_log_tau2 + kl_a


    @torch.no_grad()
    def mean_params(self):
        """
        Return mean parameters under q: τ²_mean, a_mean.

        Returns:
            tau2_mean: scalar E_q[τ²]
            a_mean:    length-2 vector E_q[(ρ₀, ν)]
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        if self.mu_a_raw is None:
            a_raw = torch.zeros(0, dtype=tau2_mean.dtype, device=tau2_mean.device)
        else:
            a_raw = self.mu_a_raw
        a_mean = self._assemble_a(a_raw)
        return tau2_mean, a_mean
    
    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        if self.mu_a_raw is None:
            return torch.zeros(0, dtype=self.mu_log_tau2.dtype, device=self.mu_log_tau2.device)
        return self.mu_a_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        if self.learn_rho0:
            names.append("rho0_raw")
        if self.learn_nu:
            names.append("nu_raw")
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]

        if self.learn_rho0 and self.learn_nu:
            blocks.append(ParamBlock(name="rho0_nu_raw", param_names=("rho0_raw", "nu_raw")))
        elif self.learn_rho0:
            blocks.append(ParamBlock.single("rho0_raw"))
        elif self.learn_nu:
            blocks.append(ParamBlock.single("nu_raw"))

        return blocks


    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        tau2, a, log_tau2, a_raw_free = self.sample_params()

        # a_raw_free is either [] (len 0), [rho0_raw], [nu_raw], or [rho0_raw, nu_raw]
        out = {"log_tau2": log_tau2.reshape(1)}
        idx = 0
        if self.learn_rho0:
            out["rho0_raw"] = a_raw_free[idx:idx+1]; idx += 1
        if self.learn_nu:
            out["nu_raw"] = a_raw_free[idx:idx+1]; idx += 1
        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {"log_tau2": self.mu_log_tau2.detach().reshape(1)}
        if self.learn_rho0 or self.learn_nu:
            # mu_a_raw stores free components in order
            idx = 0
            if self.learn_rho0:
                out["rho0_raw"] = self.mu_a_raw.detach()[idx:idx+1]; idx += 1
            if self.learn_nu:
                out["nu_raw"] = self.mu_a_raw.detach()[idx:idx+1]; idx += 1
        return out

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        tau2 = torch.exp(log_tau2)

        # rho0
        if self.learn_rho0:
            rho0 = softplus(theta["rho0_raw"])
        else:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)

        # nu
        if self.learn_nu:
            nu = softplus(theta["nu_raw"])
        else:
            nu = torch.tensor(self.fixed_nu, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)

        return {"tau2": tau2, "rho0": rho0, "nu": nu}

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        # reuse your F API: a = (rho0, nu)
        rho0 = c["rho0"].reshape(())
        nu   = c["nu"].reshape(())
        a = torch.stack([rho0, nu])

        return self.F(lam, c["tau2"], a)



class InverseLinearCARFilterFullVI(BaseSpectralFilter):
    """
    Variational inverse-linear (CAR-exact) spectral covariance filter:

        F(λ) = τ² / (λ + ρ₀)

    with τ² > 0 and ρ₀ > 0 enforced by:
        τ² = exp(log_tau2)
        ρ₀ = softplus(rho0_raw)

    Variational posteriors:
        q(log τ²)     = N(μ_τ, s_τ²)
        q(rho0_raw)   = N(μ_ρ, s_ρ²)

    Priors (on unconstrained variables):
        log τ²   ~ N(0, 1)
        rho0_raw ~ N(0, 1)
    """

    def __init__(
        self,
        mu_log_tau2: float = 0.0,
        log_std_log_tau2: float = -2.3,
        mu_rho0_raw: float = 0.0,
        log_std_rho0_raw: float = -2.3,
        fixed_rho0: float | None = None,
    ):
        super().__init__()
        self.fixed_rho0 = fixed_rho0

        # q(log τ²)
        self.mu_log_tau2 = nn.Parameter(torch.tensor([mu_log_tau2], dtype=torch.double))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([log_std_log_tau2], dtype=torch.double))

        # q(rho0_raw) unless fixed
        if fixed_rho0 is None:
            self.mu_rho0_raw = nn.Parameter(torch.tensor([mu_rho0_raw], dtype=torch.double))
            self.log_std_rho0_raw = nn.Parameter(torch.tensor([log_std_rho0_raw], dtype=torch.double))
        else:
            self.mu_rho0_raw = None
            self.log_std_rho0_raw = None

    def sample_params(self):
        """
        Reparameterized samples.

        Returns:
            tau2: scalar τ²
            rho0: scalar ρ₀
            log_tau2: scalar log τ² sample
            rho0_raw: scalar rho0_raw sample (or empty tensor if fixed)
        """
        eps_tau = torch.randn_like(self.mu_log_tau2)
        log_tau2 = self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps_tau
        tau2 = torch.exp(log_tau2)

        if self.fixed_rho0 is not None:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device)
            rho0_raw = torch.zeros(0, dtype=log_tau2.dtype, device=log_tau2.device)
            return tau2, rho0, log_tau2, rho0_raw

        eps_rho = torch.randn_like(self.mu_rho0_raw)
        rho0_raw = self.mu_rho0_raw + torch.exp(self.log_std_rho0_raw) * eps_rho
        rho0 = softplus(rho0_raw)
        return tau2, rho0, log_tau2, rho0_raw

    def F(self, lam: torch.Tensor, tau2: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        """
        Compute F(λ) elementwise.

        Args:
            lam:  [n] eigenvalues (λ ≥ 0)
            tau2: scalar τ²
            rho0: scalar ρ₀

        Returns:
            F_lam: [n] spectral variances
        """
        return tau2 / (lam + rho0)

    def kl_q_p(self) -> torch.Tensor:
        """
        KL(q(log τ²)||N(0,1)) + KL(q(rho0_raw)||N(0,1)) (if rho0 not fixed).
        """
        kl = kl_normal_std(self.mu_log_tau2, self.log_std_log_tau2).sum()
        if self.fixed_rho0 is None:
            kl = kl + kl_normal_std(self.mu_rho0_raw, self.log_std_rho0_raw).sum()
        return kl

    @torch.no_grad()
    def mean_params(self):
        """
        Return mean-ish parameters under q (using transforms of means).

        Returns:
            tau2_mean: scalar exp(mu_log_tau2)
            rho0_mean: scalar softplus(mu_rho0_raw) or fixed value
        """
        tau2_mean = torch.exp(self.mu_log_tau2)
        if self.fixed_rho0 is not None:
            rho0_mean = torch.tensor(self.fixed_rho0, dtype=tau2_mean.dtype, device=tau2_mean.device)
        else:
            rho0_mean = softplus(self.mu_rho0_raw)
        return tau2_mean, rho0_mean
    
    def init_a_raw(self) -> torch.Tensor:
        """Return raw free variable(s) for initializing MCMC (shape [d] or empty)."""
        if self.fixed_rho0 is not None or self.mu_rho0_raw is None:
            return torch.zeros(0, dtype=self.mu_log_tau2.dtype, device=self.mu_log_tau2.device)
        return self.mu_rho0_raw.detach()
    
    # --------- Universal interface ---------

    def unconstrained_names(self) -> list[str]:
        names = ["log_tau2"]
        if self.fixed_rho0 is None:
            names.append("rho0_raw")
        return names

    def blocks(self) -> list[ParamBlock]:
        blocks = [ParamBlock.single("log_tau2")]
        if self.fixed_rho0 is None:
            blocks.append(ParamBlock.single("rho0_raw"))
        return blocks

    def sample_unconstrained(self) -> dict[str, torch.Tensor]:
        tau2, rho0, log_tau2, rho0_raw = self.sample_params()
        out = {"log_tau2": log_tau2.reshape(1)}
        if self.fixed_rho0 is None:
            out["rho0_raw"] = rho0_raw.reshape(1)
        return out

    @torch.no_grad()
    def mean_unconstrained(self) -> dict[str, torch.Tensor]:
        out = {"log_tau2": self.mu_log_tau2.detach().reshape(1)}
        if self.fixed_rho0 is None and self.mu_rho0_raw is not None:
            out["rho0_raw"] = self.mu_rho0_raw.detach().reshape(1)
        return out

    def _constrain(self, theta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_tau2 = theta["log_tau2"]
        tau2 = torch.exp(log_tau2)

        if self.fixed_rho0 is not None:
            rho0 = torch.tensor(self.fixed_rho0, dtype=log_tau2.dtype, device=log_tau2.device).reshape(1)
        else:
            rho0 = softplus(theta["rho0_raw"])
        return {"tau2": tau2, "rho0": rho0}

    def spectrum_from_unconstrained(self, lam: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        c = self._constrain(theta)
        return self.F(lam, c["tau2"], c["rho0"])


