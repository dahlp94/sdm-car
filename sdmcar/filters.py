# sdmcar/filters.py

import torch
import torch.nn as nn
from .utils import softplus, kl_normal_std

class DiffusionFilterFullVI(nn.Module):
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

class MaternLikeFilterFullVI(nn.Module):
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
            raise ValueError("Choose at most one: fixed_nu or fixed)rho0 (not both)")
        
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
