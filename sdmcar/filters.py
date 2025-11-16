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
