# sdmcar/mcmc.py
#
# Collapsed Metropolis-within-Gibbs MCMC for SDM-CAR regression with
# Matérn-like spectral covariance:
#
#   F(λ) = τ² * (λ + ρ0)^(-ν)
#
# We sample the *collapsed* posterior (phi integrated out):
#
#   y | beta, hypers ~ N(X beta,  Σ(hypers))
#   Σ(hypers) = U diag(F(λ; theta) + σ²) U^T
#
# State (recommended, matches your priors):
#   beta           (p-dim)
#   s = log σ²      (1-dim)
#   t = log τ²      (1-dim)
#   a_raw = (rho0_raw, nu_raw)  (2-dim)
# with transforms:
#   τ² = exp(t)
#   ρ0 = softplus(rho0_raw)
#   ν  = softplus(nu_raw)
#
# Updates:
#   1) beta | (s,t,a_raw), y  : Gibbs (Gaussian)
#   2) s                      : RW-MH
#   3) t                      : RW-MH
#   4) a_raw                  : RW-MH (block or coordinate-wise)
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import math
import torch
from torch.distributions import Normal


# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class StepSizes:
    s: float = 0.15        # RW std for s = log sigma^2
    t: float = 0.15        # RW std for t = log tau^2
    rho_raw: float = 0.10  # RW std for rho0_raw
    nu_raw: float = 0.10   # RW std for nu_raw


@dataclass
class MCMCConfig:
    num_steps: int = 20_000
    burnin: int = 5_000
    thin: int = 10
    step: StepSizes = field(default_factory=StepSizes)
    block_a_raw: bool = True
    seed: Optional[int] = None
    # numerical stability
    var_jitter: float = 1e-12         # clamp for marginal spectral variances
    Vbeta_jitter: float = 1e-10       # jitter on V_beta_inv diagonal
    device: Optional[torch.device] = None


# -----------------------------
# Collapsed sampler
# -----------------------------

class CollapsedSDMCARMCMC:
    """
    Collapsed MCMC baseline for SDM-CAR regression.

    Inputs:
      - X_tilde = U^T X   [n, p]
      - y_tilde = U^T y   [n]
      - lam             [n]
      - beta prior: beta ~ N(m0, V0)

    sigma prior is specified in terms of s = log sigma^2:
      - logsigma2_normal : s ~ N(mu, std^2) (default)
      - logsigma_normal  : log sigma ~ N(mu, std^2)  (adds Jacobian)
      - sigma_halfcauchy : sigma ~ HalfCauchy(scale)
      - sigma2_invgamma  : sigma^2 ~ InvGamma(alpha,beta)
      - jeffreys_trunc   : uniform in s on [lo,hi]
    """

    def __init__(
        self,
        X_tilde: torch.Tensor,
        y_tilde: torch.Tensor,
        lam: torch.Tensor,
        m0: torch.Tensor,
        V0: torch.Tensor,
        sigma_prior: str = "logsigma2_normal",
        sigma_prior_params: Optional[dict] = None,
        fixed_nu: float | None = None,
        fixed_rho0: float | None = None,
        config: Optional[MCMCConfig] = None,
    ):
        self.X_tilde = X_tilde
        self.y_tilde = y_tilde
        self.lam = lam

        self.m0 = m0
        self.V0 = V0
        self.V0_inv = torch.inverse(V0)

        self.sigma_prior = sigma_prior
        self.sigma_prior_params = {} if sigma_prior_params is None else sigma_prior_params

        if fixed_nu is not None and fixed_rho0 is not None:
            raise ValueError("Choose at most one: fixed_nu or fixed_rho0.")
        self.fixed_nu = fixed_nu
        self.fixed_rho0 = fixed_rho0

        self.cfg = MCMCConfig() if config is None else config
        if self.cfg.device is None:
            self.cfg.device = self.X_tilde.device

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        self.n, self.p = self.X_tilde.shape

        # acceptance counters
        self.acc_s = 0
        self.acc_t = 0
        self.acc_a = 0
        self.tried_s = 0
        self.tried_t = 0
        self.tried_a = 0

    # -------------------------
    # helpers
    # -------------------------

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    @staticmethod
    def _as_scalar(x: torch.Tensor) -> torch.Tensor:
        # allow shape [] or [1]
        return x.reshape(())

    # -------------------------
    # spectrum + likelihood
    # -------------------------

    def compute_F(self, t: torch.Tensor, a_raw: torch.Tensor) -> torch.Tensor:
        """
        Matérn-like covariance spectrum:
            F_i = exp(t) * (λ_i + softplus(rho_raw))^(-softplus(nu_raw))
        """
        t0 = self._as_scalar(t)
        tau2 = torch.exp(t0)  # scalar

        # a_raw contains only free variables:
        # - baseline: [rho_raw, nu_raw]
        # - B1 fixed_nu: [rho_raw]
        # - B2 fixed_rho0: [nu_raw]
        idx = 0

        if self.fixed_rho0 is None:
            rho_raw = a_raw[idx]
            rho0 = self.softplus(rho_raw)
            idx += 1
        else:
            rho0 = torch.tensor(self.fixed_rho0, dtype=tau2.dtype, device=tau2.device)
        
        if self.fixed_nu is None:
            nu_raw = a_raw[idx]
            nu = self.softplus(nu_raw)
            idx += 1
        else:
            nu = torch.tensor(self.fixed_nu, dtype=tau2.dtype, device=tau2.device)
        
        # (lam + rho0)^(-nu)
        return tau2 * (self.lam + rho0).pow(-nu)

    def loglik(self, beta: torch.Tensor, s: torch.Tensor, t: torch.Tensor, a_raw: torch.Tensor) -> torch.Tensor:
        """
        Collapsed log p(y | beta, s,t,a_raw) up to constant.
        """
        s0 = self._as_scalar(s)
        sigma2 = torch.exp(s0)

        F = self.compute_F(t, a_raw)
        var = (F + sigma2).clamp_min(self.cfg.var_jitter)

        r = self.y_tilde - self.X_tilde @ beta
        ll = -0.5 * torch.sum(torch.log(var) + (r * r) / var)
        return ll

    # -------------------------
    # priors
    # -------------------------

    def logprior_beta(self, beta: torch.Tensor) -> torch.Tensor:
        d = beta - self.m0
        return -0.5 * (d.unsqueeze(0) @ self.V0_inv @ d.unsqueeze(1)).reshape(())

    def logprior_t(self, t: torch.Tensor) -> torch.Tensor:
        t0 = self._as_scalar(t)
        return -0.5 * (t0 * t0)

    def logprior_a_raw(self, a_raw: torch.Tensor) -> torch.Tensor:
        # Standard normal on each FREE raw coordinate
        return -0.5 * torch.sum(a_raw * a_raw)

    def logprior_s(self, s: torch.Tensor) -> torch.Tensor:
        """
        log p(s) where s = log sigma^2.
        Mirrors your models.py `_log_p_s`.
        """
        s0 = self._as_scalar(s)
        name = self.sigma_prior
        p = self.sigma_prior_params
        dtype = s0.dtype
        device = s0.device

        if name == "logsigma2_normal":
            mu = torch.tensor(p.get("mu", 0.0), dtype=dtype, device=device)
            std = torch.tensor(p.get("std", 1.0), dtype=dtype, device=device)
            return Normal(mu, std).log_prob(s0)

        elif name == "logsigma_normal":
            # t = log sigma ~ N(mu, std^2), where s = log sigma^2 = 2 t => t = 0.5 s
            mu = torch.tensor(p.get("mu", 0.0), dtype=dtype, device=device)
            std = torch.tensor(p.get("std", 1.0), dtype=dtype, device=device)
            t = 0.5 * s0
            # p(s) = p(t) * |dt/ds| = p(t) * 0.5
            return Normal(mu, std).log_prob(t) + math.log(0.5)

        elif name == "sigma_halfcauchy":
            # sigma ~ HalfCauchy(scale=A)
            A = torch.tensor(p.get("scale", 1.0), dtype=dtype, device=device)
            sigma = torch.exp(0.5 * s0)
            # log p(sigma) = log(2/pi) - log A - log(1+(sigma/A)^2)
            log_p_sigma = (
                torch.log(torch.tensor(2.0 / math.pi, dtype=dtype, device=device))
                - torch.log(A)
                - torch.log1p((sigma / A) ** 2)
            )
            # Jacobian: d sigma / d s = 0.5 exp(0.5 s)
            log_jac = math.log(0.5) + 0.5 * s0
            return log_p_sigma + log_jac

        elif name == "sigma2_invgamma":
            # sigma^2 ~ InvGamma(alpha=a, beta=b)
            # p(x) = b^a / Gamma(a) * x^{-(a+1)} exp(-b/x)
            # x = exp(s), Jacobian dx/ds = exp(s)
            a = torch.tensor(p.get("alpha", 2.0), dtype=dtype, device=device)
            b = torch.tensor(p.get("beta", 1.0), dtype=dtype, device=device)
            x = torch.exp(s0)
            log_p_x = a * torch.log(b) - torch.lgamma(a) - (a + 1.0) * torch.log(x) - b / x
            log_jac = s0
            return log_p_x + log_jac

        elif name == "jeffreys_trunc":
            # Jeffreys on sigma^2: p(sigma^2) ∝ 1/sigma^2 => p(s) ∝ constant
            lo = torch.tensor(p.get("lo", -20.0), dtype=dtype, device=device)
            hi = torch.tensor(p.get("hi", 20.0), dtype=dtype, device=device)
            inside = (s0 >= lo) & (s0 <= hi)
            logZ = torch.log(hi - lo)
            return torch.where(
                inside,
                -logZ,
                torch.tensor(-float("inf"), dtype=dtype, device=device),
            )

        else:
            raise ValueError(f"Unknown sigma_prior: {name}")

    # -------------------------
    # Gibbs update for beta
    # -------------------------

    def beta_posterior_params(self, inv_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute m_beta, V_beta for beta | hypers, y.

        inv_var: [n] = 1 / (F + exp(s))
        """
        # Xt_invSig_X = X_tilde^T diag(inv_var) X_tilde
        Xt_weighted = inv_var.unsqueeze(1) * self.X_tilde         # [n,p]
        Xt_invSig_X = self.X_tilde.T @ Xt_weighted                # [p,p]

        # Xt_invSig_y = X_tilde^T (inv_var * y_tilde)
        Xt_invSig_y = self.X_tilde.T @ (inv_var * self.y_tilde)   # [p]

        V_beta_inv = self.V0_inv + Xt_invSig_X
        # small jitter for stability
        V_beta_inv = V_beta_inv + self.cfg.Vbeta_jitter * torch.eye(self.p, dtype=V_beta_inv.dtype, device=V_beta_inv.device)

        V_beta = torch.inverse(V_beta_inv)
        m_beta = V_beta @ (self.V0_inv @ self.m0 + Xt_invSig_y)
        return m_beta, V_beta

    def gibbs_beta(self, s: torch.Tensor, t: torch.Tensor, a_raw: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample beta ~ p(beta | s,t,a_raw,y).
        """
        s0 = self._as_scalar(s)
        sigma2 = torch.exp(s0)

        F = self.compute_F(t, a_raw)
        var = (F + sigma2).clamp_min(self.cfg.var_jitter)
        inv_var = 1.0 / var

        m_beta, V_beta = self.beta_posterior_params(inv_var)

        # Sample beta = m + L eps
        L = torch.linalg.cholesky(V_beta)
        eps = torch.randn(self.p, dtype=m_beta.dtype, device=m_beta.device)
        beta = m_beta + L @ eps

        aux = {"m_beta": m_beta, "V_beta": V_beta, "inv_var": inv_var}
        return beta, aux

    # -------------------------
    # MH updates
    # -------------------------

    @torch.no_grad()
    def mh_update_s(self, beta, s, t, a_raw) -> torch.Tensor:
        self.tried_s += 1
        s_prop = s + self.cfg.step.s * torch.randn_like(s)

        log_post_cur = self.loglik(beta, s, t, a_raw) + self.logprior_s(s)
        log_post_prop = self.loglik(beta, s_prop, t, a_raw) + self.logprior_s(s_prop)

        log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
        if torch.log(torch.rand((), device=s.device, dtype=s.dtype)) < log_alpha:
            self.acc_s += 1
            return s_prop
        return s

    @torch.no_grad()
    def mh_update_t(self, beta, s, t, a_raw) -> torch.Tensor:
        self.tried_t += 1
        t_prop = t + self.cfg.step.t * torch.randn_like(t)

        log_post_cur = self.loglik(beta, s, t, a_raw) + self.logprior_t(t)
        log_post_prop = self.loglik(beta, s, t_prop, a_raw) + self.logprior_t(t_prop)

        log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
        if torch.log(torch.rand((), device=t.device, dtype=t.dtype)) < log_alpha:
            self.acc_t += 1
            return t_prop
        return t

    # @torch.no_grad()
    # def mh_update_a_raw(self, beta, s, t, a_raw) -> torch.Tensor:
    #     self.tried_a += 1

    #     d = a_raw.numel()
    #     if d == 0:
    #         return a_raw  # nothing to update (shouldn't happen in B1/B2)

    #     if self.cfg.block_a_raw and d > 1:
    #         # 2D block (baseline only)
    #         step = torch.tensor([self.cfg.step.rho_raw, self.cfg.step.nu_raw],
    #                             dtype=a_raw.dtype, device=a_raw.device)
    #         a_prop = a_raw + step * torch.randn_like(a_raw)
    #     else:
    #         # 1D (B1/B2) or coordinate-wise
    #         a_prop = a_raw.clone()
    #         for j in range(d):
    #             step_j = self.cfg.step.rho_raw if j == 0 else self.cfg.step.nu_raw
    #             a_prop[j] = a_prop[j] + step_j * torch.randn((), dtype=a_raw.dtype, device=a_raw.device)

    #     log_post_cur = self.loglik(beta, s, t, a_raw) + self.logprior_a_raw(a_raw)
    #     log_post_prop = self.loglik(beta, s, t, a_prop) + self.logprior_a_raw(a_prop)

    #     log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
    #     if torch.log(torch.rand((), device=a_raw.device, dtype=a_raw.dtype)) < log_alpha:
    #         self.acc_a += 1
    #         return a_prop
    #     return a_raw
    
    @torch.no_grad()
    def mh_update_a_raw(self, beta, s, t, a_raw) -> torch.Tensor:
        self.tried_a += 1

        d = a_raw.numel()
        if d == 0:
            return a_raw

        # Determine which step size(s) correspond to the free coords
        if self.fixed_nu is None and self.fixed_rho0 is None:
            # baseline: [rho_raw, nu_raw]
            steps = [self.cfg.step.rho_raw, self.cfg.step.nu_raw]
        elif self.fixed_nu is not None:
            # B1: only rho_raw is free
            steps = [self.cfg.step.rho_raw]
        else:
            # B2: only nu_raw is free
            steps = [self.cfg.step.nu_raw]

        if self.cfg.block_a_raw and d > 1:
            step = torch.tensor(steps, dtype=a_raw.dtype, device=a_raw.device)
            a_prop = a_raw + step * torch.randn_like(a_raw)
        else:
            a_prop = a_raw.clone()
            for j in range(d):
                a_prop[j] = a_prop[j] + steps[j] * torch.randn((), dtype=a_raw.dtype, device=a_raw.device)

        log_post_cur = self.loglik(beta, s, t, a_raw) + self.logprior_a_raw(a_raw)
        log_post_prop = self.loglik(beta, s, t, a_prop) + self.logprior_a_raw(a_prop)

        log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
        if torch.log(torch.rand((), device=a_raw.device, dtype=a_raw.dtype)) < log_alpha:
            self.acc_a += 1
            return a_prop
        return a_raw



    # -------------------------
    # Optional: conditional mean of phi
    # -------------------------

    @torch.no_grad()
    def phi_conditional_mean(
        self,
        beta: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        a_raw: torch.Tensor,
        U: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute E[phi | beta, hypers, y] (closed form).

        mean_phi = U ( w * (U^T r) )
        where r = y - X beta, w = F / (F + sigma^2).
        """
        s0 = self._as_scalar(s)
        sigma2 = torch.exp(s0)

        F = self.compute_F(t, a_raw)
        w = F / (F + sigma2).clamp_min(self.cfg.var_jitter)

        r = y - X @ beta
        r_tilde = U.T @ r
        mean_phi = U @ (w * r_tilde)
        return mean_phi

    # -------------------------
    # Run chain
    # -------------------------

    def run(
        self,
        init_beta: Optional[torch.Tensor] = None,
        init_s: Optional[torch.Tensor] = None,
        init_t: Optional[torch.Tensor] = None,
        init_a_raw: Optional[torch.Tensor] = None,
        init_from_conditional_beta: bool = True,
        store_phi_mean: bool = False,
        U: Optional[torch.Tensor] = None,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run Metropolis-within-Gibbs chain.

        Returns dict with chains and acceptance stats.

        If store_phi_mean=True, stores conditional mean E[phi|beta,hypers,y] per saved draw.
        Requires (U, X, y) in *original* space (not tilde).
        """
        device = self.cfg.device
        dtype = self.X_tilde.dtype

        beta = torch.zeros(self.p, dtype=dtype, device=device) if init_beta is None else init_beta.to(device=device, dtype=dtype)
        s = torch.tensor([0.0], dtype=dtype, device=device) if init_s is None else init_s.to(device=device, dtype=dtype)
        t = torch.tensor([0.0], dtype=dtype, device=device) if init_t is None else init_t.to(device=device, dtype=dtype)
        # a_raw = torch.zeros(2, dtype=dtype, device=device) if init_a_raw is None else init_a_raw.to(device=device, dtype=dtype)

        if init_a_raw is None:
            # baseline: 2 free coords; B1/B2: 1 free coord
            d = 2
            if self.fixed_nu is not None or self.fixed_rho0 is not None:
                d = 1
            a_raw = torch.zeros(d, dtype=dtype, device=device)
        else:
            a_raw = init_a_raw.to(device=device, dtype=dtype)

        if init_from_conditional_beta:
            beta, _ = self.gibbs_beta(s, t, a_raw)

        # storage
        keep_steps: List[int] = []
        chain_beta: List[torch.Tensor] = []
        chain_s: List[torch.Tensor] = []
        chain_t: List[torch.Tensor] = []
        chain_a_raw: List[torch.Tensor] = []
        chain_phi_mean: List[torch.Tensor] = []

        if store_phi_mean:
            if U is None or X is None or y is None:
                raise ValueError("store_phi_mean=True requires U, X, y.")
            U = U.to(device=device, dtype=dtype)
            X = X.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

        for step in range(1, self.cfg.num_steps + 1):
            # 1) Gibbs beta
            beta, _ = self.gibbs_beta(s, t, a_raw)

            # 2) MH hypers
            s = self.mh_update_s(beta, s, t, a_raw)
            t = self.mh_update_t(beta, s, t, a_raw)
            a_raw = self.mh_update_a_raw(beta, s, t, a_raw)

            # store
            if step > self.cfg.burnin and ((step - self.cfg.burnin) % self.cfg.thin == 0):
                keep_steps.append(step)
                chain_beta.append(beta.detach().cpu())
                chain_s.append(s.detach().cpu())
                chain_t.append(t.detach().cpu())
                chain_a_raw.append(a_raw.detach().cpu())

                if store_phi_mean:
                    phi_m = self.phi_conditional_mean(beta, s, t, a_raw, U=U, X=X, y=y)
                    chain_phi_mean.append(phi_m.detach().cpu())

        out: Dict[str, Any] = {
            "keep_steps": keep_steps,
            "beta": torch.stack(chain_beta) if chain_beta else None,       # [S, p]
            "s": torch.stack(chain_s) if chain_s else None,               # [S, 1]
            "t": torch.stack(chain_t) if chain_t else None,               # [S, 1]
            "a_raw": torch.stack(chain_a_raw) if chain_a_raw else None,   # [S, 2]
            "acc": {
                "s": (self.acc_s, self.tried_s, self.acc_s / max(1, self.tried_s)),
                "t": (self.acc_t, self.tried_t, self.acc_t / max(1, self.tried_t)),
                "a_raw": (self.acc_a, self.tried_a, self.acc_a / max(1, self.tried_a)),
            },
            "config": self.cfg,
            "sigma_prior": {"name": self.sigma_prior, "params": self.sigma_prior_params},
        }
        if store_phi_mean:
            out["phi_mean"] = torch.stack(chain_phi_mean) if chain_phi_mean else None
        return out


# -----------------------------
# Convenience constructor (optional)
# -----------------------------

def make_collapsed_mcmc_from_model(
    model,  # SpectralCAR_FullVI-like object with X_tilde, y_tilde, lam, m0, V0, sigma2_prior fields
    config: Optional[MCMCConfig] = None,
    fixed_nu=None, 
    fixed_rho0=None,
) -> CollapsedSDMCARMCMC:
    """
    Helper to build sampler directly from your SpectralCAR_FullVI instance.

    Expects the model to have:
      - X_tilde, y_tilde, lam
      - m0, V0
      - sigma2_prior, sigma2_prior_params (optional)
    """
    sigma_prior = getattr(model, "sigma2_prior", "logsigma2_normal")
    sigma_prior_params = getattr(model, "sigma2_prior_params", {"mu": 0.0, "std": 1.0})
    return CollapsedSDMCARMCMC(
        X_tilde=model.X_tilde.detach(),
        y_tilde=model.y_tilde.detach(),
        lam=model.lam.detach(),
        m0=model.m0.detach(),
        V0=model.V0.detach(),
        sigma_prior=sigma_prior,
        sigma_prior_params=sigma_prior_params,
        config=config,
        fixed_nu=fixed_nu, 
        fixed_rho0=fixed_rho0,
    )
