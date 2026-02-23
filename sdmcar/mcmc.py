# sdmcar/mcmc.py
#
# Collapsed Metropolis-within-Gibbs MCMC for Spectral CAR regression with
# a generic spectral filter module.
#
# We sample the *collapsed* posterior (phi integrated out):
#
#   y | beta, hypers ~ N(X beta,  sigma(hypers))
#   sigma(hypers) = U diag(F(λ; theta) + sigma²) U^T
#
# State:
#   beta                  (p,)
#   s = log sigma^2        scalar tensor ([1] ok)
#   theta_unconstrained    dict[str -> tensor]  (filter-defined)
#
# Updates:
#   1) beta | (s, theta), y  : Gibbs (Gaussian)
#   2) s                      : RW-MH
#   3) theta blocks            : RW-MH over filter.blocks()
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import math
import torch
from torch.distributions import Normal


# -----------------------------
# Configuration dataclass
# -----------------------------

@dataclass
class MCMCConfig:
    num_steps: int = 20_000
    burnin: int = 5_000
    thin: int = 10

    # RW std for s = log sigma^2
    step_s: float = 0.15

    # RW std per filter unconstrained parameter name
    # e.g. {"log_tau2": 0.15, "rho0_raw": 0.10, "nu_raw": 0.10}
    step_theta: Dict[str, float] = field(default_factory=dict)

    seed: Optional[int] = None
    # logging
    print_every: int = 0   # 0 disables; e.g. 2000 prints every 2000 steps

    # numerical stability
    var_jitter: float = 1e-12         # clamp for marginal spectral variances
    Vbeta_jitter: float = 1e-10       # jitter on V_beta_inv diagonal

    device: Optional[torch.device] = None


# -----------------------------
# Collapsed sampler
# -----------------------------

class CollapsedSpectralCARMCMC:
    """
    Collapsed MCMC baseline for Spectral CAR regression, filter-agnostic.

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
        filter_module,  # BaseSpectralFilter-like
        sigma_prior: str = "logsigma2_normal",
        sigma_prior_params: Optional[dict] = None,
        config: Optional[MCMCConfig] = None,
    ):
        self.X_tilde = X_tilde
        self.y_tilde = y_tilde
        self.lam = lam
        self.filter = filter_module

        self.m0 = m0
        self.V0 = V0
        self.V0_inv = torch.inverse(V0)

        self.sigma_prior = sigma_prior
        self.sigma_prior_params = {} if sigma_prior_params is None else sigma_prior_params

        self.cfg = MCMCConfig() if config is None else config
        if self.cfg.device is None:
            self.cfg.device = self.X_tilde.device

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        self.n, self.p = self.X_tilde.shape

        # default step sizes for theta if not provided
        # (safe default; you can override in benchmarks)
        for name in getattr(self.filter, "unconstrained_names", lambda: [])():
            if name not in self.cfg.step_theta:
                self.cfg.step_theta[name] = 0.10

        # acceptance counters
        self.acc_s = 0
        self.tried_s = 0

        # per-block acceptance
        self.acc_theta: Dict[str, int] = {}
        self.tried_theta: Dict[str, int] = {}
        for block in self.filter.blocks():
            self.acc_theta[block.name] = 0
            self.tried_theta[block.name] = 0

    # -------------------------
    # helpers
    # -------------------------

    @staticmethod
    def _as_scalar(x: torch.Tensor) -> torch.Tensor:
        # allow shape [] or [1]
        return x.reshape(())

    def _move_theta(self, theta: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in theta.items():
            out[k] = v.to(device=device, dtype=dtype).reshape(-1)
        return out

    # -------------------------
    # spectrum + likelihood
    # -------------------------

    def loglik(self, beta: torch.Tensor, s: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Collapsed log p(y | beta, s, theta) up to constant.
        """
        s0 = self._as_scalar(s)
        sigma2 = torch.exp(s0)

        # generic filter spectrum
        # NOTE: requires BaseSpectralFilter.spectrum(lam, theta) to exist.
        # If you didn't add that wrapper, replace with:
        #   F = self.filter.spectrum_from_unconstrained(self.lam, theta)
        F = self.filter.spectrum(self.lam, theta)
        var = (F + sigma2).clamp_min(self.cfg.var_jitter)

        r = self.y_tilde - self.X_tilde @ beta
        ll = -0.5 * torch.sum(torch.log(var) + (r * r) / var)
        return ll

    # -------------------------
    # priors
    # -------------------------

    def logprior_s(self, s: torch.Tensor) -> torch.Tensor:
        """
        log p(s) where s = log sigma^2.
        Mirrors models.py `_log_p_s`.
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
            mu = torch.tensor(p.get("mu", 0.0), dtype=dtype, device=device)
            std = torch.tensor(p.get("std", 1.0), dtype=dtype, device=device)
            t = 0.5 * s0
            return Normal(mu, std).log_prob(t) + math.log(0.5)

        elif name == "sigma_halfcauchy":
            A = torch.tensor(p.get("scale", 1.0), dtype=dtype, device=device)
            sigma = torch.exp(0.5 * s0)
            log_p_sigma = (
                torch.log(torch.tensor(2.0 / math.pi, dtype=dtype, device=device))
                - torch.log(A)
                - torch.log1p((sigma / A) ** 2)
            )
            log_jac = math.log(0.5) + 0.5 * s0
            return log_p_sigma + log_jac

        elif name == "sigma2_invgamma":
            a = torch.tensor(p.get("alpha", 2.0), dtype=dtype, device=device)
            b = torch.tensor(p.get("beta", 1.0), dtype=dtype, device=device)
            x = torch.exp(s0)
            log_p_x = a * torch.log(b) - torch.lgamma(a) - (a + 1.0) * torch.log(x) - b / x
            log_jac = s0
            return log_p_x + log_jac

        elif name == "jeffreys_trunc":
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
        Xt_weighted = inv_var.unsqueeze(1) * self.X_tilde         # [n,p]
        Xt_invSig_X = self.X_tilde.T @ Xt_weighted                # [p,p]
        Xt_invSig_y = self.X_tilde.T @ (inv_var * self.y_tilde)   # [p]

        V_beta_inv = self.V0_inv + Xt_invSig_X
        V_beta_inv = V_beta_inv + self.cfg.Vbeta_jitter * torch.eye(
            self.p, dtype=V_beta_inv.dtype, device=V_beta_inv.device
        )

        # stable solve
        L = torch.linalg.cholesky(V_beta_inv)
        V_beta = torch.cholesky_inverse(L)

        rhs = self.V0_inv @ self.m0 + Xt_invSig_y
        m_beta = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        return m_beta, V_beta

    def gibbs_beta(self, s: torch.Tensor, theta: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample beta ~ p(beta | s, theta, y).
        """
        s0 = self._as_scalar(s)
        sigma2 = torch.exp(s0)

        F = self.filter.spectrum(self.lam, theta)
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
    def mh_update_s(self, beta: torch.Tensor, s: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.tried_s += 1
        s_prop = s + self.cfg.step_s * torch.randn_like(s)

        log_post_cur = self.loglik(beta, s, theta) + self.logprior_s(s)
        log_post_prop = self.loglik(beta, s_prop, theta) + self.logprior_s(s_prop)

        log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
        if torch.log(torch.rand((), device=s.device, dtype=s.dtype)) < log_alpha:
            self.acc_s += 1
            return s_prop
        return s

    @torch.no_grad()
    def mh_update_theta(
        self,
        beta: torch.Tensor,
        s: torch.Tensor,
        theta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        RW-MH over filter-defined parameter blocks.
        step sizes are per-parameter name in cfg.step_theta.
        """
        for block in self.filter.blocks():
            bname = block.name
            pnames = list(getattr(block, "param_names", (block.name,)))  # backward compatible

            # ensure all required keys exist in theta (skip if fixed / not present)
            if any(pn not in theta for pn in pnames):
                continue

            self.tried_theta[bname] = self.tried_theta.get(bname, 0) + 1

            theta_prop = {k: v.clone() for k, v in theta.items()}
            for pn in pnames:
                step = float(self.cfg.step_theta.get(pn, 0.10))
                theta_prop[pn] = theta_prop[pn] + step * torch.randn_like(theta_prop[pn])

            log_post_cur = self.loglik(beta, s, theta) + self.filter.log_prior(theta)
            log_post_prop = self.loglik(beta, s, theta_prop) + self.filter.log_prior(theta_prop)

            log_alpha = (log_post_prop - log_post_cur).clamp_max(0.0)
            if torch.log(torch.rand((), device=s.device, dtype=s.dtype)) < log_alpha:
                theta = theta_prop
                self.acc_theta[bname] = self.acc_theta.get(bname, 0) + 1

        return theta

    # -------------------------
    # Optional: conditional mean of phi
    # -------------------------

    @torch.no_grad()
    def phi_conditional_mean(
        self,
        beta: torch.Tensor,
        s: torch.Tensor,
        theta: Dict[str, torch.Tensor],
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

        F = self.filter.spectrum(self.lam, theta)
        denom = (F + sigma2).clamp_min(self.cfg.var_jitter)
        w = F / denom

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
        init_theta_vec: Optional[torch.Tensor] = None,
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

        # theta init: prefer provided packed vec; otherwise filter.theta0()
        if init_theta_vec is not None:
            theta_vec = init_theta_vec.to(device=device, dtype=dtype).reshape(-1)
            theta = self.filter.unpack(theta_vec)
        else:
            theta0 = self.filter.theta0()  # dict in unconstrained space
            theta = self._move_theta(theta0, device=device, dtype=dtype)

        if init_from_conditional_beta:
            beta, _ = self.gibbs_beta(s, theta)

        # storage
        keep_steps: List[int] = []
        chain_beta: List[torch.Tensor] = []
        chain_s: List[torch.Tensor] = []
        chain_theta_vec: List[torch.Tensor] = []
        chain_phi_mean: List[torch.Tensor] = []

        if store_phi_mean:
            if U is None or X is None or y is None:
                raise ValueError("store_phi_mean=True requires U, X, y.")
            U = U.to(device=device, dtype=dtype)
            X = X.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

        for step in range(1, self.cfg.num_steps + 1):
            # 1) Gibbs beta
            beta, _ = self.gibbs_beta(s, theta)

            # 2) MH s
            s = self.mh_update_s(beta, s, theta)

            # 3) MH theta blocks
            theta = self.mh_update_theta(beta, s, theta)

            # progress printing
            pe = int(getattr(self.cfg, "print_every", 0))
            if pe > 0 and (step % pe == 0):
                acc_s = self.acc_s / max(1, self.tried_s)

                theta_rates = []
                for bname in self.acc_theta.keys() | self.tried_theta.keys():
                    a = self.acc_theta.get(bname, 0)
                    t = self.tried_theta.get(bname, 0)
                    theta_rates.append((bname, a / max(1, t)))
                theta_rates.sort(key=lambda x: x[0])

                theta_str = ", ".join([f"{k}={v:.3f}" for k, v in theta_rates])
                sigma2_cur = float(torch.exp(self._as_scalar(s)).detach().cpu().item())

                print(
                    f"[MCMC {step:05d}/{self.cfg.num_steps}] "
                    f"acc_s={acc_s:.3f} | acc_theta[{theta_str}] | sigma2={sigma2_cur:.4f}"
                )

            # store
            if step > self.cfg.burnin and ((step - self.cfg.burnin) % self.cfg.thin == 0):
                keep_steps.append(step)
                chain_beta.append(beta.detach().cpu())
                chain_s.append(s.detach().cpu())
                chain_theta_vec.append(self.filter.pack(theta).detach().cpu())

                if store_phi_mean:
                    phi_m = self.phi_conditional_mean(beta, s, theta, U=U, X=X, y=y)
                    chain_phi_mean.append(phi_m.detach().cpu())

        acc_theta_out = {}
        for name in self.acc_theta.keys() | self.tried_theta.keys():
            a = self.acc_theta.get(name, 0)
            t = self.tried_theta.get(name, 0)
            acc_theta_out[name] = (a, t, a / max(1, t))

        out: Dict[str, Any] = {
            "keep_steps": keep_steps,
            "beta": torch.stack(chain_beta) if chain_beta else None,              # [S, p]
            "s": torch.stack(chain_s) if chain_s else None,                      # [S, 1]
            "theta": torch.stack(chain_theta_vec) if chain_theta_vec else None,  # [S, d_theta]
            "acc": {
                "s": (self.acc_s, self.tried_s, self.acc_s / max(1, self.tried_s)),
                "theta": acc_theta_out,
            },
            "config": self.cfg,
            "sigma_prior": {"name": self.sigma_prior, "params": self.sigma_prior_params},
            "theta_names": self.filter.unconstrained_names(),
        }
        if store_phi_mean:
            out["phi_mean"] = torch.stack(chain_phi_mean) if chain_phi_mean else None
        return out


# -----------------------------
# Convenience constructor
# -----------------------------

def make_collapsed_mcmc_from_model(
    model,  # SpectralCAR_FullVI-like object with X_tilde, y_tilde, lam, m0, V0, sigma2_prior fields and .filter
    config: Optional[MCMCConfig] = None,
) -> CollapsedSpectralCARMCMC:
    """
    Helper to build sampler directly from your SpectralCAR_FullVI instance.

    Expects the model to have:
      - X_tilde, y_tilde, lam
      - m0, V0
      - sigma2_prior, sigma2_prior_params (optional)
      - filter (BaseSpectralFilter-like)
    """
    sigma_prior = getattr(model, "sigma2_prior", "logsigma2_normal")
    sigma_prior_params = getattr(model, "sigma2_prior_params", {"mu": 0.0, "std": 1.0})

    return CollapsedSpectralCARMCMC(
        X_tilde=model.X_tilde.detach(),
        y_tilde=model.y_tilde.detach(),
        lam=model.lam.detach(),
        m0=model.m0.detach(),
        V0=model.V0.detach(),
        filter_module=model.filter,
        sigma_prior=sigma_prior,
        sigma_prior_params=sigma_prior_params,
        config=config,
    )
