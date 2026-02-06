# sdmcar/models.py

import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import kl_normal_std

class SpectralCAR_FullVI(nn.Module):
    """
    Collapsed VI for Spectral CAR with a generic spectral filter module.

    - φ is collapsed analytically (never sampled)
    - q(β) is analytic Gaussian per MC hyperparameter sample
    - Full VI over (log σ²) and filter hyperparameters via the filter module

    ELBO ~ E_{q(hyper)} [ log p(y|β,F,σ²) - KL(q(β)||p(β)) ] - KL(q(hyper)||p).
    """
    def __init__(self,
                 X,
                 y,
                 lam,
                 U,
                 filter_module,
                 prior_m0=None,
                 prior_V0=None,
                 mu_log_sigma2: float = -2.3,
                 log_std_log_sigma2: float = -2.3,
                 num_mc: int = 5,
                 sigma2_prior: str = "logsigma2_normal",
                 sigma2_prior_params: dict | None = None,
                 kl_sigma_mc: int = 8):
        super().__init__()

        # Data & spectrum
        self.X = X
        self.y = y
        self.lam = lam
        self.U = U
        self.filter = filter_module
        self.num_mc = num_mc

        n, p = X.shape

        # Prior on β: N(m0, V0)
        device = X.device
        self.m0 = (torch.zeros(p, dtype=torch.double, device=device)
                   if prior_m0 is None else prior_m0)
        self.V0 = (torch.eye(p, dtype=torch.double, device=device)
                   if prior_V0 is None else prior_V0)
        self.V0_inv = torch.inverse(self.V0)

        # Fixed spectral projections
        self.X_tilde = self.U.T @ self.X    # [n, p]
        self.y_tilde = self.U.T @ self.y    # [n]

        # Store last β posterior for reporting
        self.m_beta = torch.zeros(p, dtype=torch.double, device=device)
        self.V_beta = torch.eye(p, dtype=torch.double, device=device)

        # q(log σ²) = N(μ, s²), prior N(0,1)
        self.mu_log_sigma2 = nn.Parameter(
            torch.tensor([mu_log_sigma2], dtype=torch.double, device=device)
        )
        self.log_std_log_sigma2 = nn.Parameter(
            torch.tensor([log_std_log_sigma2], dtype=torch.double, device=device)
        )
        
        # ---- Prior on s = log sigma^2 ----
        # Supported:
        #   "logsigma2_normal"  : s ~ N(mu, std^2)   (closed-form KL)
        #   "logsigma_normal"   : log sigma ~ N(mu, std^2) (MC KL)
        #   "sigma_halfcauchy"  : sigma ~ HalfCauchy(scale) (MC KL)
        #   "sigma2_invgamma"   : sigma^2 ~ InvGamma(alpha,beta) (MC KL)
        #   "jeffreys_trunc"    : p(s) uniform on [lo,hi] (MC KL)
        self.sigma2_prior = sigma2_prior
        self.sigma2_prior_params = {} if sigma2_prior_params is None else sigma2_prior_params
        self.kl_sigma_mc = int(kl_sigma_mc)
        if self.kl_sigma_mc <= 0:
            raise ValueError("kl_sigma_mc must be positive.")

    
    def _beta_update(self, inv_var, return_Xt_invSig_X: bool = False):
        """
        inv_var: [n] = 1 / (F(λ) + σ²)

        Computes:
            Vβ = (V0^{-1} + X^T Σ^{-1} X)^{-1}
            mβ = Vβ (V0^{-1} m0 + X^T Σ^{-1} y)

        using spectral trick:
            X^T Σ^{-1} X = X_tilde^T diag(inv_var) X_tilde
            X^T Σ^{-1} y = X_tilde^T (inv_var * y_tilde)

        If return_Xt_invSig_X=True, also returns Xt_invSig_X = X^T Σ^{-1} X
        for reuse in the ELBO trace term.
        """
        # X̃^T Σ^{-1} X̃
        Xt_weighted = inv_var.unsqueeze(1) * self.X_tilde        # [n, p]
        Xt_invSig_X = self.X_tilde.T @ Xt_weighted               # [p, p]

        # X̃^T Σ^{-1} ỹ
        Xt_invSig_y = self.X_tilde.T @ (inv_var * self.y_tilde)  # [p]

        # V_beta_inv = self.V0_inv + Xt_invSig_X
        # V_beta = torch.inverse(V_beta_inv)
        # # better: V_beta = torch.linalg.inv(V_beta_inv)
        # m_beta = V_beta @ (self.V0_inv @ self.m0 + Xt_invSig_y)

        eps = 1e-6
        V_beta_inv = self.V0_inv + Xt_invSig_X
        V_beta_inv = V_beta_inv + eps * torch.eye(
            V_beta_inv.shape[0],
            dtype=V_beta_inv.dtype,
            device=V_beta_inv.device
        )

        L = torch.linalg.cholesky(V_beta_inv)
        V_beta = torch.cholesky_inverse(L)

        rhs = self.V0_inv @ self.m0 + Xt_invSig_y
        m_beta = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)


        if return_Xt_invSig_X:
            return m_beta, V_beta, Xt_invSig_X
        else:
            return m_beta, V_beta    
    

    def _kl_beta(self, m_beta, V_beta):
        """
        KL(q(β)||p(β)) for Gaussians N(mβ, Vβ) vs N(m0, V0).
        """
        p = self.m0.numel()
        term1 = torch.logdet(self.V0) - torch.logdet(V_beta)
        term2 = torch.trace(self.V0_inv @ V_beta)
        dm = (m_beta - self.m0).unsqueeze(0)
        term3 = (dm @ self.V0_inv @ dm.T).squeeze()
        return 0.5 * (term1 - p + term2 + term3)
    
    def _log_q_s(self, s: torch.Tensor) -> torch.Tensor:
        """
        log q(s) for q(s)=N(mu, std^2).
        s shape: [K] or [1]
        returns: same shape
        """
        mu = self.mu_log_sigma2
        std = torch.exp(self.log_std_log_sigma2)
        return Normal(mu, std).log_prob(s)
    
    def _log_p_s(self, s: torch.Tensor) -> torch.Tensor:
        """
        log p(s) induced by the chosen prior.
        We define s = log sigma^2, sigma = exp(0.5 s), sigma^2 = exp(s).
        """
        name = self.sigma2_prior
        p = self.sigma2_prior_params

        if name == "logsigma2_normal":
            # p(s) = N(m0, s0^2)
            m0 = torch.tensor(p.get("mu", 0.0), dtype=s.dtype, device=s.device)
            s0 = torch.tensor(p.get("std", 1.0), dtype=s.dtype, device=s.device)
            return Normal(m0, s0).log_prob(s)

        elif name == "logsigma_normal":
            # Prior on t = log sigma:  t ~ N(mu, std^2)
            # s = log sigma^2 = 2t  => t = 0.5 s
            # p(s) = p(t=0.5s) * |dt/ds| = p(t) * 0.5
            mu = torch.tensor(p.get("mu", 0.0), dtype=s.dtype, device=s.device)
            std = torch.tensor(p.get("std", 1.0), dtype=s.dtype, device=s.device)
            t = 0.5 * s
            return Normal(mu, std).log_prob(t) + math.log(0.5)

        elif name == "sigma_halfcauchy":
            # Prior on sigma: sigma ~ HalfCauchy(scale=A)
            # density: p(sigma)= 2/(pi A (1+(sigma/A)^2)), sigma>0
            # convert to s: sigma=exp(0.5 s), add Jacobian |d sigma/d s| = 0.5 exp(0.5 s)
            A = torch.tensor(p.get("scale", 1.0), dtype=s.dtype, device=s.device)
            sigma = torch.exp(0.5 * s)
            log_const = math.log(2.0 / math.pi)
            log_p_sigma = log_const - torch.log(A) - torch.log1p((sigma / A) ** 2)
            log_jac = math.log(0.5) + 0.5 * s
            return log_p_sigma + log_jac

        elif name == "sigma2_invgamma":
            # Prior on sigma^2: InvGamma(alpha, beta) with density:
            # p(x) = beta^a / Gamma(a) * x^{-(a+1)} exp(-beta/x), x>0
            # x = exp(s), Jacobian |dx/ds| = exp(s)
            a = torch.tensor(p.get("alpha", 2.0), dtype=s.dtype, device=s.device)
            b = torch.tensor(p.get("beta", 1.0), dtype=s.dtype, device=s.device)
            x = torch.exp(s)
            log_p_x = a * torch.log(b) - torch.lgamma(a) - (a + 1.0) * torch.log(x) - b / x
            log_jac = s
            return log_p_x + log_jac

        elif name == "jeffreys_trunc":
            # Jeffreys on sigma^2: p(sigma^2) ∝ 1/sigma^2 => p(s) ∝ constant
            # Must truncate to be proper: s in [lo, hi]
            lo = torch.tensor(p.get("lo", -20.0), dtype=s.dtype, device=s.device)
            hi = torch.tensor(p.get("hi",  20.0), dtype=s.dtype, device=s.device)
            # uniform on [lo,hi] in s
            logZ = torch.log(hi - lo)
            inside = (s >= lo) & (s <= hi)
            return torch.where(inside, -logZ, torch.tensor(-float("inf"), dtype=s.dtype, device=s.device))

        else:
            raise ValueError(f"Unknown sigma2_prior: {name}")
    
    def _kl_sigma2(self) -> torch.Tensor:
        """
        KL(q(s)||p(s)), where s=log sigma^2.
        Closed form for logsigma2_normal with N(0,1) (or any N).
        Otherwise MC estimate.
        """
        name = self.sigma2_prior

        # closed form ONLY when p(s) is standard normal AND q is normal OR more generally normal-normal
        if name == "logsigma2_normal":
            # If user sets mu,std not (0,1), we still can do closed-form Normal-Normal KL.
            mu_p = self.sigma2_prior_params.get("mu", 0.0)
            std_p = self.sigma2_prior_params.get("std", 1.0)

            # KL(N(mu_q, std_q^2) || N(mu_p, std_p^2))
            mu_q = self.mu_log_sigma2
            std_q = torch.exp(self.log_std_log_sigma2)
            mu_p_t = torch.tensor([mu_p], dtype=mu_q.dtype, device=mu_q.device)
            std_p_t = torch.tensor([std_p], dtype=mu_q.dtype, device=mu_q.device)

            # KL = log(std_p/std_q) + (std_q^2 + (mu_q-mu_p)^2)/(2 std_p^2) - 1/2
            kl = torch.log(std_p_t / std_q) + (std_q**2 + (mu_q - mu_p_t)**2) / (2.0 * std_p_t**2) - 0.5
            return kl.sum()

        # otherwise MC
        K = self.kl_sigma_mc
        eps = torch.randn((K,) + self.mu_log_sigma2.shape, dtype=self.mu_log_sigma2.dtype, device=self.mu_log_sigma2.device)
        s = self.mu_log_sigma2 + torch.exp(self.log_std_log_sigma2) * eps  # [K,1]
        logq = self._log_q_s(s)
        logp = self._log_p_s(s)
        return (logq - logp).mean().sum()

    def elbo(self, num_mc_override: int | None = None):
        """
        Monte-Carlo estimate of ELBO over q(hyperparams):
            hyperparams = {s=log σ²} ∪ filter hyperparams θ.

        Uses paired MC samples (θ^{(k)}, s^{(k)}) ~ q(θ) q(s).

        Returns:
            elbo: scalar tensor
            stats: dict of components
        """
        # Choose MC count
        num_mc = self.num_mc if num_mc_override is None else int(num_mc_override)
        if num_mc <= 0:
            raise ValueError(f"num_mc must be positive, got {num_mc}.")

        # ---- KL(q(s) || p(s)) ----
        # This term is NOT inside E_{q(θ)q(s)}[...] in the usual ELBO decomposition.
        kl_sigma2 = self._kl_sigma2()

        mc_loglik = torch.zeros((), dtype=self.y.dtype, device=self.y.device)
        mc_kl_beta = torch.zeros((), dtype=self.y.dtype, device=self.y.device)

        last_m_beta, last_V_beta = None, None
        last_sigma2 = None

        for _ in range(num_mc):
            # ---- Sample s = log σ² from q(s) ----
            eps_sig = torch.randn_like(self.mu_log_sigma2)
            s = self.mu_log_sigma2 + torch.exp(self.log_std_log_sigma2) * eps_sig
            sigma2 = torch.exp(s).clamp_min(1e-12)  # scalar tensor
            last_sigma2 = sigma2

            # ---- Sample filter hyperparameters θ (unconstrained) from q(filter) ----
            theta = self.filter.sample_unconstrained()  # dict[name -> tensor]

            # ---- Build spectral variance diag(F(λ;θ) + σ²) ----
            F_lam = self.filter.spectrum(self.lam, theta)  # [n] # can clamp to 0.0
            var = (F_lam + sigma2).clamp_min(1e-12)    # [n]
            inv_var = 1.0 / var                       # [n]

            # ---- Exact posterior of β given (θ, s) ----
            m_beta, V_beta, Xt_invSig_X = self._beta_update(inv_var, return_Xt_invSig_X=True)

            # ---- Residual in spectral domain ----
            g = self.y_tilde - self.X_tilde @ m_beta   # [n]

            # ---- Plug-in part of log-likelihood (drop -0.5 n log 2π) ----
            loglik_plugin = -0.5 * torch.sum(torch.log(var) + g**2 * inv_var)

            # ---- Exact variance correction: -0.5 * tr(Vβ X^T Σ^{-1} X) ----
            trace_term = torch.trace(V_beta @ Xt_invSig_X)

            # ---- Exact E_q(beta|θ,s)[log p(y|β,θ,s)] (up to constant) ----
            loglik_exact = loglik_plugin - 0.5 * trace_term

            # ---- KL(q(β|θ,s) || p(β)) ----
            kl_beta = self._kl_beta(m_beta, V_beta)

            mc_loglik += loglik_exact
            mc_kl_beta += kl_beta

            last_m_beta = m_beta
            last_V_beta = V_beta

        # ---- Average over paired MC draws ----
        mc_loglik /= num_mc
        mc_kl_beta /= num_mc

        # ---- KL for filter hyperparameters ----
        kl_filter = self.filter.kl_q_p()

        # ---- Final ELBO ----
        elbo = mc_loglik - mc_kl_beta - kl_filter - kl_sigma2

        # Store last β posterior for inspection/reporting
        self.m_beta = last_m_beta.detach()
        self.V_beta = last_V_beta.detach()

        stats = {
            "mc_loglik": mc_loglik.detach(),
            "mc_kl_beta": mc_kl_beta.detach(),
            "kl_filter": kl_filter.detach(),
            "kl_sigma2": kl_sigma2.detach(),
            "sigma2_last": last_sigma2.detach() if last_sigma2 is not None else None,
            "num_mc": torch.tensor(num_mc),
        }
        return elbo, stats

    
    @torch.no_grad()
    def posterior_phi(
        self,
        mode: str = "plugin",      # "plugin" or "mc"
        num_mc: int = 32,          # used only if mode=="mc"
    ):
        """
        Posterior summaries for the spatial effect φ.

        Model:
            φ = U z,
            z | (y, β, θ, s) ~ N( μ_z, diag(v_spec) ),
            μ_z   = (F / (F + σ²)) ⊙ (Uᵀ r),
            v_spec = F σ² / (F + σ²),
            r = y - X E_q[β].

        Modes
        -----
        plugin:
            Conditional posterior given mean hyperparameters
            (θ = E_q[θ], s = E_q[s]).
            Fast, deterministic.

        mc:
            Variational marginal posterior integrating over q(θ) q(s)
            using Monte Carlo and total variance decomposition.
            Slower, uncertainty-aware.

        Returns
        -------
        mean_phi : (n,)
            Posterior mean of φ in node space.
        var_phi_diag : (n,)
            Node-wise marginal posterior variance of φ.
        """

        # residual using posterior mean of β
        r = self.y - self.X @ self.m_beta
        r_tilde = self.U.T @ r

        # --------------------------------------------------
        # (A) Plug-in hyperparameters
        # --------------------------------------------------
        if mode == "plugin":
            theta_mean = self.filter.mean_unconstrained()
            sigma2 = torch.exp(self.mu_log_sigma2).clamp_min(1e-12)
            F_lam = self.filter.spectrum(self.lam, theta_mean) # can clamp to 0.0

            denom = (F_lam + sigma2).clamp_min(1e-12)
            w = F_lam / denom

            # spectral posterior
            mu_z = w * r_tilde
            v_spec = F_lam * sigma2 / denom

            # node space
            mean_phi = self.U @ mu_z
            var_phi_diag = (self.U ** 2) @ v_spec

            return mean_phi, var_phi_diag

        # --------------------------------------------------
        # (B) MC-integrated over q(θ, s)
        # --------------------------------------------------
        elif mode == "mc":
            if num_mc <= 0:
                raise ValueError("num_mc must be positive for mode='mc'.")

            n = self.y.shape[0]
            mean_acc = torch.zeros(n, dtype=self.y.dtype, device=self.y.device)
            var_within_acc = torch.zeros(n, dtype=self.y.dtype, device=self.y.device)
            mean2_acc = torch.zeros(n, dtype=self.y.dtype, device=self.y.device)

            for _ in range(int(num_mc)):
                # sample s = log σ²
                eps = torch.randn_like(self.mu_log_sigma2)
                s = self.mu_log_sigma2 + torch.exp(self.log_std_log_sigma2) * eps
                sigma2 = torch.exp(s).clamp_min(1e-12)

                # sample θ
                theta = self.filter.sample_unconstrained()
                F_lam = self.filter.spectrum(self.lam, theta) # can clamp to 0.0

                denom = (F_lam + sigma2).clamp_min(1e-12)
                w = F_lam / denom

                mu_z = w * r_tilde
                v_spec = F_lam * sigma2 / denom

                mean_phi_k = self.U @ mu_z
                var_phi_k = (self.U ** 2) @ v_spec

                mean_acc += mean_phi_k
                mean2_acc += mean_phi_k ** 2
                var_within_acc += var_phi_k

            K = float(num_mc)

            # total variance decomposition
            mean_phi = mean_acc / K
            var_phi_diag = (
                var_within_acc / K
                + (mean2_acc / K - mean_phi ** 2)
            )

            return mean_phi, var_phi_diag

        else:
            raise ValueError("mode must be one of {'plugin', 'mc'}.")

