# sdmcar/models.py

import torch
import torch.nn as nn
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
                 num_mc: int = 5):
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

    # ----- Gaussian conjugacy for β (per sample) -----
    def _beta_update(self, inv_var):
        """
        inv_var: [n] = 1 / (F(λ) + σ²)

        Computes:
            Vβ = (V0^{-1} + X^T Σ^{-1} X)^{-1}
            mβ = Vβ (V0^{-1} m0 + X^T Σ^{-1} y)

        using spectral trick:
            X^T Σ^{-1} X = X_tilde^T diag(inv_var) X_tilde
            X^T Σ^{-1} y = X_tilde^T (inv_var * y_tilde)
        """
        Xt_invSig_X = self.X_tilde.T @ (inv_var.unsqueeze(1) * self.X_tilde)
        Xt_invSig_y = self.X_tilde.T @ (inv_var * self.y_tilde)

        V_beta_inv = self.V0_inv + Xt_invSig_X
        V_beta = torch.inverse(V_beta_inv)
        m_beta = V_beta @ (self.V0_inv @ self.m0 + Xt_invSig_y)
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

    # ----- ELBO computation -----
    def elbo(self):
        """
        Monte-Carlo estimate of ELBO over q(hyperparams):
            hyperparams = {log σ²} ∪ filter hyperparams.
        """
        # Sample q(log σ²)
        eps_sig = torch.randn_like(self.mu_log_sigma2)
        log_sigma2 = self.mu_log_sigma2 + torch.exp(self.log_std_log_sigma2) * eps_sig
        sigma2 = torch.exp(log_sigma2).clamp_min(1e-12)

        # KL for σ²
        kl_sigma2 = kl_normal_std(self.mu_log_sigma2, self.log_std_log_sigma2).sum()

        mc_loglik = 0.0
        mc_kl_beta = 0.0
        last_m_beta, last_V_beta = None, None

        for _ in range(self.num_mc):
            # Sample τ² and a from q in the filter (or more general hyperparams)
            tau2, a, log_tau2, a_raw = self.filter.sample_params()

            # Build spectral variance
            F_lam = self.filter.F(self.lam, tau2, a)      # [n]
            var = F_lam + sigma2
            inv_var = 1.0 / var

            # Analytic q(β) for this sample
            m_beta, V_beta = self._beta_update(inv_var)

            # Collapsed log-likelihood for this sample:
            # log p(y|β, F, σ²) = -0.5 Σ_i [ log(var_i) + g_i^2 / var_i ]
            g = self.y_tilde - self.X_tilde @ m_beta
            loglik = -0.5 * torch.sum(torch.log(var) + g**2 * inv_var)

            # KL(q(β)||p(β)) for this sample
            klb = self._kl_beta(m_beta, V_beta)

            mc_loglik += loglik
            mc_kl_beta += klb
            last_m_beta, last_V_beta = m_beta, V_beta  # keep last for reporting

        mc_loglik /= self.num_mc
        mc_kl_beta /= self.num_mc

        # KL for filter hyperparameters
        kl_filter = self.filter.kl_q_p()

        # ELBO = E[loglik - KL_beta] - KL_filter - KL_sigma2
        elbo = mc_loglik - mc_kl_beta - kl_filter - kl_sigma2

        # store last β posterior (for inspection/reporting)
        self.m_beta = last_m_beta.detach()
        self.V_beta = last_V_beta.detach()

        stats = {
            "mc_loglik": mc_loglik.detach(),
            "mc_kl_beta": mc_kl_beta.detach(),
            "kl_filter": kl_filter.detach(),
            "kl_sigma2": kl_sigma2.detach(),
            "sigma2": sigma2.detach(),
        }
        return elbo, stats

    @torch.no_grad()
    def posterior_phi(self, use_q_means: bool = True):
        """
        Returns posterior mean(φ) and diag var(φ).

        If use_q_means=True, uses mean params of q(hyper) to form F and σ²;
        otherwise it uses the current variational means (same here, but
        placeholder for future extensions).
        """
        if use_q_means:
            tau2_mean, a_mean = self.filter.mean_params()
            sigma2 = torch.exp(self.mu_log_sigma2)
            F_lam = self.filter.F(self.lam, tau2_mean, a_mean)
        else:
            tau2_mean, a_mean = self.filter.mean_params()
            sigma2 = torch.exp(self.mu_log_sigma2)
            F_lam = self.filter.F(self.lam, tau2_mean, a_mean)

        w = F_lam / (F_lam + sigma2)            # spectral weights
        r = self.y - self.X @ self.m_beta       # residual using posterior β mean

        mean_phi = self.U @ (w * (self.U.T @ r))
        var_phi_diag = F_lam * sigma2 / (F_lam + sigma2)
        return mean_phi, var_phi_diag
