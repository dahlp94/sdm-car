
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class SpectralCAR_HoldoutVI(nn.Module):
    """
    Collapsed VI for a SDM CAR model with a fixed holdout set.

    Model:
        y = X beta + phi + eps
        phi ~ N(0, U diag(F(lambda)) U^T)
        eps ~ N(0, sigma2 I)

    Only y_train contributes to the ELBO. The holdout responses are never stored
    or optimized by this class.

    Per hyperparameter draw, the observed-data likelihood is evaluated with the
    complementary precision identity, using an h x h Cholesky factorization,
    where h is the number of held-out areas.
    """

    def __init__(
        self,
        *,
        X: torch.Tensor,
        y: torch.Tensor,
        lam: torch.Tensor,
        U: torch.Tensor,
        filter_module: nn.Module,
        is_holdout: torch.Tensor,
        prior_m0: torch.Tensor | None = None,
        prior_V0: torch.Tensor | None = None,
        mu_log_sigma2: float = -2.3,
        log_std_log_sigma2: float = -2.3,
        num_mc: int = 5,
        sigma2_prior_params: dict | None = None,
        fixed_sigma2: float | None = None,
        jitter: float = 1e-8,
        min_variance: float = 1e-12,
    ):
        super().__init__()

        if X.ndim != 2:
            raise ValueError("X must have shape [n, p].")
        if y.ndim != 1:
            raise ValueError("y must have shape [n].")

        n, p = X.shape

        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows.")
        if lam.shape != (n,):
            raise ValueError("lam must have shape [n].")
        if U.shape != (n, n):
            raise ValueError("U must have shape [n, n].")
        if is_holdout.shape != (n,):
            raise ValueError("is_holdout must have shape [n].")
        if is_holdout.dtype != torch.bool:
            raise TypeError("is_holdout must be a boolean tensor.")
        if not (X.device == y.device == lam.device == U.device == is_holdout.device):
            raise ValueError("X, y, lam, U, and is_holdout must be on the same device.")
        if not (X.dtype == y.dtype == lam.dtype == U.dtype):
            raise ValueError("X, y, lam, and U must have the same dtype.")

        train_mask = ~is_holdout
        n_train = int(train_mask.sum().item())
        n_test = int(is_holdout.sum().item())

        if n_train == 0:
            raise ValueError("At least one training observation is required.")
        if n_test == 0:
            raise ValueError("At least one held-out observation is required.")
        if n_train <= p:
            raise ValueError("The training set should contain more rows than columns of X.")
        if not torch.isfinite(y[train_mask]).all():
            raise ValueError("Training responses contain NaN or infinite values.")

        self.filter = filter_module
        self.num_mc = int(num_mc)
        self.jitter = float(jitter)
        self.min_variance = float(min_variance)
        self.n_total = n
        self.n_train = n_train
        self.n_test = n_test
        self.p = p

        if self.num_mc <= 0:
            raise ValueError("num_mc must be positive.")
        if self.jitter < 0:
            raise ValueError("jitter must be nonnegative.")
        if self.min_variance <= 0:
            raise ValueError("min_variance must be positive.")

        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        test_idx = torch.nonzero(is_holdout, as_tuple=False).squeeze(1)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        U_train = U[train_idx, :]
        U_test = U[test_idx, :]

        self.register_buffer("train_idx", train_idx)
        self.register_buffer("test_idx", test_idx)
        self.register_buffer("X_train", X_train)
        self.register_buffer("X_test", X_test)
        self.register_buffer("y_train", y_train)
        self.register_buffer("U_train", U_train)
        self.register_buffer("U_test", U_test)
        self.register_buffer("lam", lam)

        # Fixed projections used by every ELBO evaluation.
        self.register_buffer("X_train_tilde", U_train.T @ X_train)  # [n, p]
        self.register_buffer("y_train_tilde", U_train.T @ y_train)  # [n]

        m0 = (
            torch.zeros(p, dtype=X.dtype, device=X.device)
            if prior_m0 is None
            else prior_m0
        )
        V0 = (
            torch.eye(p, dtype=X.dtype, device=X.device)
            if prior_V0 is None
            else prior_V0
        )

        if m0.shape != (p,):
            raise ValueError("prior_m0 must have shape [p].")
        if V0.shape != (p, p):
            raise ValueError("prior_V0 must have shape [p, p].")

        chol_V0 = torch.linalg.cholesky(V0)
        V0_inv = torch.cholesky_inverse(chol_V0)

        self.register_buffer("m0", m0)
        self.register_buffer("V0", V0)
        self.register_buffer("V0_inv", V0_inv)
        self.register_buffer(
            "I_test",
            torch.eye(n_test, dtype=X.dtype, device=X.device),
        )
        self.register_buffer(
            "I_beta",
            torch.eye(p, dtype=X.dtype, device=X.device),
        )

        self.fixed_sigma2 = None if fixed_sigma2 is None else float(fixed_sigma2)
        if self.fixed_sigma2 is not None and self.fixed_sigma2 <= 0:
            raise ValueError("fixed_sigma2 must be positive.")

        sigma_mu_init = (
            math.log(self.fixed_sigma2)
            if self.fixed_sigma2 is not None
            else float(mu_log_sigma2)
        )
        sigma_requires_grad = self.fixed_sigma2 is None

        self.mu_log_sigma2 = nn.Parameter(
            torch.tensor(sigma_mu_init, dtype=X.dtype, device=X.device),
            requires_grad=sigma_requires_grad,
        )
        self.log_std_log_sigma2 = nn.Parameter(
            torch.tensor(log_std_log_sigma2, dtype=X.dtype, device=X.device),
            requires_grad=sigma_requires_grad,
        )

        # First implementation: same default Normal prior used by FullVI.
        params = {} if sigma2_prior_params is None else sigma2_prior_params
        self.sigma2_prior_mu = float(params.get("mu", 0.0))
        self.sigma2_prior_std = float(params.get("std", 1.0))
        if self.sigma2_prior_std <= 0:
            raise ValueError("sigma2 prior standard deviation must be positive.")

        self.m_beta = torch.zeros(p, dtype=X.dtype, device=X.device)
        self.V_beta = torch.eye(p, dtype=X.dtype, device=X.device)

    def _fixed_sigma_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.fixed_sigma2,
            dtype=self.y_train.dtype,
            device=self.y_train.device,
        )

    def _sample_sigma2(self) -> torch.Tensor:
        if self.fixed_sigma2 is not None:
            return self._fixed_sigma_tensor()

        eps = torch.randn_like(self.mu_log_sigma2)
        s = self.mu_log_sigma2 + torch.exp(self.log_std_log_sigma2) * eps
        return torch.exp(s).clamp_min(self.min_variance)

    def _kl_sigma2(self) -> torch.Tensor:
        if self.fixed_sigma2 is not None:
            return torch.zeros(
                (),
                dtype=self.y_train.dtype,
                device=self.y_train.device,
            )

        mu_q = self.mu_log_sigma2
        std_q = torch.exp(self.log_std_log_sigma2)

        mu_p = torch.tensor(
            self.sigma2_prior_mu,
            dtype=mu_q.dtype,
            device=mu_q.device,
        )
        std_p = torch.tensor(
            self.sigma2_prior_std,
            dtype=mu_q.dtype,
            device=mu_q.device,
        )

        return (
            torch.log(std_p / std_q)
            + (std_q.square() + (mu_q - mu_p).square())
            / (2.0 * std_p.square())
            - 0.5
        )

    def _kl_beta(
        self,
        m_beta: torch.Tensor,
        V_beta: torch.Tensor,
    ) -> torch.Tensor:
        term1 = torch.logdet(self.V0) - torch.logdet(V_beta)
        term2 = torch.trace(self.V0_inv @ V_beta)
        delta = m_beta - self.m0
        term3 = delta @ self.V0_inv @ delta
        return 0.5 * (term1 - self.p + term2 + term3)

    @torch.no_grad()
    def plugin_hyperparams(self):
        theta_u = self.filter.mean_unconstrained()
        theta_c = self.filter._constrain(theta_u)

        if self.fixed_sigma2 is not None:
            sigma2 = self._fixed_sigma_tensor()
        else:
            std_s = torch.exp(self.log_std_log_sigma2)
            sigma2 = torch.exp(
                self.mu_log_sigma2 + 0.5 * std_s.square()
            ).clamp_min(self.min_variance)

        return theta_u, theta_c, sigma2

    def _observed_terms(
        self,
        F_lam: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute quantities involving Sigma_OO^{-1} without constructing Sigma_OO.

        Full covariance:
            C = U diag(v) U^T,  v = F + sigma2
        Full precision:
            P = U diag(1/v) U^T

        Identity:
            C_OO^{-1}
              = P_OO - P_OH P_HH^{-1} P_HO

            log|C_OO|
              = sum(log(v)) + log|P_HH|
        """
        v = (F_lam + sigma2).clamp_min(self.min_variance)
        inv_v = 1.0 / v

        # P_HH = U_H diag(inv_v) U_H^T
        P_HH = (
            self.U_test * inv_v.unsqueeze(0)
        ) @ self.U_test.T
        P_HH = 0.5 * (P_HH + P_HH.T)

        diag_scale = torch.diagonal(P_HH).mean().detach().clamp_min(1.0)
        P_HH_stable = P_HH + self.jitter * diag_scale * self.I_test
        chol_H = torch.linalg.cholesky(P_HH_stable)

        weighted_X = inv_v.unsqueeze(1) * self.X_train_tilde
        weighted_y = inv_v * self.y_train_tilde

        # P_HO X_O and P_HO y_O
        PHO_X = self.U_test @ weighted_X
        PHO_y = self.U_test @ weighted_y

        solve_X = torch.cholesky_solve(PHO_X, chol_H)
        solve_y = torch.cholesky_solve(
            PHO_y.unsqueeze(1),
            chol_H,
        ).squeeze(1)

        Xt_Sinv_X = (
            self.X_train_tilde.T @ weighted_X
            - PHO_X.T @ solve_X
        )
        Xt_Sinv_X = 0.5 * (Xt_Sinv_X + Xt_Sinv_X.T)

        Xt_Sinv_y = (
            self.X_train_tilde.T @ weighted_y
            - PHO_X.T @ solve_y
        )

        logdet_full = torch.log(v).sum()
        logdet_PHH = 2.0 * torch.log(torch.diagonal(chol_H)).sum()
        logdet_observed = logdet_full + logdet_PHH

        return {
            "v": v,
            "inv_v": inv_v,
            "chol_H": chol_H,
            "PHO_X": PHO_X,
            "PHO_y": PHO_y,
            "Xt_Sinv_X": Xt_Sinv_X,
            "Xt_Sinv_y": Xt_Sinv_y,
            "logdet_observed": logdet_observed,
        }

    def _beta_update_from_terms(
        self,
        terms: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        precision_beta = self.V0_inv + terms["Xt_Sinv_X"]
        precision_beta = 0.5 * (precision_beta + precision_beta.T)
        precision_beta = precision_beta + self.jitter * self.I_beta

        chol_beta = torch.linalg.cholesky(precision_beta)
        V_beta = torch.cholesky_inverse(chol_beta)

        rhs = self.V0_inv @ self.m0 + terms["Xt_Sinv_y"]
        m_beta = torch.cholesky_solve(
            rhs.unsqueeze(1),
            chol_beta,
        ).squeeze(1)

        return m_beta, V_beta

    def _residual_quadratic(
        self,
        m_beta: torch.Tensor,
        terms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        (y_O - X_O m)^T Sigma_OO^{-1} (y_O - X_O m)
        """
        residual_tilde = (
            self.y_train_tilde
            - self.X_train_tilde @ m_beta
        )
        base = torch.sum(
            terms["inv_v"] * residual_tilde.square()
        )

        PHO_residual = (
            terms["PHO_y"]
            - terms["PHO_X"] @ m_beta
        )
        correction = PHO_residual @ torch.cholesky_solve(
            PHO_residual.unsqueeze(1),
            terms["chol_H"],
        ).squeeze(1)

        return base - correction

    def elbo(
        self,
        num_mc_override: int | None = None,
    ):
        num_mc = (
            self.num_mc
            if num_mc_override is None
            else int(num_mc_override)
        )
        if num_mc <= 0:
            raise ValueError("num_mc must be positive.")

        mc_loglik = torch.zeros(
            (),
            dtype=self.y_train.dtype,
            device=self.y_train.device,
        )
        mc_kl_beta = torch.zeros_like(mc_loglik)

        last_m_beta = None
        last_V_beta = None
        last_sigma2 = None

        for _ in range(num_mc):
            sigma2 = self._sample_sigma2()
            theta = self.filter.sample_unconstrained()
            F_lam = self.filter.spectrum(
                self.lam,
                theta,
            ).clamp_min(0.0)

            terms = self._observed_terms(F_lam, sigma2)
            m_beta, V_beta = self._beta_update_from_terms(terms)

            quad = self._residual_quadratic(m_beta, terms)
            trace_term = torch.trace(
                V_beta @ terms["Xt_Sinv_X"]
            )

            # The constant -0.5*n_train*log(2*pi) is omitted.
            loglik = -0.5 * (
                terms["logdet_observed"]
                + quad
                + trace_term
            )

            mc_loglik = mc_loglik + loglik
            mc_kl_beta = mc_kl_beta + self._kl_beta(
                m_beta,
                V_beta,
            )

            last_m_beta = m_beta
            last_V_beta = V_beta
            last_sigma2 = sigma2

        mc_loglik = mc_loglik / float(num_mc)
        mc_kl_beta = mc_kl_beta / float(num_mc)

        kl_filter = self.filter.kl_q_p()
        kl_sigma2 = self._kl_sigma2()

        elbo = (
            mc_loglik
            - mc_kl_beta
            - kl_filter
            - kl_sigma2
        )

        self.m_beta = last_m_beta.detach()
        self.V_beta = last_V_beta.detach()

        stats = {
            "mc_loglik": mc_loglik.detach(),
            "mc_kl_beta": mc_kl_beta.detach(),
            "kl_filter": kl_filter.detach(),
            "kl_sigma2": kl_sigma2.detach(),
            "sigma2_last": last_sigma2.detach(),
            "num_mc": torch.tensor(
                num_mc,
                device=self.y_train.device,
            ),
        }
        return elbo, stats

    @torch.no_grad()
    def beta_posterior_plugin(self):
        theta_u, _, sigma2 = self.plugin_hyperparams()
        F_lam = self.filter.spectrum(
            self.lam,
            theta_u,
        ).clamp_min(0.0)

        terms = self._observed_terms(F_lam, sigma2)
        m_beta, V_beta = self._beta_update_from_terms(terms)

        return {
            "mean": m_beta,
            "cov": V_beta,
            "sd": torch.sqrt(
                torch.diagonal(V_beta).clamp_min(0.0)
            ),
            "F_lam": F_lam,
            "sigma2": sigma2,
            "terms": terms,
        }

    def _predict_component(
        self,
        *,
        terms: dict[str, torch.Tensor],
        m_beta: torch.Tensor,
        V_beta: torch.Tensor,
        sample: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Predict y_H | y_O for one hyperparameter setting.

        Precision-form conditional mean:
            E[y_H | y_O, beta]
              = X_H beta - P_HH^{-1} P_HO (y_O - X_O beta)
        """
        solve_y = torch.cholesky_solve(
            terms["PHO_y"].unsqueeze(1),
            terms["chol_H"],
        ).squeeze(1)

        solve_X = torch.cholesky_solve(
            terms["PHO_X"],
            terms["chol_H"],
        )

        A_beta = self.X_test + solve_X
        pred_mean = -solve_y + A_beta @ m_beta

        conditional_cov = torch.cholesky_inverse(
            terms["chol_H"]
        )
        beta_var_diag = torch.sum(
            (A_beta @ V_beta) * A_beta,
            dim=1,
        )
        pred_var = (
            torch.diagonal(conditional_cov)
            + beta_var_diag
        ).clamp_min(0.0)

        out = {
            "mean": pred_mean,
            "var": pred_var,
            "sd": torch.sqrt(pred_var),
        }

        if sample:
            chol_beta = torch.linalg.cholesky(
                V_beta + self.jitter * self.I_beta
            )
            beta_draw = (
                m_beta
                + chol_beta
                @ torch.randn(
                    self.p,
                    dtype=m_beta.dtype,
                    device=m_beta.device,
                )
            )

            z = torch.randn(
                self.n_test,
                dtype=m_beta.dtype,
                device=m_beta.device,
            )
            conditional_noise = torch.linalg.solve_triangular(
                terms["chol_H"].T,
                z.unsqueeze(1),
                upper=True,
            ).squeeze(1)

            out["draw"] = (
                -solve_y
                + A_beta @ beta_draw
                + conditional_noise
            )

        return out

    @torch.no_grad()
    def predict_plugin(self) -> dict[str, torch.Tensor]:
        beta = self.beta_posterior_plugin()
        pred = self._predict_component(
            terms=beta["terms"],
            m_beta=beta["mean"],
            V_beta=beta["cov"],
            sample=False,
        )

        pred["q025"] = pred["mean"] - 1.96 * pred["sd"]
        pred["q975"] = pred["mean"] + 1.96 * pred["sd"]
        pred["beta_mean"] = beta["mean"]
        pred["beta_cov"] = beta["cov"]
        pred["sigma2"] = beta["sigma2"]
        pred["F_lam"] = beta["F_lam"]
        pred["test_idx"] = self.test_idx
        return pred

    @torch.no_grad()
    def predict_vi_mc(
        self,
        num_mc: int = 256,
    ) -> dict[str, torch.Tensor]:
        K = int(num_mc)
        if K < 2:
            raise ValueError("predict_vi_mc requires num_mc >= 2.")

        component_means = []
        component_vars = []
        predictive_draws = []

        for _ in range(K):
            sigma2 = self._sample_sigma2()
            theta = self.filter.sample_unconstrained()
            F_lam = self.filter.spectrum(
                self.lam,
                theta,
            ).clamp_min(0.0)

            terms = self._observed_terms(F_lam, sigma2)
            m_beta, V_beta = self._beta_update_from_terms(terms)

            pred_k = self._predict_component(
                terms=terms,
                m_beta=m_beta,
                V_beta=V_beta,
                sample=True,
            )

            component_means.append(pred_k["mean"])
            component_vars.append(pred_k["var"])
            predictive_draws.append(pred_k["draw"])

        means = torch.stack(component_means, dim=0)
        vars_ = torch.stack(component_vars, dim=0)
        draws = torch.stack(predictive_draws, dim=0)

        mean = means.mean(dim=0)
        var = (
            (vars_ + means.square()).mean(dim=0)
            - mean.square()
        ).clamp_min(0.0)

        return {
            "mean": mean,
            "var": var,
            "sd": torch.sqrt(var),
            "q025": torch.quantile(draws, 0.025, dim=0),
            "q975": torch.quantile(draws, 0.975, dim=0),
            "draws": draws,
            "component_means": means,
            "component_vars": vars_,
            "test_idx": self.test_idx,
        }

    @staticmethod
    @torch.no_grad()
    def score_predictions(
        y_test: torch.Tensor,
        prediction: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        y_test = y_test.reshape(-1)
        pred_mean = prediction["mean"].reshape(-1)

        if y_test.shape != pred_mean.shape:
            raise ValueError(
                "y_test must have the same length as prediction['mean']."
            )

        error = pred_mean - y_test
        rmse = torch.sqrt(torch.mean(error.square()))
        mae = torch.mean(torch.abs(error))

        centered = y_test - y_test.mean()
        denominator = torch.sum(centered.square())
        if denominator <= 0:
            r2 = torch.tensor(
                float("nan"),
                dtype=y_test.dtype,
                device=y_test.device,
            )
        else:
            r2 = 1.0 - torch.sum(error.square()) / denominator

        out = {
            "n_test": int(y_test.numel()),
            "rmse": float(rmse.item()),
            "mae": float(mae.item()),
            "r2": float(r2.item()),
        }

        if "q025" in prediction and "q975" in prediction:
            covered = (
                (y_test >= prediction["q025"])
                & (y_test <= prediction["q975"])
            )
            out["coverage_95"] = float(
                covered.double().mean().item()
            )
            out["mean_interval_width"] = float(
                (
                    prediction["q975"]
                    - prediction["q025"]
                ).mean().item()
            )

        return out
