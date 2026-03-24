import math

import pytest
import torch
import torch.nn as nn

from sdmcar.models import SpectralCAR_FullVI


torch.set_default_dtype(torch.double)


class DummyFilter(nn.Module):
    """
    Minimal filter stub for testing SpectralCAR_FullVI utilities.

    Unconstrained variational params:
        log_tau2_q ~ Normal(mu, std^2)
        rho_raw_q  ~ Normal(mu, std^2)

    Constrained params:
        tau2 = exp(log_tau2)
        rho0 = sigmoid(rho_raw)

    Spectrum:
        F(lam) = tau2 / (1 + rho0 * lam)
    """

    def __init__(self, device="cpu", dtype=torch.double):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        self.mu_log_tau2 = nn.Parameter(torch.tensor([math.log(0.7)], dtype=dtype, device=self.device))
        self.log_std_log_tau2 = nn.Parameter(torch.tensor([-1.2], dtype=dtype, device=self.device))

        self.mu_rho_raw = nn.Parameter(torch.tensor([0.25], dtype=dtype, device=self.device))
        self.log_std_rho_raw = nn.Parameter(torch.tensor([-1.0], dtype=dtype, device=self.device))

    def mean_unconstrained(self):
        return {
            "log_tau2": self.mu_log_tau2.detach().clone(),
            "rho_raw": self.mu_rho_raw.detach().clone(),
        }

    def sample_unconstrained(self):
        eps1 = torch.randn_like(self.mu_log_tau2)
        eps2 = torch.randn_like(self.mu_rho_raw)
        return {
            "log_tau2": self.mu_log_tau2 + torch.exp(self.log_std_log_tau2) * eps1,
            "rho_raw": self.mu_rho_raw + torch.exp(self.log_std_rho_raw) * eps2,
        }

    def _constrain(self, theta_dict):
        return {
            "tau2": torch.exp(theta_dict["log_tau2"]).clamp_min(1e-12),
            "rho0": torch.sigmoid(theta_dict["rho_raw"]),
        }

    def spectrum(self, lam, theta_unconstrained):
        c = self._constrain(theta_unconstrained)
        tau2 = c["tau2"]
        rho0 = c["rho0"]
        return tau2 / (1.0 + rho0 * lam)

    def kl_q_p(self):
        # Not important for these tests; just return a scalar tensor
        return torch.zeros((), dtype=self.dtype, device=self.device)


@pytest.fixture
def small_model():
    device = torch.device("cpu")
    dtype = torch.double

    n = 6
    p = 2

    # Use identity eigenbasis to keep the setup simple and deterministic.
    U = torch.eye(n, dtype=dtype, device=device)
    lam = torch.linspace(0.1, 1.0, n, dtype=dtype, device=device)

    x = torch.linspace(-1.0, 1.0, n, dtype=dtype, device=device)
    X = torch.stack([torch.ones(n, dtype=dtype, device=device), x], dim=1)

    beta_true = torch.tensor([0.8, -0.3], dtype=dtype, device=device)
    y = X @ beta_true + 0.1 * torch.randn(n, dtype=dtype, device=device)

    filter_module = DummyFilter(device=device, dtype=dtype)

    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        mu_log_sigma2=-1.1,
        log_std_log_sigma2=-0.8,
        num_mc=4,
    ).to(device)

    return model


def test_plugin_hyperparams_returns_expected_shapes_and_values(small_model):
    model = small_model

    theta_u, theta_c, sigma2_plugin = model.plugin_hyperparams()

    assert isinstance(theta_u, dict)
    assert isinstance(theta_c, dict)
    assert "log_tau2" in theta_u
    assert "rho_raw" in theta_u
    assert "tau2" in theta_c
    assert "rho0" in theta_c

    assert theta_u["log_tau2"].shape == (1,)
    assert theta_u["rho_raw"].shape == (1,)
    assert theta_c["tau2"].shape == (1,)
    assert theta_c["rho0"].shape == (1,)
    assert sigma2_plugin.shape == (1,)

    std_s = torch.exp(model.log_std_log_sigma2.detach())
    expected_sigma2 = torch.exp(model.mu_log_sigma2.detach() + 0.5 * std_s**2).clamp_min(1e-12)

    assert torch.allclose(sigma2_plugin, expected_sigma2, atol=1e-12, rtol=1e-12)
    assert torch.all(theta_c["tau2"] > 0.0)
    assert torch.all((theta_c["rho0"] > 0.0) & (theta_c["rho0"] < 1.0))


def test_sample_vi_hyperparams_shapes(small_model):
    model = small_model
    K = 11

    out = model.sample_vi_hyperparams(num_mc=K)

    assert set(out.keys()) == {"s", "sigma2", "theta_unconstrained", "theta_constrained"}

    assert out["s"].shape == (K,)
    assert out["sigma2"].shape == (K,)

    theta_u = out["theta_unconstrained"]
    theta_c = out["theta_constrained"]

    assert set(theta_u.keys()) == {"log_tau2", "rho_raw"}
    assert set(theta_c.keys()) == {"tau2", "rho0"}

    assert theta_u["log_tau2"].shape == (K,)
    assert theta_u["rho_raw"].shape == (K,)
    assert theta_c["tau2"].shape == (K,)
    assert theta_c["rho0"].shape == (K,)

    assert torch.all(out["sigma2"] > 0.0)
    assert torch.all(theta_c["tau2"] > 0.0)
    assert torch.all((theta_c["rho0"] > 0.0) & (theta_c["rho0"] < 1.0))


def test_sample_vi_hyperparams_sigma2_matches_exp_of_s(small_model):
    model = small_model
    out = model.sample_vi_hyperparams(num_mc=20)

    assert torch.allclose(out["sigma2"], torch.exp(out["s"]).clamp_min(1e-12), atol=1e-12, rtol=1e-12)


def test_sample_vi_hyperparams_invalid_num_mc_raises(small_model):
    model = small_model

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.sample_vi_hyperparams(num_mc=0)

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.sample_vi_hyperparams(num_mc=-3)


def test_beta_posterior_plugin_uses_unified_plugin_sigma2(small_model):
    model = small_model

    m_beta, V_beta, sigma2_plugin, F_lam = model.beta_posterior_plugin()
    _, _, sigma2_expected = model.plugin_hyperparams()

    p = model.X.shape[1]
    n = model.y.shape[0]

    assert m_beta.shape == (p,)
    assert V_beta.shape == (p, p)
    assert F_lam.shape == (n,)
    assert sigma2_plugin.shape == (1,)

    assert torch.allclose(sigma2_plugin, sigma2_expected, atol=1e-12, rtol=1e-12)
    assert torch.all(F_lam > 0.0)

    # Covariance should be symmetric positive definite (up to numerical tolerance).
    assert torch.allclose(V_beta, V_beta.T, atol=1e-10, rtol=1e-10)
    eigvals = torch.linalg.eigvalsh(V_beta)
    assert torch.all(eigvals > 0.0)


def test_posterior_phi_plugin_returns_valid_shapes_and_nonnegative_variances(small_model):
    model = small_model

    mean_phi, var_phi_diag = model.posterior_phi(mode="plugin")

    n = model.y.shape[0]
    assert mean_phi.shape == (n,)
    assert var_phi_diag.shape == (n,)
    assert torch.all(var_phi_diag >= -1e-12)


def test_predictive_metrics_plugin_runs_and_returns_finite_values(small_model):
    model = small_model

    out = model.predictive_metrics(mode="plugin")

    assert set(out.keys()) == {"rmse_y", "lpd"}
    assert isinstance(out["rmse_y"], float)
    assert isinstance(out["lpd"], float)
    assert math.isfinite(out["rmse_y"])
    assert math.isfinite(out["lpd"])
    assert out["rmse_y"] >= 0.0

def test_summarize_draws_tensor_scalar_matches_manual_computations(small_model):
    model = small_model

    x = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.double)
    out = model._summarize_draws_tensor(x)

    assert set(out.keys()) == {"mean", "sd", "q025", "q975"}

    assert out["mean"].shape == ()
    assert out["sd"].shape == ()
    assert out["q025"].shape == ()
    assert out["q975"].shape == ()

    assert torch.allclose(out["mean"], x.mean(), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["sd"], x.std(unbiased=True), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["q025"], torch.quantile(x, 0.025), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["q975"], torch.quantile(x, 0.975), atol=1e-12, rtol=1e-12)


def test_summarize_draws_tensor_vector_matches_manual_computations(small_model):
    model = small_model

    x = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [4.0, 40.0],
            [7.0, 70.0],
        ],
        dtype=torch.double,
    )
    out = model._summarize_draws_tensor(x)

    assert out["mean"].shape == (2,)
    assert out["sd"].shape == (2,)
    assert out["q025"].shape == (2,)
    assert out["q975"].shape == (2,)

    assert torch.allclose(out["mean"], x.mean(dim=0), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["sd"], x.std(dim=0, unbiased=True), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["q025"], torch.quantile(x, 0.025, dim=0), atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["q975"], torch.quantile(x, 0.975, dim=0), atol=1e-12, rtol=1e-12)


def test_summarize_draws_tensor_invalid_zero_dim_raises(small_model):
    model = small_model

    with pytest.raises(ValueError, match="at least one draw dimension"):
        model._summarize_draws_tensor(torch.tensor(3.14, dtype=torch.double))


def test_sigma2_posterior_vi_returns_expected_structure_without_draws(small_model):
    model = small_model

    out = model.sigma2_posterior_vi(num_mc=25, return_draws=False)

    assert set(out.keys()) == {"plugin", "mc"}
    assert set(out["mc"].keys()) == {"mean", "sd", "q025", "q975"}

    assert out["plugin"].shape == (1,)
    assert out["mc"]["mean"].shape == ()
    assert out["mc"]["sd"].shape == ()
    assert out["mc"]["q025"].shape == ()
    assert out["mc"]["q975"].shape == ()

    assert torch.all(out["plugin"] > 0.0)
    assert out["mc"]["mean"].item() > 0.0
    assert out["mc"]["sd"].item() >= 0.0
    assert out["mc"]["q025"].item() > 0.0
    assert out["mc"]["q975"].item() > 0.0
    assert out["mc"]["q025"].item() <= out["mc"]["mean"].item() <= out["mc"]["q975"].item()


def test_sigma2_posterior_vi_return_draws_matches_summary(small_model):
    model = small_model

    out = model.sigma2_posterior_vi(num_mc=40, return_draws=True)

    assert set(out.keys()) == {"plugin", "mc", "draws"}
    assert out["draws"].shape == (40,)
    assert torch.all(out["draws"] > 0.0)

    manual = model._summarize_draws_tensor(out["draws"])

    assert torch.allclose(out["mc"]["mean"], manual["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["sd"], manual["sd"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q025"], manual["q025"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q975"], manual["q975"], atol=1e-12, rtol=1e-12)

    _, _, sigma2_plugin = model.plugin_hyperparams()
    assert torch.allclose(out["plugin"], sigma2_plugin, atol=1e-12, rtol=1e-12)


def test_theta_posterior_vi_returns_expected_structure_without_draws(small_model):
    model = small_model

    out = model.theta_posterior_vi(num_mc=30, return_draws=False)

    assert set(out.keys()) == {"plugin", "mc"}
    assert set(out["plugin"].keys()) == {"tau2", "rho0"}
    assert set(out["mc"].keys()) == {"tau2", "rho0"}

    for name in ("tau2", "rho0"):
        assert set(out["mc"][name].keys()) == {"mean", "sd", "q025", "q975"}

        assert out["plugin"][name].shape == (1,)
        assert out["mc"][name]["mean"].shape == ()
        assert out["mc"][name]["sd"].shape == ()
        assert out["mc"][name]["q025"].shape == ()
        assert out["mc"][name]["q975"].shape == ()

    assert torch.all(out["plugin"]["tau2"] > 0.0)
    assert torch.all((out["plugin"]["rho0"] > 0.0) & (out["plugin"]["rho0"] < 1.0))

    assert out["mc"]["tau2"]["mean"].item() > 0.0
    assert out["mc"]["tau2"]["sd"].item() >= 0.0
    assert out["mc"]["tau2"]["q025"].item() > 0.0
    assert out["mc"]["tau2"]["q975"].item() > 0.0

    assert 0.0 < out["mc"]["rho0"]["q025"].item() < 1.0
    assert 0.0 < out["mc"]["rho0"]["mean"].item() < 1.0
    assert 0.0 < out["mc"]["rho0"]["q975"].item() < 1.0
    assert out["mc"]["rho0"]["sd"].item() >= 0.0


def test_theta_posterior_vi_return_draws_matches_summary(small_model):
    model = small_model

    out = model.theta_posterior_vi(num_mc=35, return_draws=True)

    assert set(out.keys()) == {"plugin", "mc", "draws"}
    assert set(out["draws"].keys()) == {"tau2", "rho0"}

    for name in ("tau2", "rho0"):
        assert out["draws"][name].shape == (35,)
        manual = model._summarize_draws_tensor(out["draws"][name])

        assert torch.allclose(out["mc"][name]["mean"], manual["mean"], atol=1e-12, rtol=1e-12)
        assert torch.allclose(out["mc"][name]["sd"], manual["sd"], atol=1e-12, rtol=1e-12)
        assert torch.allclose(out["mc"][name]["q025"], manual["q025"], atol=1e-12, rtol=1e-12)
        assert torch.allclose(out["mc"][name]["q975"], manual["q975"], atol=1e-12, rtol=1e-12)

    _, theta_plugin_c, _ = model.plugin_hyperparams()
    assert torch.allclose(out["plugin"]["tau2"], theta_plugin_c["tau2"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["plugin"]["rho0"], theta_plugin_c["rho0"], atol=1e-12, rtol=1e-12)


def test_theta_posterior_vi_constrained_ranges_hold_for_draws(small_model):
    model = small_model

    out = model.theta_posterior_vi(num_mc=50, return_draws=True)

    tau2_draws = out["draws"]["tau2"]
    rho0_draws = out["draws"]["rho0"]

    assert torch.all(tau2_draws > 0.0)
    assert torch.all((rho0_draws > 0.0) & (rho0_draws < 1.0))


def test_sigma2_and_theta_vi_posteriors_are_reproducible_under_fixed_seed(small_model):
    model = small_model

    torch.manual_seed(123)
    sig1 = model.sigma2_posterior_vi(num_mc=20, return_draws=True)
    torch.manual_seed(123)
    sig2 = model.sigma2_posterior_vi(num_mc=20, return_draws=True)

    assert torch.allclose(sig1["draws"], sig2["draws"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(sig1["mc"]["mean"], sig2["mc"]["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(sig1["mc"]["sd"], sig2["mc"]["sd"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(sig1["mc"]["q025"], sig2["mc"]["q025"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(sig1["mc"]["q975"], sig2["mc"]["q975"], atol=1e-12, rtol=1e-12)

    torch.manual_seed(456)
    th1 = model.theta_posterior_vi(num_mc=20, return_draws=True)
    torch.manual_seed(456)
    th2 = model.theta_posterior_vi(num_mc=20, return_draws=True)

    for name in ("tau2", "rho0"):
        assert torch.allclose(th1["draws"][name], th2["draws"][name], atol=1e-12, rtol=1e-12)
        for stat in ("mean", "sd", "q025", "q975"):
            assert torch.allclose(th1["mc"][name][stat], th2["mc"][name][stat], atol=1e-12, rtol=1e-12)

def test_beta_posterior_vi_returns_expected_structure_without_draws(small_model):
    model = small_model

    out = model.beta_posterior_vi(num_mc=30, return_draws=False)

    assert set(out.keys()) == {"plugin", "mc"}

    for layer in ("plugin", "mc"):
        assert set(out[layer].keys()) == {"mean", "cov", "sd", "q025", "q975"}

    p = model.X.shape[1]

    # plugin
    assert out["plugin"]["mean"].shape == (p,)
    assert out["plugin"]["cov"].shape == (p, p)
    assert out["plugin"]["sd"].shape == (p,)
    assert out["plugin"]["q025"].shape == (p,)
    assert out["plugin"]["q975"].shape == (p,)

    # mc
    assert out["mc"]["mean"].shape == (p,)
    assert out["mc"]["cov"].shape == (p, p)
    assert out["mc"]["sd"].shape == (p,)
    assert out["mc"]["q025"].shape == (p,)
    assert out["mc"]["q975"].shape == (p,)

    # covariance matrices should be symmetric
    assert torch.allclose(out["plugin"]["cov"], out["plugin"]["cov"].T, atol=1e-10, rtol=1e-10)
    assert torch.allclose(out["mc"]["cov"], out["mc"]["cov"].T, atol=1e-10, rtol=1e-10)

    # diagonal variances / sds nonnegative
    assert torch.all(torch.diag(out["plugin"]["cov"]) > 0.0)
    assert torch.all(torch.diag(out["mc"]["cov"]) >= -1e-10)
    assert torch.all(out["plugin"]["sd"] >= 0.0)
    assert torch.all(out["mc"]["sd"] >= 0.0)

    # intervals ordered correctly
    assert torch.all(out["plugin"]["q025"] <= out["plugin"]["mean"])
    assert torch.all(out["plugin"]["mean"] <= out["plugin"]["q975"])
    assert torch.all(out["mc"]["q025"] <= out["mc"]["mean"])
    assert torch.all(out["mc"]["mean"] <= out["mc"]["q975"])


def test_beta_posterior_vi_plugin_matches_beta_posterior_plugin_exactly(small_model):
    model = small_model

    out = model.beta_posterior_vi(num_mc=25, return_draws=False)
    m_beta, V_beta, _, _ = model.beta_posterior_plugin()

    sd_manual = torch.sqrt(torch.diag(V_beta).clamp_min(0.0))
    q025_manual = m_beta - 1.96 * sd_manual
    q975_manual = m_beta + 1.96 * sd_manual

    assert torch.allclose(out["plugin"]["mean"], m_beta, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["plugin"]["cov"], V_beta, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["plugin"]["sd"], sd_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["plugin"]["q025"], q025_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["plugin"]["q975"], q975_manual, atol=1e-12, rtol=1e-12)


def test_beta_posterior_vi_return_draws_matches_manual_mc_summary(small_model):
    model = small_model

    out = model.beta_posterior_vi(num_mc=40, return_draws=True)

    assert set(out.keys()) == {"plugin", "mc", "draws"}
    assert set(out["draws"].keys()) == {"mean", "cov"}

    p = model.X.shape[1]
    K = 40

    mean_draws = out["draws"]["mean"]
    cov_draws = out["draws"]["cov"]

    assert mean_draws.shape == (K, p)
    assert cov_draws.shape == (K, p, p)

    # Manual recomputation of marginalized beta posterior moments
    mean_manual = mean_draws.mean(dim=0)
    second_moment = cov_draws + mean_draws.unsqueeze(2) @ mean_draws.unsqueeze(1)
    cov_manual = second_moment.mean(dim=0) - mean_manual.unsqueeze(1) @ mean_manual.unsqueeze(0)
    cov_manual = 0.5 * (cov_manual + cov_manual.T)

    sd_manual = torch.sqrt(torch.diag(cov_manual).clamp_min(0.0))
    q025_manual = mean_manual - 1.96 * sd_manual
    q975_manual = mean_manual + 1.96 * sd_manual

    assert torch.allclose(out["mc"]["mean"], mean_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["cov"], cov_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["sd"], sd_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q025"], q025_manual, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q975"], q975_manual, atol=1e-12, rtol=1e-12)


def test_beta_posterior_vi_draw_covariances_are_symmetric_positive_definite(small_model):
    model = small_model

    out = model.beta_posterior_vi(num_mc=20, return_draws=True)
    cov_draws = out["draws"]["cov"]

    for k in range(cov_draws.shape[0]):
        Vk = cov_draws[k]
        assert torch.allclose(Vk, Vk.T, atol=1e-10, rtol=1e-10)
        eigvals = torch.linalg.eigvalsh(Vk)
        assert torch.all(eigvals > 0.0)


def test_beta_posterior_vi_mc_covariance_is_positive_semidefinite(small_model):
    model = small_model

    out = model.beta_posterior_vi(num_mc=50, return_draws=False)
    cov_mc = out["mc"]["cov"]

    eigvals = torch.linalg.eigvalsh(cov_mc)
    assert torch.all(eigvals >= -1e-8)


def test_beta_posterior_vi_is_reproducible_under_fixed_seed(small_model):
    model = small_model

    torch.manual_seed(789)
    out1 = model.beta_posterior_vi(num_mc=25, return_draws=True)

    torch.manual_seed(789)
    out2 = model.beta_posterior_vi(num_mc=25, return_draws=True)

    for layer in ("plugin", "mc"):
        for key in ("mean", "cov", "sd", "q025", "q975"):
            assert torch.allclose(out1[layer][key], out2[layer][key], atol=1e-12, rtol=1e-12)

    assert torch.allclose(out1["draws"]["mean"], out2["draws"]["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out1["draws"]["cov"], out2["draws"]["cov"], atol=1e-12, rtol=1e-12)


def test_beta_posterior_vi_invalid_num_mc_raises(small_model):
    model = small_model

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.beta_posterior_vi(num_mc=0)

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.beta_posterior_vi(num_mc=-2)

def test_spectrum_posterior_vi_returns_expected_structure_without_draws(small_model):
    model = small_model

    out = model.spectrum_posterior_vi(num_mc=30, return_draws=False)

    assert set(out.keys()) == {"plugin", "mc"}
    assert set(out["mc"].keys()) == {"mean", "sd", "q025", "q975"}

    n = model.lam.shape[0]

    assert out["plugin"].shape == (n,)
    assert out["mc"]["mean"].shape == (n,)
    assert out["mc"]["sd"].shape == (n,)
    assert out["mc"]["q025"].shape == (n,)
    assert out["mc"]["q975"].shape == (n,)

    assert torch.all(out["plugin"] > 0.0)
    assert torch.all(out["mc"]["mean"] > 0.0)
    assert torch.all(out["mc"]["sd"] >= 0.0)

    assert torch.all(out["mc"]["q025"] <= out["mc"]["mean"])
    assert torch.all(out["mc"]["mean"] <= out["mc"]["q975"])


def test_spectrum_posterior_vi_plugin_matches_direct_plugin_spectrum(small_model):
    model = small_model

    out = model.spectrum_posterior_vi(num_mc=20, return_draws=False)

    theta_plugin_u, _, _ = model.plugin_hyperparams()
    F_manual = model.filter.spectrum(model.lam, theta_plugin_u).clamp_min(0.0)

    assert torch.allclose(out["plugin"], F_manual, atol=1e-12, rtol=1e-12)


def test_spectrum_posterior_vi_return_draws_matches_summary(small_model):
    model = small_model

    out = model.spectrum_posterior_vi(num_mc=40, return_draws=True)

    assert set(out.keys()) == {"plugin", "mc", "draws"}

    K = 40
    n = model.lam.shape[0]

    assert out["draws"].shape == (K, n)
    assert torch.all(out["draws"] > 0.0)

    manual = model._summarize_draws_tensor(out["draws"])

    assert torch.allclose(out["mc"]["mean"], manual["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["sd"], manual["sd"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q025"], manual["q025"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q975"], manual["q975"], atol=1e-12, rtol=1e-12)


def test_spectrum_posterior_vi_matches_manual_sampling_under_fixed_seed(small_model):
    model = small_model
    K = 25

    torch.manual_seed(2024)
    out = model.spectrum_posterior_vi(num_mc=K, return_draws=True)

    torch.manual_seed(2024)
    F_list = []
    for _ in range(K):
        theta = model.filter.sample_unconstrained()
        F_k = model.filter.spectrum(model.lam, theta).clamp_min(0.0)
        F_list.append(F_k)
    F_manual_draws = torch.stack(F_list, dim=0)
    manual = model._summarize_draws_tensor(F_manual_draws)

    assert torch.allclose(out["draws"], F_manual_draws, atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["mean"], manual["mean"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["sd"], manual["sd"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q025"], manual["q025"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(out["mc"]["q975"], manual["q975"], atol=1e-12, rtol=1e-12)


def test_spectrum_posterior_vi_is_reproducible_under_fixed_seed(small_model):
    model = small_model

    torch.manual_seed(31415)
    out1 = model.spectrum_posterior_vi(num_mc=35, return_draws=True)

    torch.manual_seed(31415)
    out2 = model.spectrum_posterior_vi(num_mc=35, return_draws=True)

    assert torch.allclose(out1["plugin"], out2["plugin"], atol=1e-12, rtol=1e-12)

    for stat in ("mean", "sd", "q025", "q975"):
        assert torch.allclose(out1["mc"][stat], out2["mc"][stat], atol=1e-12, rtol=1e-12)

    assert torch.allclose(out1["draws"], out2["draws"], atol=1e-12, rtol=1e-12)


def test_spectrum_posterior_vi_invalid_num_mc_raises(small_model):
    model = small_model

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.spectrum_posterior_vi(num_mc=0)

    with pytest.raises(ValueError, match="num_mc must be positive"):
        model.spectrum_posterior_vi(num_mc=-4)