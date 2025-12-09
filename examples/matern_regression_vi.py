# examples/matern_regression_vi.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.filters import MaternLikeFilterFullVI

torch.set_default_dtype(torch.double)


class MaternRegressionVI(nn.Module):
    """
    Variational regression model with a Matérn-like spectral prior on φ.

    Model:
        y | beta, phi, sigma2 ~ N(X beta + phi, sigma2 I)
        phi = U z
        z | theta ~ N(0, diag(F(λ; theta)))
        beta ~ N(0, sigma2_beta I)
        sigma2 ~ Inv-Gamma(a0, b0)
        theta (unconstrained) ~ N(0, I)   [handled by MaternLikeFilterFullVI]

    Variational family (mean-field):
        q(beta) = N(m_beta, diag(v_beta))
        q(z)    = N(m_z, diag(v_z))
        s = log sigma2, q(s) = N(mu_s, v_s)
        q(theta) given by MaternLikeFilterFullVI (Gaussian in unconstrained coords)
    """
    def __init__(self, n: int, p: int):
        super().__init__()

        # q(beta)
        self.m_beta = nn.Parameter(torch.zeros(p))
        self.log_v_beta = nn.Parameter(torch.full((p,), -3.0))  # exp(-3) ~ 0.05

        # q(z)
        self.m_z = nn.Parameter(torch.zeros(n))
        self.log_v_z = nn.Parameter(torch.full((n,), -3.0))

        # q(log sigma2)
        self.mu_s = nn.Parameter(torch.tensor(0.0))
        self.log_v_s = nn.Parameter(torch.tensor(-3.0))

    def elbo(self,
             y: torch.Tensor,
             X: torch.Tensor,
             U: torch.Tensor,
             lam: torch.Tensor,
             matern: MaternLikeFilterFullVI,
             a0: float,
             b0: float,
             sigma2_beta: float,
             num_mc_theta: int = 5) -> torch.Tensor:
        """
        Monte Carlo ELBO (over θ) with analytic expectations over (β, z, s).

        Args:
            y: [n]
            X: [n, p]
            U: [n, n] Laplacian eigenvectors
            lam: [n] Laplacian eigenvalues
            matern: MaternLikeFilterFullVI module (q(theta))
            a0, b0: IG prior parameters for sigma2
            sigma2_beta: prior variance for beta
            num_mc_theta: # of Monte Carlo samples from q(theta)

        Returns:
            Scalar ELBO tensor.
        """
        n = y.numel()

        # Variational variances
        v_beta = torch.exp(self.log_v_beta)   # [p]
        v_z = torch.exp(self.log_v_z)         # [n]
        v_s = torch.exp(self.log_v_s)         # scalar

        # sigma^2 moments under q(s)
        mu_s = self.mu_s
        alpha = torch.exp(-mu_s + 0.5 * v_s)  # E_q[exp(-s)] = E_q[1/sigma2]

        # Mean residual r_bar = y - X m_beta - U m_z
        r_bar = y - X @ self.m_beta - U @ self.m_z

        # R = E[||y - X beta - U z||^2]
        #   = ||r_bar||^2 + sum_j v_beta_j ||X_j||^2 + sum_i v_z_i
        R = r_bar.pow(2).sum()
        R += torch.sum(v_beta * (X**2).sum(dim=0))
        R += torch.sum(v_z)

        # --- Data term (up to constant -0.5 n log(2π)) ---
        data_term = -0.5 * n * mu_s - 0.5 * alpha * R

        # --- Beta prior term (ridge), ignoring normalizing constant ---
        beta_prior = -0.5 * (self.m_beta.pow(2).sum() + v_beta.sum()) / sigma2_beta

        # --- sigma^2 IG prior term, up to constant ---
        sigma_prior = -(a0 + 1) * mu_s - b0 * alpha

        # --- z prior term: E_{q(theta)} E_{q(z)}[log p(z | theta)] ---
        # For each theta sample, we use analytic expectation over z.
        z2_plus_var = self.m_z.pow(2) + v_z    # [n]

        z_prior_mc = 0.0
        for _ in range(num_mc_theta):
            tau2, a, log_tau2, a_raw = matern.sample_params()
            rho0, nu = a.unbind(-1)  # scalars

            F_theta = tau2 * (lam + rho0).pow(-nu)  # [n]
            F_theta = torch.clamp(F_theta, min=1e-10)

            z_prior_theta = -0.5 * torch.sum(z2_plus_var / F_theta + torch.log(F_theta))
            z_prior_mc = z_prior_mc + z_prior_theta

        z_prior_mc = z_prior_mc / num_mc_theta

        # --- KL for theta (q(theta) || p(theta)) ---
        kl_theta = matern.kl_q_p()

        # --- Entropies of q(beta), q(z), q(s) (diagonal Gaussians) ---
        H_beta = 0.5 * torch.sum(1.0 + torch.log(v_beta))
        H_z = 0.5 * torch.sum(1.0 + torch.log(v_z))
        H_s = 0.5 * (1.0 + torch.log(v_s))

        elbo = data_term + beta_prior + sigma_prior + z_prior_mc + H_beta + H_z + H_s - kl_theta
        return elbo


def main():
    print("Start...")
    device = torch.device("cpu")

    # --------------------------------------------
    # 1. Grid + Laplacian + eigen-decomposition
    # --------------------------------------------
    nx, ny = 20, 20
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [n,2]
    n = coords.size(0)

    L, W = build_laplacian_from_knn(
        coords,
        k=8,
        gamma=0.2,
        rho=0.95,
    )

    lam, U = laplacian_eigendecomp(L)
    lam = lam.to(device)
    U = U.to(device)

    # --------------------------------------------
    # 2. Define CAR prior (true) and simulate phi_true
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)           # [n]

    eps_z = torch.randn(n, dtype=torch.double, device=device)
    z_true = torch.sqrt(F_car) * eps_z            # [n]
    phi_true = U @ z_true                         # [n]

    # --------------------------------------------
    # 3. Design matrix X, true beta, sigma2, simulate y
    # --------------------------------------------
    x_coord = coords[:, 0]
    X = torch.stack([
        torch.ones(n, dtype=torch.double, device=device),
        x_coord,
    ], dim=1)                                     # [n, 2]
    p = X.shape[1]

    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    sigma2_true = 0.1

    eps_y = torch.randn(n, dtype=torch.double, device=device)
    y = X @ beta_true + phi_true + torch.sqrt(torch.tensor(sigma2_true)) * eps_y

    # --------------------------------------------
    # 4. Priors for VI (match Gibbs / CAR-VI settings)
    # --------------------------------------------
    sigma2_beta = 10.0
    a0 = 2.0
    b0 = 0.5

    # --------------------------------------------
    # 5. Initialize VI model + Matérn VI over θ
    # --------------------------------------------
    model = MaternRegressionVI(n=n, p=p).to(device)

    # matern = MaternLikeFilterFullVI(
    #     mu_log_tau2=0.0,
    #     log_std_log_tau2=-1.0,   # somewhat informative
    #     mu_rho0_raw=0.0,
    #     log_std_rho0_raw=-1.0,
    #     mu_nu_raw=0.0,
    #     log_std_nu_raw=-1.0,
    # ).to(device)

    matern = MaternLikeFilterFullVI(
    # Center τ² near CAR truth 0.4
    mu_log_tau2=math.log(0.4),
    log_std_log_tau2=-1.5,   # fairly informative

    # Strong prior that ρ₀ is tiny (≈ ε = 1e-3)
    mu_rho0_raw=-7.0,
    log_std_rho0_raw=-2.5,   # narrow prior, keeps ρ₀ small

    # Prior that ν ≈ 1
    mu_nu_raw=0.5,
    log_std_nu_raw=-1.5,
    ).to(device)



    # optimizer = torch.optim.Adam(
    #     list(model.parameters()) + list(matern.parameters()),
    #     lr=1e-2,
    # )

    optimizer = torch.optim.Adam([
    {"params": model.parameters(), "lr": 1e-2},   # β, z, σ²
    {"params": matern.parameters(), "lr": 5e-3},  # θ = (τ², ρ₀, ν)
    ])


    num_iters = 3000
    num_mc_theta = 10   # was 5
    elbo_history = []

    # push tensors to device
    y = y.to(device)
    X = X.to(device)
    U = U.to(device)
    lam = lam.to(device)
    F_car = F_car.to(device)

    for it in range(num_iters):
        optimizer.zero_grad()
        elbo = model.elbo(
            y, X, U, lam,
            matern=matern,
            a0=a0, b0=b0,
            sigma2_beta=sigma2_beta,
            num_mc_theta=num_mc_theta,
        )
        loss = -elbo
        loss.backward()
        optimizer.step()

        elbo_history.append(elbo.item())

        if it % 200 == 0:
            with torch.no_grad():
                # Posterior means for beta, sigma2, theta
                m_beta = model.m_beta.detach()

                v_s = torch.exp(model.log_v_s.detach())
                mu_s = model.mu_s.detach()
                sigma2_mean = torch.exp(mu_s + 0.5 * v_s).item()

                tau2_mean, a_mean = matern.mean_params()
                rho0_mean, nu_mean = a_mean.unbind(-1)

            print(
                f"[{it:04d}] ELBO={elbo.item():.2f}, "
                f"beta_mean={m_beta.cpu().numpy()}, "
                f"sigma2_mean={sigma2_mean:.4f}, "
                f"tau2_mean={tau2_mean.item():.4f}, "
                f"rho0_mean={rho0_mean.item():.5f}, "
                f"nu_mean={nu_mean.item():.4f}"
            )

    # --------------------------------------------
    # 6. Posterior summaries
    # --------------------------------------------
    with torch.no_grad():
        m_beta = model.m_beta.detach()
        v_s = torch.exp(model.log_v_s.detach())
        mu_s = model.mu_s.detach()
        sigma2_mean = torch.exp(mu_s + 0.5 * v_s)

        phi_mean = (U @ model.m_z.detach()).cpu()

        tau2_hat, a_hat = matern.mean_params()
        rho0_hat, nu_hat = a_hat.unbind(-1)

    print("\nPosterior means (Matérn VI, CAR data):")
    print(f"  beta_true    = {beta_true.cpu().numpy()}")
    print(f"  beta_mean    = {m_beta.cpu().numpy()}")
    print(f"  sigma2_true  = {sigma2_true:.4f}")
    print(f"  sigma2_mean  = {sigma2_mean.item():.4f}")
    print(f"  tau2_true    = {tau2_true:.4f}")
    print(f"  tau2_hat     = {tau2_hat.item():.4f}")
    print(f"  eps_car      = {eps_car:.5f}")
    print(f"  rho0_hat     = {rho0_hat.item():.5f}")
    print(f"  nu_target    ≈ 1.0 (CAR-like)")
    print(f"  nu_hat       = {nu_hat.item():.4f}")

    # RMSE for φ
    rmse_phi = torch.sqrt(torch.mean((phi_mean - phi_true.cpu())**2))
    print(f"  RMSE(phi_mean, phi_true) = {rmse_phi.item():.4f}")

    # --------------------------------------------
    # 7. Plots: ELBO, spectral filters, φ_true vs φ_mean
    # --------------------------------------------
    fig_dir = Path("examples") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ELBO convergence
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_iters), elbo_history)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (up to const)")
    plt.title("ELBO convergence (Matérn regression VI)")
    plt.tight_layout()
    fig_path_elbo = fig_dir / "matern_regression_vi_elbo.png"
    plt.savefig(fig_path_elbo, dpi=200)
    plt.close()
    print(f"Saved ELBO plot to: {fig_path_elbo}")

    # Spectral filter comparison: CAR vs learned Matérn
    with torch.no_grad():
        F_matern_hat = tau2_hat * (lam + rho0_hat).pow(-nu_hat)

    mask = lam > 1e-6
    lam_plot = lam[mask].cpu()
    F_car_plot = F_car[mask].cpu()
    F_matern_plot = F_matern_hat[mask].cpu()

    plt.figure(figsize=(6, 4))
    plt.loglog(lam_plot, F_car_plot, 'o', markersize=3, label="True CAR: τ² / (λ + ε)")
    plt.loglog(lam_plot, F_matern_plot, '--', label="Fitted Matérn-like (VI mean)")
    plt.xlabel("Eigenvalue λ")
    plt.ylabel("Spectral variance F(λ)")
    plt.title("CAR vs Matérn-like spectral filter (regression VI)")
    plt.legend()
    plt.tight_layout()
    fig_path_filter = fig_dir / "matern_regression_vi_filter.png"
    plt.savefig(fig_path_filter, dpi=200)
    plt.close()
    print(f"Saved filter comparison plot to: {fig_path_filter}")

    # φ_true vs φ_mean
    phi_true_grid = phi_true.view(nx, ny).cpu()
    phi_mean_grid = phi_mean.view(nx, ny)

    vmax = torch.max(torch.stack([
        phi_true_grid.abs().max(),
        phi_mean_grid.abs().max()
    ])).item()
    vmin, vmax = -vmax, vmax

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(phi_true_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("φ_true (CAR)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(phi_mean_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("E[φ | y] (Matérn VI)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("φ")

    fig_path_phi = fig_dir / "matern_regression_vi_phi_true_vs_mean.png"
    plt.savefig(fig_path_phi, dpi=200)
    plt.close()
    print(f"Saved φ comparison plot to: {fig_path_phi}")


if __name__ == "__main__":
    main()
