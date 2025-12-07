# examples/car_regression_vi.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- set seeds here ----
import random
import numpy as np
seed = 1010  # pick any integer you like
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp

torch.set_default_dtype(torch.double)


class CARRegressionVI(nn.Module):
    """
    Variational inference for CAR regression with θ fixed (CAR prior).

    Model:
        y | beta, phi, sigma2 ~ N(X beta + phi, sigma2 I)
        phi = U z,   z_i ~ N(0, F_car_i)
        beta ~ N(0, sigma2_beta I)
        sigma2 ~ Inv-Gamma(a0, b0)

    Variational family (mean-field):
        q(beta) = N(m_beta, diag(v_beta))
        q(z)    = N(m_z, diag(v_z))
        s = log sigma2, q(s) = N(mu_s, v_s), sigma2 = exp(s)
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
             F_car: torch.Tensor,
             a0: float,
             b0: float,
             sigma2_beta: float) -> torch.Tensor:
        """
        Analytic ELBO up to an additive constant.

        Args:
            y: [n]
            X: [n, p]
            U: [n, n] Laplacian eigenvectors
            F_car: [n] CAR spectral variances
            a0, b0: IG prior parameters for sigma2
            sigma2_beta: prior variance for beta

        Returns:
            Scalar ELBO tensor.
        """
        n = y.numel()

        # Variational variances
        v_beta = torch.exp(self.log_v_beta)    # [p]
        v_z = torch.exp(self.log_v_z)          # [n]
        v_s = torch.exp(self.log_v_s)          # scalar

        # sigma^2 moments under q(s)
        mu_s = self.mu_s
        alpha = torch.exp(-mu_s + 0.5 * v_s)   # E_q[exp(-s)] = E_q[1/sigma2]

        # Mean residual r_bar = y - X m_beta - U m_z
        r_bar = y - X @ self.m_beta - U @ self.m_z

        # R = E[||y - X beta - U z||^2]
        #   = ||r_bar||^2 + sum_j v_beta_j ||X_j||^2 + sum_i v_z_i
        R = r_bar.pow(2).sum()
        R += torch.sum(v_beta * (X**2).sum(dim=0))
        R += torch.sum(v_z)

        # --- Data term (up to constant -0.5 n log(2π)) ---
        # E[log p(y | beta, z, s)]
        # = -0.5 n E[s] - 0.5 E[exp(-s)] E[||...||^2]
        data_term = -0.5 * n * mu_s - 0.5 * alpha * R

        # --- Beta prior term (ridge), ignoring normalizing constant ---
        # log p(beta) ∝ -0.5 / sigma2_beta * ||beta||^2
        beta_prior = -0.5 * (self.m_beta.pow(2).sum() + v_beta.sum()) / sigma2_beta

        # --- z CAR prior term, up to constant ---
        # log p(z) ∝ -0.5 * sum_i [ log F_i + z_i^2 / F_i ]
        z_prior = -0.5 * torch.sum((self.m_z.pow(2) + v_z) / F_car + torch.log(F_car))

        # --- sigma^2 IG prior term, up to constant ---
        # log p(sigma2) = const - (a0+1) log sigma2 - b0 / sigma2
        # E[log sigma2] = mu_s, E[1/sigma2] = alpha
        sigma_prior = -(a0 + 1) * mu_s - b0 * alpha

        # --- Entropies (diagonal Gaussians), up to const 0.5 d log(2π) ---
        H_beta = 0.5 * torch.sum(1.0 + torch.log(v_beta))
        H_z = 0.5 * torch.sum(1.0 + torch.log(v_z))
        H_s = 0.5 * (1.0 + torch.log(v_s))

        elbo = data_term + beta_prior + z_prior + sigma_prior + H_beta + H_z + H_s
        return elbo


def main():
    device = torch.device("cpu")

    # --------------------------------------------
    # 1. Grid + Laplacian + eigen-decomposition
    #    (same setup as in car_regression_gibbs.py)
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
    # 2. Define CAR prior (fixed theta) and simulate phi_true
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)        # [n]

    eps_z = torch.randn(n, dtype=torch.double, device=device)
    z_true = torch.sqrt(F_car) * eps_z         # [n]
    phi_true = U @ z_true                      # [n]

    # --------------------------------------------
    # 3. Design matrix X, true beta, sigma2, simulate y
    # --------------------------------------------
    x_coord = coords[:, 0]
    X = torch.stack([
        torch.ones(n, dtype=torch.double, device=device),
        x_coord,
    ], dim=1)                                  # [n, 2]
    p = X.shape[1]

    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    sigma2_true = 0.1

    eps_y = torch.randn(n, dtype=torch.double, device=device)
    y = X @ beta_true + phi_true + torch.sqrt(torch.tensor(sigma2_true)) * eps_y

    # --------------------------------------------
    # 4. Priors for VI (match Gibbs settings)
    # --------------------------------------------
    sigma2_beta = 10.0
    a0 = 2.0
    b0 = 0.5

    # --------------------------------------------
    # 5. Initialize VI model + optimizer
    # --------------------------------------------
    model = CARRegressionVI(n=n, p=p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    num_iters = 3000
    elbo_history = []

    # Precompute some constants on device
    y = y.to(device)
    X = X.to(device)
    U = U.to(device)
    F_car = F_car.to(device)

    for it in range(num_iters):
        optimizer.zero_grad()
        elbo = model.elbo(y, X, U, F_car, a0=a0, b0=b0, sigma2_beta=sigma2_beta)
        loss = -elbo
        loss.backward()
        optimizer.step()

        elbo_history.append(elbo.item())

        if it % 200 == 0:
            with torch.no_grad():
                # Posterior means
                m_beta = model.m_beta.detach()
                v_s = torch.exp(model.log_v_s.detach())
                mu_s = model.mu_s.detach()
                # E[σ²] under log-normal: exp(mu_s + 0.5 v_s)
                sigma2_mean = torch.exp(mu_s + 0.5 * v_s).item()
            print(
                f"[{it:04d}] ELBO={elbo.item():.2f}, "
                f"beta_mean={m_beta.cpu().numpy()}, "
                f"sigma2_mean={sigma2_mean:.4f}"
            )

    # --------------------------------------------
    # 6. Posterior summaries
    # --------------------------------------------
    with torch.no_grad():
        m_beta = model.m_beta.detach()
        v_s = torch.exp(model.log_v_s.detach())
        mu_s = model.mu_s.detach()
        sigma2_mean = torch.exp(mu_s + 0.5 * v_s)

        # Posterior mean phi = U m_z
        phi_mean = (U @ model.m_z.detach()).cpu()

    print("\nPosterior means (VI, CAR fixed):")
    print(f"  beta_true    = {beta_true.cpu().numpy()}")
    print(f"  beta_mean    = {m_beta.cpu().numpy()}")
    print(f"  sigma2_true  = {sigma2_true:.4f}")
    print(f"  sigma2_mean  = {sigma2_mean.item():.4f}")

    # RMSE between VI φ_mean and φ_true
    rmse_phi = torch.sqrt(torch.mean((phi_mean - phi_true.cpu())**2))
    print(f"  RMSE(phi_mean, phi_true) = {rmse_phi.item():.4f}")

    # --------------------------------------------
    # 7. Plots: ELBO and φ_true vs φ_mean
    # --------------------------------------------
    fig_dir = Path("examples") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ELBO convergence
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_iters), elbo_history)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (up to const)")
    plt.title("ELBO convergence (CAR regression VI)")
    plt.tight_layout()
    fig_path_elbo = fig_dir / "car_regression_vi_elbo.png"
    plt.savefig(fig_path_elbo, dpi=200)
    plt.close()
    print(f"Saved ELBO plot to: {fig_path_elbo}")

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
    axes[0].set_title("φ_true")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(phi_mean_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("E[φ | y] (VI)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("φ")

    fig_path_phi = fig_dir / "car_regression_vi_phi_true_vs_mean.png"
    plt.savefig(fig_path_phi, dpi=200)
    plt.close()
    print(f"Saved φ comparison plot to: {fig_path_phi}")


if __name__ == "__main__":
    main()
