# examples/car_regression_gibbs.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
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


def sample_beta(y, X, phi, sigma2, sigma2_beta):
    """
    Sample beta | y, phi, sigma2 under prior beta ~ N(0, sigma2_beta I).
    """
    n, p = X.shape
    # precision: Lambda = X^T X / sigma2 + I / sigma2_beta
    XT = X.T
    precision = (XT @ X) / sigma2 + torch.eye(p, dtype=torch.double) / sigma2_beta
    cov = torch.linalg.inv(precision)

    rhs = (XT @ (y - phi)) / sigma2
    mean = cov @ rhs

    eps = torch.randn(p, dtype=torch.double)
    L = torch.linalg.cholesky(cov)
    beta = mean + L @ eps
    return beta


def sample_phi_spectral(y, X, beta, sigma2, lam, U, F_car):
    """
    Sample phi | y, X, beta, sigma2 under spectral CAR prior.
    """
    # residual r = y - X beta
    r = y - X @ beta               # [n]
    r_tilde = U.T @ r             # [n]

    # posterior variance and mean for z in spectral basis
    # v_i = (1/F_i + 1/sigma2)^(-1)
    inv_prior = 1.0 / F_car
    inv_lik = 1.0 / sigma2
    v = 1.0 / (inv_prior + inv_lik)          # [n]
    m = v * (inv_lik * r_tilde)              # [n]

    eps = torch.randn_like(m)
    z = m + torch.sqrt(v) * eps              # [n]
    phi = U @ z                              # [n]
    return phi


def sample_sigma2_ig(y, X, beta, phi, a0, b0):
    """
    Sample sigma2 | y, X, beta, phi under Inv-Gamma(a0, b0) prior.
    Using parameterization: p(sigma2) ∝ (sigma2)^(-a0-1) exp(-b0 / sigma2).
    """
    e = y - X @ beta - phi
    ssr = torch.dot(e, e).item()

    a_post = a0 + 0.5 * y.numel()
    b_post = b0 + 0.5 * ssr

    # If sigma2 ~ IG(a_post, b_post), then 1/sigma2 ~ Gamma(a_post, rate = b_post)
    gamma_dist = torch.distributions.Gamma(concentration=a_post,
                                           rate=b_post)
    inv_sigma2 = gamma_dist.sample()
    #sigma2 = b_post / inv_sigma2
    sigma2 = 1 / inv_sigma2
    return sigma2.item()


def main():
    device = torch.device("cpu")

    # --------------------------------------------
    # 1. Grid + Laplacian + eigen-decomp
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
    tau2_true = 0.4    # was 1.0
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)      # [n]

    # simulate one phi_true
    eps_z = torch.randn(n, dtype=torch.double, device=device)
    z_true = torch.sqrt(F_car) * eps_z       # [n]
    phi_true = U @ z_true                    # [n]

    # --------------------------------------------
    # 3. Design matrix X and true beta, sigma2
    # --------------------------------------------
    # Example: intercept + x-coordinate as covariate
    x_coord = coords[:, 0]                   # [n]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device),
                     x_coord], dim=1)        # [n, 2]
    p = X.shape[1]

    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    sigma2_true = 0.1

    # simulate y
    eps_y = torch.randn(n, dtype=torch.double, device=device)
    y = X @ beta_true + phi_true + torch.sqrt(torch.tensor(sigma2_true)) * eps_y

    # --------------------------------------------
    # 4. Priors for Gibbs sampler
    # --------------------------------------------
    sigma2_beta = 10.0   # prior variance on each beta coefficient
    a0 = 2.0
    b0 = 0.5

    # --------------------------------------------
    # 5. Gibbs sampler
    # --------------------------------------------
    num_iters = 5000
    burn_in = 1000

    # init
    beta = torch.zeros(p, dtype=torch.double, device=device)
    phi = torch.zeros(n, dtype=torch.double, device=device)
    sigma2 = 1.0

    beta_samples = []
    sigma2_samples = []
    phi_samples = []

    for it in range(num_iters):
        beta = sample_beta(y, X, phi, sigma2, sigma2_beta)
        phi = sample_phi_spectral(y, X, beta, sigma2, lam, U, F_car)
        sigma2 = sample_sigma2_ig(y, X, beta, phi, a0, b0)

        if it % 500 == 0:
            print(f"[{it:04d}] sigma2={sigma2:.4f}, beta={beta.cpu().numpy()}")

        if it >= burn_in:
            beta_samples.append(beta.detach().cpu().numpy())
            sigma2_samples.append(sigma2)
            phi_samples.append(phi.detach().cpu().numpy())

    #beta_samples = torch.tensor(beta_samples)        # [N_samp, p]
    beta_samples = torch.from_numpy(np.stack(beta_samples, axis=0))
    sigma2_samples = torch.tensor(sigma2_samples)    # [N_samp]
    phi_samples = torch.from_numpy(np.stack(phi_samples, axis=0))  

    beta_mean = beta_samples.mean(0)
    sigma2_mean = sigma2_samples.mean()

    # Posterior mean φ and RMSE vs φ_true
    phi_mean = phi_samples.mean(0)            # [n]
    rmse_phi = torch.sqrt(torch.mean((phi_mean - phi_true.cpu())**2))

    print("\nPosterior means (Gibbs, CAR fixed):")
    print(f"  beta_true    = {beta_true.cpu().numpy()}")
    print(f"  beta_mean    = {beta_mean.numpy()}")
    print(f"  sigma2_true  = {sigma2_true:.4f}")
    print(f"  sigma2_mean  = {sigma2_mean.item():.4f}")
    print(f"  RMSE(phi_mean, phi_true) = {rmse_phi.item():.4f}")

    # --------------------------------------------
    # 6. Simple trace plots
    # --------------------------------------------
    fig_dir = Path("examples") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # beta traces
    fig, axes = plt.subplots(p+1, 1, figsize=(7, 2.5*(p+1)), sharex=True, constrained_layout=True)
    iters_kept = range(num_iters - burn_in)

    for j in range(p):
        axes[j].plot(iters_kept, beta_samples[:, j])
        axes[j].axhline(beta_true[j].item(), color="gray", linestyle="--", linewidth=1)
        axes[j].set_ylabel(f"beta[{j}]")
    axes[p].plot(iters_kept, sigma2_samples)
    axes[p].axhline(sigma2_true, color="gray", linestyle="--", linewidth=1)
    axes[p].set_ylabel("sigma2")
    axes[p].set_xlabel("Iteration")

    fig.suptitle("Gibbs sampler traces (CAR regression, θ fixed)", y=1.02)
    fig_path_traces = fig_dir / "car_regression_gibbs_traces.png"
    plt.savefig(fig_path_traces, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved trace plot to: {fig_path_traces}")

    
    # --------------------------------------------
    # 7. Compare φ_true vs posterior mean φ
    # --------------------------------------------
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
    axes[1].set_title("E[φ | y] (Gibbs)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("φ")

    fig_path_phi = fig_dir / "car_regression_gibbs_phi_true_vs_mean.png"
    plt.savefig(fig_path_phi, dpi=200)
    plt.close()
    print(f"Saved φ comparison plot to: {fig_path_phi}")



if __name__ == "__main__":
    main()
