import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround for OpenMP clash

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no GUI

import torch
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.filters import MaternLikeFilterFullVI

# Use double for everything
torch.set_default_dtype(torch.double)


def generate_car_spectral_samples(lam, U, tau2_true=1.0, eps=1e-3, M=200):
    """
    Generate spectral CAR samples from:

        z ~ N(0, diag(F_car)),   F_car(λ) = τ² / (λ + eps).

    Returns:
        z_true: [M, n] spectral samples
        phi:   [M, n] spatial fields (U z^T)^T
        F_car: [n] spectral variance
    """
    n = lam.numel()
    F_car = tau2_true / (lam + eps)          # [n]
    sqrt_F_car = torch.sqrt(F_car)           # [n]

    eps_std = torch.randn(M, n, dtype=torch.double, device=lam.device)
    z_true = eps_std * sqrt_F_car[None, :]   # [M, n]
    phi = (U @ z_true.T).T                  # [M, n]

    return z_true, phi, F_car


def elbo_matern_on_car(matern_module, lam, z, num_mc: int = 5):
    """
    ELBO for fitting the Matérn-like spectral filter to CAR-generated z.

    Data:
        z ~ N(0, diag(F_car(λ))) from the CAR prior.

    Model:
        F(λ; θ) = τ² (λ + ρ₀)^(-ν), with θ ~ N(0, I) in unconstrained space.

    Variational family:
        q(θ) is Gaussian in unconstrained space, parameterized by matern_module.

    ELBO(φ) = E_{q(θ)}[log p(z | θ)] - KL(q(θ) || p(θ)),
    approximated by Monte Carlo over q(θ).
    """
    device = lam.device
    z = z.to(device)
    M, n = z.shape

    elbo = 0.0
    for _ in range(num_mc):
        # Sample θ from q(θ): (log τ², rho0_raw, nu_raw)
        tau2, a, log_tau2, a_raw = matern_module.sample_params()
        rho0, nu = a.unbind(-1)  # both scalars

        # Matérn-like spectral variance F(λ; θ)
        F_lam = tau2 * (lam + rho0).pow(-nu)  # [n]
        F_lam = torch.clamp(F_lam, min=1e-10)
        F_broadcast = F_lam[None, :]         # [M, n]

        # log p(z | θ) up to constant -0.5 M n log(2π)
        #   = -0.5 * sum_{m,i} [ log F_i + z_{mi}^2 / F_i ]
        loglik = -0.5 * (torch.log(F_broadcast) + z**2 / F_broadcast).sum()

        elbo += loglik

    elbo /= num_mc

    # KL(q(θ)||p(θ)) where p(θ)=N(0,I) in unconstrained coords
    kl = matern_module.kl_q_p()

    return elbo - kl


def main():
    device = torch.device("cpu")

    # -----------------------------
    # 1. Build a grid + Laplacian
    # -----------------------------
    nx, ny = 20, 20  # 20x20 grid = 400 nodes
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")  # [nx, ny] each

    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [n, 2]
    n = coords.size(0)

    # kNN graph -> Laplacian
    L, W = build_laplacian_from_knn(
        coords,
        k=8,
        gamma=0.2,
        rho=0.95,
    )

    # Eigendecomposition
    lam, U = laplacian_eigendecomp(L)  # lam: [n], U: [n, n]
    lam = lam.to(device)
    U = U.to(device)

    # --------------------------------------------
    # 2. Generate spectral CAR samples (z, φ)
    # --------------------------------------------
    tau2_true = 1.0
    eps_car = 1e-3
    M = 300  # number of replicates

    z_true, phi_true, F_car = generate_car_spectral_samples(
        lam, U, tau2_true=tau2_true, eps=eps_car, M=M
    )

    # --------------------------------------------
    # 3. Initialize Matérn-like filter VI module
    # --------------------------------------------
    matern = MaternLikeFilterFullVI(
        mu_log_tau2=0.0,
        log_std_log_tau2=-2.3,   # moderately tight prior
        mu_rho0_raw=0.0,
        log_std_rho0_raw=-2.3,
        mu_nu_raw=0.0,
        log_std_nu_raw=-2.3,
    ).to(device)

    optimizer = Adam(matern.parameters(), lr=1e-2)

    # --------------------------------------------
    # 4. VI optimization loop: fit Matérn to CAR z's
    # --------------------------------------------
    num_iters = 1000
    num_mc = 5

    for it in range(num_iters):
        optimizer.zero_grad()
        elbo = elbo_matern_on_car(matern, lam, z_true, num_mc=num_mc)
        loss = -elbo
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            with torch.no_grad():
                tau2_mean, a_mean = matern.mean_params()
                rho0_mean, nu_mean = a_mean.unbind(-1)
            print(
                f"[{it:04d}] ELBO={elbo.item():.2f}, "
                f"tau2_mean={tau2_mean.item():.4f}, "
                f"rho0_mean={rho0_mean.item():.5f}, "
                f"nu_mean={nu_mean.item():.4f}"
            )

    # Final variational mean parameters (posterior means)
    with torch.no_grad():
        tau2_hat, a_hat = matern.mean_params()
        rho0_hat, nu_hat = a_hat.unbind(-1)

    print("\nRecovered Matérn-like parameters (mean under q):")
    print(f"  tau2_hat ≈ {tau2_hat.item():.4f} (true τ² = {tau2_true})")
    print(f"  rho0_hat ≈ {rho0_hat.item():.6f} (CAR eps = {eps_car})")
    print(f"  nu_hat   ≈ {nu_hat.item():.4f} (CAR corresponds to ν ≈ 1)")

    # # --------------------------------------------
    # # 5. Compare filters: CAR vs fitted Matérn (mean q)
    # # --------------------------------------------
    # with torch.no_grad():
    #     F_matern_hat = tau2_hat * (lam + rho0_hat).pow(-nu_hat)

    # mask = lam > 1e-6
    # lam_plot = lam[mask].cpu()
    # F_car_plot = F_car[mask].cpu()
    # F_matern_plot = F_matern_hat[mask].cpu()

    # plt.figure(figsize=(6, 4))
    # plt.loglog(lam_plot, F_car_plot, label="True CAR: τ² / (λ + ε)")
    # plt.loglog(lam_plot, F_matern_plot, "--", label="Fitted Matérn-like (VI mean)")
    # plt.xlabel("Eigenvalue λ")
    # plt.ylabel("Spectral variance F(λ)")
    # plt.title("CAR vs Matérn-like spectral filter (VI)")
    # plt.legend()
    # plt.tight_layout()

    # fig_dir = Path("examples") / "figures"
    # fig_dir.mkdir(parents=True, exist_ok=True)
    # fig_path_filter = fig_dir / "matern_recovers_car_filter_vi.png"
    # plt.savefig(fig_path_filter, dpi=200)
    # plt.close()
    # print(f"Saved filter comparison plot to: {fig_path_filter}")


    # --------------------------------------------
    # 5. Compare filters: CAR vs fitted Matérn (mean q)
    #    → 3-panel figure:
    #      (1) smooth curves
    #      (2) scatter per eigenvalue
    #      (3) spectral ratio F_matern / F_car
    # --------------------------------------------
    with torch.no_grad():
        F_matern_hat = tau2_hat * (lam + rho0_hat).pow(-nu_hat)

    mask = lam > 1e-6
    lam_plot = lam[mask].cpu()
    F_car_plot = F_car[mask].cpu()
    F_matern_plot = F_matern_hat[mask].cpu()

    R_plot = (F_matern_plot / F_car_plot).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # ---- Panel 1: smooth log–log curves ----
    ax = axes[0]
    ax.loglog(lam_plot, F_car_plot, label="True CAR: τ² / (λ + ε)")
    ax.loglog(lam_plot, F_matern_plot, "--", label="Fitted Matérn-like (VI mean)")
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("Spectral variance F(λ)")
    ax.set_title("Filters (smooth curves)")
    ax.legend(fontsize=8)

    # ---- Panel 2: scatter per eigenvalue ----
    ax = axes[1]
    ax.loglog(lam_plot, F_car_plot, 'o', markersize=3,
              label="CAR (per eigenvalue)")
    ax.loglog(lam_plot, F_matern_plot, 'x', markersize=3,
              label="Matérn (per eigenvalue)")
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("Spectral variance F(λ)")
    ax.set_title("Filters (scatter)")
    ax.legend(fontsize=8)

    # ---- Panel 3: spectral ratio ----
    ax = axes[2]
    ax.semilogx(lam_plot, R_plot)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("Ratio F_Mat / F_CAR")
    ax.set_title("Spectral ratio")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig_dir = Path("examples") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path_filter = fig_dir / "matern_recovers_car_filter_vi_3panel.png"
    fig.suptitle("CAR vs Matérn-like spectral filter (VI)", y=1.02, fontsize=12)
    plt.savefig(fig_path_filter, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-panel filter comparison plot to: {fig_path_filter}")







    # --------------------------------------------
    # 6. Sample φ fields: CAR vs fitted Matérn, plot on grid
    # --------------------------------------------
    # One sample from true CAR
    eps_car_sample = torch.randn(n, dtype=torch.double, device=device)
    phi_car_sample = U @ (torch.sqrt(F_car) * eps_car_sample)  # [n]

    # One sample from fitted Matérn-like (using VI means)
    with torch.no_grad():
        F_matern_hat = tau2_hat * (lam + rho0_hat).pow(-nu_hat)
    eps_mat = torch.randn(n, dtype=torch.double, device=device)
    phi_matern_sample = U @ (torch.sqrt(F_matern_hat) * eps_mat)  # [n]

    # Reshape to [nx, ny] grids
    phi_car_grid = phi_car_sample.view(nx, ny).cpu()
    phi_matern_grid = phi_matern_sample.view(nx, ny).cpu()

    vmax = torch.max(torch.stack([
        phi_car_grid.abs().max(), phi_matern_grid.abs().max()
    ])).item()
    vmin, vmax = -vmax, vmax

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(phi_car_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Sample φ from true CAR")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(phi_matern_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Sample φ from fitted Matérn-like (VI mean)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("φ value")

    fig_path_phi = fig_dir / "matern_recovers_car_phi_grid_vi.png"
    plt.savefig(fig_path_phi, dpi=200)
    plt.close()
    print(f"Saved CAR vs Matérn φ grid plot to: {fig_path_phi}")


if __name__ == "__main__":
    main()
