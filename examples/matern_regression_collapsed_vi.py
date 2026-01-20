# examples/matern_regression_collapsed_vi.py
#
# Collapsed (full) VI regression using:
#   - SpectralCAR_FullVI (collapses phi / z analytically)
#   - MaternLikeFilterFullVI (full-VI over theta = (log tau^2, rho0_raw, nu_raw))
#
# NOTE: SpectralCAR_FullVI currently uses a Normal prior on log(sigma^2) (N(0,1))
#       via kl_normal_std, NOT an Inv-Gamma prior. If you want IG(a0,b0) for sigma2,
#       we can add that later by changing the KL/expected prior term.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.filters import MaternLikeFilterFullVI
from sdmcar.models import SpectralCAR_FullVI
from sdmcar import diagnostics

torch.set_default_dtype(torch.double)

# ---- set seeds here ----
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    print("Start...")
    device = torch.device("cpu")

    # --------------------------------------------
    # 1) Grid + Laplacian + eigen-decomposition
    # --------------------------------------------
    nx, ny = 40, 40
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
    # 2) Generate CAR data (truth) for regression
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)  # [n]

    z_true = torch.sqrt(F_car) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true  # [n]

    # Design matrix: intercept + x-coordinate
    x_coord = coords[:, 0]
    X = torch.stack(
        [
            torch.ones(n, dtype=torch.double, device=device),
            x_coord,
        ],
        dim=1,
    )  # [n,2]
    p = X.shape[1]

    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    sigma2_true = 0.1

    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(
        n, dtype=torch.double, device=device
    )

    # --------------------------------------------
    # 3) Choose Matérn-like filter full VI module
    # --------------------------------------------
    matern = MaternLikeFilterFullVI(
        mu_log_tau2=math.log(tau2_true),
        log_std_log_tau2=-2.0,
        mu_rho0_raw=-7.0,
        log_std_rho0_raw=-2.5,
        mu_nu_raw=0.5,
        log_std_nu_raw=-2.0,
    ).to(device)

    # --------------------------------------------
    # 4) Build collapsed full VI regression model
    # --------------------------------------------
    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(p, dtype=torch.double, device=device)

    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=matern,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_true),
        log_std_log_sigma2=-2.3,
        num_mc=10,   # training MC
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # --------------------------------------------
    # 5) Optimize ELBO + log eval ELBO (smooth curve)
    # --------------------------------------------
    num_iters = 2500

    log_every = 20          # how often to compute eval ELBO + record traces
    num_mc_eval = 300       # high-MC ELBO for plotting

    # store per-iteration (train) and per-log step (eval) separately
    elbo_train_hist = []          # length = num_iters
    steps_eval = []               # length = num_iters/log_every
    elbo_eval_hist = []           # length = num_iters/log_every

    # record params at log points (not every iter, keeps memory smaller)
    tau2_trace, rho0_trace, nu_trace = [], [], []
    sigma2_trace = []
    beta_trace = []

    for it in range(num_iters):
        optimizer.zero_grad()
        elbo_train, stats = model.elbo()
        (-elbo_train).backward()
        optimizer.step()

        elbo_train_hist.append(elbo_train.item())

        # Logging block
        if (it + 1) % log_every == 0:
            with torch.no_grad():
                # Smooth(er) ELBO estimate for plotting only
                elbo_eval, _ = model.elbo(num_mc_override=num_mc_eval)

                tau2_m, a_m = model.filter.mean_params()
                rho0_m, nu_m = a_m.unbind(-1)
                sigma2_m = torch.exp(model.mu_log_sigma2).item()
                beta_m = model.m_beta.detach().cpu().clone()

            steps_eval.append(it + 1)
            elbo_eval_hist.append(elbo_eval.item())

            tau2_trace.append(tau2_m.item())
            rho0_trace.append(rho0_m.item())
            nu_trace.append(nu_m.item())
            sigma2_trace.append(sigma2_m)
            beta_trace.append(beta_m.numpy())

            print(
                f"[{it+1:04d}] "
                f"ELBO_train={elbo_train.item():.2f}  "
                f"ELBO_eval(M={num_mc_eval})={elbo_eval.item():.2f}  "
                f"loglik={stats['mc_loglik'].item():.2f}  "
                f"KLbeta={stats['mc_kl_beta'].item():.2f}  "
                f"KLfilt={stats['kl_filter'].item():.2f}  "
                f"KLsig={stats['kl_sigma2'].item():.2f}  "
                f"tau2={tau2_m.item():.3f} rho0={rho0_m.item():.5f} nu={nu_m.item():.3f}  "
                f"sigma2={sigma2_m:.4f}  beta={beta_m.numpy()}"
            )

    beta_trace = np.asarray(beta_trace)  # [T_log, p]
    iters_eval = np.asarray(steps_eval)

    # --------------------------------------------
    # 6) Posterior summaries
    # --------------------------------------------
    with torch.no_grad():
        beta_hat = model.m_beta.detach().cpu()
        beta_se = torch.sqrt(torch.diag(model.V_beta)).detach().cpu()
        tau2_hat, a_hat = model.filter.mean_params()
        rho0_hat, nu_hat = a_hat.unbind(-1)
        sigma2_hat = torch.exp(model.mu_log_sigma2).item()

        # Plug-in
        #mean_phi, var_phi_diag = model.posterior_phi(mode="plugin")
        # MC-integrated
        mean_phi, var_phi_diag = model.posterior_phi(mode="mc", num_mc=64)


        mean_phi = mean_phi.detach().cpu()

    rmse_phi = torch.sqrt(torch.mean((mean_phi - phi_true.cpu()) ** 2)).item()

    print("\nPosterior means (Collapsed Full VI, Matérn-like filter):")
    print(f"  beta_true    = {beta_true.cpu().numpy()}")
    print(f"  beta_hat     = {beta_hat.numpy()}")
    print(f"  beta_se      = {beta_se.numpy()}")
    print(f"  sigma2_true  = {sigma2_true:.4f}")
    print(f"  sigma2_hat   = {sigma2_hat:.4f}")
    print(f"  tau2_true    = {tau2_true:.4f}")
    print(f"  tau2_hat     = {tau2_hat.item():.4f}")
    print(f"  eps_car      = {eps_car:.5f}")
    print(f"  rho0_hat     = {rho0_hat.item():.6f}")
    print(f"  nu_target    ≈ 1.0 (CAR-like)")
    print(f"  nu_hat       = {nu_hat.item():.4f}")
    print(f"  RMSE(phi_mean, phi_true) = {rmse_phi:.4f}")

    # --------------------------------------------
    # 7) Plots
    # --------------------------------------------
    fig_dir = Path("examples") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ELBO plot: eval (smooth) + optional train overlay
    plt.figure(figsize=(6, 4))
    plt.plot(iters_eval, elbo_eval_hist, label=rf"ELBO eval (MC={num_mc_eval})")
    # optional: overlay noisy training curve lightly
    plt.plot(np.arange(1, num_iters + 1), elbo_train_hist, alpha=0.2, label="ELBO train (noisy)")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO convergence (Collapsed Full VI, Matérn-like)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "matern_regression_full_vi_elbo.png", dpi=200)
    plt.close()
    print(f"Saved ELBO plot to: {fig_dir / 'matern_regression_full_vi_elbo.png'}")

    # beta traces (logged points only)
    plt.figure(figsize=(6, 4))
    for j in range(p):
        (line,) = plt.plot(iters_eval, beta_trace[:, j], label=rf"$\beta_{j}$ (mean)")
        color = line.get_color()
        plt.axhline(beta_true[j].item(), linestyle="--", color=color, label=rf"$\beta_{j}$ true")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\beta$")
    plt.title(r"Posterior mean $\beta$ traces (Collapsed Full VI)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "matern_regression_full_vi_beta_traces.png", dpi=200)
    plt.close()
    print(f"Saved β traces to: {fig_dir / 'matern_regression_full_vi_beta_traces.png'}")

    # theta traces (logged points only)
    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(iters_eval, tau2_trace)
    ax1.axhline(tau2_true, linestyle="--", label=r"$\tau^2_{\mathrm{true}}$")
    ax1.set_ylabel(r"$\tau^2$")
    ax1.legend(loc="best", fontsize=8)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(iters_eval, rho0_trace)
    ax2.axhline(eps_car, linestyle="--", label=r"CAR $\epsilon$")
    ax2.set_ylabel(r"$\rho_0$")
    ax2.legend(loc="best", fontsize=8)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(iters_eval, nu_trace)
    ax3.axhline(1.0, linestyle="--", label=r"target $\nu \approx 1$")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(r"$\nu$")
    ax3.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "matern_regression_full_vi_theta_traces.png", dpi=200)
    plt.close()
    print(f"Saved θ traces to: {fig_dir / 'matern_regression_full_vi_theta_traces.png'}")

    # spectral filter comparison: CAR vs fitted Matérn (VI mean)
    with torch.no_grad():
        F_matern_hat = tau2_hat * (lam + rho0_hat).pow(-nu_hat)

    mask = lam > 1e-6
    lam_plot = lam[mask].cpu()
    F_car_plot = F_car[mask].cpu()
    F_matern_plot = F_matern_hat[mask].cpu()

    plt.figure(figsize=(6, 4))
    plt.loglog(lam_plot, F_car_plot, "o", markersize=3, label=r"True CAR: $\tau^2/(\lambda+\epsilon)$")
    plt.loglog(lam_plot, F_matern_plot, "--", label=r"Fitted Matérn-like (q-mean)")
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel(r"Spectral variance $F(\lambda)$")
    plt.title("CAR vs Matérn-like spectral filter (Collapsed Full VI)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "matern_regression_full_vi_filter.png", dpi=200)
    plt.close()
    print(f"Saved filter comparison to: {fig_dir / 'matern_regression_full_vi_filter.png'}")

    # phi_true vs posterior mean phi
    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=mean_phi.to(device),
        phi_true=phi_true,
        save_path_prefix=str(fig_dir / "matern_regression_full_vi_phi"),
    )
    print("Saved φ plots (true vs posterior mean) with prefix:",
          fig_dir / "matern_regression_full_vi_phi")

    # beta intervals
    diagnostics.plot_beta_intervals(model, save_path=str(fig_dir / "matern_regression_full_vi_beta_intervals.png"))
    print(f"Saved β intervals to: {fig_dir / 'matern_regression_full_vi_beta_intervals.png'}")


if __name__ == "__main__":
    main()
