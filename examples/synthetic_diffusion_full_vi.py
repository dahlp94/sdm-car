# examples/synthetic_diffusion_full_vi.py

import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path

print("Starting...")

from sdmcar import (
    set_seed,
    build_laplacian_from_knn,
    laplacian_eigendecomp,
    DiffusionFilterFullVI,
    SpectralCAR_FullVI,
    diagnostics,
)
from sdmcar.utils import set_default_dtype

DEVICE = torch.device("cpu")  # change to "cuda" if desired and available
GEN_SEED = 123
PLOT = True

def run_synthetic_full_vi():
    set_default_dtype(torch.double)
    set_seed(GEN_SEED)

    # ---- Build a simple grid ----
    n_side = 20
    xs, ys = torch.meshgrid(
        torch.arange(n_side, dtype=torch.double),
        torch.arange(n_side, dtype=torch.double),
        indexing="ij",
    )
    coords = torch.stack([xs.flatten(), ys.flatten()], dim=1) / n_side  # [n, 2]
    coords = coords.to(DEVICE)
    n = coords.size(0)

    # ---- Graph & spectrum ----
    L, W = build_laplacian_from_knn(coords, k=6, gamma=0.25, rho=0.95)
    lam, U = laplacian_eigendecomp(L)

    # ---- Design matrix X and true β ----
    p = 2
    X = torch.cat(
        [
            torch.ones(n, 1, dtype=torch.double, device=DEVICE),
            torch.randn(n, p - 1, dtype=torch.double, device=DEVICE),
        ],
        dim=1,
    )
    beta_true = torch.tensor(
        [1.0] + [2.0] * (p - 1), dtype=torch.double, device=DEVICE
    )

    # ---- True diffusion filter + φ + noise ----
    tau2_true, a_true = 1.0, 0.7
    F_true = tau2_true * torch.exp(-a_true * lam)
    with torch.no_grad():
        K_half = U @ torch.diag(torch.sqrt(F_true))
        phi_true = K_half @ torch.randn(n, dtype=torch.double, device=DEVICE)
        sigma2_true = 0.10
        y = (
            X @ beta_true
            + phi_true
            + math.sqrt(sigma2_true)
            * torch.randn(n, dtype=torch.double, device=DEVICE)
        )

    # ---- Full-VI model ----
    filt_vi = DiffusionFilterFullVI(
        mu_log_tau2=0.0,
        log_std_log_tau2=-2.3,
        mu_a_raw=0.4,
        log_std_a_raw=-2.3,
    )
    model = SpectralCAR_FullVI(
        X,
        y,
        lam,
        U,
        filter_module=filt_vi,
        mu_log_sigma2=-2.3,
        log_std_log_sigma2=-2.3,
        num_mc=5,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    history = {
        "step": [],
        "elbo": [],
        "loglik": [],
        "kl_beta": [],
        "kl_filt": [],
        "kl_sig": [],
        "sigma2": [],
    }

    # ---- Train ----
    for t in range(800):
        opt.zero_grad()
        elbo, parts = model.elbo()
        (-elbo).backward()
        opt.step()

        if (t + 1) % 20 == 0:
            history["step"].append(t + 1)
            history["elbo"].append(elbo.item())
            history["loglik"].append(parts["mc_loglik"].item())
            history["kl_beta"].append(parts["mc_kl_beta"].item())
            history["kl_filt"].append(parts["kl_filter"].item())
            history["kl_sig"].append(parts["kl_sigma2"].item())
            history["sigma2"].append(parts["sigma2"].item())

            print(
                f"[{t+1:4d}] ELBO={elbo.item():.3f}  "
                f"loglik={parts['mc_loglik'].item():.3f}  "
                f"KLβ={parts['mc_kl_beta'].item():.3f}  "
                f"KL_filt={parts['kl_filter'].item():.3f}  "
                f"KL_sig={parts['kl_sigma2'].item():.3f}  "
                f"σ²~{parts['sigma2'].item():.4f}"
            )

    # ---- Report parameter recovery ----
    with torch.no_grad():
        beta_hat = model.m_beta
        beta_se = torch.sqrt(torch.diag(model.V_beta))
        z = 1.96
        print("\nβ recovery (posterior mean ± 1.96 sd)")
        for j, (b, se) in enumerate(zip(beta_hat, beta_se)):
            lo, hi = b - z * se, b + z * se
            print(f"beta[{j}]: {b:.3f}  [{lo:.3f}, {hi:.3f}]")
        print("true β:", beta_true.tolist())

        tau2_mean, a_mean = model.filter.mean_params()
        sigma2_mean = torch.exp(model.mu_log_sigma2)
        print(
            f"\nMeans of q(hyper): τ²≈{tau2_mean.item():.3f}, "
            f"a≈{a_mean.item():.3f}, σ²≈{sigma2_mean.item():.3f}"
        )
        print(
            f"True hyperparams:   τ²={tau2_true:.3f}, "
            f"a={a_true:.3f}, σ²={sigma2_true:.3f}"
        )

        mean_phi, var_phi_diag = model.posterior_phi(use_q_means=True)

    results = {
        "X": X,
        "y": y,
        "lam": lam,
        "U": U,
        "coords": coords,
        "F_true": F_true,
        "phi_true": phi_true,
        "model": model,
        "history": history,
        "mean_phi": mean_phi,
        "var_phi_diag": var_phi_diag,
    }
    return results

if __name__ == "__main__":
    results = run_synthetic_full_vi()

    if PLOT:
        history = results["history"]
        lam = results["lam"]
        U = results["U"]
        model = results["model"]
        F_true = results["F_true"]
        coords = results["coords"]
        mean_phi = results["mean_phi"]
        phi_true = results["phi_true"]
        X = results["X"]
        y = results["y"]

        # Create figures directory
        fig_dir = Path("examples") / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        diagnostics.plot_elbo(history, save_path=str(fig_dir / "elbo.png"))
        diagnostics.plot_filter_recovery(lam, F_true, model,
                                        save_path=str(fig_dir / "filter_recovery.png"))
        diagnostics.plot_phi_mean_vs_true(coords, mean_phi, phi_true,
                                        save_path_prefix=str(fig_dir / "phi"))
        diagnostics.plot_residual_spectrum(lam, U, X, y, model,
                                        save_path=str(fig_dir / "residual_spectrum.png"))
        diagnostics.plot_beta_intervals(model,
                                        save_path=str(fig_dir / "beta_intervals.png"))
