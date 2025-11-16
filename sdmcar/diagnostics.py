# sdmcar/diagnostics.py

import math
import torch
import matplotlib.pyplot as plt

def plot_elbo(history, save_path: str | None = None):
    plt.figure()
    plt.plot(history["step"], history["elbo"])
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO (Full VI)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_filter_recovery(lam, F_true, model, num_samples: int = 30, save_path: str | None = None):
    with torch.no_grad():
        tau2_mean, a_mean = model.filter.mean_params()
        F_mean = model.filter.F(lam, tau2_mean, a_mean)

        F_samples = []
        for _ in range(num_samples):
            tau2_s, a_s, _, _ = model.filter.sample_params()
            F_samples.append(model.filter.F(lam, tau2_s, a_s).cpu())
        F_samples = torch.stack(F_samples, dim=0)  # [S, n]

    idx = torch.argsort(lam)
    lam_sorted = lam[idx].cpu().numpy()
    F_true_sorted = F_true[idx].cpu().numpy()
    F_mean_sorted = F_mean[idx].cpu().numpy()
    F_samp_sorted = F_samples[:, idx].numpy()

    plt.figure()
    for s in range(F_samp_sorted.shape[0]):
        plt.plot(lam_sorted, F_samp_sorted[s], alpha=0.2)
    plt.plot(lam_sorted, F_true_sorted, label="True F(λ)")
    plt.plot(lam_sorted, F_mean_sorted, label="Mean q[F(λ)]")
    plt.xlabel("λ")
    plt.ylabel("F(λ)")
    plt.title("Spectral filter: true vs. full-VI")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_phi_mean_vs_true(coords, mean_phi, phi_true, save_path_prefix: str | None = None):
    n_side = int(math.sqrt(coords.size(0)))
    phi_post = mean_phi.reshape(n_side, n_side).cpu().numpy()
    phi_true_grid = phi_true.reshape(n_side, n_side).cpu().numpy()

    plt.figure()
    plt.imshow(phi_post, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title("Posterior mean φ (Full VI)")
    plt.tight_layout()
    if save_path_prefix is not None:
        plt.savefig(save_path_prefix + "_phi_post.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(phi_true_grid, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title("True φ")
    plt.tight_layout()
    if save_path_prefix is not None:
        plt.savefig(save_path_prefix + "_phi_true.png", bbox_inches="tight")
    plt.close()

def plot_residual_spectrum(lam, U, X, y, model, save_path: str | None = None):
    with torch.no_grad():
        tau2_m, a_m = model.filter.mean_params()
        F_lam = model.filter.F(lam, tau2_m, a_m)
        sigma2_m = torch.exp(model.mu_log_sigma2)
        var = F_lam + sigma2_m
        inv_var = 1.0 / var

        m_beta, V_beta = model._beta_update(inv_var)
        g = (U.T @ y) - (U.T @ X) @ m_beta
        res_power = (g**2).cpu().numpy()

    plt.figure()
    plt.scatter(lam.cpu().numpy(), res_power, s=8)
    plt.xlabel("λ")
    plt.ylabel("g_i^2")
    plt.title("Residual spectrum (Full VI, q-means)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_beta_intervals(model, save_path: str | None = None):
    with torch.no_grad():
        beta_hat = model.m_beta
        beta_se  = torch.sqrt(torch.diag(model.V_beta))
        z = 1.96
        idx = list(range(beta_hat.numel()))
        centers = beta_hat.cpu().numpy()
        errs = (z * beta_se).cpu().numpy()

    plt.figure()
    plt.errorbar(idx, centers, yerr=errs, fmt='o')
    plt.xlabel("β index")
    plt.ylabel("Value")
    plt.title("Posterior β (95% intervals)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
