# examples/run_benchmark.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import math
import random
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.models import SpectralCAR_FullVI
from sdmcar.mcmc import MCMCConfig, make_collapsed_mcmc_from_model
from sdmcar import diagnostics
from sdmcar.diagnostics import spectrum_error_log_l1, weighted_relative_l2_spectrum_error

# IMPORTANT: importing benchmarks triggers registrations
from examples.benchmarks.registry import get_filter_spec, available_filters
import examples.benchmarks  # noqa: F401  (ensures FILTER_REGISTRY is populated)


torch.set_default_dtype(torch.double)


# -------------------------
# plotting helpers
# -------------------------
def summarize_chain(x: np.ndarray):
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    lo, hi = np.quantile(x, [0.025, 0.975])
    return mean, sd, float(lo), float(hi)


def plot_trace(x: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(7, 2.5))
    plt.plot(x, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Saved draw")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_hist_with_lines(
    x: np.ndarray,
    title: str,
    save_path: Path,
    true_val: float | None = None,
    vi_val: float | None = None,
    vi_label: str = "VI",
    show_mcmc_mean: bool = True,   # NEW
):
    plt.figure(figsize=(5, 3))
    plt.hist(x, bins=40, density=True, alpha=0.85)

    # MCMC mean
    if show_mcmc_mean:
        mcmc_mean = float(np.mean(x))
        plt.axvline(
            mcmc_mean,
            linestyle=":",
            linewidth=2,
            color="black",
            label="MCMC mean",
        )

    # True value
    if true_val is not None:
        plt.axvline(
            true_val,
            linestyle="--",
            linewidth=2,
            label="true",
        )

    # VI mean
    if vi_val is not None:
        plt.axvline(
            vi_val,
            linestyle="-",
            linewidth=2,
            label=vi_label,
        )

    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_true_vs_empirical(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    empirical_energy: torch.Tensor,
    outpath: Path,
    title: str,
    logy: bool = False,
    use_markers: bool = True,
    xlabel=r"$\lambda$",
) -> None:
    lam_np = lam.detach().cpu().numpy()
    F_np = F_true.detach().cpu().numpy()
    e_np = empirical_energy.detach().cpu().numpy()

    if logy:
        F_np = np.clip(F_np, 1e-12, None)
        e_np = np.clip(e_np, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if use_markers:
        ax.plot(
            lam_np, F_np,
            marker="o", linestyle="none",
            label="True $F_0(x)$", markersize=4, alpha=0.9,
        )
        ax.plot(
            lam_np, e_np,
            marker="x", linestyle="none",
            label="Empirical energy $z^2$", markersize=4, alpha=0.7,
        )
    else:
        ax.plot(lam_np, F_np, label="True $F_0(x)$", linewidth=2.5)
        ax.plot(lam_np, e_np, label="Empirical energy $z^2$", linewidth=1.8)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spectral value")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_true_vs_learned_spectra(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
    title: str,
    logy: bool = False,
    use_markers: bool = True,
    xlabel: str = r"$\lambda$",
) -> None:
    lam_np = lam.detach().cpu().numpy()
    true_np = F_true.detach().cpu().numpy()

    if logy:
        true_np = np.clip(true_np, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if use_markers:
        ax.plot(
            lam_np, true_np,
            marker="o", linestyle="none",
            label="True $F_0(x)$", markersize=4,
        )
    else:
        ax.plot(lam_np, true_np, label="True $F_0(x)$", linewidth=2.5)

    for label, spec in learned.items():
        spec_np = spec.detach().cpu().numpy()
        if logy:
            spec_np = np.clip(spec_np, 1e-12, None)

        if use_markers:
            ax.plot(
                lam_np, spec_np,
                marker="x", linestyle="none",
                label=label, markersize=4, alpha=0.8,
            )
        else:
            ax.plot(lam_np, spec_np, label=label, linewidth=2.0)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spectral value")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_true_vs_learned_curves(
    *,
    lam: torch.Tensor,
    true_curve: torch.Tensor,
    learned: dict[str, torch.Tensor],
    outpath: Path,
    title: str,
    ylabel: str,
    logy: bool = False,
    use_markers: bool = True,
    xlabel: str = r"$\lambda$",
) -> None:
    lam_np = lam.detach().cpu().numpy()
    true_np = true_curve.detach().cpu().numpy()

    if logy:
        true_np = np.clip(true_np, 1e-12, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if use_markers:
        ax.plot(
            lam_np,
            true_np,
            marker="o",
            linestyle="none",
            label="Truth",
            markersize=4,
        )
    else:
        ax.plot(lam_np, true_np, label="Truth", linewidth=2.5)

    for label, curve in learned.items():
        curve_np = curve.detach().cpu().numpy()
        if logy:
            curve_np = np.clip(curve_np, 1e-12, None)

        if use_markers:
            ax.plot(
                lam_np,
                curve_np,
                marker="x",
                linestyle="none",
                label=label,
                markersize=4,
                alpha=0.8,
            )
        else:
            ax.plot(lam_np, curve_np, label=label, linewidth=2.0)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_spectrum_with_bands(
    *,
    lam: torch.Tensor,
    F_true: torch.Tensor,
    vi_mean: torch.Tensor,
    vi_q025: torch.Tensor,
    vi_q975: torch.Tensor,
    mcmc_mean: torch.Tensor | None,
    mcmc_q025: torch.Tensor | None,
    mcmc_q975: torch.Tensor | None,
    outpath: Path,
    title: str,
    logx: bool = False,
    logy: bool = True,
    xlabel: str = r"$\lambda$",
) -> None:
    x = lam.detach().cpu().numpy()
    true_np = F_true.detach().cpu().numpy()
    vi_mean_np = vi_mean.detach().cpu().numpy()
    vi_lo_np = vi_q025.detach().cpu().numpy()
    vi_hi_np = vi_q975.detach().cpu().numpy()

    eps = 1e-12
    if logy:
        true_np = np.clip(true_np, eps, None)
        vi_mean_np = np.clip(vi_mean_np, eps, None)
        vi_lo_np = np.clip(vi_lo_np, eps, None)
        vi_hi_np = np.clip(vi_hi_np, eps, None)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(x, true_np, linewidth=2.5, label="True spectrum")
    ax.plot(x, vi_mean_np, linewidth=2.0, label="VI mean")
    ax.fill_between(x, vi_lo_np, vi_hi_np, alpha=0.25, label="VI 95% band")

    if mcmc_mean is not None and mcmc_q025 is not None and mcmc_q975 is not None:
        m_mean_np = mcmc_mean.detach().cpu().numpy()
        m_lo_np = mcmc_q025.detach().cpu().numpy()
        m_hi_np = mcmc_q975.detach().cpu().numpy()

        if logy:
            m_mean_np = np.clip(m_mean_np, eps, None)
            m_lo_np = np.clip(m_lo_np, eps, None)
            m_hi_np = np.clip(m_hi_np, eps, None)

        ax.plot(x, m_mean_np, linewidth=2.0, linestyle="--", label="MCMC mean")
        ax.fill_between(x, m_lo_np, m_hi_np, alpha=0.18, label="MCMC 95% band")

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$F(\lambda)$")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def spectral_plot_axis(
    *,
    lam: torch.Tensor,
    eps_car: float,
    filter_name: str,
) -> tuple[torch.Tensor, str, str]:
    """
    Returns:
        x_axis: tensor used for plotting
        xlabel: axis label
        axis_tag: short name for filenames
    """
    if filter_name == "multiscale_bump":
        return (
            torch.log(lam + eps_car),
            r"$\log(\lambda+\epsilon)$",
            "logfreq",
        )

    if filter_name in {"poly", "rational"}:
        x = (lam / lam.max().clamp_min(1e-12)).clamp(0.0, 1.0)
        return (
            x,
            r"$x=\lambda/\lambda_{\max}$",
            "normalized_x",
        )

    return (
        lam,
        r"$\lambda$",
        "lambda",
    )

def resolve_fixed_tokens(fixed: dict, eps_car: float) -> dict:
    """
    Allow case specs to include "eps_car" as a placeholder token.
    """
    out = {}
    for k, v in fixed.items():
        if v == "eps_car":
            out[k] = eps_car
        else:
            out[k] = v
    return out


def unpack_filter_params_from_means(filter_module):
    """
    Generic: uses mean_unconstrained() + _constrain() if available.
    Falls back to legacy mean_params() if needed.
    Returns scalars where possible.
    """
    # New-style API
    if hasattr(filter_module, "mean_unconstrained") and hasattr(filter_module, "_constrain"):
        theta_mean = filter_module.mean_unconstrained()
        c = filter_module._constrain(theta_mean)

        out = {
            "tau2": c.get("tau2", None),
            "rho0": c.get("rho0", None),
            "nu":   c.get("nu", None),
        }

        # Optional extras you may want later
        # poly/rational:
        if "a" in c:
            out["a"] = c["a"]
        if "b" in c:
            out["b"] = c["b"]

        # reshape scalars
        for k in ("tau2", "rho0", "nu"):
            if out.get(k, None) is not None:
                out[k] = out[k].reshape(())

        return out

    # Legacy API fallback
    if hasattr(filter_module, "mean_params"):
        tau2_m, a_m = filter_module.mean_params()
        a_flat = a_m.reshape(-1)

        rho0_m = None
        nu_m = None
        if a_flat.numel() == 2:
            rho0_m = a_flat[0]
            nu_m = a_flat[1]
        elif a_flat.numel() == 1:
            rho0_m = a_flat[0]

        return {
            "tau2": tau2_m.reshape(()),
            "rho0": None if rho0_m is None else rho0_m.reshape(()),
            "nu": None if nu_m is None else nu_m.reshape(()),
        }

    raise AttributeError(
        f"{type(filter_module).__name__} must implement either "
        f"(mean_unconstrained + _constrain) or mean_params()."
    )


def decode_theta_chain(out, filter_module):
    """
    Returns:
      raw: dict[name -> np.ndarray shape [S]]
      constrained: dict[key -> np.ndarray shape [S] or [S,d]]
    """
    theta_mat = out["theta"].detach().cpu().numpy()   # [S, d_theta]
    names = out["theta_names"]

    raw = {names[j]: theta_mat[:, j].copy() for j in range(len(names))}

    # device/dtype from filter
    try:
        p0 = next(filter_module.parameters())
        device, dtype = p0.device, p0.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.double

    constrained = {}
    S = theta_mat.shape[0]

    for i in range(S):
        theta_dict = {
            names[j]: torch.tensor([theta_mat[i, j]], dtype=dtype, device=device)
            for j in range(len(names))
        }
        c = filter_module._constrain(theta_dict)

        for k, v in c.items():
            v_np = v.detach().cpu().numpy()
            v_np = np.asarray(v_np)

            # scalar
            if v_np.size == 1:
                if k not in constrained:
                    constrained[k] = np.zeros(S, dtype=float)
                constrained[k][i] = float(v_np.reshape(-1)[0])

            # vector / matrix -> store as [S, ...]
            else:
                v_shape = v_np.shape
                if k not in constrained:
                    constrained[k] = np.zeros((S,) + v_shape, dtype=float)
                constrained[k][i, ...] = v_np

    return raw, constrained


def tensor_to_scalar(x):
    """
    Convert a torch tensor / numpy scalar / python scalar to float.
    """
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


# -------------------------
# Core: run a single case
# -------------------------
def run_case(
    *,
    case_spec,
    filter_name: str,
    F_true: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    coords: torch.Tensor,
    phi_true: torch.Tensor,
    beta_true: torch.Tensor,
    sigma2_true: float,
    tau2_true: float,
    eps_car: float,
    prior_V0: torch.Tensor,
    device: torch.device,
    fig_dir: Path,
    vi_num_iters: int = 2500,
    vi_num_mc: int = 10,
    vi_lr: float = 1e-2,
    mcmc_num_steps: int = 30000,
    mcmc_burnin: int = 10000,
    mcmc_thin: int = 10,
):
    fixed = resolve_fixed_tokens(case_spec.fixed, eps_car=eps_car)

    case_dir = fig_dir / filter_name / case_spec.display_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"FILTER: {filter_name} | CASE: {case_spec.display_name}")
    print(f"fixed: {fixed}")
    print("=" * 80)

    # -------------------------
    # Build filter (VI)
    # -------------------------
    filter_module = case_spec.build_filter(
        tau2_true=tau2_true,
        eps_car=eps_car,
        device=device,
        lam_max=float(lam.max().item()),
        **fixed,
    )

    # -------------------------
    # Build VI model
    # -------------------------
    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_true),
        log_std_log_sigma2=-2.3,
        num_mc=vi_num_mc,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vi_lr)

    # -------------------------
    # Train VI
    # -------------------------
    elbo_hist = []
    log_every = 50

    for it in range(vi_num_iters):
        optimizer.zero_grad()
        elbo, stats = model.elbo()
        (-elbo).backward()
        optimizer.step()
        elbo_hist.append(elbo.item())

        if (it + 1) % log_every == 0:
            with torch.no_grad():
                theta_plugin_u, theta_plugin_c, sigma2_plugin_train = model.plugin_hyperparams()

                tau2_m = theta_plugin_c.get("tau2", None)
                rho0_m = theta_plugin_c.get("rho0", None)
                nu_m = theta_plugin_c.get("nu", None)

                mu_s = model.mu_log_sigma2.detach()
                std_s = torch.exp(model.log_std_log_sigma2.detach())

                sigma2_median = torch.exp(mu_s).item()
                sigma2_mean = torch.exp(mu_s + 0.5 * std_s**2).item()
                beta_m = model.m_beta.detach().cpu().numpy()

            nu_str = "NA" if nu_m is None else f"{nu_m.item():.3f}"
            rho0_str = "NA" if rho0_m is None else f"{rho0_m.item():.6f}"

            print(
                f"[VI {it+1:04d}] ELBO={elbo.item():.2f} "
                f"loglik={stats['mc_loglik'].item():.2f} "
                f"KLbeta={stats['mc_kl_beta'].item():.2f} "
                f"KLfilt={stats['kl_filter'].item():.2f} "
                f"KLsig={stats['kl_sigma2'].item():.2f} "
                f"tau2={tau2_m.item():.3f} rho0={rho0_str} nu={nu_str} "
                f"sigma2_med={sigma2_median:.4f} sigma2_mean={sigma2_mean:.4f} beta={beta_m}"
            )

    # save VI ELBO
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, vi_num_iters + 1), elbo_hist)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (training MC estimate)")
    plt.title(f"VI ELBO (train) — {filter_name}/{case_spec.display_name}")
    plt.tight_layout()
    plt.savefig(case_dir / "vi_elbo_train.png", dpi=200)
    plt.close()

    # -------------------------
    # VI summaries (plugin + MC)
    # -------------------------
    with torch.no_grad():
        beta_vi = model.beta_posterior_vi(num_mc=128, return_draws=False)
        sigma2_vi = model.sigma2_posterior_vi(num_mc=128, return_draws=False)
        theta_vi = model.theta_posterior_vi(num_mc=128, return_draws=False)
        spectrum_vi = model.spectrum_posterior_vi(num_mc=128, return_draws=False)

        # beta
        beta_vi_plugin = beta_vi["plugin"]["mean"].detach().cpu()
        beta_vi_plugin_sd = beta_vi["plugin"]["sd"].detach().cpu()
        beta_vi_plugin_q025 = beta_vi["plugin"]["q025"].detach().cpu()
        beta_vi_plugin_q975 = beta_vi["plugin"]["q975"].detach().cpu()

        beta_vi_mc = beta_vi["mc"]["mean"].detach().cpu()
        beta_vi_mc_sd = beta_vi["mc"]["sd"].detach().cpu()
        beta_vi_mc_q025 = beta_vi["mc"]["q025"].detach().cpu()
        beta_vi_mc_q975 = beta_vi["mc"]["q975"].detach().cpu()

        # sigma2
        sigma2_vi_plugin = tensor_to_scalar(sigma2_vi["plugin"])
        sigma2_vi_mc_mean = tensor_to_scalar(sigma2_vi["mc"]["mean"])
        sigma2_vi_mc_sd = tensor_to_scalar(sigma2_vi["mc"]["sd"])
        sigma2_vi_mc_q025 = tensor_to_scalar(sigma2_vi["mc"]["q025"])
        sigma2_vi_mc_q975 = tensor_to_scalar(sigma2_vi["mc"]["q975"])

        # theta plugin summaries (constrained scale)
        tau2_vi = theta_vi["plugin"].get("tau2", None)
        rho0_vi = theta_vi["plugin"].get("rho0", None)
        nu_vi = theta_vi["plugin"].get("nu", None)

        # phi
        mean_phi_vi_plugin, var_phi_vi_plugin = model.posterior_phi(mode="plugin")
        mean_phi_vi, var_phi_vi = model.posterior_phi(mode="mc", num_mc=128)

        mean_phi_vi_plugin = mean_phi_vi_plugin.detach().cpu()
        var_phi_vi_plugin = var_phi_vi_plugin.detach().cpu()
        mean_phi_vi = mean_phi_vi.detach().cpu()
        var_phi_vi = var_phi_vi.detach().cpu()

        # spectrum
        F_plugin = spectrum_vi["plugin"].detach()
        F_vi_mc_mean = spectrum_vi["mc"]["mean"].detach()
        F_vi_mc_sd = spectrum_vi["mc"]["sd"].detach()
        F_vi_mc_q025 = spectrum_vi["mc"]["q025"].detach()
        F_vi_mc_q975 = spectrum_vi["mc"]["q975"].detach()

    rmse_phi_vi_plugin = float(torch.sqrt(torch.mean((mean_phi_vi_plugin - phi_true.cpu()) ** 2)).item())
    rmse_phi_vi = float(torch.sqrt(torch.mean((mean_phi_vi - phi_true.cpu()) ** 2)).item())


    # -------------------------
    # Run MCMC initialized from VI means
    # -------------------------
    cfg = MCMCConfig(
        num_steps=mcmc_num_steps,
        burnin=mcmc_burnin,
        thin=mcmc_thin,
        step_s=float(case_spec.step_s),
        step_theta=case_spec.get_step_theta(model.filter),
        seed=0,
        device=device,
        print_every=5000,
    )

    sampler = make_collapsed_mcmc_from_model(model, config=cfg)

    init_s = model.mu_log_sigma2.detach()

    theta0 = model.filter.mean_unconstrained()              # dict
    init_theta_vec = model.filter.pack(theta0).detach()     # packed vector

    print("\nRunning MCMC...")
    out = sampler.run(
        init_s=init_s,
        init_theta_vec=init_theta_vec,
        init_from_conditional_beta=True,
        store_phi_mean=True,
        U=U,
        X=X,
        y=y,
    )


    acc = out["acc"]
    print("Acceptance rates:")
    print("  s:", acc["s"])
    print("  theta blocks:")
    for k, v in acc["theta"].items():
        print(f"    {k}: {v}")

    
    # --------------------------------------------
    # Predictive metrics (VI plugin + VI MC + MCMC)
    # --------------------------------------------
    pred = diagnostics.predictive_report(
        model=model,
        out=out,
        num_mc_vi=128,       # VI predictive MC draws
        max_draws_mcmc=1000, # cap LPD draws for speed
    )

    print("\nPredictive metrics:")
    print(
        f"  RMSE_y: VI(plugin)={pred['rmse_y_vi_plugin']:.4f} | "
        f"VI(mc)={pred['rmse_y_vi_mc']:.4f} | "
        f"MCMC={pred['rmse_y_mcmc']:.4f}"
    )
    print(
        f"  LPD/obs: VI(plugin)={pred['lpd_vi_plugin']:.4f} | "
        f"VI(mc)={pred['lpd_vi_mc']:.4f} | "
        f"MCMC={pred['lpd_mcmc']:.4f}"
    )


    # chains extraction
    beta_chain = out["beta"].detach().cpu().numpy()
    s_chain = out["s"].detach().cpu().numpy().reshape(-1)
    sigma2_chain = np.exp(s_chain)

    theta_raw, theta_constr = decode_theta_chain(out, model.filter)

    # -------------------------
    # MCMC spectrum posterior bands
    # -------------------------
    with torch.no_grad():
        theta_mcmc = out["theta"].detach().cpu().numpy()
        theta_names = out["theta_names"]

        F_mcmc_draws = []
        for i in range(theta_mcmc.shape[0]):
            theta_i = {
                theta_names[j]: torch.tensor(
                    [theta_mcmc[i, j]],
                    dtype=lam.dtype,
                    device=lam.device,
                )
                for j in range(len(theta_names))
            }
            F_i = model.filter.spectrum(lam, theta_i).clamp_min(1e-12)
            F_mcmc_draws.append(F_i)

        F_mcmc_draws = torch.stack(F_mcmc_draws, dim=0)  # [S, n]
        F_mcmc_mean = F_mcmc_draws.mean(dim=0)
        F_mcmc_q025 = torch.quantile(F_mcmc_draws, 0.025, dim=0)
        F_mcmc_q975 = torch.quantile(F_mcmc_draws, 0.975, dim=0)

    # integrating ridge diagnostics
    from sdmcar.diagnostics import ridge_report, plot_corr_heatmap, spectrum_draw_sd

    theta_draws = out["theta"].detach().cpu().numpy()
    theta_names = out["theta_names"]

    # filter-aware highlights, but safe if missing
    highlight_pairs = [
        ("rho0_raw", "nu_raw"),
        ("log_tau2", "rho0_raw"),
        ("log_tau2", "nu_raw"),
        ("rho_raw", "log_tau2"),   # Leroux case
    ]

    rep = ridge_report(theta_draws, theta_names, topk=10, highlight_pairs=highlight_pairs)

    print("\nPosterior correlation ridge summary:")
    print(f"  max |corr| = {rep['max_abs_corr']:.3f}")
    print(f"  mean |corr| = {rep['mean_abs_corr']:.3f}")
    for a, b, corr, abs_corr in rep["top_pairs"]:
        print(f"  {a:>14s} vs {b:<14s} corr={corr:+.3f} |corr|={abs_corr:.3f}")

    if rep["highlights"]:
        print("  highlighted:")
        for a, b, corr in rep["highlights"]:
            print(f"    {a} vs {b}: corr={corr:+.3f}")

    plot_corr_heatmap(
        rep["R"], rep["names"],
        title=f"Posterior corr — {filter_name}/{case_spec.display_name}",
        save_path=str(case_dir / "posterior_corr_heatmap.png"),
        max_vars=18,
    )


    # these exist for most of your current filters
    tau2_chain = theta_constr.get("tau2", None)  # np.ndarray [S] or None
    rho0_chain = theta_constr.get("rho0", None)
    nu_chain   = theta_constr.get("nu", None)
    a_chain = theta_constr.get("a", None)        # [S, degree+1]



    phi_mean_chain = out["phi_mean"].detach().cpu().numpy()  # [S,n]
    phi_mean_mcmc = np.mean(phi_mean_chain, axis=0)
    rmse_phi_mcmc = float(np.sqrt(np.mean((phi_mean_mcmc - phi_true.cpu().numpy()) ** 2)))
    print(f"MCMC RMSE(phi_mean, phi_true) = {rmse_phi_mcmc:.4f}")

    # -------------------------
    # Latent signal (eta) RMSE
    # -------------------------

    # true eta
    eta_true = (X @ beta_true).detach().cpu() + phi_true.detach().cpu()

    # VI plugin
    eta_vi_plugin = (X @ beta_vi_plugin).detach().cpu() + mean_phi_vi_plugin
    rmse_eta_vi_plugin = float(
        torch.sqrt(torch.mean((eta_vi_plugin - eta_true) ** 2)).item()
    )

    # VI MC
    eta_vi_mc = (X @ beta_vi_mc).detach().cpu() + mean_phi_vi
    rmse_eta_vi = float(
        torch.sqrt(torch.mean((eta_vi_mc - eta_true) ** 2)).item()
    )

    # MCMC
    beta_mean_mcmc = np.mean(beta_chain, axis=0)

    eta_mcmc = (
        X.detach().cpu().numpy() @ beta_mean_mcmc
        + phi_mean_mcmc
    )

    rmse_eta_mcmc = float(
        np.sqrt(np.mean((eta_mcmc - eta_true.numpy()) ** 2))
    )

    print(
        f"RMSE(eta): VI(plugin)={rmse_eta_vi_plugin:.4f} | "
        f"VI(mc)={rmse_eta_vi:.4f} | "
        f"MCMC={rmse_eta_mcmc:.4f}"
    )

    mean_sd_logF, sd_curve = spectrum_draw_sd(
        lam,
        model.filter,
        out["theta"],
        out["theta_names"],
    )

    print(f"Functional ridge (mean SD log F) = {mean_sd_logF:.4f}")


    # family-specific param transforms
    named = {}
    if case_spec.transform_chain is not None:
        named = case_spec.transform_chain(out, fixed_resolved=fixed, eps_car=eps_car)
    

    # lam_flat = lam.reshape(-1)
    # _, idx = torch.sort(lam_flat)

    F_true_local = F_true.detach().to(device=lam.device, dtype=lam.dtype).reshape(-1)

    plot_x, plot_xlabel, plot_axis_tag = spectral_plot_axis(
        lam=lam,
        eps_car=eps_car,
        filter_name=filter_name,
    )

    plot_spectrum_with_bands(
        lam=plot_x,
        F_true=F_true_local,
        vi_mean=F_vi_mc_mean.to(device),
        vi_q025=F_vi_mc_q025.to(device),
        vi_q975=F_vi_mc_q975.to(device),
        mcmc_mean=F_mcmc_mean.to(device),
        mcmc_q025=F_mcmc_q025.to(device),
        mcmc_q975=F_mcmc_q975.to(device),
        outpath=case_dir / f"spectrum_posterior_bands_{plot_axis_tag}_logy.png",
        title=f"Posterior spectrum bands — {filter_name}/{case_spec.display_name}",
        logy=True,
        xlabel=plot_xlabel,
    )

    z_true = U.T @ phi_true
    empirical_energy = z_true.pow(2).detach()

    true_total_var = F_true_local + sigma2_true
    vi_plugin_total_var = F_plugin.to(device) + sigma2_vi_plugin
    vi_mc_total_var = F_vi_mc_mean.to(device) + sigma2_vi_mc_mean

    # -----------------------------
    # Diagnostics: sigma^2 + scale
    # -----------------------------
    print(
        f"[DIAG] sigma2: "
        f"true={sigma2_true:.4f} | "
        f"VI_plugin={sigma2_vi_plugin:.4f} | "
        f"VI_mc={sigma2_vi_mc_mean:.4f}"
    )

    scale_true = F_true_local.sum()
    scale_hat_vi = F_vi_mc_mean.to(device).sum()
    scale_hat_plugin = F_plugin.to(device).sum()

    print(
        f"[DIAG] spectral scale ratio: "
        f"VI_plugin={scale_hat_plugin/scale_true:.4f} | "
        f"VI_mc={scale_hat_vi/scale_true:.4f}"
    )

    plot_true_vs_learned_curves(
        lam=plot_x,
        true_curve=true_total_var,
        learned={
            "VI plugin": vi_plugin_total_var,
            "VI MC mean": vi_mc_total_var,
        },
        outpath=case_dir / f"true_vs_vi_total_modal_variance_{plot_axis_tag}.png",
        title=f"True vs learned total modal variance — {filter_name}/{case_spec.display_name}",
        ylabel=r"$F(\lambda) + \sigma^2$",
        logy=False,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    plot_true_vs_learned_curves(
        lam=plot_x,
        true_curve=true_total_var,
        learned={
            "VI plugin": vi_plugin_total_var,
            "VI MC mean": vi_mc_total_var,
        },
        outpath=case_dir / f"true_vs_vi_total_modal_variance_{plot_axis_tag}_logy.png",
        title=f"True vs learned total modal variance log-y — {filter_name}/{case_spec.display_name}",
        ylabel=r"$F(\lambda) + \sigma^2$",
        logy=True,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    plot_true_vs_empirical(
        lam=plot_x,
        F_true=F_true_local,
        empirical_energy=empirical_energy,
        outpath=case_dir / f"true_vs_empirical_spectrum_{plot_axis_tag}.png",
        title=f"True spectrum vs empirical energy — {filter_name}/{case_spec.display_name}",
        logy=False,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    plot_true_vs_empirical(
        lam=plot_x,
        F_true=F_true_local,
        empirical_energy=empirical_energy,
        outpath=case_dir / f"true_vs_empirical_spectrum_{plot_axis_tag}_logy.png",
        title=f"True spectrum vs empirical energy log-y — {filter_name}/{case_spec.display_name}",
        logy=True,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    plot_true_vs_learned_spectra(
        lam=plot_x,
        F_true=F_true_local,
        learned={
            "VI plugin": F_plugin.to(device),
            "VI MC mean": F_vi_mc_mean.to(device),
        },
        outpath=case_dir / f"true_vs_vi_learned_spectrum_{plot_axis_tag}.png",
        title=f"True vs VI learned spectrum — {filter_name}/{case_spec.display_name}",
        logy=False,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    plot_true_vs_learned_spectra(
        lam=plot_x,
        F_true=F_true_local,
        learned={
            "VI plugin": F_plugin.to(device),
            "VI MC mean": F_vi_mc_mean.to(device),
        },
        outpath=case_dir / f"true_vs_vi_learned_spectrum_{plot_axis_tag}_logy.png",
        title=f"True vs VI learned spectrum log-y — {filter_name}/{case_spec.display_name}",
        logy=True,
        use_markers=True,
        xlabel=plot_xlabel,
    )

    diagnostics.plot_spectrum_recovery(
        lam=lam,
        F_true=F_true_local,
        filter_module=model.filter,
        vi_theta=model.plugin_hyperparams()[0],
        mcmc_theta=out["theta"].detach().cpu().numpy(),
        mcmc_theta_names=out["theta_names"],
        title=f"Spectrum recovery — {filter_name}/{case_spec.display_name}",
        save_path=case_dir / "spectrum_recovery.png",
        vi_band=True,
        mcmc_band=True,
    )
    # -------------------------
    # Spectrum error metrics
    # -------------------------
    spec_err_vi = spectrum_error_log_l1(
        lam=lam,
        F_true=F_true_local,
        model=model,
    )

    # use packed theta directly (out["theta"] is [S,d] torch)
    spec_err_mcmc = spectrum_error_log_l1(
        lam=lam,
        F_true=F_true_local,
        filter_module=model.filter,
        theta_mcmc=out["theta"],   # torch.Tensor [S,d]
    )

    spec_wl2_vi_plugin = weighted_relative_l2_spectrum_error(
        F_true=F_true_local,
        F_hat=F_plugin.to(device),
        weight="signal",
    )

    spec_wl2_vi_mc = weighted_relative_l2_spectrum_error(
        F_true=F_true_local,
        F_hat=F_vi_mc_mean.to(device),
        weight="signal",
    )

    spec_wl2_mcmc = weighted_relative_l2_spectrum_error(
        F_true=F_true_local,
        F_hat=F_mcmc_mean.to(device),
        weight="signal",
    )

    print(
        f"Spectrum weighted rel-L2: "
        f"VI(plugin)={spec_wl2_vi_plugin:.4f} | "
        f"VI(mc)={spec_wl2_vi_mc:.4f} | "
        f"MCMC={spec_wl2_mcmc:.4f}"
    )

    total_wl2_vi_plugin = weighted_relative_l2_spectrum_error(
        F_true=F_true_local + sigma2_true,
        F_hat=F_plugin.to(device) + sigma2_vi_plugin,
        weight="signal",
    )

    total_wl2_vi_mc = weighted_relative_l2_spectrum_error(
        F_true=F_true_local + sigma2_true,
        F_hat=F_vi_mc_mean.to(device) + sigma2_vi_mc_mean,
        weight="signal",
    )

    total_wl2_mcmc = weighted_relative_l2_spectrum_error(
        F_true=F_true_local + sigma2_true,
        F_hat=F_mcmc_mean.to(device) + float(np.mean(sigma2_chain)),
        weight="signal",
    )

    print(
        f"Total variance weighted rel-L2: "
        f"VI(plugin)={total_wl2_vi_plugin:.4f} | "
        f"VI(mc)={total_wl2_vi_mc:.4f} | "
        f"MCMC={total_wl2_mcmc:.4f}"
    )


    # -------------------------
    # Parameter recovery printout
    # -------------------------
    print("\nParameter recovery (truth vs VI plugin vs VI MC vs MCMC):")

    # beta
    for j in range(beta_chain.shape[1]):
        m, sd, lo, hi = summarize_chain(beta_chain[:, j])
        print(
            f" beta[{j}]  true={beta_true[j].item():.6g}  "
            f"VI(plugin)={beta_vi_plugin[j].item():.6g} "
            f"[{beta_vi_plugin_q025[j].item():.6g}, {beta_vi_plugin_q975[j].item():.6g}]  "
            f"VI(mc)={beta_vi_mc[j].item():.6g} "
            f"[{beta_vi_mc_q025[j].item():.6g}, {beta_vi_mc_q975[j].item():.6g}]  "
            f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
        )

    # sigma2
    m, sd, lo, hi = summarize_chain(sigma2_chain)
    print(
        f"  sigma2  true={sigma2_true:.6g}  "
        f"VI(plugin)={sigma2_vi_plugin:.6g}  "
        f"VI(mc)={sigma2_vi_mc_mean:.6g} ± {sigma2_vi_mc_sd:.4g}  "
        f"CI95=[{sigma2_vi_mc_q025:.6g}, {sigma2_vi_mc_q975:.6g}]  "
        f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
    )

    # tau2
    if tau2_chain is not None:
        m, sd, lo, hi = summarize_chain(tau2_chain)
        tau2_vi_plugin = "NA" if tau2_vi is None else f"{tensor_to_scalar(tau2_vi):.6g}"
        tau2_vi_mc = theta_vi["mc"].get("tau2", None)
        tau2_vi_mc_str = "NA"
        if tau2_vi_mc is not None:
            tau2_vi_mc_str = (
                f"{tensor_to_scalar(tau2_vi_mc['mean']):.6g} ± "
                f"{tensor_to_scalar(tau2_vi_mc['sd']):.4g}  "
                f"CI95=[{tensor_to_scalar(tau2_vi_mc['q025']):.6g}, "
                f"{tensor_to_scalar(tau2_vi_mc['q975']):.6g}]"
            )
        print(
            f"    tau2  true={tau2_true:.6g}  "
            f"VI(plugin)={tau2_vi_plugin}  "
            f"VI(mc)={tau2_vi_mc_str}  "
            f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
        )
    else:
        print("    tau2  (not present in this filter)")

    # rho0
    if rho0_chain is not None:
        m, sd, lo, hi = summarize_chain(rho0_chain)
        rho0_vi_plugin = "NA" if rho0_vi is None else f"{tensor_to_scalar(rho0_vi):.6g}"
        rho0_vi_mc = theta_vi["mc"].get("rho0", None)
        rho0_vi_mc_str = "NA"
        if rho0_vi_mc is not None:
            rho0_vi_mc_str = (
                f"{tensor_to_scalar(rho0_vi_mc['mean']):.6g} ± "
                f"{tensor_to_scalar(rho0_vi_mc['sd']):.4g}  "
                f"CI95=[{tensor_to_scalar(rho0_vi_mc['q025']):.6g}, "
                f"{tensor_to_scalar(rho0_vi_mc['q975']):.6g}]"
            )
        print(
            f"    rho0  true={eps_car:.6g}  "
            f"VI(plugin)={rho0_vi_plugin}  "
            f"VI(mc)={rho0_vi_mc_str}  "
            f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
        )

    # nu
    if nu_chain is not None:
        m, sd, lo, hi = summarize_chain(nu_chain)
        nu_vi_plugin = "NA" if nu_vi is None else f"{tensor_to_scalar(nu_vi):.6g}"
        nu_vi_mc = theta_vi["mc"].get("nu", None)
        nu_vi_mc_str = "NA"
        if nu_vi_mc is not None:
            nu_vi_mc_str = (
                f"{tensor_to_scalar(nu_vi_mc['mean']):.6g} ± "
                f"{tensor_to_scalar(nu_vi_mc['sd']):.4g}  "
                f"CI95=[{tensor_to_scalar(nu_vi_mc['q025']):.6g}, "
                f"{tensor_to_scalar(nu_vi_mc['q975']):.6g}]"
            )
        print(
            f"      nu  true={1.0:.6g}  "
            f"VI(plugin)={nu_vi_plugin}  "
            f"VI(mc)={nu_vi_mc_str}  "
            f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
        )

    a_chain = theta_constr.get("a", None)
    if a_chain is not None:
        print("  poly coeff mean (MCMC):", np.mean(a_chain, axis=0))
        if "a" in theta_vi["plugin"]:
            print("  poly coeff plugin (VI):", theta_vi["plugin"]["a"].detach().cpu().numpy())
        if "a" in theta_vi["mc"]:
            print("  poly coeff mean (VI-MC):", theta_vi["mc"]["a"]["mean"].detach().cpu().numpy())


    # -------------------------
    # Save plots
    # -------------------------
    for j in range(beta_chain.shape[1]):
        plot_trace(
            beta_chain[:, j],
            f"{filter_name}/{case_spec.display_name}: trace beta[{j}]",
            case_dir / f"trace_beta{j}.png",
        )

    plot_trace(
        sigma2_chain,
        f"{filter_name}/{case_spec.display_name}: trace sigma2",
        case_dir / "trace_sigma2.png",
    )
    
    if tau2_chain is not None:
        plot_trace(tau2_chain, f"{filter_name}/{case_spec.display_name}: trace tau2", case_dir / "trace_tau2.png")
    if rho0_chain is not None:
        plot_trace(rho0_chain, f"{filter_name}/{case_spec.display_name}: trace rho0", case_dir / "trace_rho0.png")
    if nu_chain is not None:
        plot_trace(nu_chain, f"{filter_name}/{case_spec.display_name}: trace nu", case_dir / "trace_nu.png")


    for j in range(beta_chain.shape[1]):
        plot_hist_with_lines(
            beta_chain[:, j],
            f"{filter_name}/{case_spec.display_name}: posterior beta[{j}]",
            case_dir / f"hist_beta{j}.png",
            true_val=float(beta_true[j].item()),
            vi_val=float(beta_vi_plugin[j].item()),
        )

    plot_hist_with_lines(
        sigma2_chain,
        f"{filter_name}/{case_spec.display_name}: posterior sigma2",
        case_dir / "hist_sigma2.png",
        true_val=float(sigma2_true),
        vi_val=float(sigma2_vi_mc_mean),
        vi_label="VI MC mean"
    )
    
    # tau2 (only if present)
    if tau2_chain is not None:
        plot_hist_with_lines(
            tau2_chain,
            f"{filter_name}/{case_spec.display_name}: posterior tau2",
            case_dir / "hist_tau2.png",
            true_val=float(tau2_true),
            vi_val=None if tau2_vi is None else tensor_to_scalar(tau2_vi),
        )

    # rho0 (prefer decoded chain from theta_constr if available; fall back to `named`)
    rho0_vals = rho0_chain if rho0_chain is not None else named.get("rho0", None)
    if rho0_vals is not None:
        plot_hist_with_lines(
            rho0_vals,
            f"{filter_name}/{case_spec.display_name}: posterior rho0",
            case_dir / "hist_rho0.png",
            true_val=float(eps_car),
            vi_val=None if rho0_vi is None else tensor_to_scalar(rho0_vi),
        )

    # nu (prefer decoded chain from theta_constr if available; fall back to `named`)
    nu_vals = nu_chain if nu_chain is not None else named.get("nu", None)
    if nu_vals is not None:
        plot_hist_with_lines(
            nu_vals,
            f"{filter_name}/{case_spec.display_name}: posterior nu",
            case_dir / "hist_nu.png",
            true_val=1.0,
            vi_val=None if nu_vi is None else tensor_to_scalar(nu_vi),
        )
    
    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=mean_phi_vi_plugin.to(device),
        phi_true=phi_true,
        save_path_prefix=str(case_dir / "phi_vi_plugin"),
    )

    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=mean_phi_vi.to(device),
        phi_true=phi_true,
        save_path_prefix=str(case_dir / "phi_vi"),
    )
    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=torch.from_numpy(phi_mean_mcmc).to(device=device, dtype=torch.double),
        phi_true=phi_true,
        save_path_prefix=str(case_dir / "phi_mcmc"),
    )

    # compact summary
    summary = {
        "filter": filter_name,
        "case": case_spec.display_name,
        "rmse_phi_vi_plugin": rmse_phi_vi_plugin,
        "rmse_phi_vi": rmse_phi_vi,
        "rmse_phi_mcmc": rmse_phi_mcmc,
        "spec_err_vi": spec_err_vi,
        "spec_err_mcmc": spec_err_mcmc,
        "spec_wl2_vi_plugin": spec_wl2_vi_plugin,
        "spec_wl2_vi_mc": spec_wl2_vi_mc,
        "spec_wl2_mcmc": spec_wl2_mcmc,
        "total_wl2_vi_plugin": spec_wl2_vi_plugin,
        "total_wl2_vi_mc": spec_wl2_vi_mc,
        "total_wl2_mcmc": spec_wl2_mcmc,
        "acc_s": acc["s"][2],
        "acc_theta": {k: v[2] for k, v in acc["theta"].items()},
        "vi_plugin": {
            "sigma2": sigma2_vi_plugin,
            **({"tau2": tensor_to_scalar(tau2_vi)} if tau2_vi is not None else {}),
            **({"rho0": tensor_to_scalar(rho0_vi)} if rho0_vi is not None else {}),
            **({"nu": tensor_to_scalar(nu_vi)} if nu_vi is not None else {}),
        },
        "vi_mc": {
            "sigma2_mean": sigma2_vi_mc_mean,
            "sigma2_sd": sigma2_vi_mc_sd,
            "sigma2_q025": sigma2_vi_mc_q025,
            "sigma2_q975": sigma2_vi_mc_q975,
            **(
                {
                    "tau2_mean": tensor_to_scalar(theta_vi["mc"]["tau2"]["mean"]),
                    "tau2_sd": tensor_to_scalar(theta_vi["mc"]["tau2"]["sd"]),
                } if "tau2" in theta_vi["mc"] else {}
            ),
            **(
                {
                    "rho0_mean": tensor_to_scalar(theta_vi["mc"]["rho0"]["mean"]),
                    "rho0_sd": tensor_to_scalar(theta_vi["mc"]["rho0"]["sd"]),
                } if "rho0" in theta_vi["mc"] else {}
            ),
            **(
                {
                    "nu_mean": tensor_to_scalar(theta_vi["mc"]["nu"]["mean"]),
                    "nu_sd": tensor_to_scalar(theta_vi["mc"]["nu"]["sd"]),
                } if "nu" in theta_vi["mc"] else {}
            ),
        },
        "mcmc_means": {
            "sigma2": float(np.mean(sigma2_chain)),
            **({"tau2": float(np.mean(tau2_chain))} if tau2_chain is not None else {}),
            **({"rho0": float(np.mean(rho0_chain))} if rho0_chain is not None else {}),
            **({"nu": float(np.mean(nu_chain))} if nu_chain is not None else {}),
        },
        "ridge": {
            "max_abs_corr": rep["max_abs_corr"],
            "mean_abs_corr": rep["mean_abs_corr"],
            "top_pairs": rep["top_pairs"][:10],
            "highlights": rep["highlights"],
        },
        "rmse_eta_vi_plugin": rmse_eta_vi_plugin,
        "rmse_eta_vi": rmse_eta_vi,
        "rmse_eta_mcmc": rmse_eta_mcmc,
    }

    # Add predictive metrics in the summary.
    summary.update(pred)

    summary.update({
        "sigma2_true": float(sigma2_true),
        "sigma2_vi_plugin": float(sigma2_vi_plugin),
        "sigma2_vi_mc": float(sigma2_vi_mc_mean),

        "scale_ratio_vi_plugin": float(scale_hat_plugin / scale_true),
        "scale_ratio_vi_mc": float(scale_hat_vi / scale_true),
    })

    # -------------------------
    # Write per-case metrics JSON
    # -------------------------
    import json
    metrics_path = case_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "filter": summary["filter"],
                "case": summary["case"],
                "rmse_phi_vi_plugin": summary["rmse_phi_vi_plugin"],
                "rmse_phi_vi": summary["rmse_phi_vi"],
                "rmse_phi_mcmc": summary["rmse_phi_mcmc"],
                "spec_err_vi": summary["spec_err_vi"],
                "spec_err_mcmc": summary["spec_err_mcmc"],
                "spec_wl2_vi_plugin": summary["spec_wl2_vi_plugin"],
                "spec_wl2_vi_mc": summary["spec_wl2_vi_mc"],
                "spec_wl2_mcmc": summary["spec_wl2_mcmc"],
                "total_wl2_vi_plugin": summary["spec_wl2_vi_plugin"],
                "total_wl2_vi_mc": summary["spec_wl2_vi_mc"],
                "total_wl2_mcmc": summary["spec_wl2_mcmc"],
                "acc_s": summary["acc_s"],
                "acc_theta": summary["acc_theta"],
                "vi_plugin": summary["vi_plugin"],
                "vi_mc": summary["vi_mc"],
                "mcmc_means": summary["mcmc_means"],
                "rmse_y_vi_plugin": summary["rmse_y_vi_plugin"],
                "rmse_y_vi_mc": summary["rmse_y_vi_mc"],
                "lpd_vi_plugin": summary["lpd_vi_plugin"],
                "lpd_vi_mc": summary["lpd_vi_mc"],
                "rmse_y_mcmc": summary["rmse_y_mcmc"],
                "lpd_mcmc": summary["lpd_mcmc"],
                "rmse_eta_vi_plugin": summary["rmse_eta_vi_plugin"],
                "rmse_eta_vi": summary["rmse_eta_vi"],
                "rmse_eta_mcmc": summary["rmse_eta_mcmc"],
            },
            f,
            indent=2,
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", required=True, help=f"Filter family. Available: {available_filters()}")
    parser.add_argument("--cases", nargs="+", default=["all"], help="Case IDs (e.g., all, baseline B1 B2)")
    parser.add_argument("--outdir", default=str(Path("examples") / "figures" / "benchmarks"),
                        help="Output directory for figures")

    # VI / MCMC knobs (optional)
    parser.add_argument("--vi_iters", type=int, default=2500)
    parser.add_argument("--vi_mc", type=int, default=10)
    parser.add_argument("--vi_lr", type=float, default=1e-2)
    parser.add_argument("--mcmc_steps", type=int, default=30000)
    parser.add_argument("--mcmc_burnin", type=int, default=10000)
    parser.add_argument("--mcmc_thin", type=int, default=10)

    parser.add_argument(
        "--truth",
        default="icar",
        choices=["icar", "leroux", "multiscale_bump", "poly", "rational", "diffusion"],
        help="Data-generating spectral truth.",
    )

    parser.add_argument(
    "--fast",
    action="store_true",
    help="Fast mode for CI / smoke tests (fewer VI iters and MCMC steps)",
)

    # print("--- I am here ---")
    # print()
    # print(available_filters())
    # print()
    args = parser.parse_args()

    # -------------------------
    # Fast mode overrides (CI / smoke tests)
    # -------------------------
    if args.fast:
        print("[FAST MODE] Using reduced iteration counts")

        # VI
        args.vi_iters = min(args.vi_iters, 600)     # 400–800 sweet spot
        args.vi_mc = min(args.vi_mc, 5)

        # MCMC
        args.mcmc_steps = min(args.mcmc_steps, 8000)
        args.mcmc_burnin = min(args.mcmc_burnin, 2000)
        args.mcmc_thin = max(args.mcmc_thin, 5)


    spec = get_filter_spec(args.filter)
    requested = set(args.cases)
    if "all" in requested:
        case_ids = list(spec.cases.keys())
    else:
        case_ids = list(requested)
        missing = [c for c in case_ids if c not in spec.cases]
        if missing:
            raise ValueError(f"Unknown case(s) for filter '{args.filter}': {missing}. "
                             f"Available: {list(spec.cases.keys())}")

    # Seeds + device
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    fig_dir = Path(args.outdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # 1) Grid + Laplacian + eigendecomp
    # --------------------------------------------
    nx, ny = 40, 40
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    n = coords.size(0)

    L, W = build_laplacian_from_knn(coords, k=8, gamma=0.2, rho=0.95)
    lam, U = laplacian_eigendecomp(L)
    lam = lam.to(device)
    U = U.to(device)

    # --------------------------------------------
    # 2) Generate truth
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3

    lam_max = lam.max().clamp_min(1e-12)
    x = (lam / lam_max).clamp(0.0, 1.0)

    if args.truth == "icar":
        F_true = tau2_true / (lam + eps_car)
    
    elif args.truth == "leroux":
        tau2_true = 0.4
        rho_true = 0.95

        F_true = tau2_true / ((1.0 - rho_true) + rho_true * lam)
        F_true = F_true.clamp_min(1e-12)

    elif args.truth == "multiscale_bump":
        t = torch.log(lam + eps_car)

        lo = math.log(eps_car)
        hi = math.log(float(lam.max().item()) + eps_car)

        m1 = lo + 0.45 * (hi - lo)
        m2 = lo + 0.80 * (hi - lo)

        s1 = 0.055 * (hi - lo)
        s2 = 0.045 * (hi - lo)

        F_true = tau2_true * (
            1.00 * torch.exp(-0.5 * ((t - m1) / s1) ** 2)
            + 0.75 * torch.exp(-0.5 * ((t - m2) / s2) ** 2)
            + 0.02
        ).clamp_min(1e-12)
    
    elif args.truth == "poly":
        # Example: decreasing polynomial
        tau2_true = 0.4

        a_true = torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=lam.dtype, device=lam.device)

        basis = 1.0 - x  # match "decreasing" mode

        P = torch.zeros_like(x)
        for k in range(len(a_true)):
            P += a_true[k] * (basis ** k)

        F_true = tau2_true * P.clamp_min(1e-12)
    
    elif args.truth == "rational":
        tau2_true = 0.4

        a_true = torch.tensor([1.0], dtype=lam.dtype, device=lam.device)
        b_true = torch.tensor([0.05, 2.0], dtype=lam.dtype, device=lam.device)

        P = a_true[0] * torch.ones_like(x)
        Q = b_true[0] + b_true[1] * x

        F_true = tau2_true * P / (Q + 1e-12)
        F_true = F_true.clamp_min(1e-12)

    elif args.truth == "diffusion":
        tau2_true = 0.4
        kappa_true = 4.0

        F_true = tau2_true * torch.exp(-kappa_true * x)
        F_true = F_true.clamp_min(1e-12)
    
    else:
        raise ValueError(f"Unknown truth: {args.truth}")
    
    # --------------------------------------------
    # DEBUG
    print(f"[TRUTH] truth={args.truth}")
    print(f"[TRUTH] F_true min={F_true.min().item():.6g}, max={F_true.max().item():.6g}")


    z_true = torch.sqrt(F_true) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true

    x_coord = coords[:, 0]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device), x_coord], dim=1) # X = [1, x_coord]
    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    sigma2_true = 0.1
    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(n, dtype=torch.double, device=device)

    # Prior on beta
    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(X.shape[1], dtype=torch.double, device=device)

    # --------------------------------------------
    # 3) Run selected cases
    # --------------------------------------------
    summaries = []
    for case_id in case_ids:
        summaries.append(
            run_case(
                case_spec=spec.cases[case_id],
                filter_name=spec.filter_name, F_true=F_true,
                X=X, y=y, lam=lam, U=U, coords=coords,
                phi_true=phi_true, beta_true=beta_true,
                sigma2_true=sigma2_true, tau2_true=tau2_true,
                eps_car=eps_car, prior_V0=prior_V0, device=device,
                fig_dir=fig_dir,
                vi_num_iters=args.vi_iters,
                vi_num_mc=args.vi_mc,
                vi_lr=args.vi_lr,
                mcmc_num_steps=args.mcmc_steps,
                mcmc_burnin=args.mcmc_burnin,
                mcmc_thin=args.mcmc_thin,
            )
        )

    # --------------------------------------------
    # 4) Print summary
    # --------------------------------------------
    if len(summaries) >= 1:
        print("\n" + "=" * 80)
        print("CROSS-CASE SUMMARY")
        print("=" * 80)
        for s in summaries:
            m = s["mcmc_means"]
            extras = ", ".join([f"{k}={v:.4g}" for k, v in m.items() if k not in ("sigma2", "tau2")])
            tail = f", {extras}" if extras else ""
            theta_acc_str = ", ".join([f"{k}={v:.3f}" for k, v in s["acc_theta"].items()])
            print(
                f"{s['filter']}/{s['case']:<22} | "
                f"phi_RMSE(VI-plugin)={s['rmse_phi_vi_plugin']:.4f}  "
                f"phi_RMSE(VI-mc)={s['rmse_phi_vi']:.4f}  "
                f"phi_RMSE(MCMC)={s['rmse_phi_mcmc']:.4f} | "
                f"acc_s={s['acc_s']:.3f} | acc_theta[{theta_acc_str}] | "
                f"MCMC mean sigma2={m['sigma2']:.4g}{tail}"
            )



if __name__ == "__main__":
    main()
