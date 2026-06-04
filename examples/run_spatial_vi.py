# examples/run_bcef_vi.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.benchmarks.registry import get_filter_spec, available_filters

# If this import path is different in your repo, copy the same import
# used in examples/run_benchmark.py
from sdmcar import SpectralCAR_FullVI
from sdmcar.mcmc import MCMCConfig, make_collapsed_mcmc_from_model

torch.set_default_dtype(torch.double)

def seconds_to_str(x: float) -> str:
    if x < 60:
        return f"{x:.2f}s"
    if x < 3600:
        return f"{x / 60:.2f}min"
    return f"{x / 3600:.2f}hr"


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_to_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0])
    return float(x)


def load_graph_pt(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find graph file: {path}")

    # PyTorch 2.6 changed torch.load default behavior.
    # This graph file was created locally by us, so weights_only=False is fine.
    obj = torch.load(path, map_location="cpu", weights_only=False)

    required = ["y", "X", "coords", "lam", "U", "info"]
    missing = [k for k in required if k not in obj]

    if missing:
        raise ValueError(f"Missing keys {missing}. Found keys: {list(obj.keys())}")

    return obj


def fit_vi(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    filter_name: str,
    case_id: str,
    num_mc: int,
    vi_iters: int,
    vi_lr: float,
    outdir: Path,
    fixed_sigma2: float | None = None,
):
    device = y.device

    filter_spec = get_filter_spec(filter_name)

    if case_id not in filter_spec.cases:
        raise ValueError(
            f"Unknown case_id={case_id} for filter={filter_name}. "
            f"Available cases: {list(filter_spec.cases.keys())}"
        )

    case_spec = filter_spec.cases[case_id]

    print("\n[FILTER]")
    print(f"filter_name = {filter_name}")
    print(f"case_id     = {case_id}")
    print(f"display     = {case_spec.display_name}")
    print(f"case fixed  = {case_spec.fixed}")

    filter_module = case_spec.build_filter(
        tau2_true=1.0,
        eps_car=1e-3,
        device=device,
        lam_max=float(lam.max().item()),
        **case_spec.fixed,
    )

    # OLS initialization scale.
    beta_ols = torch.linalg.lstsq(X, y).solution
    resid_ols = y - X @ beta_ols
    sigma2_ols = torch.var(resid_ols, unbiased=True).item()

    print("\n[OLS INIT]")
    print(f"beta_ols   = {beta_ols.detach().cpu().numpy()}")
    print(f"sigma2_ols = {sigma2_ols:.6f}")

    p = X.shape[1]
    prior_V0 = 10.0 * torch.eye(p, dtype=y.dtype, device=device)

    sigma2_init = fixed_sigma2 if fixed_sigma2 is not None else sigma2_ols
    sigma2_init = max(float(sigma2_init), 1e-6)

    model = SpectralCAR_FullVI(
        X=X,
        y=y,
        lam=lam,
        U=U,
        filter_module=filter_module,
        prior_m0=None,
        prior_V0=prior_V0,
        mu_log_sigma2=math.log(sigma2_init),
        log_std_log_sigma2=-2.3,
        num_mc=num_mc,
        fixed_sigma2=fixed_sigma2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vi_lr)

    elbo_hist: list[float] = []
    history: list[dict] = []

    log_every = 50

    print("\n[VI TRAINING]")
    print(f"num_mc      = {num_mc}")
    print(f"vi_iters    = {vi_iters}")
    print(f"vi_lr       = {vi_lr}")
    print(f"fixed_sigma2 = {fixed_sigma2}")

    sync_if_cuda(device)
    t0 = time.perf_counter()

    for it in range(vi_iters):
        optimizer.zero_grad()

        # Your class returns: elbo, stats
        elbo, stats = model.elbo()

        loss = -elbo
        loss.backward()
        optimizer.step()

        elbo_hist.append(float(elbo.detach().cpu()))

        if (it + 1) % log_every == 0 or it == 0 or (it + 1) == vi_iters:
            elapsed = time.perf_counter() - t0

            with torch.no_grad():
                theta_plugin_u, theta_plugin_c, sigma2_plugin_train = model.plugin_hyperparams()

                tau2_m = theta_plugin_c.get("tau2", None)
                rho0_m = theta_plugin_c.get("rho0", None)
                nu_m = theta_plugin_c.get("nu", None)
                scale_m = theta_plugin_c.get("scale", None)
                theta_m = theta_plugin_c.get("theta", None)
                w_m = theta_plugin_c.get("w", None)

                if fixed_sigma2 is None:
                    mu_s = model.mu_log_sigma2.detach()
                    std_s = torch.exp(model.log_std_log_sigma2.detach())
                    sigma2_median = torch.exp(mu_s).item()
                    sigma2_mean = torch.exp(mu_s + 0.5 * std_s**2).item()
                else:
                    sigma2_median = float(fixed_sigma2)
                    sigma2_mean = float(fixed_sigma2)

                beta_m = model.m_beta.detach().cpu().numpy()

            tau2_str = "NA" if tau2_m is None else f"{tau2_m.item():.4f}"
            rho0_str = "NA" if rho0_m is None else f"{rho0_m.item():.6f}"
            nu_str = "NA" if nu_m is None else f"{nu_m.item():.4f}"
            scale_str = "NA" if scale_m is None else f"{scale_m.item():.4f}"

            if theta_m is not None:
                theta_np = theta_m.detach().cpu().numpy()
                theta_str = ", ".join(
                    [f"theta{j}={theta_np[j]:+.3f}" for j in range(min(2, len(theta_np)))]
                )
            else:
                theta_str = "NA"

            if w_m is not None:
                w_np = w_m.detach().cpu().numpy()
                w_str = (
                    f"w_l2={np.sqrt(np.sum(w_np**2)):.3f}, "
                    f"max|w|={np.max(np.abs(w_np)):.3f}"
                )
            else:
                w_str = "NA"

            row = {
                "iter": int(it + 1),
                "elbo": float(elbo.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "mc_loglik": float(stats["mc_loglik"].detach().cpu()),
                "mc_kl_beta": float(stats["mc_kl_beta"].detach().cpu()),
                "kl_filter": float(stats["kl_filter"].detach().cpu()),
                "kl_sigma2": float(stats["kl_sigma2"].detach().cpu()),
                "sigma2_median": float(sigma2_median),
                "sigma2_mean": float(sigma2_mean),
                "elapsed_seconds": float(elapsed),
            }

            history.append(row)

            print(
                f"[VI {it+1:04d}] "
                f"ELBO={row['elbo']:.2f} "
                f"loglik={row['mc_loglik']:.2f} "
                f"KLbeta={row['mc_kl_beta']:.2f} "
                f"KLfilt={row['kl_filter']:.2f} "
                f"KLsig={row['kl_sigma2']:.2f} "
                f"tau2={tau2_str} scale={scale_str} rho0={rho0_str} nu={nu_str} "
                f"{theta_str} {w_str} "
                f"sigma2_med={sigma2_median:.4f} "
                f"sigma2_mean={sigma2_mean:.4f} "
                f"beta={beta_m} "
                f"time={seconds_to_str(elapsed)}"
            )

    sync_if_cuda(device)
    vi_seconds = time.perf_counter() - t0

    print(
        f"\n[TIME] VI training: {seconds_to_str(vi_seconds)} "
        f"({vi_seconds / max(vi_iters, 1):.4f}s/iter)"
    )

    # Save ELBO plot.
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, vi_iters + 1), elbo_hist)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title(f"VI ELBO — {filter_name}/{case_spec.display_name}")
    plt.tight_layout()
    plt.savefig(outdir / "vi_elbo_train.png", dpi=200)
    plt.close()

    return model, filter_module, case_spec, history, elbo_hist, vi_seconds

def run_mcmc_from_vi(
    *,
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    U: torch.Tensor,
    device: torch.device,
    step_s: float,
    step_theta: dict[str, float],
    mcmc_steps: int,
    mcmc_burnin: int,
    mcmc_thin: int,
):
    if mcmc_steps <= 0:
        print("\n[MCMC] Skipping MCMC because mcmc_steps <= 0.")
        return None, 0.0

    cfg = MCMCConfig(
        num_steps=mcmc_steps,
        burnin=mcmc_burnin,
        thin=mcmc_thin,
        step_s=float(step_s),
        step_theta=step_theta,
        seed=0,
        device=device,
        print_every=50,
    )

    sampler = make_collapsed_mcmc_from_model(model, config=cfg)

    init_s = model.mu_log_sigma2.detach()

    theta0 = model.filter.mean_unconstrained()
    init_theta_vec = model.filter.pack(theta0).detach()

    print("\n[MCMC]")
    print(f"mcmc_steps  = {mcmc_steps}")
    print(f"burnin      = {mcmc_burnin}")
    print(f"thin        = {mcmc_thin}")
    print(f"step_s      = {step_s}")
    print(f"step_theta  = {step_theta}")

    sync_if_cuda(device)
    t0 = time.perf_counter()

    out = sampler.run(
        init_s=init_s,
        init_theta_vec=init_theta_vec,
        init_from_conditional_beta=True,
        store_phi_mean=True,
        U=U,
        X=X,
        y=y,
    )

    sync_if_cuda(device)
    mcmc_seconds = time.perf_counter() - t0

    print(
        f"[TIME] MCMC sampling: {seconds_to_str(mcmc_seconds)} "
        f"({mcmc_seconds / max(mcmc_steps, 1):.4f}s/step)"
    )

    acc = out["acc"]
    print("\n[MCMC ACCEPTANCE]")
    print("  s:", acc["s"])
    print("  theta blocks:")
    for k, v in acc["theta"].items():
        print(f"    {k}: {v}")

    return out, mcmc_seconds

@torch.no_grad()
def vi_summaries(
    *,
    model,
    y: torch.Tensor,
    X: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    num_summary_mc: int = 128,
) -> dict:
    device = y.device

    sync_if_cuda(device)
    t0 = time.perf_counter()

    beta_vi = model.beta_posterior_vi(num_mc=num_summary_mc, return_draws=False)
    sigma2_vi = model.sigma2_posterior_vi(num_mc=num_summary_mc, return_draws=False)
    theta_vi = model.theta_posterior_vi(num_mc=num_summary_mc, return_draws=False)
    spectrum_vi = model.spectrum_posterior_vi(num_mc=num_summary_mc, return_draws=False)

    beta_plugin_mean = beta_vi["plugin"]["mean"].detach()
    beta_plugin_sd = beta_vi["plugin"]["sd"].detach()
    beta_mc_mean = beta_vi["mc"]["mean"].detach()
    beta_mc_sd = beta_vi["mc"]["sd"].detach()

    sigma2_plugin = tensor_to_scalar(sigma2_vi["plugin"])
    sigma2_mc_mean = tensor_to_scalar(sigma2_vi["mc"]["mean"])
    sigma2_mc_sd = tensor_to_scalar(sigma2_vi["mc"]["sd"])
    sigma2_mc_q025 = tensor_to_scalar(sigma2_vi["mc"]["q025"])
    sigma2_mc_q975 = tensor_to_scalar(sigma2_vi["mc"]["q975"])

    mean_phi_plugin, var_phi_plugin = model.posterior_phi(mode="plugin")
    mean_phi_mc, var_phi_mc = model.posterior_phi(mode="mc", num_mc=num_summary_mc)

    F_plugin = spectrum_vi["plugin"].detach()
    F_mc_mean = spectrum_vi["mc"]["mean"].detach()
    F_mc_sd = spectrum_vi["mc"]["sd"].detach()
    F_mc_q025 = spectrum_vi["mc"]["q025"].detach()
    F_mc_q975 = spectrum_vi["mc"]["q975"].detach()

    # In-sample fitted mean.
    eta_plugin = X @ beta_plugin_mean + mean_phi_plugin
    eta_mc = X @ beta_mc_mean + mean_phi_mc

    resid_plugin = y - eta_plugin
    resid_mc = y - eta_mc

    rmse_y_plugin = torch.sqrt(torch.mean(resid_plugin**2)).item()
    rmse_y_mc = torch.sqrt(torch.mean(resid_mc**2)).item()

    # Approximate plug-in log predictive density on training data.
    # This is in-sample and should not be treated as held-out performance.
    var_y_plugin = var_phi_plugin + sigma2_plugin
    var_y_mc = var_phi_mc + sigma2_mc_mean

    lpd_plugin = torch.mean(
        -0.5 * (
            torch.log(2.0 * torch.pi * var_y_plugin)
            + resid_plugin**2 / var_y_plugin
        )
    ).item()

    lpd_mc = torch.mean(
        -0.5 * (
            torch.log(2.0 * torch.pi * var_y_mc)
            + resid_mc**2 / var_y_mc
        )
    ).item()

    sync_if_cuda(device)
    summary_seconds = time.perf_counter() - t0

    return {
        "beta_vi": beta_vi,
        "sigma2_vi": sigma2_vi,
        "theta_vi": theta_vi,
        "spectrum_vi": spectrum_vi,

        "beta_plugin_mean": beta_plugin_mean,
        "beta_plugin_sd": beta_plugin_sd,
        "beta_mc_mean": beta_mc_mean,
        "beta_mc_sd": beta_mc_sd,

        "sigma2_plugin": sigma2_plugin,
        "sigma2_mc_mean": sigma2_mc_mean,
        "sigma2_mc_sd": sigma2_mc_sd,
        "sigma2_mc_q025": sigma2_mc_q025,
        "sigma2_mc_q975": sigma2_mc_q975,

        "mean_phi_plugin": mean_phi_plugin.detach(),
        "var_phi_plugin": var_phi_plugin.detach(),
        "mean_phi_mc": mean_phi_mc.detach(),
        "var_phi_mc": var_phi_mc.detach(),

        "F_plugin": F_plugin,
        "F_mc_mean": F_mc_mean,
        "F_mc_sd": F_mc_sd,
        "F_mc_q025": F_mc_q025,
        "F_mc_q975": F_mc_q975,

        "eta_plugin": eta_plugin.detach(),
        "eta_mc": eta_mc.detach(),
        "resid_plugin": resid_plugin.detach(),
        "resid_mc": resid_mc.detach(),

        "rmse_y_plugin": float(rmse_y_plugin),
        "rmse_y_mc": float(rmse_y_mc),
        "lpd_plugin": float(lpd_plugin),
        "lpd_mc": float(lpd_mc),

        "summary_seconds": float(summary_seconds),
    }

@torch.no_grad()
def predictive_metrics_real_data(
    *,
    y: torch.Tensor,
    X: torch.Tensor,
    vi_sum: dict,
    mcmc_out=None,
) -> dict:
    """
    Real-data predictive metrics.

    Here "truth" means the observed standardized response y.

    Metrics are currently in-sample unless you later pass held-out data.
    """

    metrics = {}

    # -------------------------
    # VI plugin
    # -------------------------
    eta_plugin = vi_sum["eta_plugin"]
    resid_plugin = y - eta_plugin

    sigma2_plugin = float(vi_sum["sigma2_plugin"])
    var_y_plugin = vi_sum["var_phi_plugin"] + sigma2_plugin

    rmse_plugin = torch.sqrt(torch.mean(resid_plugin**2)).item()

    lpd_plugin = torch.mean(
        -0.5
        * (
            torch.log(2.0 * torch.pi * var_y_plugin)
            + resid_plugin**2 / var_y_plugin
        )
    ).item()

    q025_plugin = eta_plugin - 1.96 * torch.sqrt(var_y_plugin)
    q975_plugin = eta_plugin + 1.96 * torch.sqrt(var_y_plugin)

    coverage_plugin = torch.mean(
        ((y >= q025_plugin) & (y <= q975_plugin)).double()
    ).item()

    width_plugin = torch.mean(q975_plugin - q025_plugin).item()

    metrics.update(
        {
            "rmse_y_vi_plugin": float(rmse_plugin),
            "lpd_vi_plugin": float(lpd_plugin),
            "coverage95_vi_plugin": float(coverage_plugin),
            "width95_vi_plugin": float(width_plugin),
        }
    )

    # -------------------------
    # VI MC
    # -------------------------
    eta_mc = vi_sum["eta_mc"]
    resid_mc = y - eta_mc

    sigma2_mc = float(vi_sum["sigma2_mc_mean"])
    var_y_mc = vi_sum["var_phi_mc"] + sigma2_mc

    rmse_mc = torch.sqrt(torch.mean(resid_mc**2)).item()

    lpd_mc = torch.mean(
        -0.5
        * (
            torch.log(2.0 * torch.pi * var_y_mc)
            + resid_mc**2 / var_y_mc
        )
    ).item()

    q025_mc = eta_mc - 1.96 * torch.sqrt(var_y_mc)
    q975_mc = eta_mc + 1.96 * torch.sqrt(var_y_mc)

    coverage_mc = torch.mean(
        ((y >= q025_mc) & (y <= q975_mc)).double()
    ).item()

    width_mc = torch.mean(q975_mc - q025_mc).item()

    metrics.update(
        {
            "rmse_y_vi_mc": float(rmse_mc),
            "lpd_vi_mc": float(lpd_mc),
            "coverage95_vi_mc": float(coverage_mc),
            "width95_vi_mc": float(width_mc),
        }
    )

    # -------------------------
    # MCMC
    # -------------------------
    if mcmc_out is not None:
        beta_chain = mcmc_out["beta"]              # [S, p]
        s_chain = mcmc_out["s"].reshape(-1)        # [S]
        phi_chain = mcmc_out["phi_mean"]           # [S, n]

        sigma2_chain = torch.exp(s_chain)

        S = beta_chain.shape[0]
        n = y.numel()

        eta_draws = []
        for s in range(S):
            eta_s = X @ beta_chain[s] + phi_chain[s]
            eta_draws.append(eta_s)

        eta_draws = torch.stack(eta_draws, dim=0)  # [S, n]

        eta_mean = eta_draws.mean(dim=0)
        sigma2_mean = sigma2_chain.mean()

        resid_mcmc = y - eta_mean

        rmse_mcmc = torch.sqrt(torch.mean(resid_mcmc**2)).item()

        # Predictive draws: y_rep = eta_draw + epsilon_draw.
        eps = torch.randn_like(eta_draws) * torch.sqrt(sigma2_chain[:, None])
        yrep_draws = eta_draws + eps

        yrep_mean = yrep_draws.mean(dim=0)
        yrep_q025 = torch.quantile(yrep_draws, 0.025, dim=0)
        yrep_q975 = torch.quantile(yrep_draws, 0.975, dim=0)

        coverage_mcmc = torch.mean(
            ((y >= yrep_q025) & (y <= yrep_q975)).double()
        ).item()

        width_mcmc = torch.mean(yrep_q975 - yrep_q025).item()

        # Monte Carlo log predictive density.
        # log p(y_i) approx log mean_s Normal(y_i | eta_si, sigma2_s)
        log_probs = (
            -0.5 * torch.log(2.0 * torch.pi * sigma2_chain[:, None])
            -0.5 * (y[None, :] - eta_draws) ** 2 / sigma2_chain[:, None]
        )

        lpd_i = torch.logsumexp(log_probs, dim=0) - math.log(S)
        lpd_mcmc = torch.mean(lpd_i).item()

        metrics.update(
            {
                "rmse_y_mcmc": float(rmse_mcmc),
                "lpd_mcmc": float(lpd_mcmc),
                "coverage95_mcmc": float(coverage_mcmc),
                "width95_mcmc": float(width_mcmc),
                "sigma2_mcmc_mean": float(sigma2_mean.detach().cpu()),
                "num_mcmc_saved": int(S),
            }
        )

    return metrics

def save_vi_outputs(
    *,
    outdir: Path,
    model,
    filter_name: str,
    case_id: str,
    case_display_name: str,
    history: list[dict],
    elbo_hist: list[float],
    vi_seconds: float,
    mcmc_seconds: float,
    pred_metrics: dict,
    mcmc_out,
    summaries: dict,
    y: torch.Tensor,
    X: torch.Tensor,
    coords: torch.Tensor,
    lam: torch.Tensor,
    info: dict,
):
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "vi_history.json", "w") as f:
        json.dump(history, f, indent=2)

    theta_plugin_u, theta_plugin_c, sigma2_plugin_train = model.plugin_hyperparams()

    spline_diag = {}
    if "theta" in theta_plugin_c:
        theta_np = theta_plugin_c["theta"].detach().cpu().numpy()
        spline_diag["theta_plugin"] = {
            f"theta{j}": float(theta_np[j]) for j in range(len(theta_np))
        }

    if "w" in theta_plugin_c:
        w_np = theta_plugin_c["w"].detach().cpu().numpy()
        spline_diag["w_plugin"] = {
            f"w{j}": float(w_np[j]) for j in range(len(w_np))
        }
        spline_diag["w_l1"] = float(np.sum(np.abs(w_np)))
        spline_diag["w_l2"] = float(np.sqrt(np.sum(w_np**2)))
        spline_diag["max_abs_w"] = float(np.max(np.abs(w_np)))

    metrics = {
        "filter": filter_name,
        "case": case_id,
        "case_display_name": case_display_name,

        "n": int(y.numel()),
        "p": int(X.shape[1]),

        "rmse_y_vi_plugin_in_sample": summaries["rmse_y_plugin"],
        "rmse_y_vi_mc_in_sample": summaries["rmse_y_mc"],
        "lpd_vi_plugin_in_sample": summaries["lpd_plugin"],
        "lpd_vi_mc_in_sample": summaries["lpd_mc"],

        "sigma2_vi_plugin": summaries["sigma2_plugin"],
        "sigma2_vi_mc_mean": summaries["sigma2_mc_mean"],
        "sigma2_vi_mc_sd": summaries["sigma2_mc_sd"],
        "sigma2_vi_mc_q025": summaries["sigma2_mc_q025"],
        "sigma2_vi_mc_q975": summaries["sigma2_mc_q975"],

        "beta_vi_plugin_mean": [
            float(x) for x in summaries["beta_plugin_mean"].detach().cpu().reshape(-1)
        ],
        "beta_vi_plugin_sd": [
            float(x) for x in summaries["beta_plugin_sd"].detach().cpu().reshape(-1)
        ],
        "beta_vi_mc_mean": [
            float(x) for x in summaries["beta_mc_mean"].detach().cpu().reshape(-1)
        ],
        "beta_vi_mc_sd": [
            float(x) for x in summaries["beta_mc_sd"].detach().cpu().reshape(-1)
        ],

        "time_vi_train_seconds": float(vi_seconds),
        "time_vi_summary_seconds": float(summaries["summary_seconds"]),
        "time_vi_seconds_per_iter": float(vi_seconds / max(len(elbo_hist), 1)),

        "time_mcmc_seconds": float(mcmc_seconds),
        **pred_metrics,

        "response_transform": info.get("response_transform", None),
        "y_mean": info.get("y_mean", None),
        "y_sd": info.get("y_sd", None),
        "x_mean": info.get("x_mean", None),
        "x_sd": info.get("x_sd", None),

        "spline_coefficients": spline_diag,
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "metrics": metrics,
            "lam": lam.detach().cpu(),
            "coords": coords.detach().cpu(),
            "F_plugin": summaries["F_plugin"].detach().cpu(),
            "F_mc_mean": summaries["F_mc_mean"].detach().cpu(),
            "F_mc_sd": summaries["F_mc_sd"].detach().cpu(),
            "F_mc_q025": summaries["F_mc_q025"].detach().cpu(),
            "F_mc_q975": summaries["F_mc_q975"].detach().cpu(),
            "mean_phi_plugin": summaries["mean_phi_plugin"].detach().cpu(),
            "var_phi_plugin": summaries["var_phi_plugin"].detach().cpu(),
            "mean_phi_mc": summaries["mean_phi_mc"].detach().cpu(),
            "var_phi_mc": summaries["var_phi_mc"].detach().cpu(),
            "eta_plugin": summaries["eta_plugin"].detach().cpu(),
            "eta_mc": summaries["eta_mc"].detach().cpu(),
            "resid_plugin": summaries["resid_plugin"].detach().cpu(),
            "resid_mc": summaries["resid_mc"].detach().cpu(),
        },
        outdir / "vi_outputs.pt",
    )

    beta = summaries["beta_plugin_mean"].detach().cpu().numpy()
    sigma2 = summaries["sigma2_plugin"]
    rmse = summaries["rmse_y_plugin"]
    lpd = summaries["lpd_plugin"]

    print("\n[INTERPRETATION]")
    print(f"PTC coefficient beta[1] = {beta[1]:+.4f}")
    if beta[1] > 0:
        print("  Percent tree cover is positively associated with log canopy height.")
    else:
        print("  Percent tree cover is negatively associated with log canopy height.")

    print(f"sigma2 plugin = {sigma2:.4f}")
    print(f"in-sample RMSE = {rmse:.4f}")
    print(f"in-sample LPD  = {lpd:.4f}")

    F = summaries["F_plugin"].detach()
    x_axis = lam / lam.max().clamp_min(1e-12)

    low = F[x_axis <= 0.15].mean().item()
    mid = F[(x_axis > 0.15) & (x_axis <= 0.60)].mean().item()
    high = F[x_axis > 0.60].mean().item()

    print("\n[SPECTRAL MASS SUMMARY]")
    print(f"mean F low freq  x <= 0.15       : {low:.4f}")
    print(f"mean F mid freq  0.15 < x <= 0.60: {mid:.4f}")
    print(f"mean F high freq x > 0.60        : {high:.4f}")

    if low > mid and low > high:
        print("  Residual spatial variation is dominated by broad smooth patterns.")
    elif mid > low and mid > high:
        print("  Residual spatial variation has substantial middle-scale structure.")
    elif high > low and high > mid:
        print("  Residual spatial variation has strong local/high-frequency structure.")
    else:
        print("  Residual spatial variation is spread across multiple graph frequencies.")

    print("\n[METRICS]")
    print(json.dumps(metrics, indent=2))

    # Spectrum plot.
    x_axis = lam / lam.max().clamp_min(1e-12)

    plt.figure(figsize=(7, 4))
    plt.plot(
        x_axis.detach().cpu().numpy(),
        summaries["F_plugin"].detach().cpu().numpy(),
        linewidth=2,
        label="VI plugin",
    )
    plt.plot(
        x_axis.detach().cpu().numpy(),
        summaries["F_mc_mean"].detach().cpu().numpy(),
        linewidth=2,
        linestyle="--",
        label="VI MC mean",
    )
    plt.fill_between(
        x_axis.detach().cpu().numpy(),
        summaries["F_mc_q025"].detach().cpu().numpy(),
        summaries["F_mc_q975"].detach().cpu().numpy(),
        alpha=0.25,
        label="VI MC 95%",
    )
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel(r"$\widehat F(\lambda)$")
    plt.title(f"BCEF learned spectrum — {filter_name}/{case_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "spectrum_vi.png", dpi=200)
    plt.close()

    # Log spectrum plot.
    plt.figure(figsize=(7, 4))
    plt.plot(
        x_axis.detach().cpu().numpy(),
        summaries["F_plugin"].detach().cpu().numpy(),
        linewidth=2,
        label="VI plugin",
    )
    plt.plot(
        x_axis.detach().cpu().numpy(),
        summaries["F_mc_mean"].detach().cpu().numpy(),
        linewidth=2,
        linestyle="--",
        label="VI MC mean",
    )
    plt.fill_between(
        x_axis.detach().cpu().numpy(),
        summaries["F_mc_q025"].detach().cpu().numpy(),
        summaries["F_mc_q975"].detach().cpu().numpy(),
        alpha=0.25,
        label="VI MC 95%",
    )
    plt.yscale("log")
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel(r"$\widehat F(\lambda)$")
    plt.title(f"BCEF learned spectrum log-y — {filter_name}/{case_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "spectrum_vi_logy.png", dpi=200)
    plt.close()

    coords_np = coords.detach().cpu().numpy()

    # Spatial field map.
    phi_np = summaries["mean_phi_plugin"].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=phi_np, s=8)
    plt.colorbar(sc, label=r"$\widehat\phi_i$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("BCEF VI plugin spatial field")
    plt.tight_layout()
    plt.savefig(outdir / "phi_vi_plugin_map.png", dpi=200)
    plt.close()

    # Residual map.
    resid_np = summaries["resid_plugin"].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=resid_np, s=8)
    plt.colorbar(sc, label="residual")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("BCEF VI plugin residual map")
    plt.tight_layout()
    plt.savefig(outdir / "residual_vi_plugin_map.png", dpi=200)
    plt.close()

    # Fitted eta map.
    eta_np = summaries["eta_plugin"].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=eta_np, s=8)
    plt.colorbar(sc, label=r"$\widehat\eta_i$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("BCEF VI plugin fitted latent mean")
    plt.tight_layout()
    plt.savefig(outdir / "eta_vi_plugin_map.png", dpi=200)
    plt.close()

    beta_plugin = summaries["beta_plugin_mean"]
    fixed_np = (X @ beta_plugin).detach().cpu().numpy()
    phi_np = summaries["mean_phi_plugin"].detach().cpu().numpy()
    eta_np = summaries["eta_plugin"].detach().cpu().numpy()
    resid_np = summaries["resid_plugin"].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=fixed_np, s=8)
    plt.colorbar(sc, label=r"$X_i^\top\hat\beta$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("Fixed-effect component from PTC")
    plt.tight_layout()
    plt.savefig(outdir / "fixed_effect_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=phi_np, s=8)
    plt.colorbar(sc, label=r"$\widehat\phi_i$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("Spatial random effect")
    plt.tight_layout()
    plt.savefig(outdir / "spatial_effect_phi_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=eta_np, s=8)
    plt.colorbar(sc, label=r"$X_i^\top\hat\beta+\widehat\phi_i$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("Fitted latent mean")
    plt.tight_layout()
    plt.savefig(outdir / "latent_mean_eta_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=resid_np, s=8)
    plt.colorbar(sc, label=r"$y_i-\widehat\eta_i$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("SDM-CAR residual map")
    plt.tight_layout()
    plt.savefig(outdir / "sdmcar_residual_map.png", dpi=200)
    plt.close()

    print("\n[SAVED]")
    print(outdir)

def plot_bcef_raw_data(
    *,
    outdir: Path,
    coords: torch.Tensor,
    y: torch.Tensor,
    X: torch.Tensor,
    graph: dict,
):
    outdir.mkdir(parents=True, exist_ok=True)

    coords_np = coords.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    ptc_std_np = X[:, 1].detach().cpu().numpy()

    height_raw = graph.get("height_raw", None)
    ptc_raw = graph.get("ptc_raw", None)

    if "home_value_raw" in graph:
        height_raw = graph["home_value_raw"]
        ptc_raw = graph.get("income_raw", None)
        response_label = "median home value"
        covariate_label = "median household income"

    if height_raw is not None:
        height_np = height_raw.detach().cpu().numpy()

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=height_np, s=8)
        plt.colorbar(sc, label="forest canopy height")
        plt.xlabel("scaled coord x")
        plt.ylabel("scaled coord y")
        plt.title("Observed forest canopy height")
        plt.tight_layout()
        plt.savefig(outdir / "raw_height_map.png", dpi=200)
        plt.close()

    if ptc_raw is not None:
        ptc_np = ptc_raw.detach().cpu().numpy()

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=ptc_np, s=8)
        plt.colorbar(sc, label="percent tree cover")
        plt.xlabel("scaled coord x")
        plt.ylabel("scaled coord y")
        plt.title("Percent tree cover")
        plt.tight_layout()
        plt.savefig(outdir / "ptc_map.png", dpi=200)
        plt.close()

        if height_raw is not None:
            plt.figure(figsize=(5, 4))
            plt.scatter(ptc_np, height_np, s=8, alpha=0.5)
            plt.xlabel("Percent tree cover")
            plt.ylabel("Forest canopy height")
            plt.title("Canopy height vs percent tree cover")
            plt.tight_layout()
            plt.savefig(outdir / "height_vs_ptc.png", dpi=200)
            plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=y_np, s=8)
    plt.colorbar(sc, label="standardized log(FCH + 1)")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("Transformed response")
    plt.tight_layout()
    plt.savefig(outdir / "transformed_y_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=ptc_std_np, s=8)
    plt.colorbar(sc, label="standardized PTC")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("Standardized percent tree cover")
    plt.tight_layout()
    plt.savefig(outdir / "standardized_ptc_map.png", dpi=200)
    plt.close()

@torch.no_grad()
def ols_baseline_report(
    *,
    outdir: Path,
    y: torch.Tensor,
    X: torch.Tensor,
    coords: torch.Tensor,
):
    outdir.mkdir(parents=True, exist_ok=True)

    beta_ols = torch.linalg.lstsq(X, y).solution
    yhat_ols = X @ beta_ols
    resid_ols = y - yhat_ols

    sigma2_ols = torch.var(resid_ols, unbiased=True)
    rmse_ols = torch.sqrt(torch.mean(resid_ols**2))

    lpd_ols = torch.mean(
        -0.5
        * (
            torch.log(2.0 * torch.pi * sigma2_ols)
            + resid_ols**2 / sigma2_ols
        )
    )

    metrics = {
        "beta_ols": [float(v) for v in beta_ols.detach().cpu()],
        "sigma2_ols": float(sigma2_ols.detach().cpu()),
        "rmse_ols": float(rmse_ols.detach().cpu()),
        "lpd_ols": float(lpd_ols.detach().cpu()),
    }

    with open(outdir / "ols_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[OLS BASELINE]")
    print(json.dumps(metrics, indent=2))

    coords_np = coords.detach().cpu().numpy()
    resid_np = resid_ols.detach().cpu().numpy()
    yhat_np = yhat_ols.detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=yhat_np, s=8)
    plt.colorbar(sc, label=r"$X\hat\beta$")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("OLS fitted mean from PTC only")
    plt.tight_layout()
    plt.savefig(outdir / "ols_fitted_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(coords_np[:, 0], coords_np[:, 1], c=resid_np, s=8)
    plt.colorbar(sc, label="OLS residual")
    plt.xlabel("scaled coord x")
    plt.ylabel("scaled coord y")
    plt.title("OLS residual map")
    plt.tight_layout()
    plt.savefig(outdir / "ols_residual_map.png", dpi=200)
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph_pt",
        type=str,
        required=True,
        help="Path to saved BCEF graph .pt file.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        help=f"Filter family. Available: {available_filters()}",
    )

    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="Case ID for the chosen filter.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path("examples") / "figures" / "bcef_vi")
    )
    
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--mcmc_burnin", type=int, default=1000)
    parser.add_argument("--mcmc_thin", type=int, default=2)

    parser.add_argument("--vi_iters", type=int, default=1000)
    parser.add_argument("--vi_mc", type=int, default=3)
    parser.add_argument("--vi_lr", type=float, default=1e-2)
    parser.add_argument("--summary_mc", type=int, default=128)

    parser.add_argument(
        "--fix_sigma2",
        type=float,
        default=None,
        help="Optional fixed sigma2. Leave omitted for sigma2-free VI.",
    )

    args = parser.parse_args()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    graph = load_graph_pt(args.graph_pt)

    y = graph["y"].to(device=device, dtype=torch.double)
    X = graph["X"].to(device=device, dtype=torch.double)
    coords = graph["coords"].to(device=device, dtype=torch.double)
    lam = graph["lam"].to(device=device, dtype=torch.double)
    U = graph["U"].to(device=device, dtype=torch.double)
    info = graph.get("info", {})

    print("\n[LOADED GRAPH]")
    print(f"graph_pt     = {args.graph_pt}")
    print(f"y shape      = {tuple(y.shape)}")
    print(f"X shape      = {tuple(X.shape)}")
    print(f"coords shape = {tuple(coords.shape)}")
    print(f"lam shape    = {tuple(lam.shape)}")
    print(f"U shape      = {tuple(U.shape)}")
    print(f"lambda min   = {lam.min().item():.6e}")
    print(f"lambda max   = {lam.max().item():.6e}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_bcef_raw_data(
        outdir=outdir / "data_maps",
        coords=coords,
        y=y,
        X=X,
        graph=graph,
    )

    ols_metrics = ols_baseline_report(
        outdir=outdir / "ols_baseline",
        y=y,
        X=X,
        coords=coords,
    )

    case_t0 = time.perf_counter()

    model, filter_module, case_spec, history, elbo_hist, vi_seconds = fit_vi(
        y=y,
        X=X,
        lam=lam,
        U=U,
        filter_name=args.filter,
        case_id=args.case,
        num_mc=args.vi_mc,
        vi_iters=args.vi_iters,
        vi_lr=args.vi_lr,
        outdir=outdir,
        fixed_sigma2=args.fix_sigma2,
    )

    summaries = vi_summaries(
        model=model,
        y=y,
        X=X,
        lam=lam,
        U=U,
        num_summary_mc=args.summary_mc,
    )

    # -------------------------------------------------
    # MCMC + predictive metrics
    # -------------------------------------------------
    step_theta = case_spec.get_step_theta(model.filter)

    mcmc_out, mcmc_seconds = run_mcmc_from_vi(
        model=model,
        X=X,
        y=y,
        U=U,
        device=device,
        step_s=0.14 if args.fix_sigma2 is None else 0.0,
        step_theta=step_theta,
        mcmc_steps=args.mcmc_steps,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
    )

    pred_metrics = predictive_metrics_real_data(
        y=y,
        X=X,
        vi_sum=summaries,
        mcmc_out=mcmc_out,
    )

    save_vi_outputs(
        outdir=outdir,
        model=model,
        filter_name=args.filter,
        case_id=args.case,
        case_display_name=case_spec.display_name,
        history=history,
        elbo_hist=elbo_hist,
        vi_seconds=vi_seconds,
        mcmc_seconds=mcmc_seconds,
        pred_metrics=pred_metrics,
        mcmc_out=mcmc_out,
        summaries=summaries,
        y=y,
        X=X,
        coords=coords,
        lam=lam,
        info=info,
    )

    case_seconds = time.perf_counter() - case_t0

    print(
        f"\n[TIME] Case total: {seconds_to_str(case_seconds)} | "
        f"VI train={seconds_to_str(vi_seconds)} | "
        f"VI summaries={seconds_to_str(summaries['summary_seconds'])}"
    )


if __name__ == "__main__":
    main()