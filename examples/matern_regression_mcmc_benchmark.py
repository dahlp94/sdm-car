# examples/matern_regression_mcmc_benchmark.py
#
# Baseline + identifiability ablations:
#   - Baseline: learn (tau2, rho0, nu)
#   - B1: fix nu=1, learn (tau2, rho0)
#   - B2: fix rho0=eps_car, learn (tau2, nu)
#
# For EACH case:
#   1) run VI training (true training ELBO only)
#   2) run collapsed MCMC initialized from VI means
#   3) save plots + print parameter recovery table + phi RMSE
#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.filters import MaternLikeFilterFullVI
from sdmcar.models import SpectralCAR_FullVI
from sdmcar.mcmc import MCMCConfig, StepSizes, make_collapsed_mcmc_from_model
from sdmcar import diagnostics

torch.set_default_dtype(torch.double)


# -------------------------
# Small plotting helpers
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
):
    plt.figure(figsize=(5, 3))
    plt.hist(x, bins=40, density=True, alpha=0.85)
    if true_val is not None:
        plt.axvline(true_val, linestyle="--", linewidth=2, label="true")
    if vi_val is not None:
        plt.axvline(vi_val, linestyle="-", linewidth=2, label="VI mean")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def steps_for_case(case_name: str) -> StepSizes:
    # These are safe defaults based on our observed acceptances.
    if case_name == "baseline":
        return StepSizes(
            s=0.16,         # log sigma2
            t=0.35,         # log tau2
            rho_raw=0.9,   # rho0_raw
            nu_raw=0.0,    # nu_raw
        )

    if case_name == "B1_fix_nu_1":
        # a_raw is effectively just rho_raw here. Our acceptance was ~0.879 → step too small.
        return StepSizes(
            s=0.12,
            t=0.35,
            rho_raw=0.60,   # <-- key change: increase a lot (try 0.6; if still >0.7, go 0.8-1.0)
            nu_raw=0.0,     # unused / ignored if nu fixed, but ok to set 0
        )

    if case_name == "B2_fix_rho0_eps":
        # a_raw is effectively nu_raw here. Our acceptance ~0.55; modestly bump if you want.
        return StepSizes(
            s=0.12,
            t=0.35,
            rho_raw=0.0,    # unused / ignored if rho fixed
            nu_raw=0.25,    # slightly bigger than baseline
        )

    raise ValueError(f"Unknown case_name: {case_name}")


# -------------------------
# Core: run a single case
# -------------------------
def run_case(
    case_name: str,
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
    fixed_nu: float | None = None,
    fixed_rho0: float | None = None,
    vi_num_iters: int = 2500,
    vi_num_mc: int = 10,
    vi_lr: float = 1e-2,
    mcmc_num_steps: int = 30000,
    mcmc_burnin: int = 10000,
    mcmc_thin: int = 10,
):
    case_dir = fig_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"CASE: {case_name}  (fixed_nu={fixed_nu}, fixed_rho0={fixed_rho0})")
    print("=" * 80)

    # -------------------------
    # Build filter (VI)
    # -------------------------
    # Use same initializations as our baseline VI example where applicable.
    if fixed_nu is None and fixed_rho0 is None:
        # Baseline: learn (tau2, rho0, nu)
        matern = MaternLikeFilterFullVI(
            mu_log_tau2=math.log(tau2_true),
            log_std_log_tau2=-2.0,
            mu_rho0_raw=-7.0,
            log_std_rho0_raw=-2.5,
            mu_nu_raw=0.5,
            log_std_nu_raw=-2.0,
            fixed_nu=None,
            fixed_rho0=None,
        ).to(device)
    elif fixed_nu is not None:
        # B1: fix nu, learn (tau2, rho0)
        matern = MaternLikeFilterFullVI(
            mu_log_tau2=math.log(tau2_true),
            log_std_log_tau2=-2.0,
            mu_rho0_raw=-7.0,
            log_std_rho0_raw=-2.5,
            fixed_nu=float(fixed_nu),
            fixed_rho0=None,
        ).to(device)
    else:
        # B2: fix rho0, learn (tau2, nu)
        matern = MaternLikeFilterFullVI(
            mu_log_tau2=math.log(tau2_true),
            log_std_log_tau2=-2.0,
            mu_nu_raw=0.5,
            log_std_nu_raw=-2.0,
            fixed_nu=None,
            fixed_rho0=float(fixed_rho0),
        ).to(device)

    # -------------------------
    # Build VI model
    # -------------------------
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
        num_mc=vi_num_mc,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vi_lr)

    # -------------------------
    # Train VI (true training ELBO only)
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
                tau2_m, a_m = model.filter.mean_params()
                rho0_m, nu_m = a_m.unbind(-1)
                sigma2_m = torch.exp(model.mu_log_sigma2).item()
                beta_m = model.m_beta.detach().cpu().numpy()
            print(
                f"[VI {it+1:04d}] ELBO={elbo.item():.2f} "
                f"loglik={stats['mc_loglik'].item():.2f} "
                f"KLbeta={stats['mc_kl_beta'].item():.2f} "
                f"KLfilt={stats['kl_filter'].item():.2f} "
                f"KLsig={stats['kl_sigma2'].item():.2f} "
                f"tau2={tau2_m.item():.3f} rho0={rho0_m.item():.6f} nu={nu_m.item():.3f} "
                f"sigma2={sigma2_m:.4f} beta={beta_m}"
            )

    # save VI ELBO
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, vi_num_iters + 1), elbo_hist)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO (training MC estimate)")
    plt.title(f"VI ELBO (train) — {case_name}")
    plt.tight_layout()
    plt.savefig(case_dir / "vi_elbo_train.png", dpi=200)
    plt.close()

    # VI summaries
    with torch.no_grad():
        beta_vi = model.m_beta.detach().cpu()
        beta_vi_se = torch.sqrt(torch.diag(model.V_beta)).detach().cpu()
        tau2_vi, a_vi = model.filter.mean_params()
        rho0_vi, nu_vi = a_vi.unbind(-1)
        sigma2_vi = torch.exp(model.mu_log_sigma2).item()

        # phi: prefer MC-integrated if our posterior_phi supports it
        try:
            mean_phi_vi, var_phi_vi = model.posterior_phi(mode="mc", num_mc=64)
        except TypeError:
            mean_phi_vi, var_phi_vi = model.posterior_phi(use_q_means=True)

        mean_phi_vi = mean_phi_vi.detach().cpu()

    rmse_phi_vi = float(torch.sqrt(torch.mean((mean_phi_vi - phi_true.cpu()) ** 2)).item())

    # -------------------------
    # Run MCMC initialized from VI means
    # -------------------------
    step = steps_for_case(case_name)

    cfg = MCMCConfig(
        num_steps=mcmc_num_steps,
        burnin=mcmc_burnin,
        thin=mcmc_thin,
        step=step,
        block_a_raw=True,
        seed=0,
        device=device,
    )
    sampler = make_collapsed_mcmc_from_model(
        model,
        config=cfg,
        fixed_nu=fixed_nu,
        fixed_rho0=fixed_rho0,
    )

    init_s = model.mu_log_sigma2.detach()
    init_t = model.filter.mu_log_tau2.detach()
    init_a_raw = model.filter.mu_a_raw.detach()  # dim 2 (baseline) or dim 1 (B1/B2)

    print("\nRunning MCMC...")
    out = sampler.run(
        init_s=init_s,
        init_t=init_t,
        init_a_raw=init_a_raw,
        init_from_conditional_beta=True,
        store_phi_mean=True,
        U=U,
        X=X,
        y=y,
    )

    acc = out["acc"]
    print("Acceptance rates:")
    print("  s    :", acc["s"])
    print("  t    :", acc["t"])
    print("  a_raw:", acc["a_raw"])

    beta_chain = out["beta"].numpy()          # [S,p]
    s_chain = out["s"].numpy().reshape(-1)    # [S]
    t_chain = out["t"].numpy().reshape(-1)    # [S]
    a_raw_chain = out["a_raw"].numpy()        # [S,d] where d=2 baseline, d=1 B1/B2
    phi_mean_chain = out["phi_mean"].numpy()  # [S,n]

    sigma2_chain = np.exp(s_chain)
    tau2_chain = np.exp(t_chain)

    # Transform a_raw -> (rho0, nu) respecting constraints
    if fixed_nu is None and fixed_rho0 is None:
        rho0_chain = F.softplus(torch.from_numpy(a_raw_chain[:, 0])).numpy()
        nu_chain = F.softplus(torch.from_numpy(a_raw_chain[:, 1])).numpy()
    elif fixed_nu is not None:
        rho0_chain = F.softplus(torch.from_numpy(a_raw_chain[:, 0])).numpy()
        nu_chain = np.full_like(rho0_chain, float(fixed_nu))
    else:
        rho0_chain = np.full_like(tau2_chain, float(fixed_rho0))
        nu_chain = F.softplus(torch.from_numpy(a_raw_chain[:, 0])).numpy()

    phi_mean_mcmc = np.mean(phi_mean_chain, axis=0)
    rmse_phi_mcmc = float(np.sqrt(np.mean((phi_mean_mcmc - phi_true.cpu().numpy()) ** 2)))

    print(f"MCMC RMSE(phi_mean, phi_true) = {rmse_phi_mcmc:.4f}")

    # -------------------------
    # Parameter recovery printout
    # -------------------------
    print("\nParameter recovery (truth vs VI mean vs MCMC):")
    # beta
    for j in range(beta_chain.shape[1]):
        m, sd, lo, hi = summarize_chain(beta_chain[:, j])
        print(
            f" beta[{j}]  true={beta_true[j].item():.6g}  "
            f"VI={beta_vi[j].item():.6g}  "
            f"MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]"
        )
    # sigma2
    m, sd, lo, hi = summarize_chain(sigma2_chain)
    print(f"  sigma2  true={sigma2_true:.6g}  VI={sigma2_vi:.6g}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")
    # tau2
    m, sd, lo, hi = summarize_chain(tau2_chain)
    print(f"    tau2  true={tau2_true:.6g}  VI={tau2_vi.item():.6g}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")
    # rho0
    m, sd, lo, hi = summarize_chain(rho0_chain)
    print(f"    rho0  true={eps_car:.6g}  VI={rho0_vi.item():.6g}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")
    # nu
    m, sd, lo, hi = summarize_chain(nu_chain)
    print(f"      nu  true={1.0:.6g}  VI={nu_vi.item():.6g}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")

    # -------------------------
    # Save plots
    # -------------------------
    # traces
    plot_trace(beta_chain[:, 0], f"{case_name}: trace beta[0]", case_dir / "trace_beta0.png")
    plot_trace(beta_chain[:, 1], f"{case_name}: trace beta[1]", case_dir / "trace_beta1.png")
    plot_trace(sigma2_chain, f"{case_name}: trace sigma2", case_dir / "trace_sigma2.png")
    plot_trace(tau2_chain, f"{case_name}: trace tau2", case_dir / "trace_tau2.png")
    plot_trace(rho0_chain, f"{case_name}: trace rho0", case_dir / "trace_rho0.png")
    plot_trace(nu_chain, f"{case_name}: trace nu", case_dir / "trace_nu.png")

    # histograms
    plot_hist_with_lines(beta_chain[:, 0], f"{case_name}: posterior beta[0]", case_dir / "hist_beta0.png",
                         true_val=float(beta_true[0].item()), vi_val=float(beta_vi[0].item()))
    plot_hist_with_lines(beta_chain[:, 1], f"{case_name}: posterior beta[1]", case_dir / "hist_beta1.png",
                         true_val=float(beta_true[1].item()), vi_val=float(beta_vi[1].item()))
    plot_hist_with_lines(sigma2_chain, f"{case_name}: posterior sigma2", case_dir / "hist_sigma2.png",
                         true_val=float(sigma2_true), vi_val=float(sigma2_vi))
    plot_hist_with_lines(tau2_chain, f"{case_name}: posterior tau2", case_dir / "hist_tau2.png",
                         true_val=float(tau2_true), vi_val=float(tau2_vi.item()))
    plot_hist_with_lines(rho0_chain, f"{case_name}: posterior rho0", case_dir / "hist_rho0.png",
                         true_val=float(eps_car), vi_val=float(rho0_vi.item()))
    plot_hist_with_lines(nu_chain, f"{case_name}: posterior nu", case_dir / "hist_nu.png",
                         true_val=1.0, vi_val=float(nu_vi.item()))

    # phi maps (VI and MCMC)
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

    # Return compact summary for cross-case comparison
    summary = {
        "case": case_name,
        "rmse_phi_vi": rmse_phi_vi,
        "rmse_phi_mcmc": rmse_phi_mcmc,
        "acc_s": acc["s"][2],
        "acc_t": acc["t"][2],
        "acc_a": acc["a_raw"][2],
        "vi": {
            "beta": beta_vi.numpy(),
            "sigma2": sigma2_vi,
            "tau2": float(tau2_vi.item()),
            "rho0": float(rho0_vi.item()),
            "nu": float(nu_vi.item()),
        },
        "mcmc": {
            "beta_mean": np.mean(beta_chain, axis=0),
            "sigma2_mean": float(np.mean(sigma2_chain)),
            "tau2_mean": float(np.mean(tau2_chain)),
            "rho0_mean": float(np.mean(rho0_chain)),
            "nu_mean": float(np.mean(nu_chain)),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["baseline", "B1", "B2"],
        choices=["baseline", "B1", "B2"],
        help="Which cases to run",
    )
    args = parser.parse_args()
    run_set = set(args.cases)

    # -----------------------
    # Seeds + device
    # -----------------------
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    fig_dir = Path("examples") / "figures" / "identifiability_ablations"
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
    # 2) Generate CAR truth
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)

    z_true = torch.sqrt(F_car) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true

    x_coord = coords[:, 0]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device), x_coord], dim=1)
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

    if "baseline" in run_set:
        summaries.append(
            run_case(
                case_name="baseline",
                X=X, y=y, lam=lam, U=U, coords=coords,
                phi_true=phi_true, beta_true=beta_true,
                sigma2_true=sigma2_true, tau2_true=tau2_true,
                eps_car=eps_car, prior_V0=prior_V0, device=device,
                fig_dir=fig_dir,
                fixed_nu=None, fixed_rho0=None,
            )
        )

    if "B1" in run_set:
        summaries.append(
            run_case(
                case_name="B1_fix_nu_1",
                X=X, y=y, lam=lam, U=U, coords=coords,
                phi_true=phi_true, beta_true=beta_true,
                sigma2_true=sigma2_true, tau2_true=tau2_true,
                eps_car=eps_car, prior_V0=prior_V0, device=device,
                fig_dir=fig_dir,
                fixed_nu=1.0, fixed_rho0=None,
            )
        )

    if "B2" in run_set:
        summaries.append(
            run_case(
                case_name="B2_fix_rho0_eps",
                X=X, y=y, lam=lam, U=U, coords=coords,
                phi_true=phi_true, beta_true=beta_true,
                sigma2_true=sigma2_true, tau2_true=tau2_true,
                eps_car=eps_car, prior_V0=prior_V0, device=device,
                fig_dir=fig_dir,
                fixed_nu=None, fixed_rho0=eps_car,
            )
        )

    # --------------------------------------------
    # 4) Print cross-case summary
    # --------------------------------------------
    if len(summaries) > 1:
        print("\n" + "=" * 80)
        print("CROSS-CASE SUMMARY")
        print("=" * 80)
        for s in summaries:
            print(
                f"{s['case']:>14} | "
                f"phi_RMSE(VI)={s['rmse_phi_vi']:.4f}  phi_RMSE(MCMC)={s['rmse_phi_mcmc']:.4f} | "
                f"acc(s,t,a)=({s['acc_s']:.3f},{s['acc_t']:.3f},{s['acc_a']:.3f}) | "
                f"MCMC mean (tau2,rho0,nu)=({s['mcmc']['tau2_mean']:.3f},{s['mcmc']['rho0_mean']:.4f},{s['mcmc']['nu_mean']:.3f})"
            )

if __name__ == "__main__":
    main()
