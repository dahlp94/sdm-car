# examples/run_misspec_demo.py
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

# ensure registry is populated
from examples.benchmarks.registry import get_filter_spec, available_filters
import examples.benchmarks  # noqa: F401


torch.set_default_dtype(torch.double)


# -------------------------
# Truth spectra (misspec)
# -------------------------
def truth_spectrum(lam: torch.Tensor, kind: str, *, eps_car: float) -> torch.Tensor:
    """
    Return F_true(lam) [n] for misspecified truth.

    Recommended kinds:
      - mix2 : CAR-like + diffusion tail (two-scale)
      - floor: CAR-like + flat floor
      - bump : CAR-like with mid-frequency bump
    """
    lam = lam.clamp_min(0.0)
    kind = kind.lower()

    if kind == "mix2":
        # F = tau1^2/(lam+eps) + tau2^2 * exp(-a lam)
        tau1 = 0.20
        tau2 = 0.25
        a = 8.0
        F = (tau1 / (lam + eps_car)) + tau2 * torch.exp(-a * lam)
        return F.clamp_min(1e-12)

    if kind == "floor":
        # F = tau^2/(lam+eps) + c  (CAR cannot create a flat floor)
        tau = 0.20
        c = 0.03
        F = (tau / (lam + eps_car)) + c
        return F.clamp_min(1e-12)

    if kind == "bump":
        # CAR-like + bump around lambda ~ mu (mid-frequency energy)
        tau = 0.18
        mu = 0.35 * float(lam.max().detach().cpu())
        sig = 0.12 * float(lam.max().detach().cpu())
        b = 0.9
        bump = 1.0 + b * torch.exp(-0.5 * ((lam - mu) / max(sig, 1e-12)) ** 2)
        F = (tau / (lam + eps_car)) * bump
        return F.clamp_min(1e-12)

    raise ValueError(f"Unknown truth kind '{kind}'. Choose from: mix2, floor, bump.")


# -------------------------
# Spectrum diagnostics
# -------------------------
@torch.no_grad()
def spectrum_vi_mean(filter_module, lam: torch.Tensor) -> torch.Tensor:
    theta_mean = filter_module.mean_unconstrained()
    return filter_module.spectrum(lam, theta_mean)


@torch.no_grad()
def spectrum_mcmc_mean(filter_module, lam: torch.Tensor, theta_chain: torch.Tensor, *, batch: int = 256) -> torch.Tensor:
    """
    theta_chain: [S, d_theta] packed
    returns: mean over draws of F(lam; theta_s)
    """
    device = next(filter_module.parameters()).device if any(True for _ in filter_module.parameters()) else lam.device
    dtype = lam.dtype
    lam = lam.to(device=device, dtype=dtype)

    S = theta_chain.shape[0]
    acc = torch.zeros_like(lam)
    count = 0

    for i in range(0, S, batch):
        chunk = theta_chain[i:i+batch].to(device=device, dtype=dtype)
        for j in range(chunk.shape[0]):
            theta = filter_module.unpack(chunk[j])
            acc += filter_module.spectrum(lam, theta)
            count += 1

    return acc / max(count, 1)



def plot_spectrum_curves(lam_np: np.ndarray, curves: dict[str, np.ndarray], save_path: Path, *, ylog: bool):
    plt.figure(figsize=(6.2, 4.2))
    for name, y in curves.items():
        plt.plot(lam_np, y, linewidth=2.0 if name == "truth" else 1.6, label=name)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$F(\lambda)$")
    plt.title("Spectrum comparison")
    if ylog:
        plt.yscale("log")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def rmse_log_spectrum(F_hat: torch.Tensor, F_true: torch.Tensor) -> float:
    eps = 1e-12
    a = torch.log(F_hat.clamp_min(eps))
    b = torch.log(F_true.clamp_min(eps))
    return float(torch.sqrt(torch.mean((a - b) ** 2)).detach().cpu().item())


# -------------------------
# Run one fit
# -------------------------
def run_fit(
    *,
    spec_name: str,
    case_id: str,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    coords: torch.Tensor,
    phi_true: torch.Tensor,
    beta_true: torch.Tensor,
    F_true: torch.Tensor,
    sigma2_true: float,
    tau2_init: float,
    eps_car: float,
    prior_V0: torch.Tensor,
    device: torch.device,
    outdir: Path,
    vi_iters: int,
    vi_mc: int,
    vi_lr: float,
    mcmc_steps: int,
    mcmc_burnin: int,
    mcmc_thin: int,
):
    spec = get_filter_spec(spec_name)
    if case_id not in spec.cases:
        print(f"  - skipping {spec_name}/{case_id} (case not defined)")
        return None

    case_spec = spec.cases[case_id]
    case_dir = outdir / spec_name / case_spec.display_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"FIT: {spec_name} | CASE: {case_spec.display_name}")
    print("=" * 80)

    # Build filter (VI)
    filter_module = case_spec.build_filter(
        tau2_true=tau2_init,
        eps_car=eps_car,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **case_spec.fixed,
    )

    # VI model
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
        num_mc=vi_mc,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)

    # Train VI
    for it in range(vi_iters):
        opt.zero_grad()
        elbo, _ = model.elbo()
        (-elbo).backward()
        opt.step()

    # VI phi + spectrum
    with torch.no_grad():
        phi_vi, _ = model.posterior_phi(mode="mc", num_mc=64)
        rmse_phi_vi = float(torch.sqrt(torch.mean((phi_vi.detach().cpu() - phi_true.cpu()) ** 2)).item())

        F_vi = spectrum_vi_mean(model.filter, lam).detach()
        rmse_logF_vi = rmse_log_spectrum(F_vi, F_true)

    # MCMC (init from VI)
    cfg = MCMCConfig(
        num_steps=mcmc_steps,
        burnin=mcmc_burnin,
        thin=mcmc_thin,
        step_s=float(case_spec.step_s),
        step_theta=case_spec.get_step_theta(model.filter),
        seed=0,
        device=device,
    )
    sampler = make_collapsed_mcmc_from_model(model, config=cfg)

    init_s = model.mu_log_sigma2.detach()
    theta0 = model.filter.mean_unconstrained()
    init_theta_vec = model.filter.pack(theta0).detach()

    out = sampler.run(
        init_s=init_s,
        init_theta_vec=init_theta_vec,
        init_from_conditional_beta=True,
        store_phi_mean=True,
        U=U,
        X=X,
        y=y,
    )

    # MCMC phi mean + spectrum mean
    phi_mean_chain = out["phi_mean"]  # [S,n]
    phi_mcmc = phi_mean_chain.mean(dim=0)
    rmse_phi_mcmc = float(torch.sqrt(torch.mean((phi_mcmc - phi_true.cpu()) ** 2)).item())

    theta_chain = out["theta"]  # [S,d]
    F_mcmc = spectrum_mcmc_mean(model.filter, lam, theta_chain).detach()
    rmse_logF_mcmc = rmse_log_spectrum(F_mcmc, F_true)

    # Save spectrum plots
    lam_np = lam.detach().cpu().numpy()
    curves = {
        "truth": F_true.detach().cpu().numpy(),
        "VI": F_vi.detach().cpu().numpy(),
        "MCMC": F_mcmc.detach().cpu().numpy(),
    }
    plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_linear.png", ylog=False)
    plot_spectrum_curves(lam_np, curves, case_dir / "spectrum_log.png", ylog=True)

    # Save phi plots
    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=phi_vi.to(device),
        phi_true=phi_true,
        save_path_prefix=str(case_dir / "phi_vi"),
    )
    diagnostics.plot_phi_mean_vs_true(
        coords=coords,
        mean_phi=phi_mcmc.to(device=device, dtype=torch.double),
        phi_true=phi_true,
        save_path_prefix=str(case_dir / "phi_mcmc"),
    )

    print(f"  phi RMSE  : VI={rmse_phi_vi:.4f} | MCMC={rmse_phi_mcmc:.4f}")
    print(f"  logF RMSE : VI={rmse_logF_vi:.4f} | MCMC={rmse_logF_mcmc:.4f}")
    print(f"  acc_s={out['acc']['s'][2]:.3f}  acc_theta=" +
          ", ".join([f"{k}={v[2]:.3f}" for k, v in out["acc"]["theta"].items()]))

    return {
        "filter": spec_name,
        "case": case_spec.display_name,
        "rmse_phi_vi": rmse_phi_vi,
        "rmse_phi_mcmc": rmse_phi_mcmc,
        "rmse_logF_vi": rmse_logF_vi,
        "rmse_logF_mcmc": rmse_logF_mcmc,
    }

# Ran these:
# python -m examples.run_misspec_demo --truth floor --filters car --cases baseline
# python -m examples.run_misspec_demo --truth floor --filters rational --cases flex_22


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, choices=["mix2", "floor", "bump"])
    p.add_argument("--filters", nargs="+", required=True, help=f"Filter families to fit. Available: {available_filters()}")
    p.add_argument("--cases", nargs="+", default=["baseline"], help="Case IDs to try for each filter (skips missing).")
    p.add_argument("--outdir", default=str(Path("examples") / "figures" / "misspec"))

    p.add_argument("--vi_iters", type=int, default=2000)
    p.add_argument("--vi_mc", type=int, default=10)
    p.add_argument("--vi_lr", type=float, default=1e-2)
    p.add_argument("--mcmc_steps", type=int, default=30000)
    p.add_argument("--mcmc_burnin", type=int, default=10000)
    p.add_argument("--mcmc_thin", type=int, default=10)

    args = p.parse_args()

    # Seeds + device
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    outdir = Path(args.outdir) / f"truth_{args.truth}"
    outdir.mkdir(parents=True, exist_ok=True)

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
    # 2) Generate misspecified truth
    # --------------------------------------------
    eps_car = 1e-3
    F_true = truth_spectrum(lam, args.truth, eps_car=eps_car).to(device)

    z_true = torch.sqrt(F_true) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true

    x_coord = coords[:, 0]
    X = torch.stack([torch.ones(n, dtype=torch.double, device=device), x_coord], dim=1)
    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)

    sigma2_true = 0.10
    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(n, dtype=torch.double, device=device)

    # Prior on beta
    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(X.shape[1], dtype=torch.double, device=device)

    # print("\nAVAILABLE FILTERS/CASES:")
    # for f in available_filters():
    #     spec = get_filter_spec(f)
    #     print(f"  {f}: {list(spec.cases.keys())}")
    # print()


    # --------------------------------------------
    # 3) Fit selected filters/cases
    # --------------------------------------------
    summaries = []
    tau2_init = 0.4  # just an initialization hint

    for filt in args.filters:
        # validate early
        _ = get_filter_spec(filt)

        for case_id in args.cases:
            s = run_fit(
                spec_name=filt,
                case_id=case_id,
                X=X, y=y, lam=lam, U=U, coords=coords,
                phi_true=phi_true, beta_true=beta_true,
                F_true=F_true,
                sigma2_true=sigma2_true,
                tau2_init=tau2_init,
                eps_car=eps_car,
                prior_V0=prior_V0,
                device=device,
                outdir=outdir,
                vi_iters=args.vi_iters,
                vi_mc=args.vi_mc,
                vi_lr=args.vi_lr,
                mcmc_steps=args.mcmc_steps,
                mcmc_burnin=args.mcmc_burnin,
                mcmc_thin=args.mcmc_thin,
            )
            if s is not None:
                summaries.append(s)

    # --------------------------------------------
    # 4) Print leaderboard
    # --------------------------------------------
    if summaries:
        print("\n" + "=" * 80)
        print(f"MISSPEC SUMMARY (truth={args.truth})")
        print("=" * 80)
        for s in sorted(summaries, key=lambda d: d["rmse_logF_mcmc"]):
            print(
                f"{s['filter']}/{s['case']:<18} | "
                f"logF_RMSE(VI)={s['rmse_logF_vi']:.4f}  "
                f"logF_RMSE(MCMC)={s['rmse_logF_mcmc']:.4f} | "
                f"phi_RMSE(VI)={s['rmse_phi_vi']:.4f}  "
                f"phi_RMSE(MCMC)={s['rmse_phi_mcmc']:.4f}"
            )


if __name__ == "__main__":
    main()
