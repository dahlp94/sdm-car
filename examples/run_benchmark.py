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
from sdmcar.diagnostics import spectrum_error_log_l1

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
            label="VI mean",
        )

    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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
    theta_mat = out["theta"].numpy()   # [S, d_theta]
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
                means = unpack_filter_params_from_means(model.filter)
                tau2_m = means["tau2"]
                rho0_m = means["rho0"]
                nu_m = means["nu"]

                sigma2_m = torch.exp(model.mu_log_sigma2).item()
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
                f"sigma2={sigma2_m:.4f} beta={beta_m}"
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

    # VI summaries
    with torch.no_grad():
        beta_vi = model.m_beta.detach().cpu()
        means = unpack_filter_params_from_means(model.filter)
        tau2_vi = means["tau2"]
        rho0_vi = means["rho0"]
        nu_vi = means["nu"]
        sigma2_vi = torch.exp(model.mu_log_sigma2).item()

        # phi
        try:
            mean_phi_vi, _ = model.posterior_phi(mode="mc", num_mc=64)
        except TypeError:
            mean_phi_vi, _ = model.posterior_phi(use_q_means=True)

        mean_phi_vi = mean_phi_vi.detach().cpu()

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


    # chains extraction
    beta_chain = out["beta"].numpy()
    s_chain = out["s"].numpy().reshape(-1)
    sigma2_chain = np.exp(s_chain)

    theta_raw, theta_constr = decode_theta_chain(out, model.filter)

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



    phi_mean_chain = out["phi_mean"].numpy()  # [S,n]
    phi_mean_mcmc = np.mean(phi_mean_chain, axis=0)
    rmse_phi_mcmc = float(np.sqrt(np.mean((phi_mean_mcmc - phi_true.cpu().numpy()) ** 2)))
    print(f"MCMC RMSE(phi_mean, phi_true) = {rmse_phi_mcmc:.4f}")

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
    

    lam_flat = lam.reshape(-1)
    _, idx = torch.sort(lam_flat)

    F_true_local = F_true.to(device=lam.device, dtype=lam.dtype).reshape(-1)


    diagnostics.plot_spectrum_recovery(
        lam=lam,
        F_true=F_true,
        filter_module=model.filter,
        vi_theta=model.filter.mean_unconstrained(),
        mcmc_theta=out["theta"].numpy(),
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
    if tau2_chain is not None:
        m, sd, lo, hi = summarize_chain(tau2_chain)
        print(f"    tau2  true={tau2_true:.6g}  VI={tau2_vi.item():.6g}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")
    else:
        print("    tau2  (not present in this filter)")


    if rho0_chain is not None:
        m, sd, lo, hi = summarize_chain(rho0_chain)
        rho0_vi_str = "NA" if rho0_vi is None else f"{rho0_vi.item():.6g}"
        print(f"    rho0  true={eps_car:.6g}  VI={rho0_vi_str}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")

    if nu_chain is not None:
        m, sd, lo, hi = summarize_chain(nu_chain)
        nu_vi_str = "NA" if nu_vi is None else f"{nu_vi.item():.6g}"
        print(f"      nu  true={1.0:.6g}  VI={nu_vi_str}  MCMC={m:.6g} ± {sd:.4g}  CI95=[{lo:.6g}, {hi:.6g}]")
    
    a_chain = theta_constr.get("a", None)
    if a_chain is not None:
        print("  poly coeff mean:", np.mean(a_chain, axis=0))


    # -------------------------
    # Save plots
    # -------------------------
    plot_trace(beta_chain[:, 0], f"{filter_name}/{case_spec.display_name}: trace beta[0]", case_dir / "trace_beta0.png")
    plot_trace(beta_chain[:, 1], f"{filter_name}/{case_spec.display_name}: trace beta[1]", case_dir / "trace_beta1.png")
    plot_trace(sigma2_chain, f"{filter_name}/{case_spec.display_name}: trace sigma2", case_dir / "trace_sigma2.png")
    
    if tau2_chain is not None:
        plot_trace(tau2_chain, f"{filter_name}/{case_spec.display_name}: trace tau2", case_dir / "trace_tau2.png")
    if rho0_chain is not None:
        plot_trace(rho0_chain, f"{filter_name}/{case_spec.display_name}: trace rho0", case_dir / "trace_rho0.png")
    if nu_chain is not None:
        plot_trace(nu_chain, f"{filter_name}/{case_spec.display_name}: trace nu", case_dir / "trace_nu.png")


    plot_hist_with_lines(beta_chain[:, 0], f"{filter_name}/{case_spec.display_name}: posterior beta[0]",
                         case_dir / "hist_beta0.png", true_val=float(beta_true[0].item()), vi_val=float(beta_vi[0].item()))
    plot_hist_with_lines(beta_chain[:, 1], f"{filter_name}/{case_spec.display_name}: posterior beta[1]",
                         case_dir / "hist_beta1.png", true_val=float(beta_true[1].item()), vi_val=float(beta_vi[1].item()))
    plot_hist_with_lines(sigma2_chain, f"{filter_name}/{case_spec.display_name}: posterior sigma2",
                         case_dir / "hist_sigma2.png", true_val=float(sigma2_true), vi_val=float(sigma2_vi))
    
    # tau2 (only if present)
    if tau2_chain is not None:
        plot_hist_with_lines(
            tau2_chain,
            f"{filter_name}/{case_spec.display_name}: posterior tau2",
            case_dir / "hist_tau2.png",
            true_val=float(tau2_true),
            vi_val=float(tau2_vi.item()),
        )

    # rho0 (prefer decoded chain from theta_constr if available; fall back to `named`)
    rho0_vals = rho0_chain if rho0_chain is not None else named.get("rho0", None)
    if rho0_vals is not None:
        plot_hist_with_lines(
            rho0_vals,
            f"{filter_name}/{case_spec.display_name}: posterior rho0",
            case_dir / "hist_rho0.png",
            true_val=float(eps_car),
            vi_val=None if rho0_vi is None else float(rho0_vi.item()),
        )

    # nu (prefer decoded chain from theta_constr if available; fall back to `named`)
    nu_vals = nu_chain if nu_chain is not None else named.get("nu", None)
    if nu_vals is not None:
        plot_hist_with_lines(
            nu_vals,
            f"{filter_name}/{case_spec.display_name}: posterior nu",
            case_dir / "hist_nu.png",
            true_val=1.0,
            vi_val=None if nu_vi is None else float(nu_vi.item()),
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
        "rmse_phi_vi": rmse_phi_vi,
        "rmse_phi_mcmc": rmse_phi_mcmc,
        "spec_err_vi": spec_err_vi,
        "spec_err_mcmc": spec_err_mcmc,
        "acc_s": acc["s"][2],
        "acc_theta": {k: v[2] for k, v in acc["theta"].items()},
        "mcmc_means": {
            "sigma2": float(np.mean(sigma2_chain)),
            **({ "tau2": float(np.mean(tau2_chain)) } if tau2_chain is not None else {}),
            **({ "rho0": float(np.mean(rho0_chain)) } if rho0_chain is not None else {}),
            **({ "nu": float(np.mean(nu_chain)) } if nu_chain is not None else {}),
        },
        "ridge": {
            "max_abs_corr": rep["max_abs_corr"],
            "mean_abs_corr": rep["mean_abs_corr"],
            "top_pairs": rep["top_pairs"][:10],
            "highlights": rep["highlights"],
        },
    }

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
                "rmse_phi_vi": summary["rmse_phi_vi"],
                "rmse_phi_mcmc": summary["rmse_phi_mcmc"],
                "spec_err_vi": summary["spec_err_vi"],
                "spec_err_mcmc": summary["spec_err_mcmc"],
                "acc_s": summary["acc_s"],
                "acc_theta": summary["acc_theta"],
                "mcmc_means": summary["mcmc_means"],
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
    # 2) Generate CAR truth
    # --------------------------------------------
    tau2_true = 0.4
    eps_car = 1e-3
    F_car = tau2_true / (lam + eps_car)
    F_true = F_car

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
                f"phi_RMSE(VI)={s['rmse_phi_vi']:.4f}  "
                f"phi_RMSE(MCMC)={s['rmse_phi_mcmc']:.4f} | "
                f"acc_s={s['acc_s']:.3f} | acc_theta[{theta_acc_str}] | "
                f"MCMC mean sigma2={m['sigma2']:.4g}{tail}"
            )



if __name__ == "__main__":
    main()
