# examples/run_car_recovery_experiment.py

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import math
import random
import argparse

import numpy as np
import pandas as pd
import torch

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from examples.benchmarks.registry import get_filter_spec
import examples.benchmarks  # noqa: F401

from examples.run_benchmark import run_case


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# -------------------------
# graph
# -------------------------
def build_graph(device):
    nx, ny = 40, 40
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    L, W = build_laplacian_from_knn(coords, k=8, gamma=0.2, rho=0.95)
    lam, U = laplacian_eigendecomp(L)

    return {
        "coords": coords,
        "lam": lam.to(device),
        "U": U.to(device),
    }

def make_truth_spectrum(
    truth_name: str,
    lam: torch.Tensor,
    tau2_true: float,
    eps_car: float,
    rho_true: float = 0.8,
):
    """
    Build the true spectrum F_true(lam) for the chosen truth family.

    truth_name:
        - "icar_like"
        - "leroux"
    """
    if truth_name == "icar_like":
        F_true = tau2_true / (lam + eps_car)

    elif truth_name == "leroux":
        # Leroux-style spectral form:
        # F(lam) = tau2 / ((1-rho) + rho * lam)
        F_true = tau2_true / ((1.0 - rho_true) + rho_true * lam)

    else:
        raise ValueError(f"Unknown truth_name: {truth_name}")

    return F_true.clamp_min(1e-12)

# -------------------------
# truth generator (CAR only)
# -------------------------
def generate_dataset(graph, seed, truth_name="icar_like", rho_true=0.8):
    set_seed(seed)

    lam = graph["lam"]
    U = graph["U"]
    coords = graph["coords"]
    device = lam.device

    n = coords.shape[0]

    tau2_true = 0.4
    eps_car = 1e-3
    sigma2_true = 0.1

    F_true = make_truth_spectrum(
        truth_name=truth_name,
        lam=lam,
        tau2_true=tau2_true,
        eps_car=eps_car,
        rho_true=rho_true,
    )

    z_true = torch.sqrt(F_true) * torch.randn(n, dtype=torch.double, device=device)
    phi_true = U @ z_true

    # x_coord = coords[:, 0]
    # X = torch.stack(
    #     [torch.ones(n, dtype=torch.double, device=device), x_coord],
    #     dim=1,
    # )
    X = torch.ones(n, 1, dtype=torch.double, device=device)

    # beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    beta_true = torch.tensor([1.0], dtype=torch.double, device=device)

    y = (
        X @ beta_true
        + phi_true
        + math.sqrt(sigma2_true) * torch.randn(n, dtype=torch.double, device=device)
    )

    prior_V0 = 10.0 * torch.eye(X.shape[1], dtype=torch.double, device=device)

    return {
        "truth_name": truth_name,
        "rho_true": rho_true,
        "F_true": F_true,
        "phi_true": phi_true,
        "X": X,
        "y": y,
        "beta_true": beta_true,
        "sigma2_true": sigma2_true,
        "tau2_true": tau2_true,
        "eps_car": eps_car,
        "prior_V0": prior_V0,
    }


# -------------------------
# models to fit
# -------------------------
def get_fit_configs():
    return [
        {"filter": "classic_car", "case": "baseline"},
        {"filter": "leroux", "case": "learn_rho"},
        {"filter": "invlinear_car", "case": "baseline"},
        {"filter": "invlinear_car", "case": "fix_rho0"},
        {"filter": "rational", "case": "car_like_01"},
    ]


# -------------------------
# run experiment
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="examples/figures/car_recovery_minimal")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--truths", nargs="+", default=["icar_like", "leroux"])
    parser.add_argument("--rho_true", type=float, default=0.8)

    args = parser.parse_args()

    device = torch.device("cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(device)
    fit_configs = get_fit_configs()
    rows = []

    for truth_name in args.truths:
        print("\n" + "#" * 80)
        print(f"TRUTH: {truth_name}")
        print("#" * 80)

        for seed in args.seeds:
            print("\n" + "=" * 80)
            print(f"Running seed {seed}")
            print("=" * 80)

            dataset = generate_dataset(
                graph,
                seed=seed,
                truth_name=truth_name,
                rho_true=args.rho_true,
            )

            for cfg in fit_configs:
                spec = get_filter_spec(cfg["filter"])
                case_spec = spec.cases[cfg["case"]]

                print(f"  Fitting {cfg['filter']} / {cfg['case']}")

                summary = run_case(
                    case_spec=case_spec,
                    filter_name=spec.filter_name,
                    F_true=dataset["F_true"],
                    X=dataset["X"],
                    y=dataset["y"],
                    lam=graph["lam"],
                    U=graph["U"],
                    coords=graph["coords"],
                    phi_true=dataset["phi_true"],
                    beta_true=dataset["beta_true"],
                    sigma2_true=dataset["sigma2_true"],
                    tau2_true=dataset["tau2_true"],
                    eps_car=dataset["eps_car"],
                    prior_V0=dataset["prior_V0"],
                    device=device,
                    fig_dir=outdir / truth_name / f"seed_{seed}",
                    vi_num_iters=600 if args.fast else 2500,
                    vi_num_mc=5 if args.fast else 10,
                    vi_lr=1e-2,
                    mcmc_num_steps=8000 if args.fast else 30000,
                    mcmc_burnin=2000 if args.fast else 10000,
                    mcmc_thin=10,
                )

                row = {
                    "truth": truth_name,
                    "seed": seed,
                    "rho_true": args.rho_true if truth_name == "leroux" else np.nan,
                    "model": cfg["filter"],
                    "case": cfg["case"],
                    "rmse_phi_vi_plugin": summary["rmse_phi_vi_plugin"],
                    "rmse_phi_vi_mc": summary["rmse_phi_vi"],
                    "rmse_phi_mcmc": summary["rmse_phi_mcmc"],
                    "spec_err_vi": summary["spec_err_vi"],
                    "spec_err_mcmc": summary["spec_err_mcmc"],
                    "rmse_y_vi_plugin": summary["rmse_y_vi_plugin"],
                    "rmse_y_vi_mc": summary["rmse_y_vi_mc"],
                    "rmse_y_mcmc": summary["rmse_y_mcmc"],
                    "lpd_vi_plugin": summary["lpd_vi_plugin"],
                    "lpd_vi_mc": summary["lpd_vi_mc"],
                    "lpd_mcmc": summary["lpd_mcmc"],
                    "rmse_eta_vi_plugin": summary["rmse_eta_vi_plugin"],
                    "rmse_eta_vi_mc": summary["rmse_eta_vi"],
                    "rmse_eta_mcmc": summary["rmse_eta_mcmc"],
                }

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "results_by_seed.csv", index=False)

    group_cols = ["truth", "model", "case"]

    metric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["seed", "rho_true"]
    ]

    print(df.dtypes)
    print(df.columns.tolist())

    agg = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten MultiIndex columns
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns.to_flat_index()
    ]

    agg.to_csv(outdir / "results_aggregated.csv", index=False)

    print("\nSaved raw results:")
    print(df)
    print("\nSaved aggregated results:")
    print(agg)


if __name__ == "__main__":
    main()