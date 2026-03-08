# examples/run_scaling_demo.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import math
import random
import argparse

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from sdmcar.graph import build_laplacian_from_knn, laplacian_eigendecomp
from sdmcar.models import SpectralCAR_FullVI

# ensure registry is populated
from examples.benchmarks.registry import get_filter_spec
import examples.benchmarks  # noqa: F401


torch.set_default_dtype(torch.double)


# -------------------------
# Dataclasses
# -------------------------
@dataclass(frozen=True)
class BenchmarkSpec:
    method: str             # "exact" or "cheby"
    grid_nx: int
    grid_ny: int
    filter_name: str
    case_id: str
    seed: int


@dataclass
class BenchmarkResult:
    method: str
    filter_name: str
    case_id: str
    seed: int

    grid_nx: int
    grid_ny: int
    n: int
    num_edges: int

    graph_time_sec: float
    eig_time_sec: Optional[float]
    fit_time_sec: float
    total_time_sec: float

    memory_est_mb: Optional[float]

    phi_rmse: Optional[float]
    pll: Optional[float]
    sigma2_hat: Optional[float]

    approx_error_spec: Optional[float] = None
    approx_error_phi: Optional[float] = None
    degree_K: Optional[int] = None


# -------------------------
# Basic helpers
# -------------------------
def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def now() -> float:
    return time.perf_counter()


def make_grid_coords(nx: int, ny: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, nx, dtype=torch.double, device=device)
    ys = torch.linspace(0.0, 1.0, ny, dtype=torch.double, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    return coords


def make_design_matrix(coords: torch.Tensor) -> torch.Tensor:
    x_coord = coords[:, 0]
    X = torch.stack(
        [
            torch.ones(coords.shape[0], dtype=torch.double, device=coords.device),
            x_coord,
        ],
        dim=1,
    )
    return X


def car_like_truth_spectrum(
    lam: torch.Tensor,
    *,
    eps_car: float,
    tau2_true: float,
) -> torch.Tensor:
    lam = lam.clamp_min(0.0)
    F = tau2_true / (lam + eps_car)
    return F.clamp_min(1e-12)


def simulate_data(
    *,
    X: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    sigma2_true: float,
    beta_true: torch.Tensor,
    eps_car: float,
    tau2_true: float,
    signal: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        y
        phi_true
        F_true
    """
    if signal != "moderate_car":
        raise ValueError(f"Unknown signal '{signal}'. Expected 'moderate_car'.")

    F_true = car_like_truth_spectrum(
        lam,
        eps_car=eps_car,
        tau2_true=tau2_true,
    )
    z_true = torch.sqrt(F_true) * torch.randn_like(lam)
    phi_true = U @ z_true
    y = X @ beta_true + phi_true + math.sqrt(sigma2_true) * torch.randn(
        X.shape[0],
        dtype=torch.double,
        device=X.device,
    )
    return y, phi_true, F_true


# -------------------------
# VI phi helper
# -------------------------
@torch.no_grad()
def phi_full_from_full_data(
    *,
    U: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    F: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    r = y - X @ beta
    Ut_r = U.T @ r
    shrink = (F / (F + sigma2)).clamp(0.0, 1.0)
    mu_z = shrink * Ut_r
    return U @ mu_z


@torch.no_grad()
def spectrum_vi_mc_mean(
    filter_module,
    lam: torch.Tensor,
    *,
    S: int = 128,
) -> torch.Tensor:
    acc = torch.zeros_like(lam)
    for _ in range(S):
        theta = filter_module.sample_unconstrained()
        acc += filter_module.spectrum(lam, theta).clamp_min(1e-12)
    return acc / float(S)


@torch.no_grad()
def compute_phi_vi_full_plugin(
    *,
    model: SpectralCAR_FullVI,
    lam: torch.Tensor,
    U: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    num_mc_F: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    beta_mean, _, sigma2_plugin, _ = model.beta_posterior_plugin()
    beta_mean = beta_mean.reshape(-1)
    sigma2 = float(sigma2_plugin.detach().cpu())

    F_used = spectrum_vi_mc_mean(model.filter, lam, S=num_mc_F).detach()
    phi_mean = phi_full_from_full_data(
        U=U,
        X=X,
        y=y,
        beta=beta_mean,
        F=F_used.to(lam.device),
        sigma2=sigma2,
    )
    return phi_mean.detach(), F_used.detach(), sigma2


# -------------------------
# Memory helper
# -------------------------
def estimate_exact_memory_mb(
    *,
    L: Optional[torch.Tensor],
    W: Optional[torch.Tensor],
    U: Optional[torch.Tensor],
    lam: Optional[torch.Tensor],
    X: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
) -> float:
    total_bytes = 0
    for obj in [L, W, U, lam, X, y]:
        if obj is not None:
            total_bytes += obj.numel() * obj.element_size()
    return total_bytes / (1024.0 ** 2)


# -------------------------
# Exact benchmark runner
# -------------------------
def run_exact_benchmark(
    *,
    spec: BenchmarkSpec,
    device: torch.device,
    k: int,
    gamma: float,
    rho_graph: float,
    vi_iters: int,
    vi_lr: float,
    vi_mc: int,
    sigma2_true: float,
    tau2_true: float,
    eps_car: float,
    signal: str,
    outdir: Path,
    vi_log_every: int,
) -> BenchmarkResult:
    set_all_seeds(spec.seed)

    coords = make_grid_coords(spec.grid_nx, spec.grid_ny, device)
    n = coords.shape[0]

    # graph construction
    t_graph = now()
    L, W = build_laplacian_from_knn(coords, k=k, gamma=gamma, rho=rho_graph)
    graph_time = now() - t_graph

    # eigendecomposition
    t_eig = now()
    lam, U = laplacian_eigendecomp(L)
    lam = lam.to(device)
    U = U.to(device)
    eig_time = now() - t_eig

    # Data Generating Process (DGP)
    X = make_design_matrix(coords)
    beta_true = torch.tensor([1.0, -0.5], dtype=torch.double, device=device)
    y, phi_true, _ = simulate_data(
        X=X,
        U=U,
        lam=lam,
        sigma2_true=sigma2_true,
        beta_true=beta_true,
        eps_car=eps_car,
        tau2_true=tau2_true,
        signal=signal,
    )

    # prior on beta
    sigma2_beta = 10.0
    prior_V0 = sigma2_beta * torch.eye(X.shape[1], dtype=torch.double, device=device)

    # filter / model
    filter_spec = get_filter_spec(spec.filter_name)
    if spec.case_id not in filter_spec.cases:
        raise ValueError(
            f"Case '{spec.case_id}' not found for filter '{spec.filter_name}'. "
            f"Available: {list(filter_spec.cases.keys())}"
        )
    case_spec = filter_spec.cases[spec.case_id]

    filter_module = case_spec.build_filter(
        tau2_true=tau2_true,
        eps_car=eps_car,
        lam_max=float(lam.max().detach().cpu()),
        device=device,
        **case_spec.fixed,
    )

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

    # VI fit
    t_fit = now()
    opt = torch.optim.Adam(model.parameters(), lr=vi_lr)
    for it in range(1, vi_iters + 1):
        opt.zero_grad()
        elbo, _ = model.elbo()
        (-elbo).backward()
        opt.step()

        if vi_log_every and (it % vi_log_every == 0 or it == 1 or it == vi_iters):
            print(
                f"  [VI] n={n:>5} iter {it:>5}/{vi_iters}  "
                f"ELBO={float(elbo.detach().cpu()):.3f}"
            )
    fit_time = now() - t_fit

    # posterior summaries
    with torch.no_grad():
        phi_mean, _, sigma2_hat = compute_phi_vi_full_plugin(
            model=model,
            lam=lam,
            U=U,
            X=X,
            y=y,
            num_mc_F=128,
        )
        phi_rmse = float(torch.sqrt(torch.mean((phi_mean - phi_true) ** 2)).cpu().item())

    total_time = graph_time + eig_time + fit_time
    memory_est_mb = estimate_exact_memory_mb(
        L=L,
        W=W,
        U=U,
        lam=lam,
        X=X,
        y=y,
    )

    num_edges = int((W > 0).sum().item())

    return BenchmarkResult(
        method=spec.method,
        filter_name=spec.filter_name,
        case_id=spec.case_id,
        seed=spec.seed,
        grid_nx=spec.grid_nx,
        grid_ny=spec.grid_ny,
        n=n,
        num_edges=num_edges,
        graph_time_sec=float(graph_time),
        eig_time_sec=float(eig_time),
        fit_time_sec=float(fit_time),
        total_time_sec=float(total_time),
        memory_est_mb=float(memory_est_mb),
        phi_rmse=float(phi_rmse),
        pll=None,
        sigma2_hat=float(sigma2_hat),
        approx_error_spec=None,
        approx_error_phi=None,
        degree_K=None,
    )


# -------------------------
# Chebyshev placeholder
# -------------------------
def run_cheby_benchmark(
    *,
    spec: BenchmarkSpec,
    device: torch.device,
    k: int,
    gamma: float,
    rho_graph: float,
    vi_iters: int,
    vi_lr: float,
    vi_mc: int,
    sigma2_true: float,
    tau2_true: float,
    eps_car: float,
    signal: str,
    degree_K: int,
    outdir: Path,
    vi_log_every: int,
) -> BenchmarkResult:
    raise NotImplementedError("Chebyshev benchmark not implemented yet.")


# -------------------------
# Save / plotting helpers
# -------------------------
def save_results(results: List[BenchmarkResult], save_dir: Path) -> pd.DataFrame:
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(save_dir / "scaling_results.csv", index=False)
    return df


def write_summary(df: pd.DataFrame, save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Scaling benchmark summary\n")
        f.write("=" * 80 + "\n")
        if df.empty:
            f.write("No results.\n")
        else:
            f.write(df.to_string(index=False))
            f.write("\n")


def plot_runtime(df: pd.DataFrame, save_path: Path) -> None:
    if df.empty:
        return

    plt.figure(figsize=(6.0, 4.2))
    for method, dsub in df.groupby("method"):
        dsub = dsub.sort_values("n")
        plt.plot(dsub["n"], dsub["total_time_sec"], marker="o", label=method)
    plt.xlabel("n")
    plt.ylabel("total runtime (sec)")
    plt.title("Scaling benchmark: total runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_exact_breakdown(df: pd.DataFrame, save_path: Path) -> None:
    dsub = df[df["method"] == "exact"].copy()
    if dsub.empty:
        return

    dsub = dsub.sort_values("n")
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(dsub["n"], dsub["graph_time_sec"], marker="o", label="graph")
    plt.plot(dsub["n"], dsub["eig_time_sec"], marker="o", label="eigendecomp")
    plt.plot(dsub["n"], dsub["fit_time_sec"], marker="o", label="fit")
    plt.xlabel("n")
    plt.ylabel("runtime (sec)")
    plt.title("Exact SDM-CAR runtime breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_memory(df: pd.DataFrame, save_path: Path) -> None:
    if df.empty or "memory_est_mb" not in df.columns:
        return

    plt.figure(figsize=(6.0, 4.2))
    for method, dsub in df.groupby("method"):
        dsub = dsub.sort_values("n")
        plt.plot(dsub["n"], dsub["memory_est_mb"], marker="o", label=method)
    plt.xlabel("n")
    plt.ylabel("estimated memory (MB)")
    plt.title("Scaling benchmark: memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    # methods
    p.add_argument("--methods", nargs="+", default=["exact"], help="Benchmark methods: exact, cheby")

    # graph sizes as side lengths
    p.add_argument("--grid_sizes", nargs="+", type=int, default=[20, 30, 40, 50, 70])

    # graph construction
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--rho_graph", type=float, default=0.95)

    # model / filter
    p.add_argument("--filter", default="multiscale_bump")
    p.add_argument("--case", default="k2")

    # DGP
    p.add_argument("--signal", default="moderate_car", choices=["moderate_car"])
    p.add_argument("--sigma2_true", type=float, default=0.10)
    p.add_argument("--tau2_true", type=float, default=0.20)
    p.add_argument("--eps_car", type=float, default=1e-3)

    # VI
    p.add_argument("--vi_iters", type=int, default=1000)
    p.add_argument("--vi_lr", type=float, default=1e-2)
    p.add_argument("--vi_mc", type=int, default=8)
    p.add_argument("--vi_log_every", type=int, default=100)

    # approximation placeholder
    p.add_argument("--degree_list", nargs="+", type=int, default=[5, 10, 20])

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(Path("examples") / "figures" / "scaling_demo"))

    args = p.parse_args()

    device = torch.device("cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: List[BenchmarkResult] = []

    for g in args.grid_sizes:
        for method in args.methods:
            spec = BenchmarkSpec(
                method=method,
                grid_nx=g,
                grid_ny=g,
                filter_name=args.filter,
                case_id=args.case,
                seed=args.seed,
            )

            print("\n" + "=" * 80)
            print(f"METHOD={method} | GRID={g}x{g} | FILTER={args.filter}:{args.case}")
            print("=" * 80)

            if method == "exact":
                res = run_exact_benchmark(
                    spec=spec,
                    device=device,
                    k=args.k,
                    gamma=args.gamma,
                    rho_graph=args.rho_graph,
                    vi_iters=args.vi_iters,
                    vi_lr=args.vi_lr,
                    vi_mc=args.vi_mc,
                    sigma2_true=float(args.sigma2_true),
                    tau2_true=float(args.tau2_true),
                    eps_car=float(args.eps_car),
                    signal=args.signal,
                    outdir=outdir,
                    vi_log_every=args.vi_log_every,
                )
                results.append(res)

            elif method == "cheby":
                for K in args.degree_list:
                    res = run_cheby_benchmark(
                        spec=spec,
                        device=device,
                        k=args.k,
                        gamma=args.gamma,
                        rho_graph=args.rho_graph,
                        vi_iters=args.vi_iters,
                        vi_lr=args.vi_lr,
                        vi_mc=args.vi_mc,
                        sigma2_true=float(args.sigma2_true),
                        tau2_true=float(args.tau2_true),
                        eps_car=float(args.eps_car),
                        signal=args.signal,
                        degree_K=int(K),
                        outdir=outdir,
                        vi_log_every=args.vi_log_every,
                    )
                    results.append(res)

            else:
                raise ValueError(f"Unknown method '{method}'")

    df = save_results(results, outdir)
    write_summary(df, outdir / "summary.txt")

    if not df.empty:
        plot_runtime(df, outdir / "runtime_total.png")
        plot_memory(df, outdir / "memory.png")
        if "exact" in set(df["method"]):
            plot_exact_breakdown(df, outdir / "runtime_breakdown_exact.png")

    print("\n" + "=" * 80)
    print("SCALING BENCHMARK SUMMARY")
    print("=" * 80)
    if df.empty:
        print("No results.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()