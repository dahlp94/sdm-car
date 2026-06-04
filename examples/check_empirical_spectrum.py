# examples/check_empirical_spectrum.py
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.double)


def load_pt(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    return torch.load(path, map_location="cpu", weights_only=False)


def make_equal_count_bins(
    x: torch.Tensor,
    *,
    n_bins: int = 30,
    mask: torch.Tensor | None = None,
):
    """
    Make bins with roughly equal number of eigenvalues per bin.
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)

    idx = torch.where(mask)[0]
    x_sub = x[idx]

    order = torch.argsort(x_sub)
    idx_sorted = idx[order]

    chunks = torch.chunk(idx_sorted, n_bins)

    bins = []
    for c in chunks:
        if c.numel() > 0:
            bins.append(c)

    return bins


def binned_mean(values: torch.Tensor, bins: list[torch.Tensor]) -> torch.Tensor:
    out = []
    for idx in bins:
        out.append(values[idx].mean())
    return torch.stack(out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph_pt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--vi_outputs",
        type=str,
        required=True,
        help="Path to vi_outputs.pt from run_bcef_vi.py.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graph = load_pt(args.graph_pt)
    vi = load_pt(args.vi_outputs)

    y = graph["y"].double()
    X = graph["X"].double()
    U = graph["U"].double()
    lam = graph["lam"].double()

    metrics = vi["metrics"]

    F_hat = vi["F_plugin"].double()
    sigma2_hat = float(metrics["sigma2_vi_plugin"])

    beta_hat = torch.tensor(
        metrics["beta_vi_plugin_mean"],
        dtype=torch.double,
    )

    x_axis = lam / lam.max().clamp_min(1e-12)

    # Remove fixed effects only.
    # This residual contains spatial signal + noise.
    r = y - X @ beta_hat

    # Graph Fourier residual coefficients.
    r_tilde = U.T @ r

    # Empirical observed total spectrum.
    empirical_total = r_tilde**2

    # Fitted total spectrum.
    fitted_total = F_hat + sigma2_hat

    # Avoid zero eigenvalue / intercept mode.
    mask = lam > 1e-10

    bins = make_equal_count_bins(
        x_axis,
        n_bins=args.n_bins,
        mask=mask,
    )

    x_bin = binned_mean(x_axis, bins)

    emp_bin = binned_mean(empirical_total, bins)
    fit_bin = binned_mean(fitted_total, bins)
    F_bin = binned_mean(F_hat, bins)

    # Whitened diagnostic.
    whitened_sq = empirical_total / fitted_total.clamp_min(1e-12)
    white_bin = binned_mean(whitened_sq, bins)

    # Metrics.
    rel_l2_total = (
        torch.linalg.norm(emp_bin - fit_bin)
        / torch.linalg.norm(emp_bin).clamp_min(1e-12)
    ).item()

    mean_white = white_bin.mean().item()
    max_white_dev = torch.max(torch.abs(white_bin - 1.0)).item()

    diagnostics = {
        "sigma2_hat": sigma2_hat,
        "rel_l2_binned_total_spectrum": float(rel_l2_total),
        "mean_binned_whitened_sq": float(mean_white),
        "max_abs_deviation_binned_whitened_sq_from_1": float(max_white_dev),
        "n_bins": int(len(bins)),
    }

    with open(outdir / "empirical_spectrum_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n[EMPIRICAL SPECTRUM DIAGNOSTICS]")
    print(json.dumps(diagnostics, indent=2))

    # ------------------------------------------------------------
    # Plot 1: raw empirical periodogram vs fitted total
    # ------------------------------------------------------------
    idx_sort = torch.argsort(x_axis)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(
        x_axis[idx_sort].numpy(),
        empirical_total[idx_sort].numpy(),
        s=8,
        alpha=0.25,
        label=r"Raw empirical $\tilde r_j^2$",
    )
    plt.plot(
        x_bin.numpy(),
        emp_bin.numpy(),
        linewidth=2.5,
        label="Binned empirical total",
    )
    plt.plot(
        x_bin.numpy(),
        fit_bin.numpy(),
        linewidth=2.5,
        linestyle="--",
        label=r"Binned fitted $F+\sigma^2$",
    )
    plt.yscale("log")
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel("spectral variance")
    plt.title("Empirical observed spectrum vs fitted total spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "empirical_vs_fitted_total_spectrum.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: fitted spatial spectrum and noise floor
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        x_axis[idx_sort].numpy(),
        F_hat[idx_sort].numpy(),
        linewidth=2,
        label=r"Fitted spatial $F(\lambda)$",
    )
    plt.axhline(
        sigma2_hat,
        linestyle="--",
        linewidth=2,
        label=r"Noise floor $\sigma^2$",
    )
    plt.yscale("log")
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel("variance")
    plt.title("Fitted spatial spectrum and noise floor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fitted_F_and_noise_floor.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: whitened spectral residual check
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        x_bin.numpy(),
        white_bin.numpy(),
        marker="o",
        linewidth=2,
        label=r"Binned mean $\tilde r_j^2/(F_j+\sigma^2)$",
    )
    plt.axhline(1.0, linestyle="--", linewidth=2, label="target = 1")
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel("binned whitened squared residual")
    plt.title("Whitened spectral residual diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "whitened_spectral_residuals.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: cumulative spectral energy
    # ------------------------------------------------------------
    emp_cum = torch.cumsum(empirical_total[idx_sort], dim=0)
    fit_cum = torch.cumsum(fitted_total[idx_sort], dim=0)

    emp_cum = emp_cum / emp_cum[-1].clamp_min(1e-12)
    fit_cum = fit_cum / fit_cum[-1].clamp_min(1e-12)

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        x_axis[idx_sort].numpy(),
        emp_cum.numpy(),
        linewidth=2.5,
        label="Empirical cumulative energy",
    )
    plt.plot(
        x_axis[idx_sort].numpy(),
        fit_cum.numpy(),
        linewidth=2.5,
        linestyle="--",
        label="Fitted cumulative energy",
    )
    plt.xlabel(r"$\lambda/\lambda_{\max}$")
    plt.ylabel("cumulative fraction of energy")
    plt.title("Cumulative spectral energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cumulative_spectral_energy.png", dpi=200)
    plt.close()

    print("\n[SAVED]")
    print(outdir)


if __name__ == "__main__":
    main()