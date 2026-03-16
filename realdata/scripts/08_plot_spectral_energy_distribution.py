from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from realdata.registry import get_dataset_spec, available_datasets
from realdata.io import ensure_dir, load_spectrum


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def available_method_dirs(out_root: Path) -> list[str]:

    methods = []

    for child in sorted(out_root.iterdir()):
        if child.is_dir() and (child / "spectrum.csv").exists():
            methods.append(child.name)

    return methods


def parse_methods(methods_arg: str | None, out_root: Path):

    if methods_arg is None:
        return available_method_dirs(out_root)

    return [m.strip() for m in methods_arg.split(",") if m.strip()]


def spectral_energy(spectrum: np.ndarray):

    energy = spectrum ** 2
    total = energy.sum()

    if total == 0:
        return energy

    return energy / total


# --------------------------------------------------
# Compute energy table
# --------------------------------------------------

def build_energy_table(methods, out_root):

    rows = []

    for method in methods:

        path = out_root / method / "spectrum.csv"

        df = load_spectrum(path)

        lam = df["lambda"].values
        spec = df["spectrum"].values

        energy = spectral_energy(spec)

        for i in range(len(energy)):
            rows.append(
                {
                    "method": method,
                    "index": i,
                    "lambda": lam[i],
                    "energy": energy[i],
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Plots
# --------------------------------------------------

def plot_energy_by_lambda(df, out_path):

    plt.figure(figsize=(8,5))

    for method, sub in df.groupby("method"):

        sub = sub.sort_values("lambda")

        plt.plot(
            sub["lambda"],
            sub["energy"],
            linewidth=2,
            label=method
        )

    plt.xlabel("Laplacian eigenvalue λ")
    plt.ylabel("Spectral energy")
    plt.title("Spectral energy distribution across methods")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_energy_cdf(df, out_path):

    plt.figure(figsize=(8,5))

    for method, sub in df.groupby("method"):

        sub = sub.sort_values("lambda")

        cdf = np.cumsum(sub["energy"])

        plt.plot(
            sub["lambda"],
            cdf,
            linewidth=2,
            label=method
        )

    plt.xlabel("Laplacian eigenvalue λ")
    plt.ylabel("Cumulative spectral energy")
    plt.title("Cumulative spectral energy")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_energy_by_mode(df, out_path):

    plt.figure(figsize=(8,5))

    for method, sub in df.groupby("method"):

        sub = sub.sort_values("index")

        plt.plot(
            sub["index"],
            sub["energy"],
            linewidth=2,
            label=method
        )

    plt.xlabel("Mode index")
    plt.ylabel("Spectral energy")
    plt.title("Spectral energy by mode index")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Plot spectral energy allocation of learned filters."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Available: {available_datasets()}",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma separated method names"
    )

    parser.add_argument(
        "--use_covariates",
        action="store_true"
    )

    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)

    run_tag = "with_covariates" if args.use_covariates else "intercept_only"

    out_root = Path("realdata/figures") / dataset_spec.dataset_name / run_tag

    methods = parse_methods(args.methods, out_root)

    plot_dir = out_root / "spectral_energy"

    ensure_dir(plot_dir)

    print("Dataset:", dataset_spec.dataset_name)
    print("Methods:", methods)

    # --------------------------------------------------
    # Build energy table
    # --------------------------------------------------

    df = build_energy_table(methods, out_root)

    df.to_csv(plot_dir / "spectral_energy_table.csv", index=False)

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    plot_energy_by_lambda(
        df,
        plot_dir / "spectral_energy_vs_lambda.png"
    )

    plot_energy_by_mode(
        df,
        plot_dir / "spectral_energy_vs_mode.png"
    )

    plot_energy_cdf(
        df,
        plot_dir / "spectral_energy_cdf.png"
    )

    print("\nSaved spectral diagnostics to:")
    print(plot_dir)


if __name__ == "__main__":
    main()