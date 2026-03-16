from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot graph eigenvalues and empirical spectral energy for a real-data dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. pfas_ucmr5",
    )
    parser.add_argument(
        "--model-data",
        type=str,
        default=None,
        help="Optional explicit path to model_data.csv",
    )
    parser.add_argument(
        "--eigs",
        type=str,
        default=None,
        help="Optional explicit path to eigs.npz",
    )
    args = parser.parse_args()

    dataset = args.dataset

    model_data_path = Path(args.model_data) if args.model_data else Path(f"data/processed/{dataset}/model_data.csv")
    eig_path = Path(args.eigs) if args.eigs else Path(f"data/eigs/county_us_pfas/eigs.npz")
    out_dir = Path(f"realdata/figures/{dataset}/spectral_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_data_path.exists():
        raise FileNotFoundError(f"Model data not found: {model_data_path}")
    if not eig_path.exists():
        raise FileNotFoundError(f"Eigendecomposition file not found: {eig_path}")

    print("Loading model data:", model_data_path)
    df = pd.read_csv(model_data_path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)

    if "y" not in df.columns:
        raise KeyError(f"'y' not found in {model_data_path}. Columns: {df.columns.tolist()}")

    print("Loading eigendecomposition:", eig_path)
    eig = np.load(eig_path, allow_pickle=True)

    required = {"lam", "U", "fips"}
    missing = required - set(eig.keys())
    if missing:
        raise KeyError(f"Missing keys in {eig_path}: {missing}. Available: {list(eig.keys())}")

    lam = np.asarray(eig["lam"], dtype=float)
    U = np.asarray(eig["U"], dtype=float)
    fips_eig = pd.Series(np.asarray(eig["fips"]).astype(str)).str.zfill(5).to_numpy()

    if len(df) != len(fips_eig):
        raise ValueError(
            f"Row mismatch: model_data has {len(df)} rows but eig file has {len(fips_eig)} rows"
        )

    if not np.all(df["fips"].to_numpy() == fips_eig):
        bad = np.where(df["fips"].to_numpy() != fips_eig)[0][:10]
        raise ValueError(
            "FIPS ordering mismatch between model data and eig file. "
            f"First bad indices: {bad.tolist()}"
        )

    y = df["y"].to_numpy(dtype=float)
    y_centered = y - y.mean()

    # Graph Fourier coefficients
    coeff = U.T @ y_centered
    energy = coeff ** 2

    # Sort by eigenvalue
    idx = np.argsort(lam)
    lam_sorted = lam[idx]
    energy_sorted = energy[idx]

    # Bin spectral energy for smoother visualization
    n_bins = min(40, max(10, len(lam_sorted) // 50))
    bins = np.linspace(lam_sorted.min(), lam_sorted.max(), n_bins + 1)
    bin_ids = np.digitize(lam_sorted, bins) - 1

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            rows.append({
                "bin": b,
                "lambda_bin_center": float(lam_sorted[mask].mean()),
                "mean_energy": float(energy_sorted[mask].mean()),
                "sum_energy": float(energy_sorted[mask].sum()),
                "count": int(mask.sum()),
            })

    binned = pd.DataFrame(rows)
    binned.to_csv(out_dir / "empirical_energy_binned.csv", index=False)

    # --------------------------------------------------
    # Plot 1: eigenvalue spectrum
    # --------------------------------------------------
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(np.arange(len(lam_sorted)), lam_sorted, linewidth=1.8)
    plt.xlabel("Eigenvalue index")
    plt.ylabel(r"Eigenvalue $\lambda_k$")
    plt.title(f"Graph Laplacian spectrum: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "graph_eigenvalues.png", dpi=220)
    plt.close()

    # --------------------------------------------------
    # Plot 2: raw empirical spectral energy
    # --------------------------------------------------
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(lam_sorted, energy_sorted, linewidth=1.0)
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel(r"Empirical energy $(u_k^\top y)^2$")
    plt.title(f"Raw empirical spectral energy: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "empirical_energy_raw.png", dpi=220)
    plt.close()

    # --------------------------------------------------
    # Plot 3: binned empirical spectral energy
    # --------------------------------------------------
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(
        binned["lambda_bin_center"],
        binned["mean_energy"],
        marker="o",
        linewidth=2.0,
        markersize=4,
    )
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel("Mean empirical energy")
    plt.title(f"Binned empirical spectral energy: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "empirical_energy_binned.png", dpi=220)
    plt.close()

    # --------------------------------------------------
    # Plot 4: binned energy on log scale
    # --------------------------------------------------
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(
        binned["lambda_bin_center"],
        np.maximum(binned["mean_energy"], 1e-12),
        marker="o",
        linewidth=2.0,
        markersize=4,
    )
    plt.yscale("log")
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel("Mean empirical energy (log scale)")
    plt.title(f"Binned empirical spectral energy (log): {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "empirical_energy_binned_log.png", dpi=220)
    plt.close()

    print("\nSaved spectral diagnostics to:", out_dir)
    print("Files:")
    print("  -", out_dir / "graph_eigenvalues.png")
    print("  -", out_dir / "empirical_energy_raw.png")
    print("  -", out_dir / "empirical_energy_binned.png")
    print("  -", out_dir / "empirical_energy_binned_log.png")
    print("  -", out_dir / "empirical_energy_binned.csv")

    print("\nOutcome summary:")
    print(pd.Series(y).describe())

    print("\nTop 10 low-frequency energies:")
    low_df = pd.DataFrame({
        "lambda": lam_sorted[:10],
        "energy": energy_sorted[:10],
    })
    print(low_df)

    print("\nTotal energy:", float(energy.sum()))
    print("Fraction of energy in first 5% of modes:",
          float(energy_sorted[:max(1, len(energy_sorted)//20)].sum() / energy.sum()))


if __name__ == "__main__":
    main()