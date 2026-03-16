from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def load_data(model_data_path: Path, eig_path: Path):
    if not model_data_path.exists():
        raise FileNotFoundError(f"Model data not found: {model_data_path}")
    if not eig_path.exists():
        raise FileNotFoundError(f"Eigendecomposition file not found: {eig_path}")

    df = pd.read_csv(model_data_path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)

    if "y" not in df.columns:
        raise KeyError(f"'y' not found in {model_data_path}. Columns: {df.columns.tolist()}")

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

    return df, lam, U


def compute_spectral_quantities(y: np.ndarray, lam: np.ndarray, U: np.ndarray):
    y_centered = y - y.mean()
    coeff = U.T @ y_centered
    energy = coeff ** 2

    idx = np.argsort(lam)
    lam_sorted = lam[idx]
    coeff_sorted = coeff[idx]
    energy_sorted = energy[idx]

    total_energy = float(energy_sorted.sum())
    if total_energy <= 0:
        raise ValueError("Total spectral energy is non-positive.")

    energy_frac = energy_sorted / total_energy
    cumulative_energy = np.cumsum(energy_frac)

    return {
        "y_centered": y_centered,
        "coeff_sorted": coeff_sorted,
        "lam_sorted": lam_sorted,
        "energy_sorted": energy_sorted,
        "energy_frac": energy_frac,
        "cumulative_energy": cumulative_energy,
        "total_energy": total_energy,
    }


def summarize_bands(lam_sorted: np.ndarray, energy_frac: np.ndarray):
    n = len(lam_sorted)
    k5 = max(1, n // 20)     # first 5%
    k20 = max(1, n // 5)     # first 20%
    k50 = max(1, n // 2)     # first 50%

    low_5 = float(energy_frac[:k5].sum())
    low_20 = float(energy_frac[:k20].sum())
    low_50 = float(energy_frac[:k50].sum())
    mid_20_50 = float(energy_frac[k20:k50].sum())
    high_50_100 = float(energy_frac[k50:].sum())

    return {
        "low_5_pct_modes": low_5,
        "low_20_pct_modes": low_20,
        "low_50_pct_modes": low_50,
        "mid_20_to_50_pct_modes": mid_20_50,
        "high_50_to_100_pct_modes": high_50_100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot normalized signal spectrum and cumulative energy for a real-data dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. pfas_ucmr5_top3_log")
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
    eig_path = Path(args.eigs) if args.eigs else Path("data/eigs/county_us_pfas/eigs.npz")

    out_dir = Path(f"realdata/figures/{dataset}/spectral_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model data:", model_data_path)
    print("Loading eigendecomposition:", eig_path)

    df, lam, U = load_data(model_data_path, eig_path)
    y = df["y"].to_numpy(dtype=float)

    spec = compute_spectral_quantities(y, lam, U)
    bands = summarize_bands(spec["lam_sorted"], spec["energy_frac"])

    spectrum_df = pd.DataFrame({
        "mode_index": np.arange(len(spec["lam_sorted"])),
        "lambda": spec["lam_sorted"],
        "coeff": spec["coeff_sorted"],
        "energy": spec["energy_sorted"],
        "energy_fraction": spec["energy_frac"],
        "cumulative_energy": spec["cumulative_energy"],
    })
    spectrum_df.to_csv(out_dir / "signal_spectrum.csv", index=False)

    summary_df = pd.DataFrame([{
        "dataset": dataset,
        "n_modes": len(spec["lam_sorted"]),
        "total_energy": spec["total_energy"],
        **bands,
    }])
    summary_df.to_csv(out_dir / "signal_spectrum_summary.csv", index=False)

    # Plot 1: normalized spectral energy
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(spec["lam_sorted"], spec["energy_frac"], linewidth=1.2)
    plt.xlabel(r"Eigenvalue $\lambda_k$")
    plt.ylabel("Normalized energy")
    plt.title(f"Normalized signal spectrum: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "signal_spectrum_normalized.png", dpi=220)
    plt.close()

    # Plot 2: cumulative energy vs mode index
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(np.arange(len(spec["cumulative_energy"])), spec["cumulative_energy"], linewidth=2.0)
    plt.axvline(max(1, len(spec["cumulative_energy"]) // 20), linestyle="--", linewidth=1.0)
    plt.axvline(max(1, len(spec["cumulative_energy"]) // 5), linestyle="--", linewidth=1.0)
    plt.xlabel("Mode index (sorted by eigenvalue)")
    plt.ylabel("Cumulative energy fraction")
    plt.title(f"Cumulative spectral energy: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "signal_spectrum_cumulative.png", dpi=220)
    plt.close()

    # Plot 3: cumulative energy vs eigenvalue
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(spec["lam_sorted"], spec["cumulative_energy"], linewidth=2.0)
    plt.xlabel(r"Eigenvalue $\lambda_k$")
    plt.ylabel("Cumulative energy fraction")
    plt.title(f"Cumulative energy by graph frequency: {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "signal_spectrum_cumulative_vs_lambda.png", dpi=220)
    plt.close()

    # Plot 4: log-scale normalized spectral energy
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(spec["lam_sorted"], np.maximum(spec["energy_frac"], 1e-14), linewidth=1.2)
    plt.yscale("log")
    plt.xlabel(r"Eigenvalue $\lambda_k$")
    plt.ylabel("Normalized energy (log scale)")
    plt.title(f"Normalized signal spectrum (log): {dataset}")
    plt.tight_layout()
    plt.savefig(out_dir / "signal_spectrum_normalized_log.png", dpi=220)
    plt.close()

    print("\nSaved signal spectrum diagnostics to:", out_dir)
    print("Files:")
    print("  -", out_dir / "signal_spectrum.csv")
    print("  -", out_dir / "signal_spectrum_summary.csv")
    print("  -", out_dir / "signal_spectrum_normalized.png")
    print("  -", out_dir / "signal_spectrum_cumulative.png")
    print("  -", out_dir / "signal_spectrum_cumulative_vs_lambda.png")
    print("  -", out_dir / "signal_spectrum_normalized_log.png")

    print("\nBand summary:")
    for k, v in bands.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()