from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from realdata.registry import get_dataset_spec, available_datasets
from realdata.io import ensure_dir, load_spectrum, load_extras


def available_method_dirs(out_root: Path) -> list[str]:
    """
    Return method subdirectories that contain a spectrum.csv file.
    """
    if not out_root.exists():
        return []

    method_names = []
    for child in sorted(out_root.iterdir()):
        if child.is_dir() and (child / "spectrum.csv").exists():
            method_names.append(child.name)
    return method_names


def parse_methods_arg(methods_arg: str | None, out_root: Path) -> list[str]:
    """
    If methods are explicitly provided, use them.
    Otherwise infer all available method directories.
    """
    if methods_arg is not None:
        methods = [m.strip() for m in methods_arg.split(",") if m.strip()]
        if not methods:
            raise ValueError("Parsed empty method list from --methods.")
        return methods

    inferred = available_method_dirs(out_root)
    if not inferred:
        raise ValueError(
            f"No fitted method directories with spectrum.csv found in {out_root}"
        )
    return inferred


def safe_display_name(method_dir: Path, fallback: str) -> str:
    """
    Try to recover a human-readable display name from extras.json or method_config.json.
    Fall back to the directory name.
    """
    extras_path = method_dir / "extras.json"
    if extras_path.exists():
        extras = load_extras(extras_path)
        display_name = extras.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()

    method_config_path = method_dir / "method_config.json"
    if method_config_path.exists():
        try:
            import json
            with open(method_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            display_name = cfg.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                return display_name.strip()
        except Exception:
            pass

    return fallback


def validate_spectrum_df(df: pd.DataFrame, path: Path) -> None:
    required = {"index", "lambda", "spectrum", "normalized_spectrum"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in {path}: {missing}. "
            f"Available: {df.columns.tolist()}"
        )


def load_method_spectrum(method_dir: Path, method_name: str) -> pd.DataFrame:
    """
    Load one method's spectrum.csv and append method metadata columns.
    """
    spec_path = method_dir / "spectrum.csv"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {spec_path}")

    df = load_spectrum(spec_path)
    validate_spectrum_df(df, spec_path)

    df = df.copy()
    df["method_name"] = method_name
    df["display_name"] = safe_display_name(method_dir, fallback=method_name)
    return df


def save_combined_spectrum(
    frames: list[pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    """
    Concatenate all method spectra and save to CSV.
    """
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.to_csv(out_path, index=False)
    return combined


def plot_overlay(
    combined: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    """
    Plot lambda vs chosen spectrum column for all methods.
    """
    plt.figure(figsize=(8.0, 5.2))

    for display_name, sub in combined.groupby("display_name", sort=False):
        sub = sub.sort_values("lambda")
        x = sub["lambda"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        if logy:
            y = np.maximum(y, 1e-14)

        plt.plot(x, y, linewidth=2.0, label=display_name)

    if logy:
        plt.yscale("log")

    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_by_index_overlay(
    combined: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    """
    Plot mode index vs chosen spectrum column for all methods.
    Useful when lambda spacing is uneven.
    """
    plt.figure(figsize=(8.0, 5.2))

    for display_name, sub in combined.groupby("display_name", sort=False):
        sub = sub.sort_values("index")
        x = sub["index"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        if logy:
            y = np.maximum(y, 1e-14)

        plt.plot(x, y, linewidth=2.0, label=display_name)

    if logy:
        plt.yscale("log")

    plt.xlabel("Mode index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_summary_table(combined: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Save a compact per-method summary of learned spectrum behavior.
    """
    rows = []

    for display_name, sub in combined.groupby("display_name", sort=False):
        sub = sub.sort_values("lambda")
        lam = sub["lambda"].to_numpy(dtype=float)
        spec = sub["spectrum"].to_numpy(dtype=float)
        norm_spec = sub["normalized_spectrum"].to_numpy(dtype=float)

        mask = np.isfinite(lam) & np.isfinite(spec) & np.isfinite(norm_spec)
        lam = lam[mask]
        spec = spec[mask]
        norm_spec = norm_spec[mask]

        if len(lam) == 0:
            continue

        peak_idx = int(np.argmax(np.abs(spec)))
        peak_lambda = float(lam[peak_idx])
        peak_value = float(spec[peak_idx])

        k5 = max(1, len(spec) // 20)
        k20 = max(1, len(spec) // 5)
        k50 = max(1, len(spec) // 2)

        rows.append(
            {
                "display_name": display_name,
                "n_modes": len(spec),
                "peak_lambda": peak_lambda,
                "peak_spectrum_value": peak_value,
                "mean_spectrum": float(np.mean(spec)),
                "sd_spectrum": float(np.std(spec)),
                "mean_norm_low_5pct": float(np.mean(norm_spec[:k5])),
                "mean_norm_low_20pct": float(np.mean(norm_spec[:k20])),
                "mean_norm_high_50pct": float(np.mean(norm_spec[k50:])),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned spectral filters for fitted real-data models."
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
        help=(
            "Optional comma-separated method names. "
            "If omitted, all fitted method subdirectories are used."
        ),
    )
    parser.add_argument(
        "--use_covariates",
        action="store_true",
        help="Read outputs from the with_covariates run directory.",
    )
    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)
    run_tag = "with_covariates" if args.use_covariates else "intercept_only"

    out_root = Path("realdata/figures") / dataset_spec.dataset_name / run_tag
    if not out_root.exists():
        raise FileNotFoundError(
            f"Run directory not found: {out_root}\n"
            "Run realdata/scripts/04_fit_realdata_models.py first."
        )

    methods = parse_methods_arg(args.methods, out_root=out_root)

    plot_dir = out_root / "learned_filters"
    ensure_dir(plot_dir)

    print("Dataset:", dataset_spec.dataset_name)
    print("Outcome:", dataset_spec.outcome_label)
    print("Run tag:", run_tag)
    print("Methods:", methods)
    print("Output dir:", plot_dir)

    frames: list[pd.DataFrame] = []

    for method_name in methods:
        method_dir = out_root / method_name
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")

        print(f"Loading spectrum for: {method_name}")
        df = load_method_spectrum(method_dir, method_name=method_name)
        frames.append(df)

    combined = save_combined_spectrum(
        frames=frames,
        out_path=plot_dir / "learned_filters_combined.csv",
    )

    summary = save_summary_table(
        combined=combined,
        out_path=plot_dir / "learned_filters_summary.csv",
    )

    # --------------------------------------------------
    # Main overlay plots by lambda
    # --------------------------------------------------
    plot_overlay(
        combined=combined,
        value_col="spectrum",
        ylabel="Learned spectrum",
        title=f"Learned spectral filters: {dataset_spec.dataset_name}",
        out_path=plot_dir / "learned_filters_raw.png",
        logy=False,
    )

    plot_overlay(
        combined=combined,
        value_col="normalized_spectrum",
        ylabel="Normalized learned spectrum",
        title=f"Normalized learned spectral filters: {dataset_spec.dataset_name}",
        out_path=plot_dir / "learned_filters_normalized.png",
        logy=False,
    )

    plot_overlay(
        combined=combined,
        value_col="normalized_spectrum",
        ylabel="Normalized learned spectrum (log scale)",
        title=f"Normalized learned spectral filters (log): {dataset_spec.dataset_name}",
        out_path=plot_dir / "learned_filters_normalized_log.png",
        logy=True,
    )

    # --------------------------------------------------
    # Alternative index-based plots
    # --------------------------------------------------
    plot_by_index_overlay(
        combined=combined,
        value_col="spectrum",
        ylabel="Learned spectrum",
        title=f"Learned spectral filters by mode index: {dataset_spec.dataset_name}",
        out_path=plot_dir / "learned_filters_raw_by_index.png",
        logy=False,
    )

    plot_by_index_overlay(
        combined=combined,
        value_col="normalized_spectrum",
        ylabel="Normalized learned spectrum",
        title=f"Normalized learned filters by mode index: {dataset_spec.dataset_name}",
        out_path=plot_dir / "learned_filters_normalized_by_index.png",
        logy=False,
    )

    print("\nSaved learned-filter diagnostics to:")
    print("  -", plot_dir / "learned_filters_combined.csv")
    print("  -", plot_dir / "learned_filters_summary.csv")
    print("  -", plot_dir / "learned_filters_raw.png")
    print("  -", plot_dir / "learned_filters_normalized.png")
    print("  -", plot_dir / "learned_filters_normalized_log.png")
    print("  -", plot_dir / "learned_filters_raw_by_index.png")
    print("  -", plot_dir / "learned_filters_normalized_by_index.png")

    if not summary.empty:
        print("\nSpectrum summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()