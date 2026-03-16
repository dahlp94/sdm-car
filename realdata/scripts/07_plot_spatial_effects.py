from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geopandas as gpd
import pandas as pd

from realdata.registry import get_dataset_spec, available_datasets
from realdata.io import ensure_dir, load_predictions


# --------------------------------------------------
# Helper
# --------------------------------------------------

def available_method_dirs(out_root: Path) -> list[str]:
    if not out_root.exists():
        return []

    methods = []
    for child in sorted(out_root.iterdir()):
        if child.is_dir() and (child / "predictions.csv").exists():
            methods.append(child.name)

    return methods


def parse_methods_arg(methods_arg: str | None, out_root: Path) -> list[str]:
    if methods_arg is not None:
        return [m.strip() for m in methods_arg.split(",") if m.strip()]

    inferred = available_method_dirs(out_root)

    if not inferred:
        raise ValueError(f"No fitted method directories found in {out_root}")

    return inferred


def prepare_county_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Load county shapefile and construct zero-padded county FIPS.
    """
    counties = gpd.read_file(shapefile_path)

    required = {"STATEFP", "COUNTYFP"}
    missing = required - set(counties.columns)
    if missing:
        raise KeyError(
            f"County shapefile is missing required columns: {missing}. "
            f"Available columns: {counties.columns.tolist()}"
        )

    counties = counties.copy()
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties["COUNTYFP"] = counties["COUNTYFP"].astype(str).str.zfill(3)
    counties["fips"] = (counties["STATEFP"] + counties["COUNTYFP"]).astype(str).str.zfill(5)

    return counties


def prepare_predictions(pred_file: Path) -> pd.DataFrame:
    """
    Load predictions and force FIPS to zero-padded string.
    """
    df = load_predictions(pred_file).copy()

    if "fips" not in df.columns:
        raise KeyError(
            f"'fips' not found in predictions file: {pred_file}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    out_path: Path,
) -> None:
    if column not in gdf.columns:
        raise KeyError(
            f"Column '{column}' not found in merged GeoDataFrame. "
            f"Available columns: {gdf.columns.tolist()}"
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    gdf.plot(
        column=column,
        cmap="viridis",
        linewidth=0,
        legend=True,
        ax=ax,
    )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial effects and residual maps for fitted real-data models."
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
        help="Comma separated method names. If omitted, all fitted methods are used.",
    )

    parser.add_argument(
        "--use_covariates",
        action="store_true",
    )

    parser.add_argument(
        "--county_shapefile",
        required=True,
        help="Path to US county shapefile",
    )

    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)
    run_tag = "with_covariates" if args.use_covariates else "intercept_only"

    out_root = Path("realdata/figures") / dataset_spec.dataset_name / run_tag
    plot_dir = out_root / "spatial_maps"
    ensure_dir(plot_dir)

    methods = parse_methods_arg(args.methods, out_root)

    print("Dataset:", dataset_spec.dataset_name)
    print("Methods:", methods)
    print("Run tag:", run_tag)

    # --------------------------------------------------
    # Load shapefile
    # --------------------------------------------------
    counties = prepare_county_shapefile(args.county_shapefile)

    # --------------------------------------------------
    # Loop methods
    # --------------------------------------------------
    for method in methods:
        method_dir = out_root / method
        pred_file = method_dir / "predictions.csv"

        if not pred_file.exists():
            raise FileNotFoundError(pred_file)

        df = prepare_predictions(pred_file)

        print("\nMethod:", method)
        print("Shapefile fips dtype:", counties["fips"].dtype)
        print("Predictions fips dtype:", df["fips"].dtype)
        print("Example shapefile fips:", counties["fips"].head().tolist())
        print("Example predictions fips:", df["fips"].head().tolist())

        gdf = counties.merge(df, on="fips", how="inner")

        print(method, "n counties after merge:", len(gdf))

        if len(gdf) == 0:
            raise ValueError(
                f"No counties matched for method '{method}'. "
                "Check FIPS formatting and graph/shapefile coverage."
            )

        # ----------------------------------------
        # Spatial effect
        # ----------------------------------------
        plot_map(
            gdf,
            "spatial_effect_mean",
            f"{method} spatial random effect",
            plot_dir / f"{method}_spatial_effect.png",
        )

        # ----------------------------------------
        # Residual map
        # ----------------------------------------
        plot_map(
            gdf,
            "residual",
            f"{method} residual map",
            plot_dir / f"{method}_residual.png",
        )

        # ----------------------------------------
        # Fitted mean
        # ----------------------------------------
        plot_map(
            gdf,
            "fitted_mean",
            f"{method} fitted outcome",
            plot_dir / f"{method}_fitted.png",
        )

    print("\nSaved maps to:")
    print(plot_dir)


if __name__ == "__main__":
    main()