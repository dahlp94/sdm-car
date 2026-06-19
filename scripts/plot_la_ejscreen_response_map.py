from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import sparse


DEFAULT_OUT_DIR = Path("data/processed")


def safe_name(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def make_paths(
    *,
    out_dir: Path,
    response: str,
    transform: str,
    prefix: str | None = None,
) -> dict[str, Path]:
    if prefix is None:
        stem = f"la_ejscreen_{response.lower()}_{transform.lower()}"
        base = out_dir / stem
    else:
        base = Path(prefix)

    return {
        "base": base,
        "gpkg": Path(str(base) + "_tracts.gpkg"),
        "adj": Path(str(base) + "_queen_adjacency.npz"),
        "metadata": Path(str(base) + "_metadata.json"),
    }


def infer_y_col(
    gdf: gpd.GeoDataFrame,
    *,
    metadata_path: Path | None = None,
    y_col: str | None = None,
    response: str | None = None,
    transform: str | None = None,
) -> str:
    if y_col is not None:
        if y_col not in gdf.columns:
            raise ValueError(f"--y-col {y_col!r} not found in columns.")
        return y_col

    if metadata_path is not None and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if "y_col" in meta and meta["y_col"] in gdf.columns:
            return meta["y_col"]

    if response is not None and transform is not None:
        candidate = f"y_{response.lower()}_{transform.lower()}"
        if candidate in gdf.columns:
            return candidate

    y_cols = [c for c in gdf.columns if c.lower().startswith("y_")]
    if len(y_cols) == 1:
        return y_cols[0]

    raise ValueError(
        "Could not infer response column. "
        f"Candidate y_ columns: {y_cols}. Use --y-col explicitly."
    )


def response_label(response: str, transform: str, y_col: str) -> str:
    r = response.upper()

    if r == "PTRAF":
        label = "Traffic proximity score"
    elif r == "PTSDF":
        label = "Hazardous waste proximity score"
    elif r == "NO2":
        label = "NO₂ concentration / transformed NO₂"
    else:
        label = y_col

    if transform.lower() != "identity":
        label = f"{label} ({transform})"

    return label


def add_response_diagnostics(
    gdf: gpd.GeoDataFrame,
    y_col: str,
    *,
    n_quantiles: int = 5,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    y = pd.to_numeric(gdf[y_col], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(y).all():
        raise ValueError(f"{y_col} contains non-finite values.")

    z = (y - y.mean()) / y.std(ddof=1)

    gdf["map_y"] = y
    gdf["map_z"] = z

    # Quantile class: 1, ..., n_quantiles
    q = pd.qcut(
        y,
        q=n_quantiles,
        labels=False,
        duplicates="drop",
    )

    gdf["map_quantile"] = q.astype(int) + 1

    q10 = np.quantile(y, 0.10)
    q90 = np.quantile(y, 0.90)

    group = np.full(len(y), "Middle 80%", dtype=object)
    group[y <= q10] = "Bottom 10%"
    group[y >= q90] = "Top 10%"

    gdf["map_tail_group"] = group

    return gdf


def add_neighbor_cluster_flags(
    gdf: gpd.GeoDataFrame,
    A: sparse.spmatrix,
    *,
    z_col: str = "map_z",
    high_q: float = 0.90,
    low_q: float = 0.10,
) -> gpd.GeoDataFrame:
    """
    Simple visual cluster flag, not a formal local Moran test.

    High-high candidate:
        tract value is high and neighbor average is high.

    Low-low candidate:
        tract value is low and neighbor average is low.
    """
    gdf = gdf.copy()

    if A.shape[0] != len(gdf):
        raise ValueError(
            f"Adjacency shape {A.shape} does not match GeoDataFrame rows {len(gdf)}."
        )

    A = A.tocsr()

    z = pd.to_numeric(gdf[z_col], errors="coerce").to_numpy(dtype=float)

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)

    neigh_mean = np.asarray(A @ z).ravel() / deg_safe

    hi = np.quantile(z, high_q)
    lo = np.quantile(z, low_q)

    cluster = np.full(len(z), "Not flagged", dtype=object)

    cluster[(z >= hi) & (neigh_mean >= hi)] = "High-high candidate"
    cluster[(z <= lo) & (neigh_mean <= lo)] = "Low-low candidate"

    # Optional discordant areas
    cluster[(z >= hi) & (neigh_mean <= lo)] = "High-low candidate"
    cluster[(z <= lo) & (neigh_mean >= hi)] = "Low-high candidate"

    gdf["map_neighbor_mean_z"] = neigh_mean
    gdf["map_cluster_candidate"] = cluster

    return gdf


def save_continuous_map(
    gdf: gpd.GeoDataFrame,
    *,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    edgecolor: str = "black",
    linewidth: float = 0.03,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        linewidth=linewidth,
        edgecolor=edgecolor,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_categorical_map(
    gdf: gpd.GeoDataFrame,
    *,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "tab10",
    edgecolor: str = "black",
    linewidth: float = 0.03,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        categorical=True,
        cmap=cmap,
        linewidth=linewidth,
        edgecolor=edgecolor,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_histogram(
    gdf: gpd.GeoDataFrame,
    *,
    column: str,
    title: str,
    output_path: Path,
    xlabel: str,
) -> None:
    x = pd.to_numeric(gdf[column], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(x, bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--response",
        type=str,
        default="ptraf",
        help="Response name used in processed files: ptraf, ptsdf, no2.",
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="yeojohnson",
        help="Transform name used in processed files: identity, yeojohnson, etc.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help=(
            "Optional processed file prefix. "
            "Example: data/processed/la_ejscreen_ptraf_yeojohnson"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Processed data/output directory.",
    )

    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Optional response column. If omitted, inferred from metadata.",
    )

    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=5,
        help="Number of quantile classes for quantile map.",
    )

    parser.add_argument(
        "--high-q",
        type=float,
        default=0.90,
        help="Upper quantile for high cluster candidate flag.",
    )

    parser.add_argument(
        "--low-q",
        type=float,
        default=0.10,
        help="Lower quantile for low cluster candidate flag.",
    )

    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Do not draw tract boundaries.",
    )

    parser.add_argument(
        "--save-gpkg",
        action="store_true",
        help="Save a GPKG with map diagnostic columns.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = make_paths(
        out_dir=out_dir,
        response=args.response,
        transform=args.transform,
        prefix=args.prefix,
    )

    gpkg_path = paths["gpkg"]
    adj_path = paths["adj"]
    metadata_path = paths["metadata"]

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GPKG not found: {gpkg_path}")

    print("\nUsing files:")
    print(f"  GPKG     = {gpkg_path}")
    print(f"  ADJ      = {adj_path if adj_path.exists() else 'not found'}")
    print(f"  metadata = {metadata_path if metadata_path.exists() else 'not found'}")

    gdf = gpd.read_file(gpkg_path, layer="tracts")

    y_col = infer_y_col(
        gdf,
        metadata_path=metadata_path,
        y_col=args.y_col,
        response=args.response,
        transform=args.transform,
    )

    print("\nDataset:")
    print(f"  n     = {len(gdf)}")
    print(f"  y_col = {y_col}")

    y = pd.to_numeric(gdf[y_col], errors="coerce").to_numpy(dtype=float)

    print("\nResponse summary:")
    print(pd.Series(y).describe(percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]))

    gdf = add_response_diagnostics(
        gdf,
        y_col,
        n_quantiles=args.n_quantiles,
    )

    if adj_path.exists():
        A = sparse.load_npz(adj_path).tocsr()
        gdf = add_neighbor_cluster_flags(
            gdf,
            A,
            z_col="map_z",
            high_q=args.high_q,
            low_q=args.low_q,
        )
    else:
        print("\nAdjacency not found, skipping neighbor cluster candidate map.")

    run_tag = f"la_ejscreen_{args.response.lower()}_{args.transform.lower()}"
    label = response_label(args.response, args.transform, y_col)

    edgecolor = "black" if not args.no_edges else "none"
    linewidth = 0.03 if not args.no_edges else 0.0

    # 1. Continuous response map
    response_png = out_dir / f"{run_tag}_response_map.png"
    save_continuous_map(
        gdf,
        column="map_y",
        title=f"{label}: response map",
        output_path=response_png,
        cmap="viridis",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # 2. Standardized response map
    z_png = out_dir / f"{run_tag}_response_zscore_map.png"
    save_continuous_map(
        gdf,
        column="map_z",
        title=f"{label}: standardized response map",
        output_path=z_png,
        cmap="coolwarm",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # 3. Quantile map
    quantile_png = out_dir / f"{run_tag}_response_quantile_map.png"
    save_categorical_map(
        gdf,
        column="map_quantile",
        title=f"{label}: quantile map",
        output_path=quantile_png,
        cmap="viridis",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # 4. Top/bottom tail map
    tail_png = out_dir / f"{run_tag}_response_top_bottom_decile_map.png"
    save_categorical_map(
        gdf,
        column="map_tail_group",
        title=f"{label}: top and bottom deciles",
        output_path=tail_png,
        cmap="tab10",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # 5. Neighbor cluster candidate map
    cluster_png = None
    if "map_cluster_candidate" in gdf.columns:
        cluster_png = out_dir / f"{run_tag}_response_neighbor_cluster_candidate_map.png"
        save_categorical_map(
            gdf,
            column="map_cluster_candidate",
            title=f"{label}: simple neighbor cluster candidates",
            output_path=cluster_png,
            cmap="tab10",
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    # 6. Histogram
    hist_png = out_dir / f"{run_tag}_response_histogram.png"
    save_histogram(
        gdf,
        column="map_y",
        title=f"{label}: response histogram",
        output_path=hist_png,
        xlabel=label,
    )

    if args.save_gpkg:
        out_gpkg = out_dir / f"{run_tag}_response_map_diagnostics.gpkg"
        gdf.to_file(out_gpkg, layer="tracts", driver="GPKG")
    else:
        out_gpkg = None

    print("\nSaved maps:")
    print(f"  {response_png}")
    print(f"  {z_png}")
    print(f"  {quantile_png}")
    print(f"  {tail_png}")
    if cluster_png is not None:
        print(f"  {cluster_png}")
    print(f"  {hist_png}")

    if out_gpkg is not None:
        print("\nSaved map diagnostics GPKG:")
        print(f"  {out_gpkg}")

    if "map_cluster_candidate" in gdf.columns:
        print("\nCluster candidate counts:")
        print(gdf["map_cluster_candidate"].value_counts())

    print("\nDone.")


if __name__ == "__main__":
    main()