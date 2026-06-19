from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV

DEFAULT_OUT_DIR = Path("data/processed")


def default_covariate_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    """
    Default mean-effect covariates saved from EJScreen.
    """
    candidate_covars = [
        "ACSTOTPOP",
        "ACSTOTHH",
        "ACSTOTHU",
        "PEOPCOLORPCT",
        "LOWINCPCT",
        "UNEMPPCT",
        "DISABILITYPCT",
        "LINGISOPCT",
        "LESSHSPCT",
        "UNDER5PCT",
        "OVER64PCT",
    ]

    return [c for c in candidate_covars if c in gdf.columns]

def covariate_design(
    gdf: gpd.GeoDataFrame,
    *,
    include_coords: bool = True,
    poly_degree: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Build mean-effect design matrix from real covariates plus optional coordinate terms.
    """
    covars = default_covariate_columns(gdf)

    if not covars:
        raise ValueError(
            "No default covariates found in GPKG. "
            "Rebuild the processed dataset after saving mean covariates."
        )

    X_parts = []
    names = []

    X_cov = gdf[covars].apply(pd.to_numeric, errors="coerce").copy()

    # log-transform scale/count variables
    for c in ["ACSTOTPOP", "ACSTOTHH", "ACSTOTHU"]:
        if c in X_cov.columns:
            X_cov[c] = np.log1p(X_cov[c])

    X_parts.append(X_cov.to_numpy(dtype=float))
    names.extend(covars)

    if include_coords:
        X_coord = coordinate_design(gdf, degree=poly_degree)
        coord_names = ["x", "y"]
        if poly_degree >= 2:
            coord_names += ["x2", "xy", "y2"]
        if poly_degree >= 3:
            coord_names += ["x3", "x2y", "xy2", "y3"]

        X_parts.append(X_coord)
        names.extend(coord_names)

    X = np.column_stack(X_parts)

    return X, names

def make_paths(
    *,
    out_dir: Path,
    response: str,
    transform: str,
    prefix: str | None = None,
) -> dict[str, Path]:
    """
    Construct paths for files produced by build_la_ejscreen_traffic.py.

    Example prefix:
        data/processed/la_ejscreen_ptsdf_identity
    """
    if prefix is None:
        stem = f"la_ejscreen_{response.lower()}_{transform.lower()}"
        base = out_dir / stem
    else:
        base = Path(prefix)

    return {
        "base": base,
        "gpkg": Path(str(base) + "_tracts.gpkg"),
        "csv": Path(str(base) + "_tracts.csv"),
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
    """
    Infer transformed response column.

    Priority:
        1. Explicit --y-col
        2. Metadata y_col
        3. y_{response}_{transform}
        4. Single column starting with y_
    """
    if y_col is not None:
        if y_col not in gdf.columns:
            raise ValueError(f"--y-col {y_col!r} not found in GPKG columns.")
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
        "Could not infer response column. Use --y-col explicitly.\n"
        f"Candidate y_ columns: {y_cols}"
    )


def build_normalized_laplacian(A_sparse: sparse.spmatrix) -> np.ndarray:
    """
    Build symmetric normalized graph Laplacian:

        L_norm = I - D^{-1/2} A D^{-1/2}

    This is the same graph operator you were using in the pasted script.
    """
    A = A_sparse.tocsr()
    n = A.shape[0]

    d = np.asarray(A.sum(axis=1)).ravel()

    dinv_sqrt = np.zeros_like(d, dtype=float)
    nonzero = d > 0
    dinv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])

    Dinv = sparse.diags(dinv_sqrt)
    S = Dinv @ A @ Dinv
    Lnorm = sparse.eye(n, format="csr") - S

    return Lnorm.toarray()


def coordinate_design(
    gdf: gpd.GeoDataFrame,
    *,
    degree: int = 2,
) -> np.ndarray:
    """
    Build coordinate polynomial trend design matrix.

    degree = 1:
        x, y

    degree = 2:
        x, y, x^2, xy, y^2

    degree = 3:
        x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3
    """
    if degree not in {1, 2, 3}:
        raise ValueError("degree must be one of {1, 2, 3}")

    cent = gdf.geometry.centroid
    x = cent.x.to_numpy()
    y = cent.y.to_numpy()

    xs = (x - x.mean()) / x.std(ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)

    cols = [xs, ys]

    if degree >= 2:
        cols.extend(
            [
                xs**2,
                xs * ys,
                ys**2,
            ]
        )

    if degree >= 3:
        cols.extend(
            [
                xs**3,
                (xs**2) * ys,
                xs * (ys**2),
                ys**3,
            ]
        )

    return np.column_stack(cols)


def energy_band_summary(
    lam: np.ndarray,
    energy_prop: np.ndarray,
) -> dict[str, float]:
    """
    Split modes into low/middle/high thirds by graph eigenvalue.
    """
    q1 = np.quantile(lam, 1.0 / 3.0)
    q2 = np.quantile(lam, 2.0 / 3.0)

    low = energy_prop[lam <= q1].sum()
    mid = energy_prop[(lam > q1) & (lam <= q2)].sum()
    high = energy_prop[lam > q2].sum()

    return {
        "low": float(low),
        "middle": float(mid),
        "high": float(high),
    }


def cumulative_deviation_from_uniform(x: np.ndarray) -> float:
    """
    Given a nonnegative energy vector normalized to sum 1, compare cumulative
    energy to the uniform cumulative line.

    Smaller means closer to uniform cumulative energy.
    """
    x = np.asarray(x, dtype=float)
    x = x / x.sum()

    n = len(x)
    cum = np.cumsum(x)
    uniform = np.arange(1, n + 1) / n

    return float(np.mean(np.abs(cum - uniform)))


def spectral_energy_summary(
    *,
    name: str,
    r: np.ndarray,
    lam: np.ndarray,
    U: np.ndarray,
    out_dir: Path,
    out_prefix: str,
    n_bins: int = 30,
    zero_tol: float = 1e-10,
) -> dict[str, float | str | int]:
    """
    Compute and save spectral energy diagnostics for residual vector r.

    We save two related quantities:

    1. Raw spectral energy:
        coeff_j^2

    2. CAR-whitened spectral energy:
        lambda_j * coeff_j^2

       If residuals behave like a CAR/ICAR field with variance proportional
       to 1/lambda_j, then lambda_j * coeff_j^2 should look more uniform
       across nonzero modes.
    """
    r = np.asarray(r, dtype=float)
    r = r - r.mean()

    coeff = U.T @ r
    energy = coeff**2

    keep = lam > zero_tol
    lam2 = lam[keep]
    energy2 = energy[keep]

    if energy2.sum() <= 0:
        raise ValueError(f"Zero spectral energy for {name}.")

    energy_prop = energy2 / energy2.sum()

    car_whitened = lam2 * energy2
    car_whitened_prop = car_whitened / car_whitened.sum()

    raw_bands = energy_band_summary(lam2, energy_prop)
    car_bands = energy_band_summary(lam2, car_whitened_prop)

    raw_cum_dev = cumulative_deviation_from_uniform(energy_prop)
    car_cum_dev = cumulative_deviation_from_uniform(car_whitened_prop)

    print(f"\n{name}")
    print("-" * len(name))
    print("Raw spectral energy:")
    print(f"  Low-frequency energy:    {raw_bands['low']:.4f}")
    print(f"  Middle-frequency energy: {raw_bands['middle']:.4f}")
    print(f"  High-frequency energy:   {raw_bands['high']:.4f}")
    print(f"  Cumulative deviation from uniform: {raw_cum_dev:.4f}")

    print("CAR-whitened spectral energy:")
    print(f"  Low-frequency energy:    {car_bands['low']:.4f}")
    print(f"  Middle-frequency energy: {car_bands['middle']:.4f}")
    print(f"  High-frequency energy:   {car_bands['high']:.4f}")
    print(f"  Cumulative deviation from uniform: {car_cum_dev:.4f}")

    spec = pd.DataFrame(
        {
            "lambda": lam2,
            "coeff": coeff[keep],
            "energy": energy2,
            "energy_prop": energy_prop,
            "car_whitened_energy": car_whitened,
            "car_whitened_energy_prop": car_whitened_prop,
        }
    )

    spec["bin"] = pd.qcut(spec["lambda"], q=n_bins, duplicates="drop")

    binned = (
        spec.groupby("bin", observed=True)
        .agg(
            lambda_mean=("lambda", "mean"),
            lambda_min=("lambda", "min"),
            lambda_max=("lambda", "max"),
            energy_sum=("energy", "sum"),
            energy_prop_sum=("energy_prop", "sum"),
            car_whitened_energy_sum=("car_whitened_energy", "sum"),
            car_whitened_energy_prop_sum=("car_whitened_energy_prop", "sum"),
            count=("lambda", "size"),
        )
        .reset_index(drop=True)
    )

    spec_path = out_dir / f"{out_prefix}_spectral_energy.csv"
    binned_path = out_dir / f"{out_prefix}_spectral_energy_binned.csv"

    spec.to_csv(spec_path, index=False)
    binned.to_csv(binned_path, index=False)

    # Scatter plot: raw energy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(spec["lambda"], spec["energy_prop"], s=8, alpha=0.5)
    ax.set_xlabel("Graph eigenvalue lambda")
    ax.set_ylabel("Raw energy proportion")
    ax.set_title(f"Raw spectral energy: {name}")
    plt.tight_layout()
    scatter_path = out_dir / f"{out_prefix}_spectral_energy_scatter.png"
    plt.savefig(scatter_path, dpi=250)
    plt.close()

    # Binned raw energy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(binned["lambda_mean"], binned["energy_prop_sum"], marker="o")
    ax.set_xlabel("Mean graph eigenvalue in bin")
    ax.set_ylabel("Raw energy proportion in bin")
    ax.set_title(f"Binned raw spectral energy: {name}")
    plt.tight_layout()
    binned_raw_path = out_dir / f"{out_prefix}_spectral_energy_binned.png"
    plt.savefig(binned_raw_path, dpi=250)
    plt.close()

    # Binned CAR-whitened energy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        binned["lambda_mean"],
        binned["car_whitened_energy_prop_sum"],
        marker="o",
    )
    ax.axhline(1.0 / len(binned), linestyle="--", linewidth=1)
    ax.set_xlabel("Mean graph eigenvalue in bin")
    ax.set_ylabel("CAR-whitened energy proportion in bin")
    ax.set_title(f"Binned CAR-whitened energy: {name}")
    plt.tight_layout()
    binned_car_path = out_dir / f"{out_prefix}_car_whitened_energy_binned.png"
    plt.savefig(binned_car_path, dpi=250)
    plt.close()

    # Cumulative CAR-whitened energy
    cum_car = np.cumsum(car_whitened_prop)
    uniform = np.arange(1, len(cum_car) + 1) / len(cum_car)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam2, cum_car, label="Observed CAR-whitened cumulative energy")
    ax.plot(lam2, uniform, linestyle="--", label="Uniform target")
    ax.set_xlabel("Graph eigenvalue lambda")
    ax.set_ylabel("Cumulative energy")
    ax.set_title(f"CAR-whitened cumulative energy: {name}")
    ax.legend()
    plt.tight_layout()
    cum_path = out_dir / f"{out_prefix}_car_whitened_cumulative.png"
    plt.savefig(cum_path, dpi=250)
    plt.close()

    return {
        "name": name,
        "n_modes": int(len(lam2)),
        "lambda_min": float(lam2.min()),
        "lambda_max": float(lam2.max()),
        "raw_low": raw_bands["low"],
        "raw_middle": raw_bands["middle"],
        "raw_high": raw_bands["high"],
        "raw_cum_dev_uniform": raw_cum_dev,
        "car_whitened_low": car_bands["low"],
        "car_whitened_middle": car_bands["middle"],
        "car_whitened_high": car_bands["high"],
        "car_whitened_cum_dev_uniform": car_cum_dev,
        "spec_csv": str(spec_path),
        "binned_csv": str(binned_path),
        "scatter_png": str(scatter_path),
        "binned_raw_png": str(binned_raw_path),
        "binned_car_png": str(binned_car_path),
        "cum_car_png": str(cum_path),
    }


def save_map(
    *,
    gdf: gpd.GeoDataFrame,
    col: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(
        column=col,
        ax=ax,
        legend=True,
        cmap="coolwarm",
        linewidth=0.05,
        edgecolor="black",
    )
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--response",
        type=str,
        default="ptsdf",
        help="Response name used in processed files, e.g. ptsdf, ptraf, no2.",
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="identity",
        help="Transform name used in processed files, e.g. identity, yeojohnson.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help=(
            "Optional processed file prefix. "
            "Example: data/processed/la_ejscreen_ptsdf_identity"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Optional transformed response column. If omitted, inferred from metadata.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=30,
        help="Number of equal-count eigenvalue bins.",
    )

    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Coordinate polynomial degree for detrending.",
    )

    parser.add_argument(
        "--remove-k",
        type=int,
        nargs="*",
        default=[5, 10, 25, 50],
        help="Low-frequency eigenvector counts to remove as sensitivity checks.",
    )

    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip residual map PNG outputs.",
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

    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")

    print("\nUsing processed files:")
    print(f"  GPKG     = {gpkg_path}")
    print(f"  ADJ      = {adj_path}")
    print(f"  metadata = {metadata_path if metadata_path.exists() else 'not found'}")

    gdf = gpd.read_file(gpkg_path, layer="tracts")
    A = sparse.load_npz(adj_path).tocsr()

    if A.shape[0] != len(gdf):
        raise ValueError(
            f"Adjacency shape {A.shape} does not match GPKG rows {len(gdf)}."
        )

    y_col = infer_y_col(
        gdf,
        metadata_path=metadata_path,
        y_col=args.y_col,
        response=args.response,
        transform=args.transform,
    )

    y = pd.to_numeric(gdf[y_col], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(y).all():
        raise ValueError(f"Response column {y_col!r} contains non-finite values.")

    n = len(y)

    print("\nDataset:")
    print(f"  n     = {n}")
    print(f"  y_col = {y_col}")
    print(f"  y mean = {y.mean():.6f}")
    print(f"  y sd   = {y.std(ddof=1):.6f}")

    degrees = np.asarray(A.sum(axis=1)).ravel()
    print("\nGraph:")
    print(f"  shape       = {A.shape}")
    print(f"  nnz         = {A.nnz}")
    print(f"  edges       = {A.nnz // 2}")
    print(f"  degree min  = {degrees.min():.0f}")
    print(f"  degree mean = {degrees.mean():.3f}")
    print(f"  degree max  = {degrees.max():.0f}")

    # ------------------------------------------------------------
    # Graph operator and eigendecomposition
    # ------------------------------------------------------------
    print("\nBuilding normalized Laplacian...")
    Lnorm = build_normalized_laplacian(A)

    print("Computing eigen-decomposition...")
    lam, U = eigh(Lnorm)

    print("\nEigenvalue summary:")
    print(f"  lambda min       = {lam.min():.8e}")
    print(f"  lambda max       = {lam.max():.8e}")
    print(f"  near-zero count  = {(lam <= 1e-10).sum()}")

    summaries: list[dict[str, float | str | int]] = []

    run_tag = f"la_ejscreen_{args.response.lower()}_{args.transform.lower()}"

    # ------------------------------------------------------------
    # 1. Raw centered response
    # ------------------------------------------------------------
    y_centered = y - y.mean()
    gdf["resid_raw_centered"] = y_centered

    summaries.append(
        spectral_energy_summary(
            name=f"Raw centered {y_col}",
            r=y_centered,
            lam=lam,
            U=U,
            out_dir=out_dir,
            out_prefix=f"{run_tag}_raw_centered",
            n_bins=args.n_bins,
        )
    )

    # ------------------------------------------------------------
    # 2. Coordinate polynomial detrending
    # ------------------------------------------------------------
    X_poly = coordinate_design(gdf, degree=args.poly_degree)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    yhat_poly = reg.predict(X_poly)
    r_poly = y - yhat_poly

    poly_r2 = reg.score(X_poly, y)

    print("\nCoordinate polynomial trend:")
    print(f"  degree = {args.poly_degree}")
    print(f"  R^2    = {poly_r2:.6f}")

    gdf["trend_coord_poly"] = yhat_poly
    gdf["resid_coord_poly"] = r_poly

    summaries.append(
        spectral_energy_summary(
            name=f"Residual after coordinate polynomial trend degree {args.poly_degree}",
            r=r_poly,
            lam=lam,
            U=U,
            out_dir=out_dir,
            out_prefix=f"{run_tag}_coord_poly_deg{args.poly_degree}",
            n_bins=args.n_bins,
        )
    )

    # ------------------------------------------------------------
    # 3. Actual covariate mean model
    # ------------------------------------------------------------
    X_cov, covar_names = covariate_design(
        gdf,
        include_coords=True,
        poly_degree=args.poly_degree,
    )

    mean_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )

    mean_model.fit(X_cov, y)
    yhat_cov = mean_model.predict(X_cov)
    r_cov = y - yhat_cov

    cov_r2 = 1.0 - np.sum((y - yhat_cov) ** 2) / np.sum((y - y.mean()) ** 2)

    print("\nCovariate mean model:")
    print(f"  number of covariates including coordinate terms = {X_cov.shape[1]}")
    print(f"  R^2 = {cov_r2:.6f}")
    print("  covariates:")
    for c in covar_names:
        print(f"    {c}")

    gdf["trend_covariate_mean"] = yhat_cov
    gdf["resid_covariate_mean"] = r_cov

    summaries.append(
        spectral_energy_summary(
            name="Residual after covariate mean model",
            r=r_cov,
            lam=lam,
            U=U,
            out_dir=out_dir,
            out_prefix=f"{run_tag}_covariate_mean",
            n_bins=args.n_bins,
        )
    )

    # ------------------------------------------------------------
    # 3. Remove first K low-frequency graph eigenvectors
    # ------------------------------------------------------------
    # This is not the primary mean-effect model. It is a sensitivity check:
    # How much of the response lives in the very smooth graph modes?
    for K in args.remove_k:
        if K <= 0:
            continue

        if K >= n:
            print(f"Skipping K={K}; must be smaller than n={n}.")
            continue

        U_low = U[:, :K]
        smooth = U_low @ (U_low.T @ y_centered)
        r_highpass = y_centered - smooth

        resid_col = f"resid_remove_first_{K}_eig"
        gdf[resid_col] = r_highpass

        summaries.append(
            spectral_energy_summary(
                name=f"Residual after removing first {K} eigenvectors",
                r=r_highpass,
                lam=lam,
                U=U,
                out_dir=out_dir,
                out_prefix=f"{run_tag}_remove_first_{K}_eig",
                n_bins=args.n_bins,
            )
        )

    # ------------------------------------------------------------
    # Save residual maps
    # ------------------------------------------------------------
    if not args.no_maps:
        map_specs = [
            (
                "resid_raw_centered",
                f"{run_tag}_raw_centered_map.png",
                f"Raw centered {y_col}",
            ),
            (
                "resid_coord_poly",
                f"{run_tag}_coord_poly_deg{args.poly_degree}_residual_map.png",
                f"Residual after coordinate polynomial trend degree {args.poly_degree}",
            ),
        ]

        map_specs.append(
            (
                "resid_covariate_mean",
                f"{run_tag}_covariate_mean_residual_map.png",
                "Residual after covariate mean model",
            )
        )

        for K in args.remove_k:
            resid_col = f"resid_remove_first_{K}_eig"
            if resid_col in gdf.columns and K in {10, 25, 50}:
                map_specs.append(
                    (
                        resid_col,
                        f"{run_tag}_remove_first_{K}_eig_residual_map.png",
                        f"Residual after removing first {K} eigenvectors",
                    )
                )

        for col, filename, title in map_specs:
            save_map(
                gdf=gdf,
                col=col,
                out_path=out_dir / filename,
                title=title,
            )

    # ------------------------------------------------------------
    # Save comparison table and diagnostics GPKG
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(summaries)

    summary_path = out_dir / f"{run_tag}_spectral_energy_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    out_gpkg = out_dir / f"{run_tag}_spectral_diagnostics.gpkg"
    gdf.to_file(out_gpkg, layer="tracts", driver="GPKG")

    print("\nSaved comparison:")
    print(f"  {summary_path}")
    print(f"  {out_gpkg}")

    print("\nMost important columns in the comparison table:")
    print(
        summary_df[
            [
                "name",
                "raw_low",
                "raw_middle",
                "raw_high",
                "car_whitened_low",
                "car_whitened_middle",
                "car_whitened_high",
                "car_whitened_cum_dev_uniform",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()