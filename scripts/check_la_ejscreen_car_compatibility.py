from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import minimize

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV


DEFAULT_OUT_DIR = Path("data/processed")

DEFAULT_MEAN_COVARIATES = [
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
        "Could not infer transformed response column. "
        f"Candidate y_ columns: {y_cols}. Use --y-col explicitly."
    )


def default_covariate_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    return [c for c in DEFAULT_MEAN_COVARIATES if c in gdf.columns]


def attach_covariates_from_csv_if_needed(
    gdf: gpd.GeoDataFrame,
    csv_path: Path,
) -> gpd.GeoDataFrame:
    """
    If the GPKG was created before covariates were saved, try to merge them
    from the matching CSV.
    """
    covars = default_covariate_columns(gdf)
    if covars:
        return gdf

    if not csv_path.exists():
        return gdf

    cov_df = pd.read_csv(csv_path)
    covars_csv = [c for c in DEFAULT_MEAN_COVARIATES if c in cov_df.columns]

    if not covars_csv:
        return gdf

    print("\nCovariates not found in GPKG. Merging covariates from CSV:")
    for c in covars_csv:
        print(f"  {c}")

    gdf = gdf.copy()
    cov_df = cov_df.copy()

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    cov_df["GEOID"] = cov_df["GEOID"].astype(str).str.zfill(11)

    gdf = gdf.merge(
        cov_df[["GEOID"] + covars_csv],
        on="GEOID",
        how="left",
    )

    return gdf


def build_normalized_laplacian(A_sparse: sparse.spmatrix) -> np.ndarray:
    """
    Symmetric normalized graph Laplacian:

        L = I - D^{-1/2} A D^{-1/2}
    """
    A = A_sparse.tocsr()
    n = A.shape[0]

    d = np.asarray(A.sum(axis=1)).ravel()

    dinv_sqrt = np.zeros_like(d, dtype=float)
    nz = d > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(d[nz])

    Dinv = sparse.diags(dinv_sqrt)
    S = Dinv @ A @ Dinv
    L = sparse.eye(n, format="csr") - S

    return L.toarray()


def coordinate_design(
    gdf: gpd.GeoDataFrame,
    *,
    degree: int = 2,
) -> tuple[np.ndarray, list[str]]:
    if degree not in {1, 2, 3}:
        raise ValueError("degree must be one of {1, 2, 3}")

    cent = gdf.geometry.centroid
    x = cent.x.to_numpy()
    y = cent.y.to_numpy()

    xs = (x - x.mean()) / x.std(ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)

    cols = [xs, ys]
    names = ["x", "y"]

    if degree >= 2:
        cols.extend([xs**2, xs * ys, ys**2])
        names.extend(["x2", "xy", "y2"])

    if degree >= 3:
        cols.extend([xs**3, (xs**2) * ys, xs * (ys**2), ys**3])
        names.extend(["x3", "x2y", "xy2", "y3"])

    return np.column_stack(cols), names


def covariate_design(
    gdf: gpd.GeoDataFrame,
    *,
    include_coords: bool = True,
    poly_degree: int = 2,
    user_covariates: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if user_covariates is None:
        covars = default_covariate_columns(gdf)
    else:
        covars = user_covariates

    missing = [c for c in covars if c not in gdf.columns]
    if missing:
        raise ValueError(f"Requested covariates not found in GPKG/CSV: {missing}")

    if not covars:
        print("\nAvailable columns:")
        print(list(gdf.columns))
        raise ValueError(
            "No covariates found. Rebuild the processed dataset after saving "
            "mean covariates, or pass --covariates."
        )

    X_parts = []
    names = []

    X_cov = gdf[covars].apply(pd.to_numeric, errors="coerce").copy()

    # Counts are very scale-heavy. Use log1p to make them more stable.
    for c in ["ACSTOTPOP", "ACSTOTHH", "ACSTOTHU"]:
        if c in X_cov.columns:
            X_cov[c] = np.log1p(X_cov[c])

    X_parts.append(X_cov.to_numpy(dtype=float))
    names.extend(covars)

    if include_coords:
        X_coord, coord_names = coordinate_design(gdf, degree=poly_degree)
        X_parts.append(X_coord)
        names.extend(coord_names)

    X = np.column_stack(X_parts)

    return X, names


def fit_mean_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "ols",
) -> tuple[np.ndarray, float, Any]:
    if model_type == "ols":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(),
        )
    elif model_type == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 4, 30)),
        )
    else:
        raise ValueError("model_type must be 'ols' or 'ridge'.")

    model.fit(X, y)
    yhat = model.predict(X)

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    return yhat, r2, model


def broad_band_summary(
    lam: np.ndarray,
    energy_prop: np.ndarray,
) -> dict[str, float]:
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
    x = np.asarray(x, dtype=float)
    x = x / x.sum()

    n = len(x)
    cum = np.cumsum(x)
    uniform = np.arange(1, n + 1) / n

    return float(np.mean(np.abs(cum - uniform)))


def weighted_lambda(lam: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(np.sum(lam * weights))


def relative_weighted_lambda(lam: np.ndarray, weights: np.ndarray) -> float:
    wl = weighted_lambda(lam, weights)
    return float((wl - lam.min()) / (lam.max() - lam.min()))


def nonmonotonicity_scores(values: np.ndarray) -> dict[str, float]:
    """
    CAR expects a broadly decreasing spectral curve as lambda increases.
    This measures upward movement in a binned curve.
    """
    v = np.asarray(values, dtype=float)

    if len(v) <= 1:
        return {
            "nonmonotone_frac_up": np.nan,
            "nonmonotone_amp_up": np.nan,
        }

    diffs = np.diff(v)
    up = np.maximum(diffs, 0.0)

    denom = np.sum(np.abs(diffs))
    if denom <= 0:
        amp = 0.0
    else:
        amp = float(up.sum() / denom)

    return {
        "nonmonotone_frac_up": float(np.mean(diffs > 0)),
        "nonmonotone_amp_up": amp,
    }


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(max(p, 1e-8), 1 - 1e-8)
    return math.log(p / (1.0 - p))


def car_curve(
    lam: np.ndarray,
    *,
    rho: float,
    tau2: float,
    sigma2: float,
) -> np.ndarray:
    """
    Normalized-CAR-like spectral variance curve in Laplacian eigenvalues:

        tau2 / (1 - rho + rho * lambda) + sigma2
    """
    return tau2 / (1.0 - rho + rho * lam) + sigma2


def fit_best_car_curve(
    binned: pd.DataFrame,
    *,
    rho_max: float = 0.999,
    eps: float = 1e-12,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Fit best CAR-shaped spectral curve to the binned residual periodogram.

    We fit the curve to binned mean modal energy:

        P(lambda) ≈ tau2 / (1 - rho + rho lambda) + sigma2

    using squared log error.
    """
    lam = binned["lambda_mean"].to_numpy(dtype=float)
    target = binned["energy_mean"].to_numpy(dtype=float)
    weights = binned["count"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    target = np.maximum(target, eps)
    log_target = np.log(target)

    target_median = float(np.median(target))
    target_min = float(np.min(target))

    init_rho = 0.90
    init_tau2 = max(target_median * (1.0 - init_rho + init_rho * np.median(lam)), eps)
    init_sigma2 = max(0.25 * target_min, eps)

    theta0 = np.array(
        [
            logit(init_rho / rho_max),
            np.log(init_tau2),
            np.log(init_sigma2),
        ],
        dtype=float,
    )

    var_log_target = float(np.sum(weights * (log_target - np.sum(weights * log_target)) ** 2))
    var_log_target = max(var_log_target, eps)

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        rho = rho_max * sigmoid(float(theta[0]))
        tau2 = float(np.exp(theta[1]))
        sigma2 = float(np.exp(theta[2]))
        return rho, tau2, sigma2

    def objective(theta: np.ndarray) -> float:
        rho, tau2, sigma2 = unpack(theta)
        pred = car_curve(lam, rho=rho, tau2=tau2, sigma2=sigma2)
        pred = np.maximum(pred, eps)
        log_pred = np.log(pred)

        mse = float(np.sum(weights * (log_target - log_pred) ** 2))
        return mse / var_log_target

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=[
            (-12.0, 12.0),
            (-40.0, 40.0),
            (-40.0, 40.0),
        ],
    )

    rho, tau2, sigma2 = unpack(res.x)
    pred = car_curve(lam, rho=rho, tau2=tau2, sigma2=sigma2)

    fit_df = binned.copy()
    fit_df["car_fit_energy"] = pred
    fit_df["log_energy_mean"] = np.log(np.maximum(fit_df["energy_mean"], eps))
    fit_df["log_car_fit_energy"] = np.log(np.maximum(fit_df["car_fit_energy"], eps))
    fit_df["log_residual"] = fit_df["log_energy_mean"] - fit_df["log_car_fit_energy"]

    metrics = {
        "car_fit_rho": float(rho),
        "car_fit_tau2": float(tau2),
        "car_fit_sigma2": float(sigma2),
        "car_fit_error": float(objective(res.x)),
        "car_fit_success": bool(res.success),
        "car_fit_nit": int(res.nit),
    }

    return metrics, fit_df


def spectral_compatibility_report(
    *,
    name: str,
    r: np.ndarray,
    lam: np.ndarray,
    U: np.ndarray,
    out_dir: Path,
    out_prefix: str,
    n_bins: int = 30,
    rho_max: float = 0.999,
    zero_tol: float = 1e-10,
) -> dict[str, float | str | int | bool]:
    r = np.asarray(r, dtype=float)
    r = r - r.mean()

    coeff = U.T @ r
    energy = coeff**2

    keep = lam > zero_tol

    lam2 = lam[keep]
    coeff2 = coeff[keep]
    energy2 = energy[keep]

    if energy2.sum() <= 0:
        raise ValueError(f"Zero spectral energy for {name}.")

    energy_prop = energy2 / energy2.sum()

    car_whitened = lam2 * energy2
    car_whitened_prop = car_whitened / car_whitened.sum()

    raw_bands = broad_band_summary(lam2, energy_prop)
    car_bands = broad_band_summary(lam2, car_whitened_prop)

    raw_cum_dev = cumulative_deviation_from_uniform(energy_prop)
    car_cum_dev = cumulative_deviation_from_uniform(car_whitened_prop)

    hf_ratio_raw = raw_bands["high"] / max(raw_bands["low"], 1e-12)
    hf_ratio_car = car_bands["high"] / max(car_bands["low"], 1e-12)

    lf_excess_car = car_bands["low"] - (1.0 / 3.0)
    lf_ratio_car = car_bands["low"] / (1.0 / 3.0)

    wl_raw = weighted_lambda(lam2, energy_prop)
    wl_car = weighted_lambda(lam2, car_whitened_prop)

    rel_wl_raw = relative_weighted_lambda(lam2, energy_prop)
    rel_wl_car = relative_weighted_lambda(lam2, car_whitened_prop)

    spec = pd.DataFrame(
        {
            "lambda": lam2,
            "coeff": coeff2,
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
            energy_mean=("energy", "mean"),
            energy_prop_sum=("energy_prop", "sum"),
            car_whitened_energy_sum=("car_whitened_energy", "sum"),
            car_whitened_energy_mean=("car_whitened_energy", "mean"),
            car_whitened_energy_prop_sum=("car_whitened_energy_prop", "sum"),
            count=("lambda", "size"),
        )
        .reset_index(drop=True)
    )

    raw_nonmono = nonmonotonicity_scores(binned["energy_mean"].to_numpy())
    car_nonmono = nonmonotonicity_scores(
        binned["car_whitened_energy_prop_sum"].to_numpy()
    )

    car_fit_metrics, fit_df = fit_best_car_curve(
        binned,
        rho_max=rho_max,
    )

    spec_path = out_dir / f"{out_prefix}_spectrum.csv"
    binned_path = out_dir / f"{out_prefix}_binned_spectrum.csv"
    fit_path = out_dir / f"{out_prefix}_car_fit_binned.csv"

    spec.to_csv(spec_path, index=False)
    binned.to_csv(binned_path, index=False)
    fit_df.to_csv(fit_path, index=False)

    # Plot 1: empirical binned energy vs best CAR curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        fit_df["lambda_mean"],
        fit_df["energy_mean"],
        marker="o",
        label="Empirical binned periodogram",
    )
    ax.plot(
        fit_df["lambda_mean"],
        fit_df["car_fit_energy"],
        linestyle="--",
        label="Best-fit CAR spectral curve",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Mean graph eigenvalue in bin")
    ax.set_ylabel("Mean modal energy, log scale")
    ax.set_title(f"Empirical spectrum vs CAR fit: {name}")
    ax.legend()
    plt.tight_layout()
    fit_png = out_dir / f"{out_prefix}_empirical_vs_car_fit.png"
    plt.savefig(fit_png, dpi=250)
    plt.close()

    # Plot 2: CAR-whitened binned energy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        binned["lambda_mean"],
        binned["car_whitened_energy_prop_sum"],
        marker="o",
    )
    ax.axhline(1.0 / len(binned), linestyle="--", linewidth=1)
    ax.set_xlabel("Mean graph eigenvalue in bin")
    ax.set_ylabel("CAR-whitened energy proportion in bin")
    ax.set_title(f"CAR-whitened binned energy: {name}")
    plt.tight_layout()
    car_white_png = out_dir / f"{out_prefix}_car_whitened_binned.png"
    plt.savefig(car_white_png, dpi=250)
    plt.close()

    # Plot 3: cumulative CAR-whitened energy
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
    cum_png = out_dir / f"{out_prefix}_car_whitened_cumulative.png"
    plt.savefig(cum_png, dpi=250)
    plt.close()

    print(f"\n{name}")
    print("-" * len(name))
    print("Raw spectral energy:")
    print(f"  low             = {raw_bands['low']:.4f}")
    print(f"  middle          = {raw_bands['middle']:.4f}")
    print(f"  high            = {raw_bands['high']:.4f}")
    print(f"  high / low      = {hf_ratio_raw:.4f}")
    print(f"  weighted lambda = {wl_raw:.6f}")

    print("CAR-whitened spectral energy:")
    print(f"  low             = {car_bands['low']:.4f}")
    print(f"  middle          = {car_bands['middle']:.4f}")
    print(f"  high            = {car_bands['high']:.4f}")
    print(f"  low excess      = {lf_excess_car:.4f}")
    print(f"  low ratio       = {lf_ratio_car:.4f}")
    print(f"  high / low      = {hf_ratio_car:.4f}")
    print(f"  cum dev uniform = {car_cum_dev:.4f}")

    print("Best-fit CAR spectral curve:")
    print(f"  rho             = {car_fit_metrics['car_fit_rho']:.6f}")
    print(f"  tau2            = {car_fit_metrics['car_fit_tau2']:.6e}")
    print(f"  sigma2          = {car_fit_metrics['car_fit_sigma2']:.6e}")
    print(f"  fit error       = {car_fit_metrics['car_fit_error']:.6f}")
    print(f"  success         = {car_fit_metrics['car_fit_success']}")

    report = {
        "name": name,
        "n_modes": int(len(lam2)),
        "lambda_min": float(lam2.min()),
        "lambda_max": float(lam2.max()),

        "raw_low": raw_bands["low"],
        "raw_middle": raw_bands["middle"],
        "raw_high": raw_bands["high"],
        "raw_hf_ratio": float(hf_ratio_raw),
        "raw_cum_dev_uniform": float(raw_cum_dev),
        "raw_weighted_lambda": float(wl_raw),
        "raw_relative_weighted_lambda": float(rel_wl_raw),
        "raw_nonmonotone_frac_up": raw_nonmono["nonmonotone_frac_up"],
        "raw_nonmonotone_amp_up": raw_nonmono["nonmonotone_amp_up"],

        "car_whitened_low": car_bands["low"],
        "car_whitened_middle": car_bands["middle"],
        "car_whitened_high": car_bands["high"],
        "car_whitened_hf_ratio": float(hf_ratio_car),
        "car_whitened_lf_excess": float(lf_excess_car),
        "car_whitened_lf_ratio": float(lf_ratio_car),
        "car_whitened_cum_dev_uniform": float(car_cum_dev),
        "car_whitened_weighted_lambda": float(wl_car),
        "car_whitened_relative_weighted_lambda": float(rel_wl_car),
        "car_whitened_nonmonotone_frac_up": car_nonmono["nonmonotone_frac_up"],
        "car_whitened_nonmonotone_amp_up": car_nonmono["nonmonotone_amp_up"],

        **car_fit_metrics,

        "spectrum_csv": str(spec_path),
        "binned_spectrum_csv": str(binned_path),
        "car_fit_binned_csv": str(fit_path),
        "empirical_vs_car_fit_png": str(fit_png),
        "car_whitened_binned_png": str(car_white_png),
        "car_whitened_cumulative_png": str(cum_png),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--response",
        type=str,
        default="ptraf",
        help="Response name used in processed files, e.g. ptraf, ptsdf, no2.",
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="yeojohnson",
        help="Transform name used in processed files, e.g. identity, yeojohnson.",
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
        help="Coordinate polynomial degree.",
    )

    parser.add_argument(
        "--mean-model",
        type=str,
        default="ols",
        choices=["ols", "ridge"],
        help="Mean model used for covariate residualization.",
    )

    parser.add_argument(
        "--covariates",
        type=str,
        default=None,
        help=(
            "Optional comma-separated covariate list. "
            "If omitted, uses default EJScreen mean covariates."
        ),
    )

    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="Do not include coordinate polynomial terms in covariate mean model.",
    )

    parser.add_argument(
        "--rho-max",
        type=float,
        default=0.999,
        help="Maximum rho allowed in best-fit CAR spectral curve.",
    )

    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1e-10,
        help="Eigenvalue threshold for dropping constant/near-constant modes.",
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
    csv_path = paths["csv"]
    metadata_path = paths["metadata"]

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GPKG not found: {gpkg_path}")

    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")

    print("\nUsing processed files:")
    print(f"  GPKG     = {gpkg_path}")
    print(f"  CSV      = {csv_path}")
    print(f"  ADJ      = {adj_path}")
    print(f"  metadata = {metadata_path if metadata_path.exists() else 'not found'}")

    gdf = gpd.read_file(gpkg_path, layer="tracts")
    gdf = attach_covariates_from_csv_if_needed(gdf, csv_path)

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
    print(f"  n      = {n}")
    print(f"  y_col  = {y_col}")
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

    print("\nBuilding normalized Laplacian...")
    Lnorm = build_normalized_laplacian(A)

    print("Computing eigen-decomposition...")
    lam, U = eigh(Lnorm)

    print("\nEigenvalue summary:")
    print(f"  lambda min       = {lam.min():.8e}")
    print(f"  lambda max       = {lam.max():.8e}")
    print(f"  near-zero count  = {(lam <= args.zero_tol).sum()}")

    run_tag = f"la_ejscreen_{args.response.lower()}_{args.transform.lower()}"
    reports: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1. Raw centered response
    # ------------------------------------------------------------
    y_centered = y - y.mean()

    reports.append(
        spectral_compatibility_report(
            name=f"Raw centered {y_col}",
            r=y_centered,
            lam=lam,
            U=U,
            out_dir=out_dir,
            out_prefix=f"{run_tag}_raw_centered_car_compat",
            n_bins=args.n_bins,
            rho_max=args.rho_max,
            zero_tol=args.zero_tol,
        )
    )

    # ------------------------------------------------------------
    # 2. Coordinate polynomial residual
    # ------------------------------------------------------------
    X_poly, poly_names = coordinate_design(gdf, degree=args.poly_degree)
    yhat_poly, poly_r2, _ = fit_mean_model(X_poly, y, model_type="ols")
    r_poly = y - yhat_poly

    print("\nCoordinate polynomial mean model:")
    print(f"  degree = {args.poly_degree}")
    print(f"  R^2    = {poly_r2:.6f}")

    rep = spectral_compatibility_report(
        name=f"Residual after coordinate polynomial trend degree {args.poly_degree}",
        r=r_poly,
        lam=lam,
        U=U,
        out_dir=out_dir,
        out_prefix=f"{run_tag}_coord_poly_deg{args.poly_degree}_car_compat",
        n_bins=args.n_bins,
        rho_max=args.rho_max,
        zero_tol=args.zero_tol,
    )
    rep["mean_model_type"] = "coordinate_polynomial"
    rep["mean_model_r2"] = float(poly_r2)
    rep["mean_model_n_covariates"] = int(X_poly.shape[1])
    rep["mean_model_covariates"] = ",".join(poly_names)
    reports.append(rep)

    # ------------------------------------------------------------
    # 3. Covariate mean residual
    # ------------------------------------------------------------
    if args.covariates is None:
        user_covariates = None
    else:
        user_covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]

    X_cov, cov_names = covariate_design(
        gdf,
        include_coords=not args.no_coords,
        poly_degree=args.poly_degree,
        user_covariates=user_covariates,
    )

    yhat_cov, cov_r2, cov_model = fit_mean_model(
        X_cov,
        y,
        model_type=args.mean_model,
    )
    r_cov = y - yhat_cov

    print("\nCovariate mean model:")
    print(f"  model type = {args.mean_model}")
    print(f"  number of covariates including coordinate terms = {X_cov.shape[1]}")
    print(f"  R^2 = {cov_r2:.6f}")
    print("  covariates:")
    for c in cov_names:
        print(f"    {c}")

    rep = spectral_compatibility_report(
        name="Residual after covariate mean model",
        r=r_cov,
        lam=lam,
        U=U,
        out_dir=out_dir,
        out_prefix=f"{run_tag}_covariate_mean_car_compat",
        n_bins=args.n_bins,
        rho_max=args.rho_max,
        zero_tol=args.zero_tol,
    )
    rep["mean_model_type"] = args.mean_model
    rep["mean_model_r2"] = float(cov_r2)
    rep["mean_model_n_covariates"] = int(X_cov.shape[1])
    rep["mean_model_covariates"] = ",".join(cov_names)
    reports.append(rep)

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------
    summary = pd.DataFrame(reports)

    summary_path = out_dir / f"{run_tag}_car_compatibility_summary.csv"
    summary.to_csv(summary_path, index=False)

    json_path = out_dir / f"{run_tag}_car_compatibility_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("\nSaved CAR compatibility report:")
    print(f"  {summary_path}")
    print(f"  {json_path}")

    important_cols = [
        "name",
        "mean_model_r2",
        "raw_low",
        "raw_middle",
        "raw_high",
        "raw_hf_ratio",
        "car_whitened_low",
        "car_whitened_middle",
        "car_whitened_high",
        "car_whitened_lf_excess",
        "car_whitened_lf_ratio",
        "car_whitened_hf_ratio",
        "car_whitened_cum_dev_uniform",
        "car_fit_rho",
        "car_fit_tau2",
        "car_fit_sigma2",
        "car_fit_error",
    ]

    available = [c for c in important_cols if c in summary.columns]

    print("\nMost important CAR-compatibility columns:")
    print(summary[available].to_string(index=False))


if __name__ == "__main__":
    main()