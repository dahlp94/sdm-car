# scripts/02_explore_fentanyl_data.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAPH_PATH = Path("data/graph/county_graph_conus.npz")
CENTROIDS_PATH = Path("data/raw/county_centroids_conus.csv")
OUTDIR = Path("figures/fentanyl")


def normal_quantiles(n: int) -> np.ndarray:
    """
    Approximate standard normal quantiles without scipy.
    Uses a large simulated sample to avoid extra dependencies.
    """
    rng = np.random.default_rng(0)
    z = np.sort(rng.standard_normal(200_000))
    probs = (np.arange(1, n + 1) - 0.5) / n
    idx = np.clip((probs * (len(z) - 1)).astype(int), 0, len(z) - 1)
    return z[idx]


def morans_i(y: np.ndarray, W: np.ndarray) -> float:
    """
    Compute global Moran's I for a response y and adjacency/weight matrix W.
    """
    y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=float)

    n = len(y)
    z = y - y.mean()
    S0 = W.sum()

    if S0 <= 0:
        raise ValueError("Weight matrix W has zero total weight.")

    num = z @ W @ z
    den = z @ z

    if den <= 0:
        raise ValueError("Centered response has zero variance.")

    return float((n / S0) * (num / den))


def morans_i_permutation_test(
    y: np.ndarray,
    W: np.ndarray,
    n_perm: int = 999,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """
    Permutation test for Moran's I.

    Returns:
        I_obs  : observed Moran's I
        p_perm : one-sided permutation p-value for positive autocorrelation
        I_perm : permutation distribution
    """
    rng = np.random.default_rng(seed)

    I_obs = morans_i(y, W)

    I_perm = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        y_perm = rng.permutation(y)
        I_perm[b] = morans_i(y_perm, W)

    p_perm = (1.0 + np.sum(I_perm >= I_obs)) / (n_perm + 1.0)

    return I_obs, float(p_perm), I_perm


def make_histogram(y: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(y, bins=30, edgecolor="black")
    plt.xlabel("Momentum outcome y")
    plt.ylabel("Count")
    plt.title("Histogram of fentanyl mortality momentum")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def make_qqplot(y: np.ndarray, save_path: Path) -> None:
    y_sorted = np.sort(y)
    q_theory = normal_quantiles(len(y_sorted))

    # line through quartiles
    y_q1, y_q3 = np.quantile(y_sorted, [0.25, 0.75])
    z_q1, z_q3 = np.quantile(q_theory, [0.25, 0.75])
    slope = (y_q3 - y_q1) / (z_q3 - z_q1)
    intercept = y_q1 - slope * z_q1

    x_line = np.array([q_theory.min(), q_theory.max()])
    y_line = intercept + slope * x_line

    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(q_theory, y_sorted, s=12, alpha=0.7)
    plt.plot(x_line, y_line, linewidth=2)
    plt.xlabel("Theoretical Normal Quantiles")
    plt.ylabel("Observed Quantiles of y")
    plt.title("QQ plot of fentanyl mortality momentum")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def make_spatial_map(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(9.0, 5.5))
    sc = plt.scatter(
        df["lon"],
        df["lat"],
        c=df["y"],
        s=18,
        alpha=0.85,
    )
    plt.colorbar(sc, label="Momentum outcome y")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial map of fentanyl mortality momentum")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_morans_i_permutation(I_obs: float, I_perm: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(I_perm, bins=30, edgecolor="black")
    plt.axvline(I_obs, linewidth=2, linestyle="--", label=f"Observed I = {I_obs:.3f}")
    plt.xlabel("Permutation Moran's I")
    plt.ylabel("Count")
    plt.title("Moran's I permutation test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")
    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"Centroid file not found: {CENTROIDS_PATH}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    graph = np.load(GRAPH_PATH, allow_pickle=True)
    W = graph["W"].astype(float)

    cent = pd.read_csv(CENTROIDS_PATH, dtype={"fips": str})

    df = pd.DataFrame({
        "fips": pd.Series(graph["fips"]).astype(str).str.zfill(5),
        "county": graph["county"],
        "state": graph["state"],
        "y": graph["y"].astype(float),
    })

    cent["fips"] = cent["fips"].astype(str).str.zfill(5)

    merged = df.merge(cent[["fips", "lon", "lat"]], on="fips", how="left")

    missing_coords = merged["lon"].isna().sum() + merged["lat"].isna().sum()
    if missing_coords > 0:
        raise ValueError("Some modeled counties are missing centroid coordinates.")

    y = merged["y"].to_numpy()

    print("Exploration summary")
    print("-------------------")
    print(f"n counties: {len(merged)}")
    print(merged["y"].describe())

    print("\nTop 10 counties by y:")
    print(
        merged.sort_values("y", ascending=False)[["fips", "county", "state", "y"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nBottom 10 counties by y:")
    print(
        merged.sort_values("y")[["fips", "county", "state", "y"]]
        .head(10)
        .to_string(index=False)
    )

    I_obs, p_perm, I_perm = morans_i_permutation_test(y, W, n_perm=999, seed=0)

    print("\nMoran's I")
    print("---------")
    print(f"Observed Moran's I: {I_obs:.6f}")
    print(f"Permutation p-value (positive autocorrelation): {p_perm:.6f}")

    make_histogram(y, OUTDIR / "hist_y.png")
    make_qqplot(y, OUTDIR / "qq_y.png")
    make_spatial_map(merged, OUTDIR / "map_y.png")
    plot_morans_i_permutation(I_obs, I_perm, OUTDIR / "morans_i_perm.png")

    print(f"\nSaved figures to: {OUTDIR}")
    print(" - hist_y.png")
    print(" - qq_y.png")
    print(" - map_y.png")
    print(" - morans_i_perm.png")


if __name__ == "__main__":
    main()