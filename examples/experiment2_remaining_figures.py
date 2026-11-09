r"""Generate dissertation figures 2-4 from FINAL Experiment 2 CSV files.

Usage (PowerShell, from the project root):
    python -m examples.experiment2_remaining_figures `
        --experiment-dir "$base\experiment2_N144_final_two_start" `
        --output-dir "$base\experiment2_N144_final_two_start\required_plots"

Only reads existing CSVs; does not fit or modify either spatial model.
Saves publication-resolution PNG and vector PDF versions of each figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works without a graphical desktop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

LEROUX = "Leroux CAR"
SDM = "Adaptive P-spline SDM-CAR"
EXPECTED_SEEDS = set(range(1001, 1051))


def read_final_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return df


def check_complete(df: pd.DataFrame, name: str, rows_per_seed: int) -> None:
    actual = set(df["seed"].astype(int).unique())
    if actual != EXPECTED_SEEDS:
        raise ValueError(
            f"{name}: expected seeds 1001-1050, but missing "
            f"{sorted(EXPECTED_SEEDS - actual)} and extra {sorted(actual - EXPECTED_SEEDS)}"
        )
    counts = df.groupby("seed").size()
    if not (counts == rows_per_seed).all():
        raise ValueError(f"{name}: each seed must have {rows_per_seed} rows")


def paired_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    tmp = df.pivot(index="seed", columns="model", values=metric)
    if tmp.isna().any().any() or not {LEROUX, SDM}.issubset(tmp.columns):
        raise ValueError(f"Incomplete model pairing for {metric}")
    if len(tmp) != 50:
        raise ValueError(f"Expected 50 paired datasets for {metric}")
    return tmp.sort_index()


def average_and_bands(data: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and empirical 2.5/97.5 percentiles *across datasets* per graph mode."""
    mat = data.pivot(index="seed", columns="mode", values=column)
    if mat.shape != (50, 144) or mat.isna().any().any():
        raise ValueError(f"Expected complete 50 x 144 spectral bank for {column}")
    vals = mat.to_numpy(dtype=float)
    return (np.mean(vals, axis=0),
            np.quantile(vals, 0.025, axis=0),
            np.quantile(vals, 0.975, axis=0))


def format_figure(ax):
    ax.grid(True, color="0.88", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def save_both(fig, output_dir: Path, basename: str) -> None:
    for ext in ("png", "pdf"):
        path = output_dir / f"{basename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def figure2_precision_and_variance(
    precision: pd.DataFrame, spectra: pd.DataFrame, output_dir: Path,
    leroux_color: str, sdm_color: str,
) -> None:
    """Figure 2: two spectral scales with *cross-dataset* dispersion bands."""
    check_complete(precision, "precision recovery", 144)
    check_complete(spectra, "modal variance recovery", 144)
    if precision.duplicated(["seed", "mode"]).any() or spectra.duplicated(["seed", "mode"]).any():
        raise ValueError("Duplicate (seed, mode) records in spectral tables")
    truth = precision.loc[precision.seed == 1001].sort_values("mode")
    x = truth["lambda"].to_numpy(dtype=float)
    q_true = truth["q_true"].to_numpy(dtype=float)
    variance_truth = spectra.loc[spectra.seed == 1001].sort_values("mode")
    if not np.allclose(variance_truth["lambda"], x, rtol=0, atol=1e-10):
        raise ValueError("Modal eigenvalues do not match between tables")
    F_true = variance_truth["F_true"].to_numpy(dtype=float)
    if not np.allclose(q_true * F_true, 1, atol=1e-8):
        raise ValueError("Generating precision and variance are inconsistent")

    q_l, q_l_lo, q_l_hi = average_and_bands(precision, "q_leroux_mean")
    q_s, q_s_lo, q_s_hi = average_and_bands(precision, "q_sdm_mean")
    F_l, F_l_lo, F_l_hi = average_and_bands(spectra, "F_leroux_median")
    F_s, F_s_lo, F_s_hi = average_and_bands(spectra, "F_sdm_median")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35), layout="constrained")
    ax = axes[0]
    ax.fill_between(x, q_l_lo, q_l_hi, color=leroux_color, alpha=0.12, linewidth=0, zorder=1)
    ax.fill_between(x, q_s_lo, q_s_hi, color=sdm_color, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(x, q_true, color="black", linewidth=2.1, label="Generating truth", zorder=4)
    ax.plot(x, q_l, color=leroux_color, linewidth=1.8, label="Leroux-CAR", zorder=3)
    ax.plot(x, q_s, color=sdm_color, linewidth=1.8, label="SDM-CAR", zorder=3)
    ax.set(xlabel="Graph Laplacian eigenvalue", ylabel="Modal precision")
    ax.set_title("(a) Precision recovery", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    format_figure(ax)

    ax = axes[1]
    ax.fill_between(x, np.maximum(F_l_lo, 1e-12), F_l_hi,
                    color=leroux_color, alpha=0.12, linewidth=0, zorder=1)
    ax.fill_between(x, np.maximum(F_s_lo, 1e-12), F_s_hi,
                    color=sdm_color, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(x, F_true, color="black", linewidth=2.1, label="Generating truth", zorder=4)
    ax.plot(x, F_l, color=leroux_color, linewidth=1.8, label="Leroux-CAR", zorder=3)
    ax.plot(x, F_s, color=sdm_color, linewidth=1.8, label="SDM-CAR", zorder=3)
    ax.set_yscale("log")
    ax.set(xlabel="Graph Laplacian eigenvalue", ylabel="Modal variance (log axis)")
    ax.set_title("(b) Modal-variance recovery", loc="left", fontsize=11)
    format_figure(ax)
    fig.text(0.5, -0.025,
             "Shading: empirical 2.5th–97.5th percentile across 50 datasets (not posterior credible intervals).",
             ha="center", fontsize=9, color="0.35")
    save_both(fig, output_dir, "exp2_fig2_spectral_recovery_50")


def figure3_joint_prediction(pred: pd.DataFrame, output_dir: Path, point_color: str) -> None:
    """Figure 3: per-dataset paired oracle KL and covariance Frobenius error."""
    check_complete(pred, "predictive metrics", 2)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.65), layout="constrained")
    definitions = [
        ("oracle_conditional_KL_per_site", "Oracle conditional KL / site"),
        ("joint_predictive_cov_relative_frobenius", "Relative conditional covariance error"),
    ]
    for i, (ax, (metric, label)) in enumerate(zip(axes, definitions)):
        pair = paired_metric(pred, metric)
        x = pair[LEROUX].to_numpy(dtype=float)
        y = pair[SDM].to_numpy(dtype=float)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        span = max(hi - lo, 1e-4)
        lo = max(0.0, lo - 0.08 * span)
        hi = hi + 0.08 * span
        ax.plot([lo, hi], [lo, hi], "--", color="0.45", linewidth=1.2, zorder=1)
        ax.scatter(x, y, s=35, alpha=0.78, edgecolor="white", linewidth=0.4,
                   color=point_color, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        n_better = int(np.count_nonzero(y < x))
        mean_diff = float(np.mean(y - x))
        ax.text(0.04, 0.97,
                f"SDM lower: {n_better}/50\nMean paired difference: {mean_diff:+.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.8)
        ax.set(xlabel=f"Leroux-CAR: {label}", ylabel=f"SDM-CAR: {label}")
        ax.set_title(f"({chr(ord('a') + i)}) {label}", loc="left", fontsize=10.5)
        format_figure(ax)
    fig.text(0.5, -0.02,
             "Each point is a dataset; points below the dashed equality line favor SDM-CAR.",
             ha="center", fontsize=9, color="0.35")
    save_both(fig, output_dir, "exp2_fig3_paired_joint_prediction")


def mean_ci95(diffs: np.ndarray) -> tuple[float, float, float]:
    n = len(diffs)
    mean = float(np.mean(diffs))
    half = float(student_t.ppf(0.975, n - 1) * np.std(diffs, ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def figure4_zero_mode(
    spectral_metrics: pd.DataFrame, spectral_curves: pd.DataFrame,
    output_dir: Path, leroux_color: str, sdm_color: str,
) -> None:
    """Figure 4: posterior-median zero mode and modal-variance error decomposition."""
    check_complete(spectral_metrics, "spectral metrics", 2)
    check_complete(spectral_curves, "spectral recovery", 144)
    zero = spectral_curves.loc[spectral_curves["mode"].eq(0)].sort_values("seed")
    if len(zero) != 50:
        raise ValueError("Expected exactly one zero-eigenmode row per dataset")
    truth_values = zero["F_true"].to_numpy(dtype=float)
    if not np.allclose(truth_values, truth_values[0], rtol=0, atol=1e-10):
        raise ValueError("Zero-mode generating variance varies by seed")
    truth_f0 = float(truth_values[0])
    f0_l = zero["F_leroux_median"].to_numpy(dtype=float)
    f0_s = zero["F_sdm_median"].to_numpy(dtype=float)
    if np.any(f0_l <= 0) or np.any(f0_s <= 0):
        raise ValueError("All modal variances must be strictly positive")

    total = paired_metric(spectral_metrics, "spectrum_relative_L2")
    pos = paired_metric(spectral_metrics, "variance_positive_mode_relative_L2")
    if not np.array_equal(total.index.to_numpy(), pos.index.to_numpy()):
        raise ValueError("Paired total and positive-mode results have different seeds")
    d_total = (total[SDM] - total[LEROUX]).to_numpy(dtype=float)
    d_pos = (pos[SDM] - pos[LEROUX]).to_numpy(dtype=float)

    rng = np.random.default_rng(20261108)  # Plot jitter only, not model randomness
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.75), layout="constrained")
    ax = axes[0]
    # Boxplots are raw posterior medians; logarithmic axis displays their high-end range.
    bp = ax.boxplot([f0_l, f0_s], positions=[0, 1], widths=0.42,
                    showfliers=False, patch_artist=False, medianprops={"color": "black"})
    for x0, vals, color in [(0, f0_l, leroux_color), (1, f0_s, sdm_color)]:
        xvals = x0 + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xvals, vals, s=19, alpha=0.62, color=color,
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(truth_f0, color="0.2", linestyle="--", linewidth=1.15,
               label=f"True $F(0)$ = {truth_f0:.2f}")
    ax.set_xticks([0, 1], ["Leroux-CAR", "SDM-CAR"])
    ax.set_yscale("log")
    ax.set(ylabel="Estimated zero-mode variance (log axis)")
    ax.set_title("(a) Zero-mode variance estimates", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    format_figure(ax)

    ax = axes[1]
    for x0, diffs, color in [(0, d_total, sdm_color), (1, d_pos, leroux_color)]:
        xvals = x0 + rng.uniform(-0.16, 0.16, size=len(diffs))
        ax.scatter(xvals, diffs, s=19, alpha=0.53, color=color,
                   edgecolor="white", linewidth=0.3, zorder=3)
        mean, ci_lo, ci_hi = mean_ci95(diffs)
        ax.errorbar(x0, mean, yerr=np.array([[mean-ci_lo], [ci_hi-mean]]),
                    fmt="D", color="black", capsize=4, markersize=5.7,
                    linewidth=1.3, zorder=5)
        print(f"{'Total variance' if x0 == 0 else 'Positive modes'}: "
              f"mean SDM−Leroux={mean:+.5f}, 95% paired CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xticks([0, 1], ["Total modal\nvariance error", "Positive-mode-only\nvariance error"])
    ax.set_ylabel("Error difference (SDM-CAR − Leroux-CAR)")
    ax.set_title("(b) Paired relative variance errors", loc="left", fontsize=11)
    # Label the largest total-error difference cases as diagnostics.
    for seed in [1030, 1042, 1050]:
        index = total.index.get_loc(seed)
        ax.annotate(str(seed), xy=(0.04, d_total[index]), xytext=(9, 0),
                    textcoords="offset points", fontsize=8, color="0.25")
    format_figure(ax)
    fig.text(0.5, -0.025,
             "Dots: individual datasets. Diamonds and whiskers: mean paired difference and 95% t interval.",
             ha="center", fontsize=9, color="0.35")
    save_both(fig, output_dir, "exp2_fig4_zero_mode_decomposition")


def main():
    parser = argparse.ArgumentParser(description="Final Experiment 2 figures 2-4 (no model fitting).")
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Folder containing master_tables and per_run")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: EXPERIMENT_DIR/required_plots")
    args = parser.parse_args()
    base = args.experiment_dir.resolve()
    output = (args.output_dir or base / "required_plots").resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = base / "master_tables"
    precision = read_final_table(tables / "all_runs__precision_recovery.csv",
        {"seed", "mode", "lambda", "q_true", "q_leroux_mean", "q_sdm_mean"})
    spectra = read_final_table(tables / "all_runs__spectral_recovery.csv",
        {"seed", "mode", "lambda", "F_true", "F_leroux_median", "F_sdm_median"})
    predictive = read_final_table(tables / "all_runs__predictive_metrics.csv",
        {"seed", "model", "oracle_conditional_KL_per_site",
         "joint_predictive_cov_relative_frobenius"})
    spectral_metrics = read_final_table(tables / "all_runs__spectral_metrics.csv",
        {"seed", "model", "spectrum_relative_L2", "variance_positive_mode_relative_L2"})
    check_complete(predictive, "predictive metrics", 2)
    check_complete(spectral_metrics, "spectral metrics", 2)
    # Match one model row for each (seed, family), without silently averaging duplicates.
    for frame, name in [(predictive, "predictive"), (spectral_metrics, "spectral")]:
        if frame.duplicated(["seed", "model"]).any():
            raise ValueError(f"Duplicate model rows in {name} metrics")
        if set(frame["model"].unique()) != {LEROUX, SDM}:
            raise ValueError(f"Unexpected fitted family names in {name} metrics")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    leroux_color, sdm_color = default_colors[0], default_colors[1]
    figure2_precision_and_variance(precision, spectra, output, leroux_color, sdm_color)
    figure3_joint_prediction(predictive, output, sdm_color)
    figure4_zero_mode(spectral_metrics, spectra, output, leroux_color, sdm_color)
    print("Complete: three figures, each saved as PNG and PDF. No model was refitted.")


if __name__ == "__main__":
    main()
