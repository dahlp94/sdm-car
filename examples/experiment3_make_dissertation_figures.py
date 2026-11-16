#!/usr/bin/env python3
r"""Create the final Experiment 3 dissertation figures from the 50-seed master CSVs.

Dependencies: numpy, pandas, matplotlib (no fitting, no PyTorch, no SciPy).

Example (PowerShell):
    python -m pip install numpy pandas matplotlib
    python .\examples\experiment3_make_dissertation_figures.py `
        --root "D:\sdm-car-results\experiment3_N144_final_power_warp" `
        --outdir "D:\sdm-car-results\experiment3_N144_final_power_warp\dissertation_figures"

The script reads only:
    master_tables/all_runs__predictive.csv
    master_tables/all_runs__covariance.csv
    master_tables/per_seed__aggregates.csv

There must be 50 independent seeds (1001--1050), 5 warp levels, and 2
model families. Four main figures plus an optional appendix figure are
produced as 300-dpi PNGs and vector PDFs, along with an audit CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on remote/headless Windows sessions.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GAMMAS = (1.00, 0.90, 0.75, 0.50, 0.35)  # left to right; warping strengthens
SEEDS = tuple(range(1001, 1051))
LEROUX = "Leroux CAR"
SDM = "Adaptive P-spline SDM-CAR"
MODELS = (LEROUX, SDM)
DISPLAY = {LEROUX: "Leroux-CAR", SDM: "SDM-CAR"}
COLORS = {LEROUX: "#1F77B4", SDM: "#D95F02"}  # agreed blue / orange
MARKERS = {LEROUX: "o", SDM: "s"}
LINESTYLES = {LEROUX: "-", SDM: "--"}
TRUE_ZERO_VARIANCE = 0.70 / (1.0 - 0.92)  # exactly 8.75


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.32,
        "lines.linewidth": 2.1,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / "master_tables" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required final-run table not found: {path}")
    return pd.read_csv(path)


def validate_panel(df: pd.DataFrame, label: str, *, scales: bool = False) -> pd.DataFrame:
    """Fail on incomplete/mixed experiments instead of drawing misleading figures."""
    df = df.copy()
    required = ["seed", "gamma", "model"] + (["scale"] if scales else [])
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["gamma"] = pd.to_numeric(df["gamma"], errors="raise").round(3)
    df["model"] = df["model"].astype(str).str.strip()

    if set(df["seed"].unique()) != set(SEEDS):
        raise ValueError(f"{label}: expected seeds 1001--1050, found {sorted(df.seed.unique())}")
    if set(df["gamma"].unique()) != set(GAMMAS):
        raise ValueError(f"{label}: wrong gamma grid: {sorted(df.gamma.unique())}")
    if set(df["model"].unique()) != set(MODELS):
        raise ValueError(f"{label}: unexpected models: {sorted(df.model.unique())}")

    keys = required
    if df.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate seed/gamma/model{'/scale' if scales else ''} rows")
    expected_scales = {"full_4x4", "subblock_2x2"}
    if scales and set(df["scale"].unique()) != expected_scales:
        raise ValueError(f"{label}: expected {expected_scales}, found {set(df.scale.unique())}")
    expected_count = len(SEEDS) * len(GAMMAS) * len(MODELS) * (2 if scales else 1)
    if len(df) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} rows, got {len(df)}")
    return df


def required_numeric(df: pd.DataFrame, label: str, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"{label}: missing metric column: {col}")
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"{label}: non-finite values in {col}")
    return df


def prepare_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels([f"{g:.2f}" for g in GAMMAS])
    ax.set_xlim(1.035, 0.315)  # 1.00 at left; 0.35 at right
    ax.set_xlabel(r"Warp exponent $\gamma$ (stronger warping $\longrightarrow$)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def model_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
               linestyle=LINESTYLES[m], linewidth=2, label=DISPLAY[m])
        for m in MODELS
    ]


def draw_means(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str,
               panel_title: str, audit_rows: list[dict], name: str,
               scale: str | None = None) -> None:
    for model in MODELS:
        part = df[df["model"] == model]
        if scale is not None:
            part = part[part["scale"] == scale]
        grouped = part.groupby("gamma", sort=False)[metric].agg(["mean", "std", "count"])
        grouped = grouped.reindex(GAMMAS)
        if grouped.isna().any().any() or not (grouped["count"] == 50).all():
            raise ValueError(f"Expected 50 non-missing observations for {metric}, {model}")
        means = grouped["mean"].to_numpy()
        ax.plot(GAMMAS, means, color=COLORS[model], marker=MARKERS[model],
                linestyle=LINESTYLES[model], markeredgecolor="white",
                markeredgewidth=0.7, label=DISPLAY[model])
        for g in GAMMAS:
            audit_rows.append({
                "figure": name,
                "panel": panel_title,
                "metric": metric,
                "scale": scale or "",
                "model": model,
                "gamma": g,
                "n_seeds": int(grouped.loc[g, "count"]),
                "mean": float(grouped.loc[g, "mean"]),
                "sd_across_seeds": float(grouped.loc[g, "std"]),
            })
    prepare_axes(ax, ylabel)
    ax.set_title(panel_title, loc="left", pad=10)
    ax.set_ylim(bottom=0)  # all plotted metrics are nonnegative


def finish_two_panel(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.legend(handles=model_legend(), loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, frameon=False, handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.4)
    save_figure(fig, outdir, stem)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        path = outdir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)


def fig1_joint_prediction(pred: pd.DataFrame, cov: pd.DataFrame,
                          outdir: Path, audit: list[dict]) -> None:
    stem = "experiment3_fig1_joint_prediction"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35))
    draw_means(axes[0], pred, "oracle_conditional_KL_per_site",
               "Oracle conditional KL per site", "(a) Oracle conditional KL",
               audit, stem)
    draw_means(axes[1], cov, "joint_predictive_cov_relative_frobenius",
               "Relative Frobenius error", "(b) Joint predictive covariance",
               audit, stem)
    finish_two_panel(fig, outdir, stem)


def fig2_latent_covariance(cov: pd.DataFrame,
                           outdir: Path, audit: list[dict]) -> None:
    stem = "experiment3_fig2_latent_covariance"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35))
    draw_means(axes[0], cov, "positive_latent_cov_relative_frobenius",
               "Relative Frobenius error", "(a) Positive-mode latent covariance",
               audit, stem)
    draw_means(axes[1], cov, "full_latent_cov_relative_frobenius",
               "Relative Frobenius error", "(b) Full latent covariance",
               audit, stem)
    finish_two_panel(fig, outdir, stem)


def fig3_zero_mode(cov: pd.DataFrame, outdir: Path, audit: list[dict]) -> None:
    stem = "experiment3_fig3_zero_mode_variance"
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    width = 0.28
    offsets = {LEROUX: -0.18, SDM: +0.18}
    for model in MODELS:
        arrays = [
            cov.loc[(cov["gamma"] == g) & (cov["model"] == model),
                    "zero_mode_variance_mean"].to_numpy()
            for g in GAMMAS
        ]
        if any(len(v) != 50 for v in arrays):
            raise ValueError(f"Incorrect zero-mode sample counts for {model}")
        positions = np.arange(len(GAMMAS)) + offsets[model]
        result = ax.boxplot(
            arrays, positions=positions, widths=width, patch_artist=True,
            manage_ticks=False, whis=1.5, showfliers=True, showmeans=True,
            boxprops={"linewidth": 1.1, "edgecolor": COLORS[model]},
            whiskerprops={"color": COLORS[model], "linewidth": 1.1},
            capprops={"color": COLORS[model], "linewidth": 1.1},
            medianprops={"color": "#151515", "linewidth": 1.6},
            meanprops={"marker": "D", "markersize": 4.1, "markerfacecolor": "white",
                       "markeredgecolor": "#151515", "linestyle": "none"},
            flierprops={"marker": "o", "markersize": 3.2,
                        "markerfacecolor": "none", "markeredgecolor": COLORS[model],
                        "alpha": 0.65},
        )
        for box in result["boxes"]:
            box.set_facecolor(COLORS[model])
            box.set_alpha(0.38)
        for g, values in zip(GAMMAS, arrays):
            audit.append({
                "figure": stem,
                "panel": "Zero-mode variance",
                "metric": "zero_mode_variance_mean",
                "scale": "",
                "model": model,
                "gamma": g,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "sd_across_seeds": float(np.std(values, ddof=1)),
            })
    ax.axhline(TRUE_ZERO_VARIANCE, color="#404040", linestyle=":", linewidth=1.8)
    ax.set_xlim(-0.58, len(GAMMAS)-0.42)
    ax.set_xticks(range(len(GAMMAS)))
    ax.set_xticklabels([f"{g:.2f}" for g in GAMMAS])
    ax.set_xlabel(r"Warp exponent $\gamma$ (stronger warping $\longrightarrow$)")
    ax.set_ylabel(r"Estimated zero-mode variance $\widehat{F}(0)$")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=COLORS[m], edgecolor=COLORS[m], alpha=0.45,
              label=DISPLAY[m]) for m in MODELS
    ]
    handles += [
        Line2D([0], [0], color="#404040", linestyle=":", linewidth=1.8,
               label=r"True $F_0(0)=8.75$"),
        Line2D([0], [0], marker="D", markersize=5, markerfacecolor="white",
               markeredgecolor="#151515", linestyle="none", label="Mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, outdir, stem)


def fig4_spatial_aggregates(agg: pd.DataFrame,
                            outdir: Path, audit: list[dict]) -> None:
    stem = "experiment3_fig4_spatial_aggregates"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35))
    for ax, scale, title in zip(
        axes, ("full_4x4", "subblock_2x2"),
        ("(a) Full $4\\times4$ block average", "(b) Mean of $2\\times2$ subblock errors")
    ):
        draw_means(ax, agg, "absolute_relative_variance_error",
                   "Mean absolute relative variance error", title,
                   audit, stem, scale=scale)
    finish_two_panel(fig, outdir, stem)


def fig_appendix_rmse_nlpd(pred: pd.DataFrame,
                            outdir: Path, audit: list[dict]) -> None:
    stem = "experiment3_appendix_rmse_nlpd"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35))
    draw_means(axes[0], pred, "response_RMSE", "Held-out response RMSE",
               "(a) Response RMSE", audit, stem)
    draw_means(axes[1], pred, "joint_response_NLPD_per_site",
               "Joint NLPD per site", "(b) Joint predictive NLPD", audit, stem)
    # Unlike a zero-baseline panel, a tight y-axis reveals modest differences.
    for ax in axes:
        ax.autoscale(axis="y", tight=False)
        all_y = np.concatenate([line.get_ydata() for line in ax.lines])
        lo, hi = float(all_y.min()), float(all_y.max())
        pad = max((hi - lo) * 0.2, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
    finish_two_panel(fig, outdir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="Experiment 3 final output folder containing master_tables")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Destination; defaults to ROOT/dissertation_figures")
    parser.add_argument("--no-appendix", action="store_true",
                        help="Skip the optional RMSE + NLPD appendix figure")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    outdir = (args.outdir if args.outdir is not None else root / "dissertation_figures")
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pred = validate_panel(read_csv(root, "all_runs__predictive.csv"), "predictive")
    cov = validate_panel(read_csv(root, "all_runs__covariance.csv"), "covariance")
    agg = validate_panel(read_csv(root, "per_seed__aggregates.csv"),
                         "aggregates", scales=True)
    pred = required_numeric(pred, "predictive", [
        "oracle_conditional_KL_per_site", "response_RMSE",
        "joint_response_NLPD_per_site",
    ])
    cov = required_numeric(cov, "covariance", [
        "joint_predictive_cov_relative_frobenius",
        "positive_latent_cov_relative_frobenius",
        "full_latent_cov_relative_frobenius",
        "zero_mode_variance_mean", "zero_mode_variance_true",
    ])
    agg = required_numeric(agg, "aggregates", ["absolute_relative_variance_error"])
    if not np.allclose(cov["zero_mode_variance_true"], TRUE_ZERO_VARIANCE,
                       atol=1e-9, rtol=0):
        raise ValueError("Reported true zero-mode variance differs from 8.75")
    if (cov["zero_mode_variance_mean"] <= 0).any():
        raise ValueError("Nonpositive estimated zero-mode variance")
    expected_keys = set(zip(pred.seed, pred.gamma, pred.model))
    if set(zip(cov.seed, cov.gamma, cov.model)) != expected_keys:
        raise ValueError("Predictive and covariance tables are not perfectly paired")
    if set(zip(agg.seed, agg.gamma, agg.model)) != expected_keys:
        raise ValueError("Aggregate table is not paired with predictive/covariance tables")
    print("Validated: 50 seeds, 5 warp levels, 2 models; 500 fits and 1000 aggregate rows.")

    setup_style()
    audit: list[dict] = []
    fig1_joint_prediction(pred, cov, outdir, audit)
    fig2_latent_covariance(cov, outdir, audit)
    fig3_zero_mode(cov, outdir, audit)
    fig4_spatial_aggregates(agg, outdir, audit)
    if not args.no_appendix:
        fig_appendix_rmse_nlpd(pred, outdir, audit)
    audit_path = outdir / "experiment3_figure_means_audit.csv"
    pd.DataFrame(audit).to_csv(audit_path, index=False, float_format="%.10g")
    print(f"Saved: {audit_path}")
    print("Finished: 4 main figures" + ("." if args.no_appendix else " + 1 appendix figure."))


if __name__ == "__main__":
    main()
