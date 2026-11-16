"""Experiment 3: publication-ready landscape numerical convergence figure.

Uses final experiment convergence-history CSV files. Does not refit any model.
Default examples match Experiment 3 appendix: Leroux seed 1022/gamma 0.75
and SDM-CAR seed 1004/gamma 1.00.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Prespecified operational stability thresholds.
ELBO_THRESHOLD = 0.010
PRECISION_THRESHOLD = 0.005
ZERO_VARIANCE_THRESHOLD = 0.020
CONSECUTIVE_THRESHOLD = 3

PANEL_SPECS = [
    ("ELBO_abs_upper_2MCSE_per_site", "ELBO change + 2 MCSE / site", ELBO_THRESHOLD, True),
    ("precision_relative_change", "Relative precision change", PRECISION_THRESHOLD, True),
    ("zero_mode_variance_relative_change", "Relative zero-mode variance change", ZERO_VARIANCE_THRESHOLD, True),
    ("pilot_consecutive_passes", "Consecutive passing checkpoints", CONSECUTIVE_THRESHOLD, False),
]

CASES = [
    ("Leroux-CAR", "leroux", 1022, 0.75, "#1f77b4"),
    ("SDM-CAR", "sdm", 1004, 1.00, "#ff7f0e"),
]


def load_case(root: Path, model_tag: str, seed: int, gamma: float) -> pd.DataFrame:
    stem = f"seed_{seed:05d}__gamma_{gamma:.3f}".replace(".", "p")
    path = (
        root / "per_run" / f"seed_{seed:05d}"
        / f"gamma_{gamma:.3f}".replace(".", "p")
        / f"{stem}__{model_tag}_convergence.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing convergence data: {path}")
    df = pd.read_csv(path)
    needed = {"iteration", "pilot_persistent_pass"} | {spec[0] for spec in PANEL_SPECS}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.sort_values("iteration").reset_index(drop=True)


def make_plot(root: Path, output: Path) -> None:
    data = []
    for name, tag, seed, gamma, color in CASES:
        df = load_case(root, tag, seed, gamma)
        persistent = df.loc[df["pilot_persistent_pass"].astype(bool), "iteration"]
        first_persistent = int(persistent.iloc[0]) if not persistent.empty else None
        final_iteration = int(df["iteration"].max())
        data.append((name, seed, gamma, color, df, first_persistent, final_iteration))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Reserve dedicated strips ABOVE each model row for headers.  No suptitle.
    fig = plt.figure(figsize=(17, 8), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        left=0.055, right=0.985, bottom=0.095, top=0.84,
        wspace=0.31, hspace=0.65,
    )

    for r, (name, seed, gamma, color, df, first_persistent, final_iteration) in enumerate(data):
        # These headers are in *separate reserved whitespace*, never inside axes.
        header_y = 0.945 if r == 0 else 0.483
        fig.text(
            0.055, header_y,
            rf"{name}   |   seed {seed}   |   $\gamma={gamma:.2f}$   |   final iteration {final_iteration:,}",
            fontsize=12, fontweight="bold", color=color, ha="left", va="top",
        )

        x = pd.to_numeric(df["iteration"], errors="raise").to_numpy()
        for c, (column, title, threshold, logarithmic) in enumerate(PANEL_SPECS):
            ax = fig.add_subplot(gs[r, c])
            y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=color, lw=1.7, solid_capstyle="round")
            ax.axhline(threshold, color="#555555", linestyle="--", lw=1.1)
            if first_persistent is not None:
                ax.axvline(first_persistent, color="#7b4fa3", ls=":", lw=1.5)
            ax.axvline(final_iteration, color="#333333", ls="-.", lw=1.2)
            ax.set_xlim(0, max(1, final_iteration * 1.035))

            if logarithmic:
                positive = y[np.isfinite(y) & (y > 0)]
                if positive.size == 0:
                    raise ValueError(f"No positive diagnostic values: {column}, {name}")
                ax.set_yscale("log")
                ax.set_ylim(max(1e-7, min(positive.min(), threshold) * .6),
                            max(positive.max(), threshold) * 1.6)
            else:
                ax.set_ylim(-0.25, max(4.5, np.nanmax(y) + .7))
                ax.set_yticks(np.arange(0, max(5, int(np.nanmax(y)) + 2), 1))

            # Only the first row needs panel headings. Eliminates duplicate headings.
            if r == 0:
                ax.set_title(title, pad=13)
            if r == 1:
                ax.set_xlabel("Iteration")
            ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
            ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.5)
            ax.tick_params(axis="x", labelrotation=0, pad=4)
            ax.tick_params(axis="y", pad=3)

    # Compact line-style legend in a separate clear bottom strip.
    legend_handles = [
        Line2D([0], [0], color="#555555", linestyle="--", lw=1.2, label="Threshold"),
        Line2D([0], [0], color="#7b4fa3", linestyle=":", lw=1.5, label="First persistent pass"),
        Line2D([0], [0], color="#333333", linestyle="-.", lw=1.2, label="Final iteration"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.52, 0.003), frameon=False, fontsize=10,
               handlelength=3.0, columnspacing=2.5)

    output.mkdir(parents=True, exist_ok=True)
    png = output / "experiment3_appendix_numerical_trajectories.png"
    pdf = output / "experiment3_appendix_numerical_trajectories.pdf"
    # No tight bbox: preserve explicitly reserved margins and avoid new layout shifts.
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    audit = pd.DataFrame([
        {
            "model": name, "seed": seed, "gamma": gamma,
            "first_persistent_pass": first_persistent,
            "final_iteration": final_it,
        }
        for name, seed, gamma, _, _, first_persistent, final_it in data
    ])
    audit.to_csv(output / "experiment3_appendix_numerical_trajectories_audit.csv", index=False)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(audit.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Experiment 3 final output folder containing per_run")
    parser.add_argument("--outdir", type=Path,
                        help="Figure folder; defaults to <root>/dissertation_figures")
    args = parser.parse_args()
    make_plot(args.root, args.outdir or args.root / "dissertation_figures")


if __name__ == "__main__":
    main()
