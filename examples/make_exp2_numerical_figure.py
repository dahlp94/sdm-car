"""Plot Experiment 2 selected-SDM convergence histories for seeds 1003 and 1004.

Reads saved per-seed convergence CSVs; does not rerun model fitting.

Example
-------
python -m examples.make_exp2_numerical_figure \
  --experiment-dir "C:/Users/pd006/Desktop/internship_search/sdm-car/experiment2_N144_final_two_start" \
  --output-dir "C:/Users/pd006/Desktop/internship_search/sdm-car/Figures"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("ELBO_abs_upper_2MCSE_per_site", 0.01,
     r"$|\Delta\widehat{\mathrm{ELBO}}|+2\,\mathrm{MCSE}$ per training site"),
    ("precision_relative_change", 0.005,
     "Relative modal-precision change"),
    ("zero_mode_variance_relative_change_exact", 0.02,
     "Zero-mode variance relative change"),
]


def load_history(experiment_dir: Path, seed: int) -> pd.DataFrame:
    path = (experiment_dir / "per_run" / f"seed_{seed:05d}"
            / f"seed_{seed:05d}__sdm_convergence.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing convergence history: {path}")
    df = pd.read_csv(path).sort_values("iteration")
    required = {"iteration"} | {metric for metric, _, _ in METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} does not contain columns: {sorted(missing)}")
    return df


def make_figure(experiment_dir: Path, output_dir: Path, seeds: list[int],
                min_iteration: int = 11000) -> None:
    if len(seeds) != 2:
        raise ValueError("Provide exactly two seeds for the two-column figure.")
    data = [load_history(experiment_dir, seed) for seed in seeds]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 8.1), sharex="col",
                             layout="constrained")

    for col, (seed, df) in enumerate(zip(seeds, data)):
        final_iteration = int(df["iteration"].max())
        for row, (metric, tolerance, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            series = df[["iteration", metric]].dropna()
            series = series.loc[(series["iteration"] >= min_iteration)
                                & np.isfinite(series[metric])]
            if series.empty:
                raise ValueError(f"No finite {metric} values for seed {seed}.")

            # Log y-axis requires strictly positive values. Tiny changes are clipped
            # only for display; convergence flags are not recalculated or changed.
            values = np.maximum(series[metric].to_numpy(dtype=float), 1e-9)
            ax.plot(series["iteration"] / 1000, values, marker="o",
                    ms=2.6, lw=1.1, label="Observed diagnostic")
            ax.axhline(tolerance, color="#b94c4c", ls="--", lw=1.2,
                       label="Tolerance")
            ax.axvline(14, color="#888888", ls=":", lw=1,
                       label="Initial SDM budget")
            ax.axvline(18, color="#555555", ls="-.", lw=1,
                       label="Original continuation cap")
            ax.set_yscale("log")
            ax.grid(True, which="major", alpha=0.20)
            ax.set_xlim(min_iteration / 1000, max(25, final_iteration / 1000 + 1))
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(f"Dataset seed {seed} (final: {final_iteration:,} iterations)",
                             fontsize=11)
            if row == 2:
                ax.set_xlabel("Optimizer iteration (thousands)")
            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=7, frameon=False)
            ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("Experiment 2: selected SDM fit convergence diagnostics", fontsize=13)
    stem = "exp2_appendix_numerical_trajectories"
    for ext in ("pdf", "png"):
        dest = output_dir / f"{stem}.{ext}"
        fig.savefig(dest, dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"Saved: {dest}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Completed experiment2_N144_final_two_start directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination directory for PNG and PDF figures")
    parser.add_argument("--seeds", type=int, nargs=2, default=[1003, 1004],
                        metavar=("SEED1", "SEED2"))
    args = parser.parse_args()
    make_figure(args.experiment_dir, args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
