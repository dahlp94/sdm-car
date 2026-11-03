"""Recreate the Experiment 1 appendix numerical trajectory figure from Step 3 CSVs.

Run from the SDM-CAR project directory, for example:
    python examples/plot_experiment1_appendix_numerical_trajectories.py

Input:  experiment1_N144_numerical_audit/audit_upload/fits/...
Output: Figures/experiment1_appendix_numerical_trajectories.png

No optimizer checkpoints, additional optimization, or regenerated data required.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = (1011, 1013)
VARIANTS = (
    ("sdm_continuation", "Original; continued", "#223e61"),
    ("sdm_restart_rng_shift", "Changed optimizer stream", "#d08523"),
    ("sdm_restart_tilt_plus", "Tilted initialization", "#36866a"),
    ("sdm_restart_tilt_plus_rng_shift", "Tilt + changed stream", "#92558b"),
)
N_TRAIN = 128
PLOT_FROM = 9000
ORIGINAL_END = 14000


def plot(audit_dir: Path, output: Path) -> None:
    # Accept either the full audit directory or its audit_upload subdirectory.
    audit_root = audit_dir / "audit_upload" if (audit_dir / "audit_upload").is_dir() else audit_dir
    source = audit_root / "fits"
    if not source.is_dir():
        raise FileNotFoundError(f"Could not find audit fit histories under {source}")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 170,
    })
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex="col")

    for i, seed in enumerate(SEEDS):
        for variant, label, color in VARIANTS:
            file = source / f"seed_{seed:05d}" / variant / "convergence_history.csv"
            if not file.is_file():
                raise FileNotFoundError(f"Missing trajectory: {file}")
            frame = pd.read_csv(file).sort_values("iteration")
            required = {"iteration", "evaluation_ELBO", "zero_variance_mean_exact_unclipped"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing columns in {file}: {sorted(required - set(frame.columns))}")
            frame = frame.loc[frame["iteration"] >= PLOT_FROM].copy()
            if frame.empty or int(frame.iloc[0]["iteration"]) != PLOT_FROM:
                raise ValueError(f"Expected an iteration-{PLOT_FROM} row in {file}")
            if "zero_exact_clip_safe_8sd" in frame.columns:
                flags = frame["zero_exact_clip_safe_8sd"].astype(str).str.lower()
                if not flags.eq("true").all():
                    raise ValueError(f"Unclipped analytic zero-mode moments may not be valid: {file}")

            x = frame["iteration"].to_numpy(dtype=int)
            # Each fit is centered at its OWN iteration-9000 ELBO.
            # Different restart checkpoints do not use the same diagnostic bank.
            elbo = frame["evaluation_ELBO"].to_numpy(dtype=float) / N_TRAIN
            delta_elbo = elbo - elbo[0]
            zero_var = frame["zero_variance_mean_exact_unclipped"].to_numpy(dtype=float)
            if not np.isfinite(delta_elbo).all() or not np.isfinite(zero_var).all():
                raise ValueError(f"Non-finite plotted values in {file}")

            if variant == "sdm_continuation":
                # Original fit: solid through 14,000; continuation: dashed afterward.
                original = x <= ORIGINAL_END
                extended = x >= ORIGINAL_END
                axs[i, 0].plot(x[original], delta_elbo[original], color=color,
                               label=label, linewidth=1.8)
                axs[i, 1].plot(x[original], zero_var[original], color=color,
                               label=label, linewidth=1.8)
                axs[i, 0].plot(x[extended], delta_elbo[extended], color=color,
                               linestyle="--", linewidth=1.4)
                axs[i, 1].plot(x[extended], zero_var[extended], color=color,
                               linestyle="--", linewidth=1.4)
            else:
                axs[i, 0].plot(x, delta_elbo, color=color, label=label, linewidth=1.6)
                axs[i, 1].plot(x, zero_var, color=color, label=label, linewidth=1.6)

        for j in (0, 1):
            axs[i, j].axvline(ORIGINAL_END, color="0.45", linestyle=":", linewidth=1)
            axs[i, j].grid(alpha=0.17)
            axs[i, j].set_title(
                f"Seed {seed}: " + ("within-run ELBO improvement" if j == 0 else "zero-mode variance")
            )
        axs[i, 0].set_ylabel(r"$\Delta$ ELBO/site from iter. 9,000")
        axs[i, 1].set_ylabel(r"$E_q[F(0)]$ (analytic)")

    axs[1, 0].set_xlabel("Optimization iteration")
    axs[1, 1].set_xlabel("Optimization iteration")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.subplots_adjust(top=0.87, wspace=0.24, hspace=0.30,
                        bottom=0.08, left=0.07, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path,
                        default=Path("experiment1_N144_numerical_audit"))
    parser.add_argument("--output", type=Path,
                        default=Path("Figures/experiment1_appendix_numerical_trajectories.png"))
    args = parser.parse_args()
    plot(args.audit_dir, args.output)


if __name__ == "__main__":
    main()
