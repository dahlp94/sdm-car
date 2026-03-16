from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from realdata.registry import get_dataset_spec, available_datasets
from realdata.io import ensure_dir


def require_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {source}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def add_missing_optional_columns(df: pd.DataFrame, optional: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in optional:
        if col not in df.columns:
            df[col] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and summarize fitted real-data benchmark results."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Available: {available_datasets()}",
    )
    parser.add_argument(
        "--use_covariates",
        action="store_true",
        help="Read outputs from the with_covariates run directory.",
    )

    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)
    run_tag = "with_covariates" if args.use_covariates else "intercept_only"

    out_root = Path("realdata/figures") / dataset_spec.dataset_name / run_tag
    ensure_dir(out_root)

    summary_path = out_root / "benchmark_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found: {summary_path}\n"
            "Run realdata/scripts/04_fit_realdata_models.py first."
        )

    df = pd.read_csv(summary_path)

    if df.empty:
        raise ValueError(f"No rows found in {summary_path}")

    required_cols = [
        "dataset_name",
        "method_name",
        "display_name",
        "family",
        "objective_name",
        "final_objective",
        "residual_mean",
        "residual_sd",
        "residual_mse",
        "spatial_effect_sd",
        "fit_time_sec",
        "n_obs",
        "num_parameters",
    ]
    require_columns(df, required_cols, summary_path)

    df = add_missing_optional_columns(df, ["notes"])

    # --------------------------------------------------
    # Rank methods by final objective
    # Higher is better for ELBO-style objectives
    # --------------------------------------------------
    ranked = df.sort_values("final_objective", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    best_obj = float(ranked.loc[0, "final_objective"])
    best_time = float(ranked["fit_time_sec"].min())

    ranked["delta_from_best"] = ranked["final_objective"] - best_obj
    ranked["objective_gap"] = best_obj - ranked["final_objective"]

    if abs(best_obj) > 1e-12:
        ranked["objective_gap_pct"] = 100.0 * ranked["objective_gap"] / abs(best_obj)
    else:
        ranked["objective_gap_pct"] = np.nan

    ranked["residual_rmse"] = np.sqrt(np.maximum(ranked["residual_mse"], 0.0))

    if best_time > 0:
        ranked["relative_fit_time"] = ranked["fit_time_sec"] / best_time
    else:
        ranked["relative_fit_time"] = np.nan

    ranked["is_best"] = ranked["rank"] == 1

    ranked = ranked[
        [
            "rank",
            "is_best",
            "dataset_name",
            "method_name",
            "display_name",
            "family",
            "objective_name",
            "final_objective",
            "delta_from_best",
            "objective_gap",
            "objective_gap_pct",
            "residual_mean",
            "residual_sd",
            "residual_mse",
            "residual_rmse",
            "spatial_effect_sd",
            "fit_time_sec",
            "relative_fit_time",
            "n_obs",
            "num_parameters",
            "notes",
        ]
    ]

    ranked_path = out_root / "benchmark_summary_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    # --------------------------------------------------
    # Compact comparison table
    # --------------------------------------------------
    compact = ranked[
        [
            "rank",
            "display_name",
            "final_objective",
            "objective_gap",
            "objective_gap_pct",
            "residual_sd",
            "residual_mse",
            "residual_rmse",
            "spatial_effect_sd",
            "fit_time_sec",
            "relative_fit_time",
        ]
    ].copy()

    compact_round_cols = [
        "final_objective",
        "objective_gap",
        "objective_gap_pct",
        "residual_sd",
        "residual_mse",
        "residual_rmse",
        "spatial_effect_sd",
        "fit_time_sec",
        "relative_fit_time",
    ]
    for col in compact_round_cols:
        compact[col] = compact[col].astype(float).round(4)

    compact_path = out_root / "benchmark_summary_compact.csv"
    compact.to_csv(compact_path, index=False)

    # --------------------------------------------------
    # Paper-friendly table
    # --------------------------------------------------
    paper = ranked[
        [
            "rank",
            "display_name",
            "final_objective",
            "objective_gap",
            "residual_rmse",
            "spatial_effect_sd",
            "fit_time_sec",
        ]
    ].copy()

    paper.columns = [
        "rank",
        "method",
        "objective",
        "gap_from_best",
        "rmse",
        "spatial_sd",
        "fit_time_sec",
    ]

    for col in ["objective", "gap_from_best", "rmse", "spatial_sd", "fit_time_sec"]:
        paper[col] = paper[col].astype(float).round(4)

    paper_path = out_root / "benchmark_summary_paper.csv"
    paper.to_csv(paper_path, index=False)

    # --------------------------------------------------
    # Console report
    # --------------------------------------------------
    print("\n" + "=" * 72)
    print("REAL-DATA BENCHMARK EVALUATION")
    print("=" * 72)
    print("Dataset:", dataset_spec.dataset_name)
    print("Outcome:", dataset_spec.outcome_label)
    print("Run tag:", run_tag)
    print("Summary file:", summary_path)
    print()

    print("Ranked benchmark summary:")
    print(ranked.to_string(index=False))

    print("\nSaved:")
    print("  -", ranked_path)
    print("  -", compact_path)
    print("  -", paper_path)

    top = ranked.iloc[0]
    print("\nBest method summary:")
    print(
        f"  {top['display_name']} | "
        f"{top['objective_name']}={top['final_objective']:.4f} | "
        f"RMSE={top['residual_rmse']:.4f} | "
        f"fit_time={top['fit_time_sec']:.2f}s"
    )

    if len(ranked) >= 2:
        second = ranked.iloc[1]
        gap = float(top["final_objective"] - second["final_objective"])

        print("\nTop-vs-second comparison:")
        print(
            f"  Best   : {top['display_name']} "
            f"({top['objective_name']}={top['final_objective']:.4f})"
        )
        print(
            f"  Second : {second['display_name']} "
            f"({second['objective_name']}={second['final_objective']:.4f})"
        )
        print(f"  Objective gap: {gap:.4f}")

    fastest_idx = ranked["fit_time_sec"].astype(float).idxmin()
    fastest = ranked.loc[fastest_idx]
    print("\nFastest method:")
    print(
        f"  {fastest['display_name']} | "
        f"fit_time={fastest['fit_time_sec']:.2f}s"
    )


if __name__ == "__main__":
    main()