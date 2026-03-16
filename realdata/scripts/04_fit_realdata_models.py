from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch

from realdata.registry import get_dataset_spec, available_datasets
from realdata.base import MethodSpec, summary_to_dict
from realdata.io import ensure_dir, save_method_result
from realdata.methods import fit_sdmcar_gaussian


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Temporary method registry
# ---------------------------------------------------------------------
# This stays local for now so we can get the real-data benchmark runner
# working immediately. Later, if we add INLA / BYM2 / graph GP, we can
# promote this into realdata/methods/registry.py.
METHOD_REGISTRY = {
    "sdmcar_classic": MethodSpec(
        method_name="sdmcar_classic",
        runner_name="sdmcar_gaussian",
        family="gaussian",
        display_name="SDM-CAR Classic CAR",
        filter_name="classic_car",
        case_id="baseline",
    ),
    "sdmcar_leroux": MethodSpec(
        method_name="sdmcar_leroux",
        runner_name="sdmcar_gaussian",
        family="gaussian",
        display_name="SDM-CAR Leroux",
        filter_name="leroux",
        case_id="learn_rho",
    ),
    "sdmcar_matern": MethodSpec(
        method_name="sdmcar_matern",
        runner_name="sdmcar_gaussian",
        family="gaussian",
        display_name="SDM-CAR Matérn",
        filter_name="matern",
        case_id="baseline",
    ),
    "sdmcar_rational": MethodSpec(
        method_name="sdmcar_rational",
        runner_name="sdmcar_gaussian",
        family="gaussian",
        display_name="SDM-CAR Rational",
        filter_name="rational",
        case_id="flex_21",
    ),
}


def available_methods() -> list[str]:
    return sorted(METHOD_REGISTRY.keys())


def get_method_spec(name: str) -> MethodSpec:
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available methods: {available_methods()}"
        )
    return METHOD_REGISTRY[name]


def parse_method_names(methods_arg: str) -> list[str]:
    raw = [m.strip() for m in methods_arg.split(",") if m.strip()]
    if not raw:
        raise ValueError(
            "No methods provided. Pass a comma-separated list via --methods."
        )

    seen = set()
    deduped = []
    for m in raw:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def validate_method_names(method_names: list[str]) -> None:
    unknown = [m for m in method_names if m not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Available methods: {available_methods()}"
        )


def write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------
def run_method(
    *,
    dataset_name: str,
    method_name: str,
    use_covariates: bool,
    vi_iters: int,
    vi_lr: float,
    vi_mc: int,
):
    dataset_spec = get_dataset_spec(dataset_name)
    method_spec = get_method_spec(method_name)

    if method_spec.runner_name == "sdmcar_gaussian":
        return fit_sdmcar_gaussian(
            dataset_spec=dataset_spec,
            method_spec=method_spec,
            use_covariates=use_covariates,
            vi_iters=vi_iters,
            vi_lr=vi_lr,
            vi_mc=vi_mc,
            device=DEVICE,
        )

    raise NotImplementedError(
        f"Runner '{method_spec.runner_name}' not implemented."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit registered real-data models on a registered dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Available: {available_datasets()}",
    )
    parser.add_argument(
        "--methods",
        required=True,
        help=(
            "Comma-separated list of method names. "
            f"Available: {available_methods()}"
        ),
    )
    parser.add_argument(
        "--use_covariates",
        action="store_true",
        help="Use default covariates from the dataset registry.",
    )
    parser.add_argument("--vi_iters", type=int, default=1500)
    parser.add_argument("--vi_lr", type=float, default=1e-2)
    parser.add_argument("--vi_mc", type=int, default=8)

    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)
    method_names = parse_method_names(args.methods)
    validate_method_names(method_names)

    run_tag = "with_covariates" if args.use_covariates else "intercept_only"
    out_root = Path("realdata/figures") / dataset_spec.dataset_name / run_tag
    ensure_dir(out_root)

    run_config = {
        "dataset_name": dataset_spec.dataset_name,
        "family": dataset_spec.family,
        "outcome_label": dataset_spec.outcome_label,
        "run_tag": run_tag,
        "use_covariates": bool(args.use_covariates),
        "methods": method_names,
        "vi_iters": int(args.vi_iters),
        "vi_lr": float(args.vi_lr),
        "vi_mc": int(args.vi_mc),
        "device": str(DEVICE),
        "output_root": str(out_root),
    }
    write_json(run_config, out_root / "run_config.json")

    methods_requested_df = pd.DataFrame(
        {
            "method_name": method_names,
            "display_name": [get_method_spec(m).display_name for m in method_names],
            "runner_name": [get_method_spec(m).runner_name for m in method_names],
            "filter_name": [get_method_spec(m).filter_name for m in method_names],
            "case_id": [get_method_spec(m).case_id for m in method_names],
        }
    )
    methods_requested_df.to_csv(out_root / "methods_requested.csv", index=False)

    print("Dataset:", dataset_spec.dataset_name)
    print("Family:", dataset_spec.family)
    print("Outcome label:", dataset_spec.outcome_label)
    print("Using covariates:", args.use_covariates)
    print("Methods:", method_names)
    print("Output root:", out_root)

    summary_rows: list[dict] = []
    failed_rows: list[dict] = []

    for method_name in method_names:
        method_spec = get_method_spec(method_name)
        method_dir = out_root / method_name
        ensure_dir(method_dir)

        print("\n" + "=" * 72)
        print(f"FITTING: {method_name}")
        print("=" * 72)

        method_meta = {
            "dataset_name": dataset_spec.dataset_name,
            "run_tag": run_tag,
            "use_covariates": bool(args.use_covariates),
            "vi_iters": int(args.vi_iters),
            "vi_lr": float(args.vi_lr),
            "vi_mc": int(args.vi_mc),
            "device": str(DEVICE),
            "method_name": method_spec.method_name,
            "display_name": method_spec.display_name,
            "runner_name": method_spec.runner_name,
            "filter_name": method_spec.filter_name,
            "case_id": method_spec.case_id,
            "family": method_spec.family,
        }
        write_json(method_meta, method_dir / "method_config.json")

        try:
            result = run_method(
                dataset_name=args.dataset,
                method_name=method_name,
                use_covariates=args.use_covariates,
                vi_iters=args.vi_iters,
                vi_lr=args.vi_lr,
                vi_mc=args.vi_mc,
            )

            save_method_result(result, method_dir)

            row = summary_to_dict(result.summary)
            row["status"] = "ok"
            row["error_message"] = ""
            summary_rows.append(row)

            pd.DataFrame([row]).to_csv(method_dir / "summary_row.csv", index=False)

            print(f"[OK] Saved method results to: {method_dir}")

        except Exception as e:
            err_msg = str(e)
            tb = traceback.format_exc()

            failure_row = {
                "dataset_name": dataset_spec.dataset_name,
                "method_name": method_spec.method_name,
                "display_name": method_spec.display_name,
                "family": method_spec.family,
                "objective_name": None,
                "final_objective": None,
                "residual_mean": None,
                "residual_sd": None,
                "residual_mse": None,
                "spatial_effect_sd": None,
                "fit_time_sec": None,
                "n_obs": None,
                "num_parameters": None,
                "notes": "fit_failed",
                "status": "failed",
                "error_message": err_msg,
            }
            failed_rows.append(failure_row)

            pd.DataFrame([failure_row]).to_csv(method_dir / "summary_row.csv", index=False)
            (method_dir / "error_traceback.txt").write_text(tb, encoding="utf-8")

            print(f"[FAILED] {method_name}")
            print(err_msg)

    # Successful runs only
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values("final_objective", ascending=False).reset_index(drop=True)
        summary_df.to_csv(out_root / "benchmark_summary.csv", index=False)
    else:
        summary_df = pd.DataFrame()
        summary_df.to_csv(out_root / "benchmark_summary.csv", index=False)

    # All runs including failures
    all_rows = summary_rows + failed_rows
    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        sort_cols = [c for c in ["status", "final_objective"] if c in all_df.columns]
        if "final_objective" in all_df.columns:
            ok_mask = all_df["status"].eq("ok") if "status" in all_df.columns else pd.Series([True] * len(all_df))
            ok_df = all_df.loc[ok_mask].sort_values("final_objective", ascending=False) if ok_mask.any() else pd.DataFrame(columns=all_df.columns)
            fail_df = all_df.loc[~ok_mask] if "status" in all_df.columns else pd.DataFrame(columns=all_df.columns)
            all_df = pd.concat([ok_df, fail_df], ignore_index=True)
        all_df.to_csv(out_root / "benchmark_summary_all.csv", index=False)
    else:
        all_df.to_csv(out_root / "benchmark_summary_all.csv", index=False)

    print("\nSaved benchmark files to:")
    print("  -", out_root / "run_config.json")
    print("  -", out_root / "methods_requested.csv")
    print("  -", out_root / "benchmark_summary.csv")
    print("  -", out_root / "benchmark_summary_all.csv")

    if not summary_df.empty:
        print("\nRanked summary (successful fits only):")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo successful fits were completed.")

    if failed_rows:
        print("\nFailed methods:")
        for row in failed_rows:
            print(f"  - {row['method_name']}: {row['error_message']}")


if __name__ == "__main__":
    main()