# examples/sweep_benchmarks.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import subprocess

from examples.benchmarks.registry import available_filters, get_filter_spec


def flatten_row(d: dict) -> dict:
    out = dict(d)
    # flatten acc_theta
    acc_theta = out.pop("acc_theta", {})
    for k, v in acc_theta.items():
        out[f"acc_theta_{k}"] = v
    # flatten mcmc_means
    mcmc_means = out.pop("mcmc_means", {})
    for k, v in mcmc_means.items():
        out[f"mcmc_mean_{k}"] = v
    return out


def main():
    outdir = Path("examples") / "figures" / "benchmarks_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "results.csv"
    rows = []

    for f in available_filters():
        spec = get_filter_spec(f)

        for cid, case in spec.cases.items():
            print("\n" + "=" * 80)
            print(f"RUNNING: filter={f} case={cid}")
            print("=" * 80)

            cmd = [
                sys.executable, "-m", "examples.run_benchmark",
                "--filter", f,
                "--cases", cid,
                "--outdir", str(outdir),
                "--fast",
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"[WARN] Failed: filter={f} case={cid}")
                continue

            case_dir = outdir / f / case.display_name
            metrics_path = case_dir / "metrics.json"
            if not metrics_path.exists():
                print(f"[WARN] Missing metrics.json: {metrics_path}")
                continue

            with open(metrics_path, "r") as fp:
                metrics = json.load(fp)

            rows.append(flatten_row(metrics))

    if not rows:
        print("No successful runs; CSV not written.")
        return

    # stable column order
    cols = sorted({k for r in rows for k in r.keys()})

    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote: {csv_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
