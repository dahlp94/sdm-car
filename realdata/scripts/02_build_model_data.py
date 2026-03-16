from __future__ import annotations

import argparse
from pathlib import Path

from realdata.registry import get_dataset_spec, available_datasets
from realdata.datasets.build_model_data import build_model_data


DEFAULT_ACS_PANEL = Path("data/raw/covariates/acs_county_covariates_2018_2024_panel.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Build graph-aligned model-ready data for a registered real-data dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name. Available: {available_datasets()}",
    )
    parser.add_argument(
        "--acs_panel",
        type=Path,
        default=DEFAULT_ACS_PANEL,
        help=f"Path to ACS panel CSV. Default: {DEFAULT_ACS_PANEL}",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress printing.",
    )

    args = parser.parse_args()

    dataset_spec = get_dataset_spec(args.dataset)

    print("Building model data for dataset:", dataset_spec.dataset_name)
    print("Outcome file:", dataset_spec.outcome_file)
    print("Model data file:", dataset_spec.model_data_file)
    print("Graph eig file:", dataset_spec.eig_file)
    print("ACS panel:", args.acs_panel)

    build_model_data(
        dataset_spec=dataset_spec,
        acs_panel_file=args.acs_panel,
        verbose=not args.quiet,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()