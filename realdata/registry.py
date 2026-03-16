"""
Dataset registry for real-data experiments.

Each entry describes:
- where the processed dataset lives
- which graph it uses
- which eigendecomposition corresponds to the graph
- what outcome column should be modeled
- what covariates are available

All real-data scripts should obtain dataset configuration from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


# ------------------------------------------------------------
# Dataset specification
# ------------------------------------------------------------

@dataclass
class RealDataSpec:
    dataset_name: str
    family: str

    # data locations
    raw_dir: Optional[Path]
    processed_dir: Path
    outcome_file: Path
    model_data_file: Path

    # graph + eigendecomposition
    graph_name: str
    graph_dir: Path
    graph_file: Path
    eig_file: Path
    nodes_file: Optional[Path]

    # modeling
    outcome_column: str
    default_covariates: List[str]

    # human-readable label
    outcome_label: str


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_ROOT = Path("data")

RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
GRAPH_ROOT = DATA_ROOT / "graph"
EIG_ROOT = DATA_ROOT / "eigs"


# ------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------

REALDATA_REGISTRY: Dict[str, RealDataSpec] = {

    # --------------------------------------------------------
    # Fentanyl mortality momentum (rate-based outcome)
    # --------------------------------------------------------
    "fentanyl_rate": RealDataSpec(

        dataset_name="fentanyl_rate",
        family="gaussian",

        raw_dir=RAW_ROOT / "fentanyl_rate",
        processed_dir=PROCESSED_ROOT / "fentanyl_rate",

        outcome_file=PROCESSED_ROOT / "fentanyl_rate" / "outcome.csv",
        model_data_file=PROCESSED_ROOT / "fentanyl_rate" / "model_data.csv",

        graph_name="county_conus",
        graph_dir=GRAPH_ROOT / "county_conus",
        graph_file=GRAPH_ROOT / "county_conus" / "adjacency.npz",
        eig_file=EIG_ROOT / "county_conus" / "eigs.npz",
        nodes_file=GRAPH_ROOT / "county_conus" / "nodes.csv",

        outcome_column="y",

        default_covariates=[
            "poverty_rate_z",
            "unemployment_rate_z",
            "low_education_rate_z",
            "log_pop_density_z",
        ],

        outcome_label="Fentanyl mortality momentum",
    ),

    # --------------------------------------------------------
    # Fentanyl share of overdose deaths
    # --------------------------------------------------------
    "fentanyl_share": RealDataSpec(

        dataset_name="fentanyl_share",
        family="gaussian",

        raw_dir=RAW_ROOT / "fentanyl_share",
        processed_dir=PROCESSED_ROOT / "fentanyl_share",

        outcome_file=PROCESSED_ROOT / "fentanyl_share" / "outcome.csv",
        model_data_file=PROCESSED_ROOT / "fentanyl_share" / "model_data.csv",

        graph_name="county_conus",
        graph_dir=GRAPH_ROOT / "county_conus",
        graph_file=GRAPH_ROOT / "county_conus" / "adjacency.npz",
        eig_file=EIG_ROOT / "county_conus" / "eigs.npz",
        nodes_file=GRAPH_ROOT / "county_conus" / "nodes.csv",

        outcome_column="y",

        default_covariates=[
            "poverty_rate_z",
            "unemployment_rate_z",
            "low_education_rate_z",
            "log_pop_density_z",
        ],

        outcome_label="Fentanyl share momentum",
    ),

    # --------------------------------------------------------
    # PFAS drinking-water contamination (UCMR5-based outcome)
    # --------------------------------------------------------
    "pfas_ucmr5": RealDataSpec(

        dataset_name="pfas_ucmr5",
        family="gaussian",

        raw_dir=RAW_ROOT / "pfas_ucmr5",
        processed_dir=PROCESSED_ROOT / "pfas_ucmr5",

        outcome_file=PROCESSED_ROOT / "pfas_ucmr5" / "outcome.csv",
        model_data_file=PROCESSED_ROOT / "pfas_ucmr5" / "model_data.csv",

        graph_name="county_us_pfas",
        graph_dir=GRAPH_ROOT / "county_us_pfas",
        graph_file=GRAPH_ROOT / "county_us_pfas" / "adjacency.npz",
        eig_file=EIG_ROOT / "county_us_pfas" / "eigs.npz",
        nodes_file=GRAPH_ROOT / "county_us_pfas" / "nodes.csv",

        outcome_column="y",

        default_covariates=[
            "poverty_rate_z",
            "unemployment_rate_z",
            "low_education_rate_z",
            "log_pop_density_z",
        ],

        outcome_label="County PFAS burden",
    ),

    "pfas_ucmr5_hotspot": RealDataSpec(

        dataset_name="pfas_ucmr5_hotspot",
        family="gaussian",

        raw_dir=RAW_ROOT / "pfas_ucmr5",
        processed_dir=PROCESSED_ROOT / "pfas_ucmr5_hotspot",

        outcome_file=PROCESSED_ROOT / "pfas_ucmr5_hotspot" / "outcome.csv",
        model_data_file=PROCESSED_ROOT / "pfas_ucmr5_hotspot" / "model_data.csv",

        graph_name="county_us_pfas",
        graph_dir=GRAPH_ROOT / "county_us_pfas",
        graph_file=GRAPH_ROOT / "county_us_pfas" / "adjacency.npz",
        eig_file=EIG_ROOT / "county_us_pfas" / "eigs.npz",
        nodes_file=GRAPH_ROOT / "county_us_pfas" / "nodes.csv",

        outcome_column="y",

        default_covariates=[
            "poverty_rate_z",
            "unemployment_rate_z",
            "low_education_rate_z",
            "log_pop_density_z",
        ],

        outcome_label="County PFAS hotspot burden",
    ),

    "pfas_ucmr5_top3_log": RealDataSpec(

        dataset_name="pfas_ucmr5_top3_log",
        family="gaussian",

        raw_dir=RAW_ROOT / "pfas_ucmr5",
        processed_dir=PROCESSED_ROOT / "pfas_ucmr5_top3_log",

        outcome_file=PROCESSED_ROOT / "pfas_ucmr5_top3_log" / "outcome.csv",
        model_data_file=PROCESSED_ROOT / "pfas_ucmr5_top3_log" / "model_data.csv",

        graph_name="county_us_pfas",
        graph_dir=GRAPH_ROOT / "county_us_pfas",
        graph_file=GRAPH_ROOT / "county_us_pfas" / "adjacency.npz",
        eig_file=EIG_ROOT / "county_us_pfas" / "eigs.npz",
        nodes_file=GRAPH_ROOT / "county_us_pfas" / "nodes.csv",

        outcome_column="y",

        default_covariates=[
            "poverty_rate_z",
            "unemployment_rate_z",
            "low_education_rate_z",
            "log_pop_density_z",
        ],

        outcome_label="County PFAS top-3 hotspot burden",
    ),


}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_dataset_spec(name: str) -> RealDataSpec:
    """
    Retrieve dataset specification.
    """
    if name not in REALDATA_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(REALDATA_REGISTRY.keys())}"
        )

    return REALDATA_REGISTRY[name]


def available_datasets() -> List[str]:
    """
    List available datasets.
    """
    return sorted(REALDATA_REGISTRY.keys())