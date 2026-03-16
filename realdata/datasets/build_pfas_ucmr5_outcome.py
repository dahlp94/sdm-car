from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw/pfas_ucmr5")

# ------------------------------------------------------------
# Choose outcome variant here
# ------------------------------------------------------------
OUTCOME_VARIANT = "top3_log"
# choices:
#   "burden_q95"
#   "hotspot_contrast"
#   "hotspot_sqrt"
#   "top3_log"
#   "top5_log"
#   "top3_contrast"

OUT_DIR = Path(f"data/processed/pfas_ucmr5_{OUTCOME_VARIANT}")
OUTCOME_FILE = OUT_DIR / "outcome.csv"

UCMR_FILE = RAW_DIR / "ucmr5_contaminant_results.csv"
GEO_FILE = RAW_DIR / "sdwa_dataset/SDWA_GEOGRAPHIC_AREAS.csv"
PWS_FILE = RAW_DIR / "sdwa_dataset/SDWA_PUB_WATER_SYSTEMS.csv"


PFAS_LIST = [
    "PFOS",
    "PFOA",
    "PFHxS",
    "PFNA",
    "HFPO-DA",
    "PFBS",
]

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}


def load_ucmr_results() -> pd.DataFrame:
    print("Loading UCMR PFAS results...")
    df = pd.read_csv(UCMR_FILE, dtype=str)
    df["Result (µg/L)"] = pd.to_numeric(df["Result (µg/L)"], errors="coerce").fillna(0.0)

    print("Total rows:", len(df))

    df = df[df["Contaminant"].isin(PFAS_LIST)].copy()
    print("PFAS rows:", len(df))

    return df


def build_system_burden(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform concentration before aggregation
    df["log_result"] = np.log1p(df["Result (µg/L)"])

    system = (
        df.groupby("PWS ID", as_index=False)
        .agg(
            system_burden=("log_result", "sum"),
            n_obs=("log_result", "size"),
            mean_result=("Result (µg/L)", "mean"),
            max_result=("Result (µg/L)", "max"),
        )
    )

    print("Unique water systems:", len(system))
    return system


def build_system_county_crosswalk() -> pd.DataFrame:
    print("Loading geographic crosswalk...")
    geo = pd.read_csv(GEO_FILE, dtype=str)
    geo = geo[geo["AREA_TYPE_CODE"] == "CN"].copy()
    print("CN rows:", len(geo))

    print("Loading public water systems table...")
    pws = pd.read_csv(PWS_FILE, dtype=str)
    pws = pws[["PWSID", "STATE_CODE"]].drop_duplicates()

    geo = geo.merge(pws, on="PWSID", how="left")

    geo["state_fips"] = geo["STATE_CODE"].map(STATE_ABBR_TO_FIPS)
    geo["county_fips"] = (
        geo["state_fips"].fillna("") +
        geo["ANSI_ENTITY_CODE"].fillna("").str.zfill(3)
    )

    geo = geo[geo["county_fips"].str.len() == 5].copy()

    # remove Puerto Rico and territories
    exclude_prefixes = {"72", "60", "66", "69", "78"}
    geo = geo[~geo["county_fips"].str[:2].isin(exclude_prefixes)].copy()

    geo = geo[["PWSID", "county_fips"]].drop_duplicates()

    # weight multi-county systems equally across served counties
    geo["n_counties_served"] = geo.groupby("PWSID")["county_fips"].transform("nunique")
    geo["county_weight"] = 1.0 / geo["n_counties_served"]

    print("System-county mappings:", len(geo))
    print("Unique mapped counties:", geo["county_fips"].nunique())

    return geo


def build_county_outcome(merged: pd.DataFrame, variant: str) -> pd.DataFrame:
    # Weighted burden for systems serving multiple counties
    merged = merged.copy()
    merged["weighted_system_burden"] = merged["system_burden"] * merged["county_weight"]

    def topk_mean(x: pd.Series, k: int = 3) -> float:
        arr = np.sort(np.asarray(x, dtype=float))
        if arr.size == 0:
            return 0.0
        return float(arr[-min(k, arr.size):].mean())

    county = (
        merged.groupby("county_fips")
        .agg(
            n_systems=("PWS ID", "nunique"),
            q50=("weighted_system_burden", lambda x: np.quantile(x, 0.50)),
            q95=("weighted_system_burden", lambda x: np.quantile(x, 0.95)),
            mean_burden=("weighted_system_burden", "mean"),
            max_burden=("weighted_system_burden", "max"),
            top3_mean=("weighted_system_burden", lambda x: topk_mean(x, k=3)),
            top5_mean=("weighted_system_burden", lambda x: topk_mean(x, k=5)),
        )
        .reset_index()
    )

    # optional stability filter
    county = county[county["n_systems"] >= 1].copy()

    if variant == "burden_q95":
        county["y"] = np.log1p(county["q95"])

    elif variant == "hotspot_contrast":
        county["y"] = np.log1p(county["q95"]) - np.log1p(county["q50"])

    elif variant == "hotspot_sqrt":
        hotspot = np.maximum(county["q95"] - county["q50"], 0.0)
        county["y"] = np.sqrt(hotspot)

    elif variant == "top3_log":
        county["y"] = np.log1p(county["top3_mean"])

    elif variant == "top5_log":
        county["y"] = np.log1p(county["top5_mean"])

    elif variant == "top3_contrast":
        county["y"] = np.log1p(county["top3_mean"]) - np.log1p(county["q50"])

    else:
        raise ValueError(f"Unknown OUTCOME_VARIANT: {variant}")

    out = county.rename(columns={"county_fips": "fips"})
    out["fips"] = out["fips"].astype(str).str.zfill(5)

    return out

def main():
    df = load_ucmr_results()
    system_burden = build_system_burden(df)
    geo = build_system_county_crosswalk()

    merged = system_burden.merge(
        geo,
        left_on="PWS ID",
        right_on="PWSID",
        how="left",
    )
    merged = merged.dropna(subset=["county_fips"]).copy()

    print("Systems with county mapping:", len(merged))

    outcome = build_county_outcome(merged, OUTCOME_VARIANT)

    print(f"\nOutcome variant: {OUTCOME_VARIANT}")
    print("Outcome summary:")
    print(outcome["y"].describe())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outcome.to_csv(OUTCOME_FILE, index=False)

    print("\nOutcome saved to:", OUTCOME_FILE)
    print("Number of counties:", len(outcome))
    print("\nFirst few FIPS:")
    print(outcome["fips"].head(10).tolist())


if __name__ == "__main__":
    main()