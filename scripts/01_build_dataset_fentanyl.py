from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/wonder_fentanyl_county_year.csv")
OUT_PATH = Path("data/processed/fentanyl_momentum_conus.csv")

PRE_YEARS = (2018, 2019, 2020)
POST_YEARS = (2022, 2023, 2024)
EPS = 1.0  # small offset for log-rate stability


def parse_county_state(county_name: str) -> tuple[str, str]:
    """
    Split 'Baldwin County, AL' -> ('Baldwin County', 'AL')
    """
    parts = str(county_name).rsplit(",", maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return str(county_name).strip(), ""


def report_stage(name: str, df: pd.DataFrame, fips_col: str | None = None) -> None:
    msg = f"[{name}] rows={len(df)}"
    if fips_col is not None and fips_col in df.columns:
        msg += f" | unique_counties={df[fips_col].nunique()}"
    print(msg)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    report_stage("raw import", df)

    # -------------------------
    # Basic cleaning
    # -------------------------
    df["Year"] = pd.to_numeric(df["Year"].astype(str).str.strip(), errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Year"]).copy()
    print(f"Dropped rows with invalid Year: {before - len(df)}")
    df["Year"] = df["Year"].astype(int)
    report_stage("after Year clean", df)

    df["County Code"] = pd.to_numeric(df["County Code"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["County Code"]).copy()
    print(f"Dropped rows with invalid County Code: {before - len(df)}")
    df["County Code"] = df["County Code"].astype(int).astype(str).str.zfill(5)
    report_stage("after County Code clean", df)

    # Parse numeric outcome columns early
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Deaths", "Population"]).copy()
    print(f"Dropped rows with invalid Deaths/Population: {before - len(df)}")
    report_stage("after Deaths/Population clean", df)

    df = df.rename(
        columns={
            "County Code": "fips",
            "County": "county_full",
            "Deaths": "deaths",
            "Population": "population",
            "Crude Rate": "crude_rate",
        }
    )

    # Keep only county rows with 5-digit FIPS
    before = len(df)
    df = df[df["fips"].str.len() == 5].copy()
    print(f"Dropped rows with non-5-digit FIPS: {before - len(df)}")
    report_stage("after FIPS length filter", df, "fips")

    # Parse county/state
    county_state = df["county_full"].apply(parse_county_state)
    df["county"] = county_state.str[0]
    df["state"] = county_state.str[1]

    # Filter contiguous U.S. only
    exclude_states = {"AK", "HI", "DC", "PR", "GU", "VI", "MP", "AS"}
    before = len(df)
    df = df[~df["state"].isin(exclude_states)].copy()
    print(f"Dropped non-CONUS rows: {before - len(df)}")
    report_stage("after CONUS filter", df, "fips")

    print("Unique years retained:", sorted(df["Year"].unique().tolist()))
    print("Year counts:")
    print(df["Year"].value_counts().sort_index())

    # -------------------------
    # Aggregate pre/post windows
    # -------------------------
    pre_raw = df[df["Year"].isin(PRE_YEARS)].copy()
    post_raw = df[df["Year"].isin(POST_YEARS)].copy()

    report_stage("pre window raw rows", pre_raw, "fips")
    report_stage("post window raw rows", post_raw, "fips")

    pre = (
        pre_raw.groupby(["fips", "county", "state"], as_index=False)
        .agg(
            pre_deaths=("deaths", "sum"),
            pre_pop=("population", "sum"),
        )
    )

    post = (
        post_raw.groupby(["fips", "county", "state"], as_index=False)
        .agg(
            post_deaths=("deaths", "sum"),
            post_pop=("population", "sum"),
        )
    )

    report_stage("pre aggregated counties", pre, "fips")
    report_stage("post aggregated counties", post, "fips")

    pre_only = sorted(set(pre["fips"]) - set(post["fips"]))
    post_only = sorted(set(post["fips"]) - set(pre["fips"]))

    print(f"Counties only in PRE window: {len(pre_only)}")
    print(f"Counties only in POST window: {len(post_only)}")
    if pre_only:
        print("First few PRE-only FIPS:", pre_only[:10])
    if post_only:
        print("First few POST-only FIPS:", post_only[:10])

    data = pre.merge(post, on=["fips", "county", "state"], how="inner")
    report_stage("after PRE/POST merge", data, "fips")

    # -------------------------
    # Construct outcome
    # -------------------------
    data["rate_pre"] = 100000.0 * data["pre_deaths"] / data["pre_pop"]
    data["rate_post"] = 100000.0 * data["post_deaths"] / data["post_pop"]

    data["log_rate_pre"] = np.log(data["rate_pre"] + EPS)
    data["log_rate_post"] = np.log(data["rate_post"] + EPS)

    data["y"] = data["log_rate_post"] - data["log_rate_pre"]
    data["crude_diff"] = data["rate_post"] - data["rate_pre"]
    data["total_deaths"] = data["pre_deaths"] + data["post_deaths"]

    before = len(data)
    data = data[data["total_deaths"] > 0].copy()
    print(f"Dropped counties with total_deaths == 0: {before - len(data)}")
    report_stage("final modeling dataset", data, "fips")

    # Sort
    data = data.sort_values(["state", "county"]).reset_index(drop=True)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)

    print(f"\nSaved processed dataset to: {OUT_PATH}")
    print("\nFinal y summary:")
    print(data["y"].describe())

    print("\nBottom 10 counties by y:")
    print(data.sort_values("y")[["fips", "county", "state", "y"]].head(10))

    print("\nTop 10 counties by y:")
    print(data.sort_values("y", ascending=False)[["fips", "county", "state", "y"]].head(10))

    print("\nPreview:")
    print(data.head())

if __name__ == "__main__":
    main()