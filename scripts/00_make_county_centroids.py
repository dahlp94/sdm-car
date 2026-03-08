# scripts/00_make_county_centroids.py

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/2025_Gaz_counties_national.txt")
OUT_PATH = Path("data/raw/county_centroids_conus.csv")


def main():

    df = pd.read_csv(RAW_PATH, sep="|", dtype={"GEOID": str})

    print("Rows in Gazetteer:", len(df))

    # Keep required columns
    df = df[["GEOID", "INTPTLAT", "INTPTLONG"]].copy()

    df = df.rename(
        columns={
            "GEOID": "fips",
            "INTPTLAT": "lat",
            "INTPTLONG": "lon",
        }
    )

    # convert coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["lat", "lon"]).copy()

    # remove non-CONUS states
    exclude_state_fips = {"02", "15", "11", "60", "66", "69", "72", "78"}

    before = len(df)
    df = df[~df["fips"].str[:2].isin(exclude_state_fips)].copy()

    print("Dropped non-CONUS counties:", before - len(df))

    df = df.sort_values("fips").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\nSaved centroid file:")
    print(OUT_PATH)
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()