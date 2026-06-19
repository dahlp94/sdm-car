from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components


# Your actual downloaded folder:
# ~/Downloads/2024/2024/2.32_August_UseMe
DEFAULT_EJ_DIR = Path.home() / "Downloads" / "2024" / "2024" / "2.32_August_UseMe"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# California tract shapefile
TRACT_URL = "https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_06_tract.zip"

def get_default_mean_covariates(df: pd.DataFrame) -> list[str]:
    """
    Default covariates for mean-effect removal.

    These are demographic / socioeconomic / population scale variables.
    They are not environmental exposure variables.
    """
    candidate_covars = [
        "ACSTOTPOP",
        "ACSTOTHH",
        "ACSTOTHU",
        "PEOPCOLORPCT",
        "LOWINCPCT",
        "UNEMPPCT",
        "DISABILITYPCT",
        "LINGISOPCT",
        "LESSHSPCT",
        "UNDER5PCT",
        "OVER64PCT",
    ]

    return [c for c in candidate_covars if c in df.columns]

def find_ejscreen_tract_file(raw_dir: Path) -> Path:
    """
    Prefer the raw tract file, not BG, not StatePct.
    We want:
        EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv.zip
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {raw_dir}")

    preferred = raw_dir / "EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv.zip"
    if preferred.exists():
        return preferred

    candidates = []
    for p in raw_dir.glob("*.csv.zip"):
        name = p.name.lower()
        if (
            "tract" in name
            and "with_as_cnmi_gu_vi" in name
            and "statepct" not in name
            and "bg" not in name
        ):
            candidates.append(p)

    if not candidates:
        print("Files found:")
        for p in raw_dir.iterdir():
            print(" ", p.name)
        raise FileNotFoundError("Could not find the raw EJScreen tract CSV zip.")

    return candidates[0]


def infer_geoid_col(df: pd.DataFrame) -> str:
    candidates = ["GEOID", "ID", "ID_2", "FIPS", "TRACT", "CENSUSTRACT"]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains(r"\d{11}", regex=True, na=False).mean() > 0.5:
            return c

    raise ValueError("Could not infer GEOID column. Inspect df.columns manually.")


def normalize_geoid(series: pd.Series) -> pd.Series:
    """
    Handles values like:
        06037101110
        1400000US06037101110
    """
    s = series.astype(str)
    out = s.str.extract(r"(\d{11})", expand=False)
    out = out.fillna(s.str.replace(r"\D", "", regex=True))
    return out.str.zfill(11)


def safe_name(x: str) -> str:
    """
    Make a string safe for filenames and column names.
    """
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def resolve_column_name(df: pd.DataFrame, col: str) -> str:
    """
    Resolve a requested column name, allowing case-insensitive matching.
    """
    if col in df.columns:
        return col

    lower_map = {c.lower(): c for c in df.columns}
    if col.lower() in lower_map:
        return lower_map[col.lower()]

    raise ValueError(
        f"Column {col!r} not found. Available columns include:\n"
        f"{list(df.columns[:80])}"
    )


def is_bad_response_column(col: str) -> bool:
    """
    Exclude columns that are probably IDs, names, flags, ranks, percentiles,
    indexes, or administrative codes rather than raw continuous responses.

    This is mainly used when response_family='all'.
    """
    c = col.upper()

    bad_exact = {
        "GEOID",
        "ID",
        "ID_2",
        "OID_",
        "FIPS",
        "TRACT",
        "CENSUSTRACT",
        "STATEFP",
        "COUNTYFP",
        "TRACTCE",
        "GEOIDFQ",
        "NAME",
        "NAMELSAD",
        "LSAD",
        "MTFCC",
        "FUNCSTAT",
        "ALAND",
        "AWATER",
        "INTPTLAT",
        "INTPTLON",
        "REGION",
        "DIVISION",
        "ST_ABBREV",
        "STATE_NAME",
        "CNTY_NAME",
        "COUNTY_NAME",
        "LOCATION",
    }

    if c in bad_exact:
        return True

    bad_tokens = [
        "PERCENTILE",
        "PCTL",
        "PCTILE",
        "STATEPCT",
        "NATPCT",
        "RANK",
        "INDEX",
        "SPL",
        "VERSION",
    ]

    if any(tok in c for tok in bad_tokens):
        return True

    # EJScreen percentile variables often start with P_.
    if c.startswith("P_"):
        return True

    # Avoid categorical flags.
    if c.endswith("FLAG") or c.startswith("FLAG"):
        return True

    return False


def get_response_family_columns(df: pd.DataFrame, family: str) -> list[str]:
    """
    Choose candidate response columns by scientific family.

    family:
        all            = broad numeric screening after exclusions
        environmental  = raw environmental burden variables
        environmental_index = D5_* environmental index-style variables
        health         = health-disparity variables
        demographic    = ACS demographic/socioeconomic variables
    """
    family = family.lower()

    if family == "all":
        return list(df.columns)

    environmental_allow = {
        # Air / pollution / hazard burden variables
        "PM25",
        "OZONE",
        "NO2",
        "DSLPM",
        "RSEI_AIR",
        "PTRAF",
        "PRE1960",
        "PNPL",
        "PRMP",
        "PTSDF",
        "UST",
        "PWDIS",
        "DWATER",
        "CANCER",
        "RESP",
    }

    environmental_index_allow = {
        "D5_PM25",
        "D5_OZONE",
        "D5_NO2",
        "D5_DSLPM",
        "D5_RSEI_AIR",
        "D5_PTRAF",
        "D5_PRE1960",
        "D5_PNPL",
        "D5_PRMP",
        "D5_PTSDF",
        "D5_UST",
        "D5_PWDIS",
        "D5_DWATER",
        "D5_CANCER",
        "D5_RESP",
    }

    health_allow = {
        "LIFEEXPPCT",
        "HEARTDISEASE",
        "HEARTDISEASEPCT",
        "ASTHMA",
        "ASTHMAPCT",
        "CANCER",
        "CANCERPCT",
    }

    demographic_allow = {
        "PEOPCOLOR",
        "PEOPCOLORPCT",
        "LOWINCOME",
        "LOWINCPCT",
        "UNEMPLOYED",
        "UNEMPPCT",
        "DISABILITY",
        "DISABILITYPCT",
        "LINGISO",
        "LINGISOPCT",
        "LESSHS",
        "LESSHSPCT",
        "UNDER5",
        "UNDER5PCT",
        "OVER64",
        "OVER64PCT",
        "DEMOGIDX_2",
        "DEMOGIDX_5",
    }

    if family == "environmental":
        allow = environmental_allow
    elif family == "environmental_index":
        allow = environmental_index_allow
    elif family == "health":
        allow = health_allow
    elif family == "demographic":
        allow = demographic_allow
    else:
        raise ValueError(
            f"Unknown response family: {family}. "
            "Use one of: all, environmental, environmental_index, health, demographic."
        )

    cols = [c for c in df.columns if c.upper() in allow]

    if not cols:
        print("\nAvailable columns:")
        print(list(df.columns))
        raise ValueError(f"No columns found for response family {family!r}.")

    return cols


def make_transformed_response(x: pd.Series, transform: str) -> pd.Series:
    """
    Apply a candidate transformation.

    Supported transforms:
        identity
        log1p
        sqrt
        cbrt
        yeojohnson
    """
    x_num = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)

    valid = x_num.replace([np.inf, -np.inf], np.nan).dropna()

    if valid.empty:
        return out

    if transform == "identity":
        out.loc[valid.index] = valid

    elif transform == "log1p":
        ok = valid >= 0
        out.loc[valid.index[ok]] = np.log1p(valid[ok])

    elif transform == "sqrt":
        ok = valid >= 0
        out.loc[valid.index[ok]] = np.sqrt(valid[ok])

    elif transform == "cbrt":
        out.loc[valid.index] = np.cbrt(valid)

    elif transform == "yeojohnson":
        try:
            y, _lambda = stats.yeojohnson(valid.values)
            out.loc[valid.index] = y
        except Exception:
            pass

    else:
        raise ValueError(f"Unknown transform: {transform}")

    return out


def normality_score(y: pd.Series, *, min_unique: int = 20) -> dict | None:
    """
    Score approximate normality.

    Lower score is better.

    This is a ranking diagnostic, not a formal hypothesis test.
    """
    y = pd.to_numeric(y, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(y)
    if n < 100:
        return None

    n_unique = y.nunique()
    if n_unique < min_unique:
        return None

    vals = y.values.astype(float)

    sd = vals.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return None

    z = (vals - vals.mean()) / sd

    skew_abs = abs(stats.skew(z, bias=False))
    excess_kurt_abs = abs(stats.kurtosis(z, fisher=True, bias=False))

    try:
        osm, osr = stats.probplot(z, dist="norm", fit=False)
        qq_corr = np.corrcoef(osm, osr)[0, 1]
        qq_r2 = float(qq_corr**2)
    except Exception:
        return None

    if not np.isfinite(qq_r2):
        return None

    top_value_frac = float(y.value_counts(normalize=True).iloc[0])

    q01, q25, q50, q75, q99 = np.quantile(vals, [0.01, 0.25, 0.50, 0.75, 0.99])
    iqr = q75 - q25

    score = (
        skew_abs
        + 0.50 * excess_kurt_abs
        + 10.0 * (1.0 - qq_r2)
        + 2.0 * max(0.0, top_value_frac - 0.05)
    )

    return {
        "n": n,
        "n_unique": int(n_unique),
        "mean": float(vals.mean()),
        "sd": float(sd),
        "q01": float(q01),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q99": float(q99),
        "iqr": float(iqr),
        "skew_abs": float(skew_abs),
        "excess_kurt_abs": float(excess_kurt_abs),
        "qq_r2": float(qq_r2),
        "top_value_frac": float(top_value_frac),
        "score": float(score),
    }


def screen_response_candidates(
    df: pd.DataFrame,
    *,
    min_nonmissing_frac: float = 0.90,
    min_unique: int = 20,
    transforms: tuple[str, ...] = ("identity", "log1p", "sqrt", "cbrt", "yeojohnson"),
    only_col: str | None = None,
    response_family: str = "environmental",
) -> pd.DataFrame:
    """
    Screen numeric EJScreen variables and transformations.

    Returns a table sorted by approximate normality.
    """
    rows = []

    if only_col is not None:
        columns = [resolve_column_name(df, only_col)]
    else:
        columns = get_response_family_columns(df, response_family)

    print("\nCandidate columns being screened:")
    for c in columns:
        print(f"  {c}")

    for col in columns:
        if only_col is None and response_family == "all" and is_bad_response_column(col):
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        nonmissing_frac = float(x.notna().mean())

        if nonmissing_frac < min_nonmissing_frac:
            continue

        if x.nunique(dropna=True) < min_unique:
            continue

        for transform in transforms:
            y = make_transformed_response(x, transform)
            metrics = normality_score(y, min_unique=min_unique)

            if metrics is None:
                continue

            rows.append(
                {
                    "column": col,
                    "transform": transform,
                    "response_family": response_family,
                    "nonmissing_frac": nonmissing_frac,
                    **metrics,
                }
            )

    if not rows:
        raise ValueError("No candidate response variables passed screening.")

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=True).reset_index(drop=True)
    return out


def build_queen_adjacency(gdf: gpd.GeoDataFrame) -> sparse.csr_matrix:
    """
    Build symmetric queen adjacency with zero diagonal.
    """
    w = Queen.from_dataframe(gdf, ids=gdf["GEOID"].tolist())
    A = w.sparse.tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ejs-dir",
        type=Path,
        default=DEFAULT_EJ_DIR,
        help="Folder containing EJScreen 2024 files.",
    )

    parser.add_argument(
        "--response-family",
        type=str,
        default="environmental",
        choices=["all", "environmental", "environmental_index", "health", "demographic"],
        help=(
            "Which family of response variables to screen. "
            "Default is 'environmental' so the script does not automatically "
            "choose demographic or health variables such as LIFEEXPPCT."
        ),
    )

    parser.add_argument(
        "--response-col",
        type=str,
        default=None,
        help=(
            "Optional response column to use. If omitted, the script screens "
            "candidate variables within --response-family and chooses the best "
            "approximate-normal response."
        ),
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="auto",
        choices=["auto", "identity", "log1p", "sqrt", "cbrt", "yeojohnson"],
        help=(
            "Transformation to use. If 'auto', the script chooses the best "
            "transformation by normality score."
        ),
    )

    parser.add_argument(
        "--min-nonmissing-frac",
        type=float,
        default=0.90,
        help="Minimum nonmissing fraction required for candidate response variables.",
    )

    parser.add_argument(
        "--min-unique",
        type=int,
        default=20,
        help="Minimum number of unique values required for candidate responses.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top response candidates to print.",
    )

    parser.add_argument(
        "--screen-only",
        action="store_true",
        help=(
            "Only screen response variables and save the screening table; "
            "do not build geometry/adjacency outputs."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory.",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = args.ejs_dir
    print(f"Using EJScreen directory:\n  {raw_dir}")

    # 1. Load EJScreen tract CSV zip
    ej_file = find_ejscreen_tract_file(raw_dir)
    print(f"\nUsing EJScreen file:\n  {ej_file}")

    ej = pd.read_csv(ej_file, compression="zip", low_memory=False)
    ej = ej.copy()  # defragment dataframe before adding new columns

    print(f"EJScreen shape: {ej.shape}")

    print("\nFirst 30 columns:")
    print(list(ej.columns[:30]))

    # 2. Normalize tract GEOID
    geoid_col = infer_geoid_col(ej)
    print(f"\nUsing GEOID column: {geoid_col}")

    ej["GEOID"] = normalize_geoid(ej[geoid_col])

    # Los Angeles County GEOID prefix = 06037
    ej_la = ej[ej["GEOID"].str.startswith("06037")].copy()
    print(f"LA County EJScreen tract rows: {ej_la.shape[0]}")

    if ej_la.empty:
        raise ValueError("No LA County rows found. Check GEOID parsing.")

    # 3. Screen candidate response variables
    print("\nScreening candidate response variables for approximate normality...")
    print(f"Response family: {args.response_family}")

    if args.transform == "auto":
        transforms = ("identity", "log1p", "sqrt", "cbrt", "yeojohnson")
    else:
        transforms = (args.transform,)

    if args.response_col is not None:
        resolved_response_col = resolve_column_name(ej_la, args.response_col)
        print(f"\nManual response column requested: {resolved_response_col}")

        response_screen = screen_response_candidates(
            ej_la,
            min_nonmissing_frac=args.min_nonmissing_frac,
            min_unique=args.min_unique,
            transforms=transforms,
            only_col=resolved_response_col,
            response_family=args.response_family,
        )

    else:
        response_screen = screen_response_candidates(
            ej_la,
            min_nonmissing_frac=args.min_nonmissing_frac,
            min_unique=args.min_unique,
            transforms=transforms,
            only_col=None,
            response_family=args.response_family,
        )

    screen_prefix = safe_name(args.response_family)
    out_screen = out_dir / f"la_ejscreen_{screen_prefix}_response_screen.csv"
    response_screen.to_csv(out_screen, index=False)

    print(f"\nSaved response screening table:\n  {out_screen}")

    print(f"\nTop {args.top_k} candidate responses:")
    print(response_screen.head(args.top_k).to_string(index=False))

    best = response_screen.iloc[0]
    response_col = str(best["column"])
    response_transform = str(best["transform"])

    response_safe = safe_name(response_col)
    transform_safe = safe_name(response_transform)
    y_col = f"y_{response_safe}_{transform_safe}"

    print("\nSelected response:")
    print(f"  response family   = {args.response_family}")
    print(f"  column            = {response_col}")
    print(f"  transform         = {response_transform}")
    print(f"  transformed y col = {y_col}")
    print(f"  normality score   = {best['score']:.6f}")
    print(f"  QQ R^2            = {best['qq_r2']:.6f}")
    print(f"  |skewness|        = {best['skew_abs']:.6f}")
    print(f"  |excess kurtosis| = {best['excess_kurt_abs']:.6f}")
    print(f"  top value frac    = {best['top_value_frac']:.6f}")

    ej_la[response_col] = pd.to_numeric(ej_la[response_col], errors="coerce")
    ej_la[y_col] = make_transformed_response(ej_la[response_col], response_transform)

    mean_covariates = get_default_mean_covariates(ej_la)

    print("\nMean-effect covariates to save:")
    for c in mean_covariates:
        print(f"  {c}")

    for c in mean_covariates:
        ej_la[c] = pd.to_numeric(ej_la[c], errors="coerce")

    if args.screen_only:
        print("\n--screen-only was used. Stopping after response screening.")
        return

    # 4. Load California census tract geometries
    print("\nLoading Census TIGER/Line California tracts...")
    tracts = gpd.read_file(TRACT_URL)
    tracts = tracts[tracts["COUNTYFP"] == "037"].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    # Project to California Albers for stable geometry operations
    tracts = tracts.to_crs("EPSG:3310")

    print(f"LA County TIGER tracts: {tracts.shape[0]}")

    # 5. Join EJScreen data to tract polygons
    save_cols = ["GEOID", response_col, y_col] + mean_covariates

    gdf = tracts.merge(
        ej_la[save_cols],
        on="GEOID",
        how="inner",
    )

    gdf = gdf.dropna(subset=[y_col]).copy()
    print(f"Joined LA tracts with nonmissing response: {gdf.shape[0]}")

    if gdf.empty:
        raise ValueError("Join failed. GEOIDs did not match between EJScreen and TIGER.")

    # 6. Build queen adjacency
    print("\nBuilding queen adjacency...")
    A = build_queen_adjacency(gdf)

    # 7. Keep largest connected component
    n_components, labels = connected_components(A, directed=False)
    print(f"Connected components: {n_components}")

    if n_components > 1:
        counts = np.bincount(labels)
        largest = counts.argmax()
        keep = labels == largest

        print(f"Keeping largest connected component: {keep.sum()} / {len(keep)} tracts")

        gdf = gdf.iloc[np.where(keep)[0]].copy()

        # Rebuild queen adjacency after dropping disconnected tracts
        A = build_queen_adjacency(gdf)

        n_components2, _ = connected_components(A, directed=False)
        print(f"Connected components after keeping largest component: {n_components2}")

    # 8. Save outputs
    out_prefix = f"la_ejscreen_{response_safe}_{transform_safe}"

    out_gpkg = out_dir / f"{out_prefix}_tracts.gpkg"
    out_csv = out_dir / f"{out_prefix}_tracts.csv"
    out_adj = out_dir / f"{out_prefix}_queen_adjacency.npz"
    out_meta = out_dir / f"{out_prefix}_metadata.json"

    gdf.to_file(out_gpkg, layer="tracts", driver="GPKG")

    gdf[["GEOID", response_col, y_col] + mean_covariates].to_csv(out_csv, index=False)

    sparse.save_npz(out_adj, A)

    meta = {
        "response_family": args.response_family,
        "response_col": response_col,
        "response_transform": response_transform,
        "y_col": y_col,
        "n_tracts": int(gdf.shape[0]),
        "adjacency_shape": list(A.shape),
        "adjacency_nnz": int(A.nnz),
        "screening_file": str(out_screen),
        "gpkg_file": str(out_gpkg),
        "csv_file": str(out_csv),
        "adjacency_file": str(out_adj),
        "normality_score": float(best["score"]),
        "qq_r2": float(best["qq_r2"]),
        "skew_abs": float(best["skew_abs"]),
        "excess_kurt_abs": float(best["excess_kurt_abs"]),
        "top_value_frac": float(best["top_value_frac"]),
        "mean_covariates": mean_covariates,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  {out_gpkg}")
    print(f"  {out_csv}")
    print(f"  {out_adj}")
    print(f"  {out_meta}")

    print("\nRaw selected response summary:")
    print(gdf[response_col].describe())

    print(f"\nTransformed response summary: {y_col}")
    print(gdf[y_col].describe())

    print("\nApproximate normality metrics for final transformed response:")
    final_metrics = normality_score(gdf[y_col], min_unique=args.min_unique)
    print(final_metrics)

    degrees = np.asarray(A.sum(axis=1)).ravel()

    print("\nGraph summary:")
    print(f"  n nodes     = {A.shape[0]}")
    print(f"  nnz         = {A.nnz}")
    print(f"  undirected edges approx = {A.nnz // 2}")
    print(f"  degree min  = {degrees.min():.0f}")
    print(f"  degree mean = {degrees.mean():.3f}")
    print(f"  degree max  = {degrees.max():.0f}")


if __name__ == "__main__":
    main()