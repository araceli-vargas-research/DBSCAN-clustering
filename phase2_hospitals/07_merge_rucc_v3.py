#!/usr/bin/env python3
"""
07_merge_rucc.py

Merge USDA ERS 2023 Rural-Urban Continuum Codes (RUCC) onto the
hospital-cluster competition panel from Step 06.

Place directly inside:
    phase2_hospitals/

Expected input:
    data/spatial/hospital_cluster_competition_panel.csv

Downloads:
    USDA ERS 2023 RUCC CSV

Creates:
    data/processed/rucc_2023.csv
    data/spatial/hospital_cluster_rucc_panel.csv
    data/spatial/cluster_rucc_competition.csv
    data/spatial/rucc_merge_unmatched.csv
    data/spatial/rucc_hospital_summary.csv
    data/spatial/rucc_cluster_summary.csv

Definitions:
    RUCC 1-3 = metro
    RUCC 4-9 = nonmetro

Cluster rurality:
    metro_only  = all matched member hospitals are metro
    rural_only  = all matched member hospitals are nonmetro
    mixed       = cluster contains both metro and nonmetro hospitals

This avoids arbitrarily labeling a spatial cluster "rural" just because
one member hospital is rural.

Run:
    pip3 install pandas requests
    python3 07_merge_rucc.py
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPATIAL_DIR = DATA_DIR / "spatial"

INPUT_FILE = (
    SPATIAL_DIR / "hospital_cluster_competition_panel.csv"
)

RUCC_FILE = PROCESSED_DIR / "rucc_2023.csv"

HOSPITAL_OUTPUT = (
    SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
)

CLUSTER_OUTPUT = (
    SPATIAL_DIR / "cluster_rucc_competition.csv"
)

UNMATCHED_OUTPUT = (
    SPATIAL_DIR / "rucc_merge_unmatched.csv"
)

HOSPITAL_SUMMARY_OUTPUT = (
    SPATIAL_DIR / "rucc_hospital_summary.csv"
)

CLUSTER_SUMMARY_OUTPUT = (
    SPATIAL_DIR / "rucc_cluster_summary.csv"
)

RUCC_URL = (
    "https://www.ers.usda.gov/media/5768/"
    "2023-rural-urban-continuum-codes.csv?v=52934"
)


STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "PUERTO RICO": "PR",
    "VIRGIN ISLANDS": "VI",
    "U.S. VIRGIN ISLANDS": "VI",
    "GUAM": "GU",
    "AMERICAN SAMOA": "AS",
    "NORTHERN MARIANA ISLANDS": "MP",
    "COMMONWEALTH OF THE NORTHERN MARIANA ISLANDS": "MP",
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    s = str(value).upper().strip()

    s = unicodedata.normalize(
        "NFKD",
        s,
    )
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    s = s.replace("&", " AND ")

    s = re.sub(
        r"[^A-Z0-9 ]+",
        " ",
        s,
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s


def normalize_state(value) -> str:
    s = normalize_text(value)

    if len(s) == 2:
        return s

    return STATE_NAME_TO_ABBR.get(
        s,
        s,
    )


def normalize_county(value) -> str:
    s = normalize_text(value)

    suffixes = [
        " COUNTY",
        " PARISH",
        " BOROUGH",
        " CENSUS AREA",
        " MUNICIPIO",
        " MUNICIPALITY",
        " CITY AND BOROUGH",
        " CITY AND COUNTY",
    ]

    changed = True
    while changed:
        changed = False

        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[
                    :-len(suffix)
                ].strip()
                changed = True
                break

    # Independent-city forms may appear as "X CITY".
    if s.endswith(" CITY"):
        s = s[:-5].strip()

    return s


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    norm = {
        re.sub(
            r"[^a-z0-9]+",
            "",
            str(c).lower(),
        ): c
        for c in df.columns
    }

    for candidate in candidates:
        key = re.sub(
            r"[^a-z0-9]+",
            "",
            candidate.lower(),
        )

        if key in norm:
            return norm[key]

    raise KeyError(
        "Could not find any of these columns: "
        + ", ".join(candidates)
        + "\nAvailable columns:\n"
        + "\n".join(
            f"  {c}"
            for c in df.columns
        )
    )


def download_rucc() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading official USDA ERS "
        "2023 RUCC CSV..."
    )

    headers = {
        "User-Agent": (
            "hospital-spatial-competition-research/1.0"
        )
    }

    response = requests.get(
        RUCC_URL,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    RUCC_FILE.write_bytes(
        response.content
    )


def read_rucc_csv(path: Path) -> pd.DataFrame:
    """
    USDA county-name files may be distributed in Windows-1252/Latin-1
    rather than UTF-8 because of characters such as ñ.
    Try a small ordered set of encodings and report which one worked.
    """
    encodings = [
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
                encoding=encoding,
            )
            print(
                f"RUCC file decoded successfully "
                f"with encoding={encoding}"
            )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def load_rucc() -> pd.DataFrame:
    if not RUCC_FILE.exists():
        download_rucc()

    try:
        rucc_raw = read_rucc_csv(
            RUCC_FILE
        )
    except Exception:
        print(
            "Existing RUCC file could not be read. "
            "Downloading a fresh copy."
        )
        download_rucc()

        rucc_raw = read_rucc_csv(
            RUCC_FILE
        )

    print("\nUSDA RUCC columns:")
    print(
        " | ".join(
            str(c)
            for c in rucc_raw.columns
        )
    )

    state_col = find_column(
        rucc_raw,
        [
            "State",
            "State_Name",
            "State Name",
        ],
    )

    county_col = find_column(
        rucc_raw,
        [
            "County_Name",
            "County Name",
            "County",
        ],
    )

    fips_col = find_column(
        rucc_raw,
        [
            "FIPS",
            "FIPS_Code",
            "FIPS Code",
        ],
    )

    # USDA currently distributes this file in a long Attribute/Value
    # layout. Older/alternate versions may have a direct RUCC column,
    # so support both formats.
    try:
        rucc_col = find_column(
            rucc_raw,
            [
                "RUCC_2023",
                "RUCC 2023",
                "RUCC2023",
                "Rural-urban Continuum Code 2023",
            ],
        )

        rucc_source = rucc_raw.copy()
        rucc_values = pd.to_numeric(
            rucc_source[rucc_col],
            errors="coerce",
        )

    except KeyError:
        attribute_col = find_column(
            rucc_raw,
            [
                "Attribute",
            ],
        )

        value_col = find_column(
            rucc_raw,
            [
                "Value",
            ],
        )

        attrs = (
            rucc_raw[attribute_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        def attr_key(x: str) -> str:
            return re.sub(
                r"[^a-z0-9]+",
                "",
                str(x).lower(),
            )

        candidates = [
            a for a in attrs
            if (
                "rural" in attr_key(a)
                and "urban" in attr_key(a)
                and "continuum" in attr_key(a)
                and "2023" in attr_key(a)
            )
        ]

        if not candidates:
            candidates = [
                a for a in attrs
                if (
                    "rucc" in attr_key(a)
                    and "2023" in attr_key(a)
                )
            ]

        if not candidates:
            print("\nAvailable USDA Attribute values:")
            for a in attrs[:100]:
                print(f"  {a}")

            raise KeyError(
                "Could not identify the 2023 RUCC attribute "
                "in the USDA Attribute/Value file."
            )

        # Prefer the most explicit candidate if more than one appears.
        chosen_attribute = sorted(
            candidates,
            key=lambda x: (
                "code" not in attr_key(x),
                len(str(x)),
            ),
        )[0]

        print(
            f"Detected long-format RUCC attribute: "
            f"{chosen_attribute}"
        )

        rucc_source = rucc_raw.loc[
            rucc_raw[attribute_col]
            .astype(str)
            .eq(chosen_attribute)
        ].copy()

        rucc_values = pd.to_numeric(
            rucc_source[value_col],
            errors="coerce",
        )

    rucc = pd.DataFrame(
        {
            "county_fips": (
                rucc_source[fips_col]
                .astype(str)
                .str.replace(
                    r"\.0$",
                    "",
                    regex=True,
                )
                .str.zfill(5)
            ),
            "rucc_state_raw": (
                rucc_source[state_col]
            ),
            "rucc_county_raw": (
                rucc_source[county_col]
            ),
            "rucc_2023": rucc_values,
        }
    )

    rucc["state_key"] = (
        rucc["rucc_state_raw"]
        .map(normalize_state)
    )

    rucc["county_key"] = (
        rucc["rucc_county_raw"]
        .map(normalize_county)
    )

    rucc["metro_status"] = np.where(
        rucc["rucc_2023"].between(
            1,
            3,
        ),
        "metro",
        np.where(
            rucc["rucc_2023"].between(
                4,
                9,
            ),
            "nonmetro",
            pd.NA,
        ),
    )

    rucc["rural"] = np.where(
        rucc["rucc_2023"].between(
            4,
            9,
        ),
        1,
        np.where(
            rucc["rucc_2023"].between(
                1,
                3,
            ),
            0,
            np.nan,
        ),
    )

    # Verify merge keys are unique.
    dupes = rucc.duplicated(
        [
            "state_key",
            "county_key",
        ],
        keep=False,
    )

    if dupes.any():
        print(
            "\nWARNING: duplicate normalized "
            "RUCC county keys detected:"
        )
        print(
            rucc.loc[
                dupes,
                [
                    "county_fips",
                    "rucc_state_raw",
                    "rucc_county_raw",
                    "state_key",
                    "county_key",
                    "rucc_2023",
                ],
            ]
            .sort_values(
                [
                    "state_key",
                    "county_key",
                ]
            )
            .head(30)
            .to_string(
                index=False
            )
        )

    return rucc


def classify_cluster(group: pd.DataFrame) -> pd.Series:
    matched = group.loc[
        group["rural"].notna()
    ].copy()

    n_total = len(group)
    n_matched = len(matched)

    if n_matched == 0:
        cluster_rurality = "unmatched"
        rural_share = np.nan
        rural_count = 0
        metro_count = 0

    else:
        rural_count = int(
            (matched["rural"] == 1).sum()
        )
        metro_count = int(
            (matched["rural"] == 0).sum()
        )

        rural_share = (
            rural_count / n_matched
        )

        if rural_count == n_matched:
            cluster_rurality = "rural_only"
        elif metro_count == n_matched:
            cluster_rurality = "metro_only"
        else:
            cluster_rurality = "mixed"

    return pd.Series(
        {
            "member_hospitals": n_total,
            "rucc_matched_hospitals": n_matched,
            "rural_hospitals": rural_count,
            "metro_hospitals": metro_count,
            "rural_share": rural_share,
            "cluster_rurality": cluster_rurality,
        }
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}\n\n"
            "Run python3 06_cluster_competition_v2.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    rucc = load_rucc()

    print()
    print("=" * 72)
    print("MERGE USDA 2023 RUCC")
    print("=" * 72)
    print(
        f"Hospital-cluster rows: {len(df):,}"
    )
    print(
        f"RUCC county records:    {len(rucc):,}"
    )

    df["state_key"] = (
        df["state"]
        .map(normalize_state)
    )

    df["county_key"] = (
        df["county"]
        .map(normalize_county)
    )

    merge_cols = [
        "county_fips",
        "rucc_state_raw",
        "rucc_county_raw",
        "state_key",
        "county_key",
        "rucc_2023",
        "metro_status",
        "rural",
    ]

    merged = df.merge(
        rucc[merge_cols],
        on=[
            "state_key",
            "county_key",
        ],
        how="left",
        validate="many_to_one",
    )

    matched = (
        merged["rucc_2023"]
        .notna()
    )

    print(
        f"RUCC matched rows:       "
        f"{matched.sum():,} / {len(merged):,}"
    )
    print(
        f"Hospital-row RUCC match rate: "
        f"{100 * matched.mean():.2f}%"
    )

    merged.to_csv(
        HOSPITAL_OUTPUT,
        index=False,
    )

    unmatched = (
        merged.loc[
            ~matched,
            [
                c for c in [
                    "facility_id",
                    "facility_name",
                    "year",
                    "state",
                    "county",
                    "state_key",
                    "county_key",
                    "cluster_id",
                ]
                if c in merged.columns
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "state",
                "county",
                "facility_name",
            ],
            na_position="last",
        )
    )

    unmatched.to_csv(
        UNMATCHED_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # HOSPITAL-LEVEL RURAL / METRO SUMMARY
    # -------------------------------------------------------------

    hospital_summary = (
        merged.loc[
            merged["rucc_2023"].notna()
        ]
        .groupby(
            [
                "year",
                "metro_status",
            ],
            dropna=False,
        )
        .agg(
            hospitals=(
                "facility_id",
                "size",
            ),
            median_nn1_km=(
                "nn1_distance_km",
                "median",
            ),
            median_nearest_cluster_km=(
                "nearest_competing_cluster_km",
                "median",
            ),
        )
        .reset_index()
    )

    hospital_summary.to_csv(
        HOSPITAL_SUMMARY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # CLUSTER RURALITY
    # -------------------------------------------------------------

    cluster_group_cols = [
        "year",
        "snapshot",
        "cluster_id",
    ]

    cluster_rurality = (
        merged.groupby(
            cluster_group_cols,
            dropna=False,
        )
        .apply(
            classify_cluster,
            include_groups=False,
        )
        .reset_index()
    )

    cluster_metrics = (
        merged[
            [
                c for c in [
                    "year",
                    "snapshot",
                    "cluster_id",
                    "cluster_size_cluster",
                    "cluster_size",
                    "within_cluster_mean_km",
                    "within_cluster_max_km",
                    "nearest_cluster_id",
                    "nearest_competing_cluster_km",
                    "nearest_competing_cluster_miles",
                    "isolation_ratio",
                    "isolation_ratio_max",
                    "multi_hospital_cluster",
                    "singleton_cluster",
                ]
                if c in merged.columns
            ]
        ]
        .drop_duplicates(
            cluster_group_cols
        )
    )

    clusters = cluster_metrics.merge(
        cluster_rurality,
        on=cluster_group_cols,
        how="left",
        validate="one_to_one",
    )

    clusters.to_csv(
        CLUSTER_OUTPUT,
        index=False,
    )

    cluster_summary = (
        clusters.loc[
            clusters["cluster_rurality"].isin(
                [
                    "metro_only",
                    "rural_only",
                    "mixed",
                ]
            )
        ]
        .groupby(
            [
                "year",
                "cluster_rurality",
            ],
            dropna=False,
        )
        .agg(
            service_clusters=(
                "cluster_id",
                "nunique",
            ),
            median_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                "median",
            ),
            mean_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                "mean",
            ),
            median_isolation_ratio=(
                "isolation_ratio",
                "median",
            ),
        )
        .reset_index()
    )

    cluster_summary.to_csv(
        CLUSTER_SUMMARY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # PRINT LATEST-YEAR RESULTS
    # -------------------------------------------------------------

    latest_year = pd.to_numeric(
        merged["year"],
        errors="coerce",
    ).max()

    latest_clusters = (
        cluster_summary.loc[
            cluster_summary["year"]
            == latest_year
        ]
        .copy()
        .sort_values(
            "cluster_rurality"
        )
    )

    print()
    print("=" * 72)
    print("RUCC MERGE COMPLETE")
    print("=" * 72)

    print("\nCreated:")
    for path in [
        RUCC_FILE,
        HOSPITAL_OUTPUT,
        CLUSTER_OUTPUT,
        UNMATCHED_OUTPUT,
        HOSPITAL_SUMMARY_OUTPUT,
        CLUSTER_SUMMARY_OUTPUT,
    ]:
        print(
            f"  {path.relative_to(PROJECT_DIR)}"
        )

    print()
    print(
        f"Latest-year cluster comparison "
        f"({int(latest_year)}):"
    )

    if len(latest_clusters):
        print(
            latest_clusters.to_string(
                index=False
            )
        )
    else:
        print(
            "No latest-year RUCC cluster "
            "summary available."
        )

    print()
    print(
        "Interpretation reminder: RUCC 1-3 = metro; "
        "RUCC 4-9 = nonmetro."
    )


if __name__ == "__main__":
    main()
