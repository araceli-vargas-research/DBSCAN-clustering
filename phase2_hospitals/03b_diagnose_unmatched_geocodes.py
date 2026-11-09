#!/usr/bin/env python3
"""
03b_diagnose_unmatched_geocodes.py

Diagnose which hospital addresses failed geocoding before building
Euclidean distances.

Place directly inside:
    phase2_hospitals/

Input:
    data/spatial/hospital_panel_geocoded.csv
    data/spatial/hospital_address_geocodes.csv

Creates:
    data/spatial/unmatched_geocodes.csv
    data/spatial/unmatched_by_state.csv
    data/spatial/unmatched_by_year.csv
    data/spatial/unmatched_address_patterns.csv

Run:
    python3 03b_diagnose_unmatched_geocodes.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"

PANEL_FILE = SPATIAL_DIR / "hospital_panel_geocoded.csv"
CROSSWALK_FILE = SPATIAL_DIR / "hospital_address_geocodes.csv"

OUT_UNMATCHED = SPATIAL_DIR / "unmatched_geocodes.csv"
OUT_STATE = SPATIAL_DIR / "unmatched_by_state.csv"
OUT_YEAR = SPATIAL_DIR / "unmatched_by_year.csv"
OUT_PATTERNS = SPATIAL_DIR / "unmatched_address_patterns.csv"


def boolish(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def address_pattern(address: str) -> str:
    x = (address or "").upper().strip()

    if not x:
        return "missing_address"

    if "PO BOX" in x or "P.O. BOX" in x:
        return "po_box"

    if re.search(r"\b(SUITE|STE|UNIT|BLDG|BUILDING|FLOOR|FL|ROOM|RM)\b", x):
        return "secondary_unit"

    if re.search(r"\b(HWY|HIGHWAY|ROUTE|RTE|STATE ROAD|COUNTY ROAD|CR)\b", x):
        return "highway_route"

    if re.search(r"\bNORTH|SOUTH|EAST|WEST|NE|NW|SE|SW\b", x):
        return "directional"

    if not re.search(r"\d", x):
        return "no_street_number"

    return "other"


def main() -> None:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PANEL_FILE}. Run 03_geocode_hospitals.py first."
        )

    panel = pd.read_csv(
        PANEL_FILE,
        dtype=str,
        low_memory=False,
    )

    success = boolish(panel["geocode_success"])
    panel["geocode_success_bool"] = success

    unmatched = panel.loc[
        ~success,
        [
            "facility_id",
            "facility_name",
            "year",
            "snapshot",
            "address",
            "city",
            "state",
            "zip_code",
            "hospital_type",
            "hospital_ownership",
            "address_key",
        ],
    ].copy()

    unmatched["address_pattern"] = (
        unmatched["address"]
        .fillna("")
        .map(address_pattern)
    )

    # One row per unique failed physical address.
    unique_unmatched = (
        unmatched.sort_values(["state", "facility_id", "year"])
        .drop_duplicates(subset=["address_key"])
        .reset_index(drop=True)
    )

    unique_unmatched.to_csv(
        OUT_UNMATCHED,
        index=False,
    )

    # State-level rates use all hospital-year observations.
    state = (
        panel.groupby("state", dropna=False)
        .agg(
            hospital_rows=("facility_id", "size"),
            matched_rows=("geocode_success_bool", "sum"),
        )
        .reset_index()
    )

    state["unmatched_rows"] = (
        state["hospital_rows"] - state["matched_rows"]
    )
    state["match_rate_pct"] = (
        100 * state["matched_rows"] / state["hospital_rows"]
    ).round(2)

    state = state.sort_values(
        ["match_rate_pct", "hospital_rows"],
        ascending=[True, False],
    )

    state.to_csv(
        OUT_STATE,
        index=False,
    )

    year = (
        panel.groupby(["year", "snapshot"], dropna=False)
        .agg(
            hospital_rows=("facility_id", "size"),
            matched_rows=("geocode_success_bool", "sum"),
        )
        .reset_index()
    )

    year["unmatched_rows"] = (
        year["hospital_rows"] - year["matched_rows"]
    )
    year["match_rate_pct"] = (
        100 * year["matched_rows"] / year["hospital_rows"]
    ).round(2)

    year.to_csv(
        OUT_YEAR,
        index=False,
    )

    patterns = (
        unique_unmatched.groupby(
            "address_pattern",
            dropna=False,
        )
        .size()
        .reset_index(name="unique_unmatched_addresses")
        .sort_values(
            "unique_unmatched_addresses",
            ascending=False,
        )
    )

    patterns.to_csv(
        OUT_PATTERNS,
        index=False,
    )

    print("=" * 72)
    print("UNMATCHED GEOCODE DIAGNOSTICS")
    print("=" * 72)
    print(f"Hospital-year rows:              {len(panel):,}")
    print(f"Unmatched hospital-year rows:    {(~success).sum():,}")
    print(f"Unique unmatched addresses:      {len(unique_unmatched):,}")

    print("\nAddress patterns among unique failures:")
    print(patterns.to_string(index=False))

    print("\n10 states/territories with lowest match rates:")
    print(
        state[
            [
                "state",
                "hospital_rows",
                "unmatched_rows",
                "match_rate_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nCreated:")
    for path in [
        OUT_UNMATCHED,
        OUT_STATE,
        OUT_YEAR,
        OUT_PATTERNS,
    ]:
        print(f"  {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
