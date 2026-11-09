#!/usr/bin/env python3
"""
02_build_hospital_panel.py

Build a standardized 2019-2026 CMS hospital panel from the raw annual snapshots.

Place directly inside:
    phase2_hospitals/

Expected input:
    phase2_hospitals/data/raw/hospital_*.csv

Creates:
    phase2_hospitals/data/processed/hospital_panel.csv
    phase2_hospitals/data/processed/hospital_panel_schema_report.csv
    phase2_hospitals/data/processed/hospital_panel_year_summary.csv

Run:
    python3 02_build_hospital_panel.py

Requirements:
    pip3 install pandas
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_PANEL = PROCESSED_DIR / "hospital_panel.csv"
OUTPUT_SCHEMA = PROCESSED_DIR / "hospital_panel_schema_report.csv"
OUTPUT_SUMMARY = PROCESSED_DIR / "hospital_panel_year_summary.csv"


def snake_case(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# Canonical column targets with historical aliases.
ALIASES = {
    "facility_id": [
        "facility_id",
        "provider_id",
        "provider_number",
        "facilityid",
        "providerid",
        "providernumber",
    ],
    "facility_name": [
        "facility_name",
        "hospital_name",
        "provider_name",
        "facilityname",
        "hospitalname",
        "providername",
    ],
    "address": [
        "address",
        "facility_address",
        "hospital_address",
        "provider_address",
    ],
    "city": [
        "city_town",
        "citytown",
        "city",
        "provider_city",
    ],
    "state": [
        "state",
        "provider_state",
    ],
    "zip_code": [
        "zip_code",
        "zipcode",
        "zip",
        "provider_zip_code",
    ],
    "county": [
        "county_parish",
        "countyparish",
        "county",
        "county_name",
    ],
    "telephone_number": [
        "telephone_number",
        "phone_number",
        "phone",
    ],
    "hospital_type": [
        "hospital_type",
        "type_of_hospital",
    ],
    "hospital_ownership": [
        "hospital_ownership",
        "ownership",
        "hospital_owner",
    ],
    "emergency_services": [
        "emergency_services",
        "emergency_service",
    ],
    "hospital_overall_rating": [
        "hospital_overall_rating",
        "overall_rating",
    ],
}


def find_alias(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {snake_case(c): c for c in columns}

    for alias in aliases:
        alias_norm = snake_case(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]

    return None


def extract_snapshot(path: Path) -> str:
    match = re.search(r"hospital_(\d{4}-\d{2}-\d{2})\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not parse snapshot date from {path.name}")
    return match.group(1)


def standardize_file(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    snapshot = extract_snapshot(path)
    year = int(snapshot[:4])

    print(f"\nReading {path.name} ...")

    # Read everything as text first so IDs/ZIPs keep leading zeroes.
    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
        encoding_errors="replace",
    )

    original_columns = list(df.columns)
    df.columns = [snake_case(c) for c in df.columns]

    schema_rows = []

    out = pd.DataFrame(index=df.index)
    out["snapshot"] = snapshot
    out["year"] = year

    for canonical, aliases in ALIASES.items():
        source = find_alias(list(df.columns), aliases)

        if source is not None:
            out[canonical] = df[source]
            status = "matched"
        else:
            out[canonical] = pd.NA
            status = "missing"

        schema_rows.append(
            {
                "snapshot": snapshot,
                "year": year,
                "canonical_column": canonical,
                "source_column": source if source is not None else "",
                "status": status,
            }
        )

    # Preserve a few useful raw fields not covered above.
    # This helps avoid losing information if CMS changes fields later.
    for optional in [
        "meets_criteria_for_birthing_friendly_designation",
        "meets_criteria_for_promoting_interoperability_of_ehrs",
        "maternity_care_birthing_friendly_facility",
    ]:
        if optional in df.columns:
            out[optional] = df[optional]

    # Clean key string fields.
    for col in [
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip_code",
        "county",
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
    ]:
        out[col] = out[col].astype("string").str.strip()

    # ZIP normalization: keep first 5 digits if ZIP+4 appears.
    out["zip_code"] = (
        out["zip_code"]
        .str.extract(r"(\d{5})", expand=False)
        .astype("string")
    )

    # Useful geographic address string for later geocoding.
    out["full_address"] = (
        out["address"].fillna("")
        + ", "
        + out["city"].fillna("")
        + ", "
        + out["state"].fillna("")
        + " "
        + out["zip_code"].fillna("")
        + ", USA"
    ).str.replace(r"\s+", " ", regex=True).str.strip(" ,")

    print(
        f"  rows: {len(out):,} | "
        f"facility IDs present: {out['facility_id'].notna().sum():,}"
    )

    return out, schema_rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("hospital_????-??-??.csv"))

    if not files:
        raise FileNotFoundError(
            f"No hospital snapshot files found in {RAW_DIR}\n"
            "Expected files like hospital_2020-10-16.csv"
        )

    print(f"Found {len(files)} hospital snapshot files.")

    panels = []
    schema_report = []

    for path in files:
        panel_part, schema_rows = standardize_file(path)
        panels.append(panel_part)
        schema_report.extend(schema_rows)

    panel = pd.concat(
        panels,
        ignore_index=True,
        sort=False,
    )

    # Basic panel diagnostics.
    panel["facility_year_key"] = (
        panel["facility_id"].fillna("MISSING")
        + "_"
        + panel["year"].astype(str)
    )

    year_summary = (
        panel.groupby(["year", "snapshot"], dropna=False)
        .agg(
            rows=("facility_id", "size"),
            unique_facilities=("facility_id", "nunique"),
            missing_facility_id=("facility_id", lambda s: s.isna().sum()),
            missing_address=("address", lambda s: s.isna().sum()),
            missing_zip=("zip_code", lambda s: s.isna().sum()),
            missing_county=("county", lambda s: s.isna().sum()),
        )
        .reset_index()
    )

    appearances = (
        panel.dropna(subset=["facility_id"])
        .groupby("facility_id")["year"]
        .nunique()
    )

    print("\n" + "=" * 72)
    print("PANEL CHECK")
    print("=" * 72)
    print(f"Total rows:              {len(panel):,}")
    print(f"Unique hospital IDs:     {panel['facility_id'].nunique():,}")
    print(f"Years:                   {sorted(panel['year'].unique().tolist())}")
    print(
        f"Hospitals in 2+ years:   "
        f"{(appearances >= 2).sum():,}"
    )
    print(
        f"Hospitals in all years:  "
        f"{(appearances == panel['year'].nunique()).sum():,}"
    )

    duplicates = panel["facility_year_key"].duplicated(keep=False).sum()
    print(f"Duplicate hospital-years:{duplicates:>9,}")

    # Save.
    panel.to_csv(
        OUTPUT_PANEL,
        index=False
    )

    pd.DataFrame(schema_report).to_csv(
        OUTPUT_SCHEMA,
        index=False
    )

    year_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print("\nCreated:")
    print(f"  {OUTPUT_PANEL.relative_to(PROJECT_DIR)}")
    print(f"  {OUTPUT_SCHEMA.relative_to(PROJECT_DIR)}")
    print(f"  {OUTPUT_SUMMARY.relative_to(PROJECT_DIR)}")

    print("\nYear summary:")
    print(year_summary.to_string(index=False))


if __name__ == "__main__":
    main()
