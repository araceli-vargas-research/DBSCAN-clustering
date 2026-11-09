#!/usr/bin/env python3
"""
03c_retry_unmatched_geocodes.py

Conservatively retry failed U.S. hospital geocodes after address cleaning.

This script:
- keeps all successful geocodes from 03_geocode_hospitals.py
- retries only currently unmatched addresses
- strips suite/unit/floor/building fragments
- normalizes punctuation and common route/highway abbreviations
- skips Puerto Rico and U.S. territories for separate handling
- only accepts actual Census coordinate matches
- updates the geocode crosswalk and full geocoded panel

Requirements:
    pip3 install pandas requests
"""

from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
BATCH_DIR = SPATIAL_DIR / "census_retry_batches"

CROSSWALK_FILE = SPATIAL_DIR / "hospital_address_geocodes.csv"
PANEL_FILE = SPATIAL_DIR / "hospital_panel_geocoded.csv"

OUT_CROSSWALK = CROSSWALK_FILE
OUT_PANEL = PANEL_FILE
OUT_RETRY = SPATIAL_DIR / "retry_geocode_results.csv"
OUT_STILL_UNMATCHED = SPATIAL_DIR / "still_unmatched_after_retry.csv"

CENSUS_BATCH_URL = (
    "https://geocoding.geo.census.gov/geocoder/"
    "locations/addressbatch"
)
BENCHMARK = "Public_AR_Current"

BATCH_SIZE = 500
MAX_RETRIES = 4
RETRY_SECONDS = 5

SEPARATE_STATES = {"PR", "GU", "VI", "AS", "MP"}


def boolish(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def clean_component(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_street(address: str) -> str:
    """
    Conservative address cleaning.
    We do NOT invent street numbers, cities, or ZIPs.
    """
    x = clean_component(address).upper()

    # Normalize apostrophes/odd punctuation.
    x = (
        x.replace("’", "'")
         .replace("–", "-")
         .replace("—", "-")
    )

    # Remove common secondary-unit fragments.
    # Keep everything before the unit marker.
    x = re.split(
        r"\b(?:SUITE|STE|UNIT|APT|APARTMENT|"
        r"BLDG|BUILDING|FLOOR|FL|ROOM|RM|DEPT)\b",
        x,
        maxsplit=1,
    )[0]

    # Normalize highway/route words while preserving route numbers.
    replacements = [
        (r"\bSTATE ROUTE\b", "STATE RTE"),
        (r"\bSTATE HIGHWAY\b", "STATE HWY"),
        (r"\bUS HIGHWAY\b", "US HWY"),
        (r"\bU\.?S\.? HIGHWAY\b", "US HWY"),
        (r"\bU\.?S\.? ROUTE\b", "US RTE"),
        (r"\bCOUNTY ROAD\b", "COUNTY RD"),
        (r"\bHIGHWAY\b", "HWY"),
        (r"\bROUTE\b", "RTE"),
    ]

    for pattern, replacement in replacements:
        x = re.sub(pattern, replacement, x)

    # Remove punctuation that commonly interferes with matching.
    x = re.sub(r"[#,;]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def prepare_retry(crosswalk: pd.DataFrame) -> pd.DataFrame:
    success = boolish(crosswalk["geocode_success"])

    retry = crosswalk.loc[
        ~success,
        [
            "address_key",
            "address",
            "city",
            "state",
            "zip_code",
        ],
    ].copy()

    retry["state"] = retry["state"].fillna("").str.upper().str.strip()

    retry["retry_eligible"] = ~retry["state"].isin(SEPARATE_STATES)

    retry["retry_address"] = retry["address"].fillna("").map(clean_street)
    retry["retry_city"] = retry["city"].fillna("").str.strip()
    retry["retry_zip"] = (
        retry["zip_code"]
        .fillna("")
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
        .fillna("")
    )

    # Do not send unusable or unchanged blank streets.
    retry["retry_eligible"] &= retry["retry_address"].ne("")

    return retry


def write_batch(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for row in df.itertuples(index=False):
            writer.writerow(
                [
                    row.address_key,
                    row.retry_address,
                    row.retry_city,
                    row.state,
                    row.retry_zip,
                ]
            )


def post_batch(
    path: Path,
    session: requests.Session,
) -> bytes:

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with path.open("rb") as f:
                r = session.post(
                    CENSUS_BATCH_URL,
                    files={
                        "addressFile": (
                            path.name,
                            f,
                            "text/csv",
                        )
                    },
                    data={"benchmark": BENCHMARK},
                    timeout=180,
                )

            r.raise_for_status()

            if not r.content.strip():
                raise RuntimeError("Empty Census response")

            return r.content

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                print(
                    f"    attempt {attempt} failed: {exc}; "
                    f"retrying in {RETRY_SECONDS}s"
                )
                time.sleep(RETRY_SECONDS)

    raise RuntimeError(
        f"Census retry failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def parse_response(payload: bytes) -> pd.DataFrame:
    text = payload.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))

    result = []

    for row in rows:
        if not row:
            continue

        row = row + [""] * max(0, 8 - len(row))

        key = row[0].strip()
        match_status = row[2].strip()
        match_type = row[3].strip()
        matched_address = row[4].strip()
        coordinates = row[5].strip()
        tigerline_id = row[6].strip()
        side = row[7].strip()

        lat = pd.NA
        lon = pd.NA

        if coordinates:
            parts = coordinates.split(",")

            if len(parts) >= 2:
                try:
                    lon = float(parts[0].strip())
                    lat = float(parts[1].strip())
                except ValueError:
                    pass

        result.append(
            {
                "address_key": key,
                "retry_match_status": match_status,
                "retry_match_type": match_type,
                "retry_matched_address": matched_address,
                "retry_longitude": lon,
                "retry_latitude": lat,
                "retry_tigerline_id": tigerline_id,
                "retry_side": side,
            }
        )

    return pd.DataFrame(result)


def main() -> None:
    print("03c retry script started")
    print(f"Project directory: {PROJECT_DIR}")

    if not CROSSWALK_FILE.exists():
        raise FileNotFoundError(
            "Missing hospital_address_geocodes.csv. "
            "Run 03_geocode_hospitals.py first."
        )

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            "Missing hospital_panel_geocoded.csv. "
            "Run 03_geocode_hospitals.py first."
        )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(
        CROSSWALK_FILE,
        dtype=str,
        low_memory=False,
    )

    panel = pd.read_csv(
        PANEL_FILE,
        dtype=str,
        low_memory=False,
    )

    retry = prepare_retry(crosswalk)

    eligible = retry[
        retry["retry_eligible"]
    ].copy()

    separate = retry[
        ~retry["retry_eligible"]
    ].copy()

    print("=" * 72)
    print("RETRY UNMATCHED HOSPITAL GEOCODES")
    print("=" * 72)
    print(f"Currently unmatched unique addresses: {len(retry):,}")
    print(f"Eligible for cleaned Census retry:    {len(eligible):,}")
    print(
        f"PR/territories held for separate pass:{len(separate):>9,}"
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "CMS-Hospital-Spatial-Competition-Research/1.0 "
                "(academic research)"
            )
        }
    )

    results = []

    for batch_no, start in enumerate(
        range(0, len(eligible), BATCH_SIZE),
        start=1,
    ):
        batch = eligible.iloc[
            start:start + BATCH_SIZE
        ].copy()

        input_file = (
            BATCH_DIR /
            f"retry_input_{batch_no:03d}.csv"
        )

        output_file = (
            BATCH_DIR /
            f"retry_output_{batch_no:03d}.csv"
        )

        write_batch(batch, input_file)

        print(
            f"\nBatch {batch_no}: "
            f"{len(batch):,} cleaned addresses"
        )

        payload = post_batch(
            input_file,
            session,
        )

        output_file.write_bytes(payload)

        parsed = parse_response(payload)
        results.append(parsed)

        matched = (
            pd.to_numeric(
                parsed["retry_latitude"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                parsed["retry_longitude"],
                errors="coerce",
            ).notna()
        ).sum()

        print(
            f"  newly matched: "
            f"{matched:,} / {len(batch):,}"
        )

    if results:
        results = pd.concat(
            results,
            ignore_index=True,
            sort=False,
        )
    else:
        results = pd.DataFrame(
            columns=[
                "address_key",
                "retry_match_status",
                "retry_match_type",
                "retry_matched_address",
                "retry_longitude",
                "retry_latitude",
                "retry_tigerline_id",
                "retry_side",
            ]
        )

    retry_results = retry.merge(
        results,
        on="address_key",
        how="left",
    )

    retry_results.to_csv(
        OUT_RETRY,
        index=False,
    )

    # Update ONLY rows that now have real coordinates.
    updates = retry_results.copy()

    update_mask = (
        pd.to_numeric(
            updates["retry_latitude"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            updates["retry_longitude"],
            errors="coerce",
        ).notna()
    )

    updates = updates.loc[
        update_mask
    ].copy()

    update_map = (
        updates.set_index("address_key")
        .to_dict("index")
    )

    for idx, row in crosswalk.iterrows():
        key = row["address_key"]

        if key not in update_map:
            continue

        u = update_map[key]

        crosswalk.at[idx, "latitude"] = u["retry_latitude"]
        crosswalk.at[idx, "longitude"] = u["retry_longitude"]
        crosswalk.at[idx, "matched_address"] = u["retry_matched_address"]
        crosswalk.at[idx, "match_status"] = u["retry_match_status"]
        crosswalk.at[idx, "match_type"] = u["retry_match_type"]
        crosswalk.at[idx, "tigerline_id"] = u["retry_tigerline_id"]
        crosswalk.at[idx, "geocode_success"] = True
        crosswalk.at[idx, "geocode_method"] = "census_cleaned_retry"

    if "geocode_method" not in crosswalk.columns:
        crosswalk["geocode_method"] = pd.NA

    success = (
        pd.to_numeric(
            crosswalk["latitude"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            crosswalk["longitude"],
            errors="coerce",
        ).notna()
    )

    crosswalk.loc[
        success & crosswalk["geocode_method"].isna(),
        "geocode_method",
    ] = "census_original"

    crosswalk["geocode_success"] = success

    crosswalk.to_csv(
        OUT_CROSSWALK,
        index=False,
    )

    # Refresh the full panel from the updated crosswalk.
    drop_cols = [
        "matched_address",
        "match_status",
        "match_type",
        "latitude",
        "longitude",
        "tigerline_id",
        "geocode_success",
        "geocode_method",
    ]

    panel = panel.drop(
        columns=[
            c for c in drop_cols
            if c in panel.columns
        ],
        errors="ignore",
    )

    merge_cols = [
        "address_key",
        "matched_address",
        "match_status",
        "match_type",
        "latitude",
        "longitude",
        "tigerline_id",
        "geocode_success",
        "geocode_method",
    ]

    panel = panel.merge(
        crosswalk[merge_cols],
        on="address_key",
        how="left",
        validate="many_to_one",
    )

    panel.to_csv(
        OUT_PANEL,
        index=False,
    )

    still_unmatched = crosswalk.loc[
        ~crosswalk["geocode_success"],
    ].copy()

    still_unmatched.to_csv(
        OUT_STILL_UNMATCHED,
        index=False,
    )

    newly_matched = len(updates)
    total_success = int(crosswalk["geocode_success"].sum())
    total = len(crosswalk)

    print()
    print("=" * 72)
    print("RETRY COMPLETE")
    print("=" * 72)
    print(f"New unique addresses matched: {newly_matched:,}")
    print(
        f"Total unique addresses matched: "
        f"{total_success:,} / {total:,}"
    )
    print(
        f"New unique-address match rate: "
        f"{100 * total_success / total:.2f}%"
    )
    print(
        f"Still unmatched: "
        f"{len(still_unmatched):,}"
    )

    print("\nCreated/updated:")
    print(
        f"  {OUT_CROSSWALK.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUT_PANEL.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUT_RETRY.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUT_STILL_UNMATCHED.relative_to(PROJECT_DIR)}"
    )


if __name__ == "__main__":
    main()
