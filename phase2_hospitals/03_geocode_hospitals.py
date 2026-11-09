#!/usr/bin/env python3
"""
03_geocode_hospitals.py

Geocode unique hospital addresses in the cleaned CMS hospital panel using
the U.S. Census Batch Geocoder, then merge coordinates back onto the
full hospital-year panel.

Place directly inside:
    phase2_hospitals/

Expected input:
    data/processed/hospital_panel.csv

Creates:
    data/spatial/hospital_address_geocodes.csv
    data/spatial/hospital_panel_geocoded.csv
    data/spatial/geocode_summary.csv
    data/spatial/census_batches/...

Run:
    python3 03_geocode_hospitals.py

Requirements:
    pip3 install pandas requests

Notes:
- No API key is required.
- The script geocodes UNIQUE addresses, not every hospital-year row.
- It is restart-safe: already matched addresses are reused.
- CMS hospitals can change address over time, so the address itself,
  not facility_id alone, is the geocoding key.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
BATCH_DIR = SPATIAL_DIR / "census_batches"

INPUT_PANEL = PROCESSED_DIR / "hospital_panel.csv"

GEOCODE_CROSSWALK = SPATIAL_DIR / "hospital_address_geocodes.csv"
OUTPUT_PANEL = SPATIAL_DIR / "hospital_panel_geocoded.csv"
OUTPUT_SUMMARY = SPATIAL_DIR / "geocode_summary.csv"


# ---------------------------------------------------------------------
# CENSUS GEOCODER
# ---------------------------------------------------------------------

CENSUS_BATCH_URL = (
    "https://geocoding.geo.census.gov/geocoder/"
    "locations/addressbatch"
)

BENCHMARK = "Public_AR_Current"

# Smaller than the Census maximum so failed batches are cheap to retry.
BATCH_SIZE = 1000

MAX_RETRIES = 4
RETRY_SECONDS = 5


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
    )


def normalize_zip(series: pd.Series) -> pd.Series:
    return (
        clean_text(series)
        .str.extract(r"(\d{5})", expand=False)
        .fillna("")
    )


def make_address_key(
    address: str,
    city: str,
    state: str,
    zip_code: str,
) -> str:
    """
    Stable hash of the normalized physical address.
    """
    text = "|".join(
        [
            address.strip().upper(),
            city.strip().upper(),
            state.strip().upper(),
            zip_code.strip(),
        ]
    )

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:20]


def prepare_addresses(panel: pd.DataFrame) -> pd.DataFrame:
    required = [
        "facility_id",
        "address",
        "city",
        "state",
        "zip_code",
    ]

    missing = [
        col for col in required
        if col not in panel.columns
    ]

    if missing:
        raise RuntimeError(
            "hospital_panel.csv is missing required columns: "
            + ", ".join(missing)
        )

    work = panel[
        [
            "facility_id",
            "address",
            "city",
            "state",
            "zip_code",
        ]
    ].copy()

    work["address"] = clean_text(work["address"])
    work["city"] = clean_text(work["city"])
    work["state"] = clean_text(work["state"]).str.upper()
    work["zip_code"] = normalize_zip(work["zip_code"])

    work["address_key"] = [
        make_address_key(a, c, s, z)
        for a, c, s, z in zip(
            work["address"],
            work["city"],
            work["state"],
            work["zip_code"],
        )
    ]

    # Census batch geocoder needs a usable street plus state and either
    # city or ZIP. Flag incomplete addresses rather than sending garbage.
    work["address_usable"] = (
        work["address"].ne("")
        & work["state"].ne("")
        & (
            work["city"].ne("")
            | work["zip_code"].ne("")
        )
    )

    unique = (
        work[
            [
                "address_key",
                "address",
                "city",
                "state",
                "zip_code",
                "address_usable",
            ]
        ]
        .drop_duplicates(subset=["address_key"])
        .reset_index(drop=True)
    )

    return unique


def load_existing_crosswalk() -> pd.DataFrame:
    if not GEOCODE_CROSSWALK.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            GEOCODE_CROSSWALK,
            dtype=str,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    return df


def write_census_input(
    batch: pd.DataFrame,
    path: Path,
) -> None:
    """
    Census batch input columns:
      unique_id, street, city, state, zip
    No header.
    """
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        for row in batch.itertuples(index=False):
            writer.writerow(
                [
                    row.address_key,
                    row.address,
                    row.city,
                    row.state,
                    row.zip_code,
                ]
            )


def submit_batch(
    input_path: Path,
    output_path: Path,
    session: requests.Session,
) -> bytes:

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with input_path.open("rb") as f:
                response = session.post(
                    CENSUS_BATCH_URL,
                    files={
                        "addressFile": (
                            input_path.name,
                            f,
                            "text/csv",
                        )
                    },
                    data={
                        "benchmark": BENCHMARK,
                    },
                    timeout=180,
                )

            response.raise_for_status()

            payload = response.content

            if not payload.strip():
                raise RuntimeError(
                    "Census returned an empty response."
                )

            output_path.write_bytes(payload)
            return payload

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                print(
                    f"    attempt {attempt} failed: {exc}"
                )
                print(
                    f"    retrying in {RETRY_SECONDS} seconds..."
                )
                time.sleep(RETRY_SECONDS)

    raise RuntimeError(
        f"Census geocoder failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def parse_census_response(
    payload: bytes,
) -> pd.DataFrame:
    """
    Census locations/addressbatch response is a CSV without a header.

    Typical fields:
      0 id
      1 input address
      2 match status
      3 match type
      4 matched address
      5 coordinates
      6 tigerline id
      7 side

    We parse defensively because unmatched rows may contain blanks.
    """
    text = payload.decode(
        "utf-8-sig",
        errors="replace",
    )

    rows = list(
        csv.reader(
            io.StringIO(text)
        )
    )

    parsed = []

    for row in rows:
        if not row:
            continue

        # Pad short rows.
        row = row + [""] * max(0, 8 - len(row))

        address_key = row[0].strip()
        input_address = row[1].strip()
        match_status = row[2].strip()
        match_type = row[3].strip()
        matched_address = row[4].strip()
        coordinates = row[5].strip()
        tigerline_id = row[6].strip()
        side = row[7].strip()

        longitude = pd.NA
        latitude = pd.NA

        if coordinates:
            parts = [
                x.strip()
                for x in coordinates.split(",")
            ]

            if len(parts) >= 2:
                try:
                    longitude = float(parts[0])
                    latitude = float(parts[1])
                except ValueError:
                    pass

        parsed.append(
            {
                "address_key": address_key,
                "census_input_address": input_address,
                "match_status": match_status,
                "match_type": match_type,
                "matched_address": matched_address,
                "longitude": longitude,
                "latitude": latitude,
                "tigerline_id": tigerline_id,
                "side": side,
            }
        )

    return pd.DataFrame(parsed)


def save_crosswalk(
    addresses: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:

    base = addresses.copy()

    result_cols = [
        "address_key",
        "census_input_address",
        "match_status",
        "match_type",
        "matched_address",
        "longitude",
        "latitude",
        "tigerline_id",
        "side",
    ]

    # Keep only newest result for a key if duplicates occur.
    if not results.empty:
        results = (
            results[result_cols]
            .drop_duplicates(
                subset=["address_key"],
                keep="last",
            )
        )

        base = base.merge(
            results,
            on="address_key",
            how="left",
        )
    else:
        for col in result_cols[1:]:
            base[col] = pd.NA

    base["geocode_success"] = (
        pd.to_numeric(
            base["latitude"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            base["longitude"],
            errors="coerce",
        ).notna()
    )

    base.to_csv(
        GEOCODE_CROSSWALK,
        index=False,
    )

    return base


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    if not INPUT_PANEL.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_PANEL}\n\n"
            "Run python3 02_build_hospital_panel.py first."
        )

    SPATIAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel = pd.read_csv(
        INPUT_PANEL,
        dtype=str,
        low_memory=False,
    )

    print("=" * 72)
    print("HOSPITAL GEOCODING")
    print("=" * 72)
    print(f"Hospital-year rows: {len(panel):,}")

    addresses = prepare_addresses(panel)

    print(
        f"Unique physical addresses: "
        f"{len(addresses):,}"
    )

    usable = addresses[
        addresses["address_usable"]
    ].copy()

    unusable = addresses[
        ~addresses["address_usable"]
    ].copy()

    print(
        f"Usable addresses:          "
        f"{len(usable):,}"
    )
    print(
        f"Incomplete addresses:      "
        f"{len(unusable):,}"
    )

    existing = load_existing_crosswalk()

    existing_success = set()

    if not existing.empty:
        if {
            "address_key",
            "latitude",
            "longitude",
        }.issubset(existing.columns):

            lat = pd.to_numeric(
                existing["latitude"],
                errors="coerce",
            )

            lon = pd.to_numeric(
                existing["longitude"],
                errors="coerce",
            )

            existing_success = set(
                existing.loc[
                    lat.notna() & lon.notna(),
                    "address_key",
                ].astype(str)
            )

    remaining = usable[
        ~usable["address_key"].isin(
            existing_success
        )
    ].copy()

    print(
        f"Already geocoded:          "
        f"{len(existing_success):,}"
    )
    print(
        f"Need Census lookup:        "
        f"{len(remaining):,}"
    )

    # Begin with any prior results so successful lookups survive reruns.
    prior_results = pd.DataFrame()

    if not existing.empty:
        keep = [
            c for c in [
                "address_key",
                "census_input_address",
                "match_status",
                "match_type",
                "matched_address",
                "longitude",
                "latitude",
                "tigerline_id",
                "side",
            ]
            if c in existing.columns
        ]

        prior_results = existing[keep].copy()

    all_new_results = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "CMS-Hospital-Spatial-Competition-Research/1.0 "
                "(academic research)"
            )
        }
    )

    if len(remaining) > 0:

        batch_number = 0

        for start in range(
            0,
            len(remaining),
            BATCH_SIZE,
        ):
            batch_number += 1

            batch = remaining.iloc[
                start:start + BATCH_SIZE
            ].copy()

            print()
            print(
                f"Batch {batch_number}: "
                f"{len(batch):,} addresses"
            )

            input_path = (
                BATCH_DIR
                / f"census_input_{batch_number:03d}.csv"
            )

            output_path = (
                BATCH_DIR
                / f"census_output_{batch_number:03d}.csv"
            )

            write_census_input(
                batch,
                input_path,
            )

            payload = submit_batch(
                input_path,
                output_path,
                session,
            )

            result = parse_census_response(
                payload
            )

            all_new_results.append(
                result
            )

            matched = pd.to_numeric(
                result.get(
                    "latitude",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).notna().sum()

            print(
                f"  matched coordinates: "
                f"{matched:,} / {len(batch):,}"
            )

            # Save progress after EVERY batch.
            combined_results = pd.concat(
                [prior_results]
                + all_new_results,
                ignore_index=True,
                sort=False,
            )

            crosswalk = save_crosswalk(
                addresses,
                combined_results,
            )

            print(
                f"  progress saved: "
                f"{GEOCODE_CROSSWALK.relative_to(PROJECT_DIR)}"
            )

    # Final crosswalk.
    if all_new_results:
        combined_results = pd.concat(
            [prior_results]
            + all_new_results,
            ignore_index=True,
            sort=False,
        )
    else:
        combined_results = prior_results

    crosswalk = save_crosswalk(
        addresses,
        combined_results,
    )

    # -----------------------------------------------------------------
    # Merge coordinates back onto full hospital-year panel.
    # -----------------------------------------------------------------

    panel["address"] = clean_text(panel["address"])
    panel["city"] = clean_text(panel["city"])
    panel["state"] = clean_text(panel["state"]).str.upper()
    panel["zip_code"] = normalize_zip(panel["zip_code"])

    panel["address_key"] = [
        make_address_key(a, c, s, z)
        for a, c, s, z in zip(
            panel["address"],
            panel["city"],
            panel["state"],
            panel["zip_code"],
        )
    ]

    geocode_fields = crosswalk[
        [
            "address_key",
            "matched_address",
            "match_status",
            "match_type",
            "latitude",
            "longitude",
            "tigerline_id",
            "geocode_success",
        ]
    ].copy()

    merged = panel.merge(
        geocode_fields,
        on="address_key",
        how="left",
        validate="many_to_one",
    )

    merged.to_csv(
        OUTPUT_PANEL,
        index=False,
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    merged["geocode_success"] = (
        merged["geocode_success"]
        .fillna(False)
        .astype(bool)
    )

    year_summary = (
        merged.groupby(
            ["year", "snapshot"],
            dropna=False,
        )
        .agg(
            hospital_rows=("facility_id", "size"),
            geocoded_rows=(
                "geocode_success",
                "sum",
            ),
        )
        .reset_index()
    )

    year_summary["geocode_rate_pct"] = (
        100
        * year_summary["geocoded_rows"]
        / year_summary["hospital_rows"]
    ).round(2)

    year_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    unique_success = (
        crosswalk["geocode_success"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print()
    print("=" * 72)
    print("GEOCODING COMPLETE")
    print("=" * 72)

    print(
        f"Unique addresses matched: "
        f"{unique_success:,} / "
        f"{len(crosswalk):,}"
    )

    if len(crosswalk):
        print(
            f"Unique-address match rate: "
            f"{100 * unique_success / len(crosswalk):.2f}%"
        )

    print()
    print("Created:")
    print(
        f"  {GEOCODE_CROSSWALK.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUTPUT_PANEL.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUTPUT_SUMMARY.relative_to(PROJECT_DIR)}"
    )

    print()
    print("Year summary:")
    print(
        year_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
