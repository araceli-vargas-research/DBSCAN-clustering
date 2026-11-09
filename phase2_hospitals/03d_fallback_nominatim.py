#!/usr/bin/env python3
"""
03d_fallback_nominatim.py

Fallback geocoder for hospital addresses that the U.S. Census geocoder
could not match.

Strategy:
1. Preserve all successful Census coordinates.
2. Retry only still-unmatched addresses.
3. Search OpenStreetMap Nominatim at <= 1 request/second.
4. Use structured address search first.
5. If that fails, try facility name + city/state/ZIP.
6. Validate returned country/state/ZIP conservatively.
7. Save progress after every successful or failed lookup.

Place directly inside:
    phase2_hospitals/

Run:
    python3 03d_fallback_nominatim.py

Requirements:
    pip3 install pandas requests

Important:
- Public Nominatim policy requires <= 1 request/second.
- This script caches every result so reruns do not repeat completed queries.
- Puerto Rico and territories remain flagged separately by default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"

CROSSWALK_FILE = SPATIAL_DIR / "hospital_address_geocodes.csv"
PANEL_FILE = SPATIAL_DIR / "hospital_panel_geocoded.csv"
UNMATCHED_FILE = SPATIAL_DIR / "still_unmatched_after_retry.csv"

CACHE_FILE = SPATIAL_DIR / "nominatim_fallback_cache.csv"
OUT_STILL_UNMATCHED = SPATIAL_DIR / "still_unmatched_after_nominatim.csv"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

REQUEST_DELAY = 1.1

# Handle these separately later because their address systems / rural
# classifications are different from the 50-state/DC sample.
SEPARATE_STATES = {"PR", "GU", "VI", "AS", "MP"}

USER_AGENT = (
    "CMS-Hospital-Spatial-Competition-Research/1.0 "
    "(academic research; hospital geocoding)"
)


def boolish(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def normalize_zip(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else ""


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "address_key",
                "fallback_status",
                "query_type",
                "query_text",
                "nominatim_latitude",
                "nominatim_longitude",
                "nominatim_display_name",
                "nominatim_category",
                "nominatim_type",
                "nominatim_importance",
                "returned_state",
                "returned_postcode",
                "returned_country_code",
            ]
        )

    return pd.read_csv(
        CACHE_FILE,
        dtype=str,
        low_memory=False,
    )


def save_cache(cache: pd.DataFrame) -> None:
    cache.drop_duplicates(
        subset=["address_key"],
        keep="last",
    ).to_csv(
        CACHE_FILE,
        index=False,
    )


def query_nominatim(
    session: requests.Session,
    params: dict,
) -> list[dict]:
    params = {
        **params,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 3,
        "countrycodes": "us",
    }

    r = session.get(
        NOMINATIM_URL,
        params=params,
        timeout=60,
    )
    r.raise_for_status()

    time.sleep(REQUEST_DELAY)

    return r.json()


def state_matches(expected_state: str, result: dict) -> bool:
    """
    Conservative state validation.

    Nominatim may return:
      state = "Alabama"
    rather than "AL", so we also inspect ISO3166-2 fields.
    """
    expected_state = expected_state.upper().strip()

    address = result.get("address", {}) or {}

    iso_values = [
        str(v).upper()
        for k, v in address.items()
        if str(k).upper().startswith("ISO3166-2")
    ]

    if any(v.endswith(f"-{expected_state}") for v in iso_values):
        return True

    # If no usable ISO state code exists, do not reject solely on this.
    return len(iso_values) == 0


def zip_matches(expected_zip: str, result: dict) -> bool:
    if not expected_zip:
        return True

    address = result.get("address", {}) or {}
    returned = normalize_zip(address.get("postcode", ""))

    if not returned:
        # Missing postcode in OSM result is not automatically fatal.
        return True

    return returned == expected_zip


def choose_result(
    results: list[dict],
    expected_state: str,
    expected_zip: str,
) -> dict | None:
    for result in results:
        addr = result.get("address", {}) or {}

        country_code = str(
            addr.get("country_code", "")
        ).lower()

        if country_code not in {"us"}:
            continue

        if not state_matches(expected_state, result):
            continue

        if not zip_matches(expected_zip, result):
            continue

        if "lat" not in result or "lon" not in result:
            continue

        return result

    return None


def make_cache_row(
    address_key: str,
    status: str,
    query_type: str,
    query_text: str,
    result: dict | None,
) -> dict:

    if result is None:
        return {
            "address_key": address_key,
            "fallback_status": status,
            "query_type": query_type,
            "query_text": query_text,
            "nominatim_latitude": "",
            "nominatim_longitude": "",
            "nominatim_display_name": "",
            "nominatim_category": "",
            "nominatim_type": "",
            "nominatim_importance": "",
            "returned_state": "",
            "returned_postcode": "",
            "returned_country_code": "",
        }

    address = result.get("address", {}) or {}

    return {
        "address_key": address_key,
        "fallback_status": status,
        "query_type": query_type,
        "query_text": query_text,
        "nominatim_latitude": result.get("lat", ""),
        "nominatim_longitude": result.get("lon", ""),
        "nominatim_display_name": result.get("display_name", ""),
        "nominatim_category": result.get("category", ""),
        "nominatim_type": result.get("type", ""),
        "nominatim_importance": result.get("importance", ""),
        "returned_state": address.get("state", ""),
        "returned_postcode": address.get("postcode", ""),
        "returned_country_code": address.get("country_code", ""),
    }


def main() -> None:
    print("03d Nominatim fallback started")
    print(f"Project directory: {PROJECT_DIR}")

    if not CROSSWALK_FILE.exists():
        raise FileNotFoundError(
            "Missing hospital_address_geocodes.csv."
        )

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            "Missing hospital_panel_geocoded.csv."
        )

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

    success = boolish(crosswalk["geocode_success"])

    unmatched = crosswalk.loc[
        ~success
    ].copy()

    unmatched["state"] = (
        unmatched["state"]
        .fillna("")
        .str.upper()
        .str.strip()
    )

    mainland = unmatched.loc[
        ~unmatched["state"].isin(SEPARATE_STATES)
    ].copy()

    held_separate = unmatched.loc[
        unmatched["state"].isin(SEPARATE_STATES)
    ].copy()

    # Add a representative facility name to help fallback search.
    names = (
        panel[
            ["address_key", "facility_name"]
        ]
        .dropna(subset=["address_key"])
        .drop_duplicates(
            subset=["address_key"],
            keep="last",
        )
    )

    mainland = mainland.merge(
        names,
        on="address_key",
        how="left",
    )

    cache = load_cache()

    done_keys = set(
        cache["address_key"]
        .dropna()
        .astype(str)
    )

    todo = mainland.loc[
        ~mainland["address_key"].isin(done_keys)
    ].copy()

    print("=" * 72)
    print("NOMINATIM FALLBACK")
    print("=" * 72)
    print(
        f"Still unmatched unique addresses: "
        f"{len(unmatched):,}"
    )
    print(
        f"50-state/DC addresses eligible:   "
        f"{len(mainland):,}"
    )
    print(
        f"PR/territories held separately:   "
        f"{len(held_separate):,}"
    )
    print(
        f"Already cached:                    "
        f"{len(done_keys):,}"
    )
    print(
        f"Need Nominatim lookup:             "
        f"{len(todo):,}"
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    cache_rows = cache.to_dict("records")

    for i, row in enumerate(
        todo.itertuples(index=False),
        start=1,
    ):
        key = row.address_key
        street = normalize_text(row.address)
        city = normalize_text(row.city)
        state = normalize_text(row.state)
        zip_code = normalize_zip(row.zip_code)
        facility_name = normalize_text(
            getattr(row, "facility_name", "")
        )

        print(
            f"[{i}/{len(todo)}] "
            f"{facility_name or key} — "
            f"{city}, {state}"
        )

        matched = None
        query_type = ""
        query_text = ""

        # -------------------------------------------------------------
        # Attempt 1: structured postal address
        # -------------------------------------------------------------
        structured = {
            "street": street,
            "city": city,
            "state": state,
            "postalcode": zip_code,
        }

        try:
            results = query_nominatim(
                session,
                structured,
            )

            matched = choose_result(
                results,
                state,
                zip_code,
            )

            query_type = "structured_address"
            query_text = json.dumps(
                structured,
                sort_keys=True,
            )

        except Exception as exc:
            print(f"  structured query error: {exc}")

        # -------------------------------------------------------------
        # Attempt 2: hospital/facility name + locality
        # -------------------------------------------------------------
        if matched is None and facility_name:
            q = ", ".join(
                x
                for x in [
                    facility_name,
                    street,
                    city,
                    state,
                    zip_code,
                    "USA",
                ]
                if x
            )

            try:
                results = query_nominatim(
                    session,
                    {"q": q},
                )

                matched = choose_result(
                    results,
                    state,
                    zip_code,
                )

                query_type = "facility_name_address"
                query_text = q

            except Exception as exc:
                print(f"  name query error: {exc}")

        # -------------------------------------------------------------
        # Attempt 3: facility name + city/state only
        # -------------------------------------------------------------
        if matched is None and facility_name:
            q = ", ".join(
                x
                for x in [
                    facility_name,
                    city,
                    state,
                    "USA",
                ]
                if x
            )

            try:
                results = query_nominatim(
                    session,
                    {"q": q},
                )

                matched = choose_result(
                    results,
                    state,
                    "",
                )

                query_type = "facility_name_locality"
                query_text = q

            except Exception as exc:
                print(f"  locality query error: {exc}")

        if matched is not None:
            print(
                f"  MATCH: "
                f"{matched.get('display_name', '')}"
            )

            record = make_cache_row(
                key,
                "matched",
                query_type,
                query_text,
                matched,
            )
        else:
            print("  no acceptable match")

            record = make_cache_row(
                key,
                "unmatched",
                query_type,
                query_text,
                None,
            )

        cache_rows.append(record)

        # Save after every lookup so interruption is safe.
        cache_df = pd.DataFrame(cache_rows)
        save_cache(cache_df)

    cache = load_cache()

    # -------------------------------------------------------------
    # Apply successful fallback coordinates to crosswalk.
    # -------------------------------------------------------------
    matched_cache = cache.loc[
        cache["fallback_status"].eq("matched")
    ].copy()

    matched_cache["nominatim_latitude"] = pd.to_numeric(
        matched_cache["nominatim_latitude"],
        errors="coerce",
    )

    matched_cache["nominatim_longitude"] = pd.to_numeric(
        matched_cache["nominatim_longitude"],
        errors="coerce",
    )

    matched_cache = matched_cache.loc[
        matched_cache["nominatim_latitude"].notna()
        & matched_cache["nominatim_longitude"].notna()
    ].copy()

    update_map = (
        matched_cache.set_index("address_key")
        .to_dict("index")
    )

    if "geocode_method" not in crosswalk.columns:
        crosswalk["geocode_method"] = pd.NA

    if "geocode_source_detail" not in crosswalk.columns:
        crosswalk["geocode_source_detail"] = pd.NA

    newly_applied = 0

    for idx, row in crosswalk.iterrows():
        key = str(row["address_key"])

        # Never replace an existing good Census geocode.
        current_success = (
            pd.notna(row.get("latitude"))
            and pd.notna(row.get("longitude"))
        )

        if current_success:
            continue

        if key not in update_map:
            continue

        m = update_map[key]

        crosswalk.at[idx, "latitude"] = (
            m["nominatim_latitude"]
        )
        crosswalk.at[idx, "longitude"] = (
            m["nominatim_longitude"]
        )
        crosswalk.at[idx, "matched_address"] = (
            m["nominatim_display_name"]
        )
        crosswalk.at[idx, "match_status"] = "Match"
        crosswalk.at[idx, "match_type"] = "Nominatim fallback"
        crosswalk.at[idx, "geocode_method"] = (
            "nominatim_fallback"
        )
        crosswalk.at[idx, "geocode_source_detail"] = (
            m["query_type"]
        )

        newly_applied += 1

    lat = pd.to_numeric(
        crosswalk["latitude"],
        errors="coerce",
    )
    lon = pd.to_numeric(
        crosswalk["longitude"],
        errors="coerce",
    )

    crosswalk["geocode_success"] = (
        lat.notna() & lon.notna()
    )

    # Label pre-existing matches if not already labeled.
    census_mask = (
        crosswalk["geocode_success"]
        & crosswalk["geocode_method"].isna()
    )

    crosswalk.loc[
        census_mask,
        "geocode_method",
    ] = "census"

    crosswalk.to_csv(
        CROSSWALK_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Refresh full panel.
    # -------------------------------------------------------------
    replace_cols = [
        "matched_address",
        "match_status",
        "match_type",
        "latitude",
        "longitude",
        "geocode_success",
        "geocode_method",
        "geocode_source_detail",
    ]

    panel = panel.drop(
        columns=[
            c
            for c in replace_cols
            if c in panel.columns
        ],
        errors="ignore",
    )

    panel = panel.merge(
        crosswalk[
            [
                "address_key",
                *replace_cols,
            ]
        ],
        on="address_key",
        how="left",
        validate="many_to_one",
    )

    panel.to_csv(
        PANEL_FILE,
        index=False,
    )

    still_unmatched = crosswalk.loc[
        ~boolish(crosswalk["geocode_success"])
    ].copy()

    still_unmatched.to_csv(
        OUT_STILL_UNMATCHED,
        index=False,
    )

    total_success = int(
        boolish(crosswalk["geocode_success"]).sum()
    )

    print()
    print("=" * 72)
    print("NOMINATIM FALLBACK COMPLETE")
    print("=" * 72)
    print(
        f"Nominatim matches available: "
        f"{len(matched_cache):,}"
    )
    print(
        f"New coordinates applied:     "
        f"{newly_applied:,}"
    )
    print(
        f"Total unique addresses matched: "
        f"{total_success:,} / {len(crosswalk):,}"
    )
    print(
        f"Unique-address match rate: "
        f"{100 * total_success / len(crosswalk):.2f}%"
    )
    print(
        f"Still unmatched: "
        f"{len(still_unmatched):,}"
    )

    print("\nCreated/updated:")
    print(
        f"  {CROSSWALK_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {PANEL_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {CACHE_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {OUT_STILL_UNMATCHED.relative_to(PROJECT_DIR)}"
    )


if __name__ == "__main__":
    main()
