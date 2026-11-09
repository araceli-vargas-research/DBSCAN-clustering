#!/usr/bin/env python3

"""

01_download_hospitals.py



Download one representative CMS Hospitals snapshot per year and extract

the Hospital General Information CSV.



This version handles older CMS archives that contain nested ZIP files.



Place directly inside:

    phase2_hospitals/



Run:

    python3 01_download_hospitals.py



Optional:

    python3 01_download_hospitals.py --all-snapshots



Requirements:

    pip3 install requests

"""



from __future__ import annotations



import argparse

import hashlib

import io

import json

import re

import sys

import zipfile

from datetime import datetime, timezone

from pathlib import Path



import requests





PROJECT_DIR = Path(__file__).resolve().parent

RAW_DIR = PROJECT_DIR / "data" / "raw"

MANIFEST_PATH = RAW_DIR / "hospital_download_manifest.json"



BASE_URL = (

    "https://data.cms.gov/provider-data/sites/default/files/"

    "dataset-archives/theme/hospitals/hospitals_{snapshot}.zip"

)



ANNUAL_SNAPSHOTS = [

    "2019-04-04",

    "2020-10-16",

    "2021-07-21",

    "2022-10-06",

    "2023-11-06",

    "2024-10-30",

    "2025-11-26",

    "2026-05-13",

]



ALL_SNAPSHOTS = [

    "2019-03-04",

    "2019-04-04",



    "2020-01-04",

    "2020-05-01",

    "2020-07-04",

    "2020-10-16",



    "2021-01-27",

    "2021-03-31",

    "2021-04-28",

    "2021-07-21",

    "2021-10-27",



    "2022-01-06",

    "2022-04-06",

    "2022-07-06",

    "2022-10-06",



    "2023-01-06",

    "2023-04-06",

    "2023-07-06",

    "2023-10-06",

    "2023-11-06",



    "2024-01-31",

    "2024-04-24",

    "2024-07-31",

    "2024-10-30",



    "2025-02-19",

    "2025-04-30",

    "2025-08-14",

    "2025-11-26",



    "2026-02-25",

    "2026-05-13",

]





def sha256_bytes(data: bytes) -> str:

    return hashlib.sha256(data).hexdigest()





def normalize_header(text: str) -> str:

    return re.sub(r"[^a-z0-9]+", "", text.lower())





def header_score(header: str) -> tuple[int, int]:
    """
    Return:
      core_score      = generic hospital identity/location fields
      hospital_score  = fields specific to Hospital General Information

    This prevents CMS survey/outpatient tables from being mistaken for
    Hospital General Information just because they contain facility name,
    address, state, and ZIP.
    """
    normalized = normalize_header(header)

    core_groups = [
        ("facilityid", "providerid", "providernumber"),
        ("facilityname", "hospitalname", "providername"),
        ("address",),
        ("state",),
        ("zipcode", "zip"),
    ]

    hospital_specific_groups = [
        ("hospitaltype", "typeofhospital"),
        ("hospitalownership", "ownership"),
        ("emergencyservices", "emergencyservice"),
        ("hospitaloverallrating", "overallrating"),
    ]

    core_score = sum(
        1
        for alternatives in core_groups
        if any(term in normalized for term in alternatives)
    )

    hospital_score = sum(
        1
        for alternatives in hospital_specific_groups
        if any(term in normalized for term in alternatives)
    )

    return core_score, hospital_score


def csv_name_score(name: str) -> int:
    """
    Rank likely Hospital General Information filenames.
    """
    base = Path(name).name.lower()
    score = 0

    if "hospital" in base:
        score += 2
    if "general" in base:
        score += 3
    if "information" in base:
        score += 3

    # Strongly down-rank known non-general-information concepts.
    bad_terms = [
        "survey",
        "patient",
        "outpatient",
        "asc",
        "ambulatory",
        "hcahps",
        "payment",
        "timely",
        "complication",
        "readmission",
        "infection",
        "mortality",
    ]

    for term in bad_terms:
        if term in base:
            score -= 4

    if "dictionary" in base:
        score -= 20

    return score


def count_csv_rows(payload: bytes) -> int:
    """
    Approximate row count from raw CSV bytes.
    Good enough for candidate selection; avoids fully parsing every file.
    """
    if not payload:
        return 0

    # Number of physical lines minus header.
    # Hospital General Information is one row per facility, so a true
    # annual national file should be in the thousands, not hundreds.
    return max(0, len(payload.splitlines()) - 1)


def find_hospital_csv_in_zip(
    zf: zipfile.ZipFile,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 4,
) -> tuple[str, bytes]:
    """
    Recursively search CMS archives and nested ZIPs for the true
    Hospital General Information table.

    Selection criteria:
      - identity/location fields present
      - at least one hospital-specific general-info field
      - plausible national hospital row count
      - filename used only as a tie-breaker
    """

    candidates = []
    nested_zips = []

    for member in zf.namelist():
        if member.endswith("/"):
            continue

        lower = member.lower()

        try:
            payload = zf.read(member)
        except Exception:
            continue

        display_name = f"{prefix}{member}"

        if lower.endswith(".csv"):
            if "dictionary" in lower:
                continue

            first_line = payload.splitlines()[0] if payload else b""

            # Historical CMS CSVs may be cp1252.
            try:
                header = first_line.decode("utf-8-sig")
            except UnicodeDecodeError:
                header = first_line.decode("cp1252", errors="replace")

            core_score, hospital_score = header_score(header)
            name_score = csv_name_score(member)
            row_count = count_csv_rows(payload)

            # Candidate tuple fields:
            # valid_general, hospital_score, core_score, row_count,
            # name_score, display_name, payload
            valid_general = (
                core_score >= 5
                and hospital_score >= 1
                and row_count >= 3000
            )

            candidates.append(
                (
                    valid_general,
                    hospital_score,
                    core_score,
                    row_count,
                    name_score,
                    display_name,
                    payload,
                )
            )

        elif lower.endswith(".zip") and depth < max_depth:
            nested_zips.append((display_name, payload))

    # Search nested ZIPs too; collect their winning candidates.
    nested_winners = []

    for nested_name, nested_bytes in nested_zips:
        try:
            if not zipfile.is_zipfile(io.BytesIO(nested_bytes)):
                continue

            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                nested_winners.append(
                    find_hospital_csv_in_zip(
                        nested,
                        prefix=f"{nested_name}::",
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )

        except Exception:
            pass

    # Convert nested winners to candidate-style tuples so top-level and
    # nested candidates compete together.
    for nested_name, nested_payload in nested_winners:
        first_line = nested_payload.splitlines()[0] if nested_payload else b""

        try:
            header = first_line.decode("utf-8-sig")
        except UnicodeDecodeError:
            header = first_line.decode("cp1252", errors="replace")

        core_score, hospital_score = header_score(header)
        row_count = count_csv_rows(nested_payload)

        candidates.append(
            (
                core_score >= 5 and hospital_score >= 1 and row_count >= 3000,
                hospital_score,
                core_score,
                row_count,
                csv_name_score(nested_name),
                nested_name,
                nested_payload,
            )
        )

    # Rank true general-information candidates first.
    candidates.sort(
        key=lambda x: (
            x[0],   # valid_general
            x[1],   # hospital-specific fields
            x[2],   # generic identity/location fields
            x[3],   # row count
            x[4],   # filename score
        ),
        reverse=True,
    )

    if candidates and candidates[0][0]:
        _, _, _, _, _, best_name, best_payload = candidates[0]
        return best_name, best_payload

    diagnostics = "\n".join(
        (
            f"  rows={rows} core={core}/5 hospital_specific={hospital}/4 "
            f"name_score={name_score}: {name}"
        )
        for _, hospital, core, rows, name_score, name, _ in candidates[:30]
    )

    raise RuntimeError(
        "Could not identify a valid Hospital General Information table.\n"
        "A valid candidate must have the core hospital identity/location "
        "fields, at least one hospital-specific general-information field, "
        "and at least 3,000 rows.\n"
        "Best candidates found:\n"
        f"{diagnostics}"
    )


def validate_header_bytes(payload: bytes, source_name: str) -> None:
    first_line = payload.splitlines()[0] if payload else b""

    try:
        header = first_line.decode("utf-8-sig")
    except UnicodeDecodeError:
        header = first_line.decode("cp1252", errors="replace")

    core_score, hospital_score = header_score(header)
    rows = count_csv_rows(payload)

    if core_score < 5 or hospital_score < 1 or rows < 3000:
        raise RuntimeError(
            f"{source_name}: selected file failed Hospital General "
            f"Information validation "
            f"(core={core_score}/5, "
            f"hospital_specific={hospital_score}/4, rows={rows}).\n"
            f"Header:\n{header[:1500]}"
        )



def download_snapshot(

    snapshot: str,

    session: requests.Session

) -> dict:



    url = BASE_URL.format(snapshot=snapshot)



    print()

    print("=" * 72)

    print(f"CMS Hospitals snapshot: {snapshot}")

    print(url)



    response = session.get(

        url,

        timeout=180

    )

    response.raise_for_status()



    archive_bytes = response.content



    if not zipfile.is_zipfile(io.BytesIO(archive_bytes)):

        raise RuntimeError(

            f"{snapshot}: CMS response was not a ZIP file "

            f"({len(archive_bytes):,} bytes)."

        )



    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:

        archive_member, csv_bytes = find_hospital_csv_in_zip(zf)



    validate_header_bytes(

        csv_bytes,

        archive_member

    )



    output_path = RAW_DIR / f"hospital_{snapshot}.csv"

    output_path.write_bytes(csv_bytes)



    record = {

        "snapshot": snapshot,

        "url": url,

        "archive_member": archive_member,

        "file": str(output_path.relative_to(PROJECT_DIR)),

        "bytes": len(csv_bytes),

        "sha256": sha256_bytes(csv_bytes),

        "retrieved_utc": datetime.now(timezone.utc).isoformat(),

    }



    print(f"Extracted: {archive_member}")

    print(f"Saved:     {record['file']}")

    print(f"Bytes:     {len(csv_bytes):,}")

    print(f"SHA256:    {record['sha256']}")



    return record





def main() -> None:

    parser = argparse.ArgumentParser(

        description="Download archived CMS Hospital General Information files."

    )



    parser.add_argument(

        "--all-snapshots",

        action="store_true",

        help="Download every listed CMS archive snapshot instead of one per year.",

    )



    args = parser.parse_args()



    snapshots = (

        ALL_SNAPSHOTS

        if args.all_snapshots

        else ANNUAL_SNAPSHOTS

    )



    RAW_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    session = requests.Session()

    session.headers.update(

        {

            "User-Agent": (

                "Mozilla/5.0 "

                "CMS-Hospital-Spatial-Competition-Research/1.0"

            )

        }

    )



    manifest = []

    failures = []



    for snapshot in snapshots:

        try:

            record = download_snapshot(

                snapshot,

                session

            )

            manifest.append(record)



        except Exception as exc:

            failures.append(

                {

                    "snapshot": snapshot,

                    "error": str(exc),

                }

            )



            print(

                f"\nERROR [{snapshot}]: {exc}",

                file=sys.stderr

            )



    MANIFEST_PATH.write_text(

        json.dumps(

            manifest,

            indent=2

        ),

        encoding="utf-8"

    )



    print()

    print("=" * 72)

    print(

        f"Successfully downloaded "

        f"{len(manifest)} of {len(snapshots)} snapshots."

    )

    print(

        "Manifest:",

        MANIFEST_PATH.relative_to(PROJECT_DIR)

    )



    failure_path = (

        RAW_DIR /

        "hospital_download_failures.json"

    )



    if failures:

        failure_path.write_text(

            json.dumps(

                failures,

                indent=2

            ),

            encoding="utf-8"

        )



        print(

            "Failures:",

            failure_path.relative_to(PROJECT_DIR)

        )

        sys.exit(1)



    if failure_path.exists():

        failure_path.unlink()





if __name__ == "__main__":

    main()


