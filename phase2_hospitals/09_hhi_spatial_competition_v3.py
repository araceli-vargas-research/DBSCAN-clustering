#!/usr/bin/env python3
"""
09_hhi_spatial_competition.py

Build system-level hospital concentration measures for the 2023 spatial
clusters using the AHRQ Compendium of U.S. Health Systems hospital linkage
file, then test whether spatial isolation captures a distinct dimension from
ownership concentration.

Place directly inside:
    phase2_hospitals/

Expected inputs:
    data/spatial/hospital_cluster_rucc_panel.csv

Downloads:
    AHRQ Compendium of U.S. Health Systems, 2023, Hospital Linkage File

Creates:
    data/raw/ahrq_hospital_linkage_2023.csv
    data/spatial/hospital_2023_system_linkage.csv
    data/spatial/cluster_hhi_2023.csv
    data/results/hhi_spatial_descriptives_2023.csv
    data/results/hhi_spatial_regressions_2023.csv
    data/results/hhi_spatial_summary_2023.txt

Key concentration measures:
    hhi_count_10000
        HHI based on hospital-count shares by health system within each
        10-km service cluster.

    hhi_beds_10000
        HHI based on bed shares by health system within each service cluster.

    top_system_share_count
        Largest system's share of hospitals in the cluster.

    top_system_share_beds
        Largest system's share of beds in the cluster.

Standalone hospitals:
    Hospitals not linked to an AHRQ health system are treated as separate
    independent firms, each with its own firm ID. This avoids incorrectly
    pooling all independents into one "non-system" owner.

Important:
    This is a 2023 cross-section because the AHRQ linkage file is for 2023.
    It should NOT be mechanically applied to 2019-2026 as if system ownership
    never changed.

Run:
    pip3 install pandas numpy requests statsmodels
    python3 09_hhi_spatial_competition.py
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
RESULTS_DIR = PROJECT_DIR / "data" / "results"

INPUT_FILE = (
    SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
)

AHRQ_FILE = (
    RAW_DIR / "ahrq_hospital_linkage_2023.csv"
)

HOSPITAL_LINKED_OUTPUT = (
    SPATIAL_DIR / "hospital_2023_system_linkage.csv"
)

CLUSTER_HHI_OUTPUT = (
    SPATIAL_DIR / "cluster_hhi_2023.csv"
)

DESCRIPTIVES_OUTPUT = (
    RESULTS_DIR / "hhi_spatial_descriptives_2023.csv"
)

REGRESSIONS_OUTPUT = (
    RESULTS_DIR / "hhi_spatial_regressions_2023.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR / "hhi_spatial_summary_2023.txt"
)

AHRQ_URL = (
    "https://www.ahrq.gov/sites/default/files/"
    "wysiwyg/chsp/compendium/"
    "chsp-hospital-linkage-2023.csv"
)

AHRQ_LANDING_PAGE = (
    "https://www.ahrq.gov/chsp/data-resources/compendium-2023.html"
)

LOCAL_DOWNLOAD_CANDIDATES = [
    Path.home() / "Downloads" / "chsp-hospital-linkage-2023.csv",
    Path.home() / "Downloads" / "ahrq_hospital_linkage_2023.csv",
]

TARGET_YEAR = 2023
PRIMARY_EPS = 10


def download_ahrq() -> None:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # First, if the user already downloaded the official CSV in a browser,
    # use it directly.
    for candidate in LOCAL_DOWNLOAD_CANDIDATES:
        if candidate.exists():
            print(
                f"Using browser-downloaded AHRQ file: {candidate}"
            )
            AHRQ_FILE.write_bytes(
                candidate.read_bytes()
            )
            return

    print(
        "Downloading AHRQ 2023 Hospital Linkage File..."
    )

    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/26.0 Safari/605.1.15"
        ),
        "Accept": (
            "text/csv,text/plain,application/octet-stream,"
            "*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": AHRQ_LANDING_PAGE,
    }

    try:
        # Visit the official landing page first so any cookies/session
        # requirements are satisfied before requesting the CSV.
        landing = session.get(
            AHRQ_LANDING_PAGE,
            headers={
                **headers,
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            },
            timeout=60,
        )
        landing.raise_for_status()

        response = session.get(
            AHRQ_URL,
            headers=headers,
            timeout=120,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = (
            response.headers.get(
                "Content-Type",
                "",
            ).lower()
        )

        if (
            len(response.content) < 1000
            or "text/html" in content_type
        ):
            raise RuntimeError(
                "AHRQ returned an HTML/block page instead of the CSV."
            )

        AHRQ_FILE.write_bytes(
            response.content
        )

        print(
            f"Saved: {AHRQ_FILE.relative_to(PROJECT_DIR)}"
        )
        return

    except Exception as exc:
        print()
        print(
            "Automatic AHRQ download was blocked:"
        )
        print(f"  {exc}")
        print()
        print(
            "AHRQ allows the file through its browser download page."
        )
        print(
            "Download the official 2023 Hospital Linkage CSV and leave it in:"
        )
        print(
            f"  {LOCAL_DOWNLOAD_CANDIDATES[0]}"
        )
        print()
        print(
            "Then rerun this same script; it will detect and copy the file "
            "automatically."
        )
        raise RuntimeError(
            "AHRQ programmatic download blocked; browser download required."
        ) from exc


def read_ahrq() -> pd.DataFrame:
    if not AHRQ_FILE.exists():
        download_ahrq()

    # AHRQ CSVs may contain Windows-1252 characters.
    last_error = None

    for encoding in [
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]:
        try:
            df = pd.read_csv(
                AHRQ_FILE,
                dtype=str,
                low_memory=False,
                encoding=encoding,
            )

            print(
                f"AHRQ decoded with encoding={encoding}"
            )
            return df

        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def normalize_ccn(series: pd.Series) -> pd.Series:
    """
    Normalize CMS Certification Numbers to six-character strings.
    """
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    # Keep digits only where possible.
    s = s.str.replace(
        r"[^0-9]",
        "",
        regex=True,
    )

    return s.str.zfill(6)


def hhi_from_shares(shares: pd.Series) -> float:
    x = pd.to_numeric(
        shares,
        errors="coerce",
    ).dropna()

    if len(x) == 0:
        return np.nan

    return float(
        10000.0 * np.square(x).sum()
    )



def cluster_rurality_from_members(g: pd.DataFrame) -> str:
    """
    Reconstruct cluster rurality from hospital-level RUCC labels.

    rural == 1 -> nonmetro (RUCC 4-9)
    rural == 0 -> metro    (RUCC 1-3)
    """
    if "rural" not in g.columns:
        return "unmatched"

    vals = pd.to_numeric(
        g["rural"],
        errors="coerce",
    ).dropna()

    if len(vals) == 0:
        return "unmatched"

    has_rural = bool((vals == 1).any())
    has_metro = bool((vals == 0).any())

    if has_rural and has_metro:
        return "mixed"
    if has_rural:
        return "rural_only"
    return "metro_only"


def build_cluster_hhi(
    hospitals: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cluster_id, g in hospitals.groupby(
        "cluster_id",
        dropna=False,
    ):
        g = g.copy()

        # ---------------------------------------------------------
        # COUNT-BASED SYSTEM SHARES
        # ---------------------------------------------------------
        count_by_firm = (
            g.groupby(
                "firm_id",
                dropna=False,
            )
            .size()
            .rename("hospital_count")
            .reset_index()
        )

        total_hospitals = int(
            count_by_firm[
                "hospital_count"
            ].sum()
        )

        count_by_firm[
            "share_count"
        ] = (
            count_by_firm[
                "hospital_count"
            ]
            / total_hospitals
        )

        hhi_count = hhi_from_shares(
            count_by_firm[
                "share_count"
            ]
        )

        top_share_count = float(
            count_by_firm[
                "share_count"
            ].max()
        )

        # ---------------------------------------------------------
        # BED-WEIGHTED SYSTEM SHARES
        # ---------------------------------------------------------
        beds = pd.to_numeric(
            g["hos_beds"],
            errors="coerce",
        )

        g["hos_beds_num"] = beds

        valid_beds = g.loc[
            g["hos_beds_num"].notna()
            & (g["hos_beds_num"] > 0)
        ].copy()

        if len(valid_beds):
            beds_by_firm = (
                valid_beds.groupby(
                    "firm_id",
                    dropna=False,
                )["hos_beds_num"]
                .sum()
                .rename("beds")
                .reset_index()
            )

            total_beds = float(
                beds_by_firm[
                    "beds"
                ].sum()
            )

            beds_by_firm[
                "share_beds"
            ] = (
                beds_by_firm[
                    "beds"
                ]
                / total_beds
            )

            hhi_beds = hhi_from_shares(
                beds_by_firm[
                    "share_beds"
                ]
            )

            top_share_beds = float(
                beds_by_firm[
                    "share_beds"
                ].max()
            )

            bed_coverage = (
                len(valid_beds)
                / len(g)
            )

        else:
            total_beds = np.nan
            hhi_beds = np.nan
            top_share_beds = np.nan
            bed_coverage = 0.0

        n_system_linked = int(
            g["health_sys_id"]
            .notna()
            .sum()
        )

        n_independent = int(
            g["health_sys_id"]
            .isna()
            .sum()
        )

        rows.append(
            {
                "year": TARGET_YEAR,
                "cluster_id": cluster_id,
                "cluster_size": len(g),
                "number_of_firms": int(
                    count_by_firm[
                        "firm_id"
                    ].nunique()
                ),
                "system_linked_hospitals": (
                    n_system_linked
                ),
                "independent_hospitals": (
                    n_independent
                ),
                "hhi_count_10000": hhi_count,
                "top_system_share_count": (
                    top_share_count
                ),
                "total_beds_observed": (
                    total_beds
                ),
                "bed_data_coverage": (
                    bed_coverage
                ),
                "hhi_beds_10000": (
                    hhi_beds
                ),
                "top_system_share_beds": (
                    top_share_beds
                ),
                "nearest_competing_cluster_km": (
                    pd.to_numeric(
                        g[
                            "nearest_competing_cluster_km"
                        ],
                        errors="coerce",
                    ).iloc[0]
                ),
                "within_cluster_mean_km": (
                    pd.to_numeric(
                        g[
                            "within_cluster_mean_km"
                        ],
                        errors="coerce",
                    ).iloc[0]
                ),
                "isolation_ratio": (
                    pd.to_numeric(
                        g[
                            "isolation_ratio"
                        ],
                        errors="coerce",
                    ).iloc[0]
                ),
                "cluster_rurality": (
                    cluster_rurality_from_members(g)
                ),
            }
        )

    return pd.DataFrame(rows)


def extract_result(
    model,
    term: str,
    model_name: str,
    outcome: str,
) -> dict:

    ci = model.conf_int()

    return {
        "model": model_name,
        "outcome": outcome,
        "term": term,
        "coefficient": float(
            model.params[term]
        ),
        "robust_se_hc1": float(
            model.bse[term]
        ),
        "p_value": float(
            model.pvalues[term]
        ),
        "ci_lower": float(
            ci.loc[term, 0]
        ),
        "ci_upper": float(
            ci.loc[term, 1]
        ),
        "n": int(model.nobs),
        "r_squared": float(
            model.rsquared
        ),
        "adj_r_squared": float(
            model.rsquared_adj
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input:\n{INPUT_FILE}\n\n"
            "Run Step 07 first."
        )

    hospital = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    hospital["year"] = pd.to_numeric(
        hospital["year"],
        errors="coerce",
    )

    # Step 07 panel is the 10-km primary specification already.
    hospital_2023 = hospital.loc[
        hospital["year"] == TARGET_YEAR
    ].copy()

    print("=" * 76)
    print("2023 SYSTEM HHI + SPATIAL COMPETITION")
    print("=" * 76)
    print(
        f"2023 hospital rows in spatial panel: "
        f"{len(hospital_2023):,}"
    )

    ahrq = read_ahrq()

    print(
        f"AHRQ hospital-linkage rows: "
        f"{len(ahrq):,}"
    )

    required = [
        "ccn",
        "health_sys_id",
        "health_sys_name",
        "hos_beds",
    ]

    missing = [
        c for c in required
        if c not in ahrq.columns
    ]

    if missing:
        raise KeyError(
            "AHRQ file missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns:\n"
            + "\n".join(
                str(c)
                for c in ahrq.columns
            )
        )

    hospital_2023["ccn_key"] = (
        normalize_ccn(
            hospital_2023[
                "facility_id"
            ]
        )
    )

    ahrq["ccn_key"] = normalize_ccn(
        ahrq["ccn"]
    )

    ahrq_keep = (
        ahrq[
            [
                "ccn_key",
                "health_sys_id",
                "health_sys_name",
                "hos_beds",
            ]
        ]
        .drop_duplicates(
            "ccn_key"
        )
        .copy()
    )

    linked = hospital_2023.merge(
        ahrq_keep,
        on="ccn_key",
        how="left",
        validate="many_to_one",
    )

    linkage_rate = (
        linked["ccn_key"].isin(
            set(
                ahrq_keep[
                    "ccn_key"
                ]
            )
        )
    ).mean()

    system_rate = (
        linked["health_sys_id"]
        .notna()
        .mean()
    )

    print(
        f"CCN found in AHRQ linkage: "
        f"{100 * linkage_rate:.2f}%"
    )

    print(
        f"Hospitals linked to health system: "
        f"{100 * system_rate:.2f}%"
    )

    # Standalone hospitals must remain separate firms.
    linked["firm_id"] = (
        linked["health_sys_id"]
        .astype("string")
    )

    independent_mask = (
        linked["health_sys_id"]
        .isna()
    )

    linked.loc[
        independent_mask,
        "firm_id",
    ] = (
        "INDEPENDENT_"
        + linked.loc[
            independent_mask,
            "facility_id",
        ].astype(str)
    )

    linked["firm_name"] = (
        linked["health_sys_name"]
        .astype("string")
    )

    if "rural" in linked.columns:
        linked["rural"] = pd.to_numeric(
            linked["rural"],
            errors="coerce",
        )

    linked.loc[
        independent_mask,
        "firm_name",
    ] = (
        "Independent hospital: "
        + linked.loc[
            independent_mask,
            "facility_name",
        ].astype(str)
    )

    linked.to_csv(
        HOSPITAL_LINKED_OUTPUT,
        index=False,
    )

    clusters = build_cluster_hhi(
        linked
    )

    # Only pure rural / pure metro for rural comparison.
    clusters["rural"] = np.where(
        clusters["cluster_rurality"]
        == "rural_only",
        1,
        np.where(
            clusters[
                "cluster_rurality"
            ]
            == "metro_only",
            0,
            np.nan,
        ),
    )

    clusters["log_nearest_cluster_km"] = (
        np.where(
            pd.to_numeric(
                clusters[
                    "nearest_competing_cluster_km"
                ],
                errors="coerce",
            ) > 0,
            np.log(
                pd.to_numeric(
                    clusters[
                        "nearest_competing_cluster_km"
                    ],
                    errors="coerce",
                )
            ),
            np.nan,
        )
    )

    clusters["log_isolation_ratio"] = (
        np.where(
            pd.to_numeric(
                clusters[
                    "isolation_ratio"
                ],
                errors="coerce",
            ) > 0,
            np.log(
                pd.to_numeric(
                    clusters[
                        "isolation_ratio"
                    ],
                    errors="coerce",
                )
            ),
            np.nan,
        )
    )

    clusters["log_cluster_size"] = (
        np.log(
            pd.to_numeric(
                clusters[
                    "cluster_size"
                ],
                errors="coerce",
            )
        )
    )

    clusters["hhi_count_scaled"] = (
        pd.to_numeric(
            clusters[
                "hhi_count_10000"
            ],
            errors="coerce",
        )
        / 1000.0
    )

    clusters["hhi_beds_scaled"] = (
        pd.to_numeric(
            clusters[
                "hhi_beds_10000"
            ],
            errors="coerce",
        )
        / 1000.0
    )

    clusters.to_csv(
        CLUSTER_HHI_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # DESCRIPTIVES
    # -------------------------------------------------------------

    pure = clusters.loc[
        clusters["cluster_rurality"]
        .isin(
            [
                "metro_only",
                "rural_only",
            ]
        )
    ].copy()


    print()
    print("Cluster rurality counts:")
    print(
        clusters["cluster_rurality"]
        .value_counts(dropna=False)
        .to_string()
    )

    if pure.empty:
        raise RuntimeError(
            "No pure rural/metro clusters available after RUCC classification."
        )

    descriptives = (
        pure.groupby(
            "cluster_rurality",
            dropna=False,
        )
        .agg(
            clusters=(
                "cluster_id",
                "nunique",
            ),
            median_cluster_size=(
                "cluster_size",
                "median",
            ),
            median_hhi_count=(
                "hhi_count_10000",
                "median",
            ),
            mean_hhi_count=(
                "hhi_count_10000",
                "mean",
            ),
            median_hhi_beds=(
                "hhi_beds_10000",
                "median",
            ),
            median_nearest_cluster_km=(
                "nearest_competing_cluster_km",
                "median",
            ),
            median_isolation_ratio=(
                "isolation_ratio",
                "median",
            ),
        )
        .reset_index()
    )

    descriptives.to_csv(
        DESCRIPTIVES_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # CORRELATIONS
    # -------------------------------------------------------------

    multi = pure.loc[
        pure["cluster_size"] >= 2
    ].copy()

    correlation_count = (
        multi[
            [
                "hhi_count_10000",
                "nearest_competing_cluster_km",
                "isolation_ratio",
            ]
        ]
        .corr()
    )

    # -------------------------------------------------------------
    # REGRESSIONS
    # These test whether HHI and spatial isolation are empirically
    # distinct dimensions. They do NOT yet prove outcome effects.
    # -------------------------------------------------------------

    regression_rows = []

    m1_data = pure.dropna(
        subset=[
            "log_nearest_cluster_km",
            "hhi_count_scaled",
            "rural",
            "log_cluster_size",
        ]
    )

    m1 = smf.ols(
        "log_nearest_cluster_km "
        "~ hhi_count_scaled + rural + log_cluster_size",
        data=m1_data,
    ).fit(
        cov_type="HC1"
    )

    for term in [
        "hhi_count_scaled",
        "rural",
    ]:
        regression_rows.append(
            extract_result(
                m1,
                term,
                "M1",
                "log_nearest_cluster_km",
            )
        )

    m2_data = multi.dropna(
        subset=[
            "log_isolation_ratio",
            "hhi_count_scaled",
            "rural",
            "log_cluster_size",
        ]
    )

    m2 = smf.ols(
        "log_isolation_ratio "
        "~ hhi_count_scaled + rural + log_cluster_size",
        data=m2_data,
    ).fit(
        cov_type="HC1"
    )

    for term in [
        "hhi_count_scaled",
        "rural",
    ]:
        regression_rows.append(
            extract_result(
                m2,
                term,
                "M2",
                "log_isolation_ratio",
            )
        )

    # Bed-weighted robustness where bed data exists.
    m3_data = multi.dropna(
        subset=[
            "log_isolation_ratio",
            "hhi_beds_scaled",
            "rural",
            "log_cluster_size",
        ]
    )

    m3 = smf.ols(
        "log_isolation_ratio "
        "~ hhi_beds_scaled + rural + log_cluster_size",
        data=m3_data,
    ).fit(
        cov_type="HC1"
    )

    for term in [
        "hhi_beds_scaled",
        "rural",
    ]:
        regression_rows.append(
            extract_result(
                m3,
                term,
                "M3",
                "log_isolation_ratio",
            )
        )

    regressions = pd.DataFrame(
        regression_rows
    )

    regressions.to_csv(
        REGRESSIONS_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # TEXT SUMMARY
    # -------------------------------------------------------------

    lines = []

    lines.append(
        "2023 SYSTEM HHI + SPATIAL COMPETITION"
    )
    lines.append(
        "=" * 76
    )
    lines.append(
        f"2023 hospital rows: {len(linked):,}"
    )
    lines.append(
        f"CCN linkage rate: {100 * linkage_rate:.2f}%"
    )
    lines.append(
        f"System-affiliated hospital share: {100 * system_rate:.2f}%"
    )
    lines.append("")
    lines.append(
        "Pure rural/metro descriptives:"
    )
    lines.append(
        descriptives.to_string(
            index=False
        )
    )
    lines.append("")
    lines.append(
        "Multi-hospital-cluster correlation matrix:"
    )
    lines.append(
        correlation_count.to_string()
    )
    lines.append("")
    lines.append(
        "Regression coefficients:"
    )
    lines.append(
        regressions.to_string(
            index=False
        )
    )
    lines.append("")
    lines.append(
        "Caution: these regressions test whether ownership concentration "
        "and spatial isolation are distinct dimensions. A hospital outcome "
        "or price measure is still needed to test whether spatial isolation "
        "adds explanatory power beyond HHI for antitrust-relevant outcomes."
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("STEP 09 COMPLETE")
    print("=" * 76)

    print("\nCreated:")
    for path in [
        AHRQ_FILE,
        HOSPITAL_LINKED_OUTPUT,
        CLUSTER_HHI_OUTPUT,
        DESCRIPTIVES_OUTPUT,
        REGRESSIONS_OUTPUT,
        SUMMARY_OUTPUT,
    ]:
        print(
            f"  {path.relative_to(PROJECT_DIR)}"
        )

    print()
    print("2023 rural vs metro concentration:")
    print(
        descriptives.to_string(
            index=False
        )
    )

    print()
    print(
        "Multi-hospital-cluster correlations:"
    )
    print(
        correlation_count.round(3).to_string()
    )

    print()
    print("Regression terms:")
    print(
        regressions[
            [
                "model",
                "outcome",
                "term",
                "coefficient",
                "robust_se_hc1",
                "p_value",
                "n",
            ]
        ]
        .round(
            {
                "coefficient": 4,
                "robust_se_hc1": 4,
                "p_value": 5,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
