#!/usr/bin/env python3
"""
11_hcahps_outcome_models.py

Download/extract the 2023 CMS HCAHPS hospital file and test whether
spatial isolation adds information beyond HHI for patient experience.

Place directly inside:
    phase2_hospitals/

Expected inputs:
    data/spatial/hospital_cluster_rucc_panel.csv
    data/spatial/cluster_hhi_2023.csv

Downloads:
    CMS Hospital archive snapshot 2023-11-06

Creates:
    data/raw/hcahps_2023.csv
    data/results/hcahps_models_2023.csv
    data/results/hcahps_model_comparison_2023.csv
    data/results/hcahps_descriptives_2023.csv
    data/results/hcahps_model_summary_2023.txt

Primary outcomes:
    1) Percent of patients giving hospital rating 9 or 10
    2) Percent of patients who would definitely recommend the hospital

Model families for each outcome:
    H1 HHI-only
    H2 Spatial-only
    H3 HHI + spatial
    H4 HHI + spatial + rural interaction
    H5 Bed-HHI + spatial

Standard errors:
    clustered by 10-km spatial cluster_id

Run:
    pip3 install pandas numpy requests statsmodels
    python3 11_hcahps_outcome_models.py
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import zipfile
import re

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
RESULTS_DIR = PROJECT_DIR / "data" / "results"

HOSPITAL_PANEL = (
    SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
)

CLUSTER_HHI_FILE = (
    SPATIAL_DIR / "cluster_hhi_2023.csv"
)

HCAHPS_FILE = (
    RAW_DIR / "hcahps_2023.csv"
)

COEF_OUTPUT = (
    RESULTS_DIR / "hcahps_models_2023.csv"
)

COMPARE_OUTPUT = (
    RESULTS_DIR / "hcahps_model_comparison_2023.csv"
)

DESC_OUTPUT = (
    RESULTS_DIR / "hcahps_descriptives_2023.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR / "hcahps_model_summary_2023.txt"
)

TARGET_YEAR = 2023
SNAPSHOT = "2023-11-06"

CMS_ARCHIVE_URL = (
    "https://data.cms.gov/provider-data/sites/default/files/"
    f"dataset-archives/theme/hospitals/hospitals_{SNAPSHOT}.zip"
)


def normalize_name(x: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).lower(),
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {
        normalize_name(c): c
        for c in df.columns
    }

    for cand in candidates:
        key = normalize_name(cand)
        if key in lookup:
            return lookup[key]

    raise KeyError(
        "Could not find any of these columns: "
        + ", ".join(candidates)
        + "\nAvailable:\n"
        + "\n".join(
            f"  {c}" for c in df.columns
        )
    )


def normalize_ccn(series: pd.Series) -> pd.Series:
    """
    Normalize CMS Certification Numbers while preserving letters.

    Some provider IDs contain alphabetic suffixes (for example VA
    facilities). Stripping those letters can create false duplicate keys.
    """
    s = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True,
        )
    )

    numeric_only = s.str.fullmatch(r"\d+")

    s.loc[numeric_only] = (
        s.loc[numeric_only]
        .str.zfill(6)
    )

    return s


def download_hcahps() -> None:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading CMS hospitals archive {SNAPSHOT}..."
    )

    response = requests.get(
        CMS_ARCHIVE_URL,
        headers={
            "User-Agent": (
                "hospital-spatial-competition-research/1.0"
            )
        },
        timeout=180,
    )
    response.raise_for_status()

    zf = zipfile.ZipFile(
        BytesIO(response.content)
    )

    csv_names = [
        n for n in zf.namelist()
        if n.lower().endswith(".csv")
    ]

    candidates = []

    for name in csv_names:
        try:
            with zf.open(name) as f:
                sample = pd.read_csv(
                    f,
                    dtype=str,
                    nrows=5,
                    low_memory=False,
                    encoding="utf-8-sig",
                )

            cols_norm = {
                normalize_name(c)
                for c in sample.columns
            }

            has_facility = any(
                x in cols_norm
                for x in [
                    "facilityid",
                    "providerid",
                ]
            )

            has_measure = any(
                x in cols_norm
                for x in [
                    "hcahpsmeasureid",
                    "measureid",
                ]
            )

            has_answer = any(
                x in cols_norm
                for x in [
                    "hcahpsanswerpercent",
                    "patientssurveyanswerpercent",
                    "score",
                ]
            )

            if has_facility and has_measure and has_answer:
                candidates.append(
                    (
                        name,
                        list(sample.columns),
                    )
                )

        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            "Could not identify an HCAHPS CSV in the CMS archive."
        )

    # Prefer filenames that explicitly say HCAHPS.
    candidates.sort(
        key=lambda x: (
            "hcahps" not in x[0].lower(),
            len(x[0]),
        )
    )

    chosen = candidates[0][0]

    print(
        f"Extracting HCAHPS file: {chosen}"
    )

    raw = zf.read(chosen)

    HCAHPS_FILE.write_bytes(raw)

    print(
        f"Saved: {HCAHPS_FILE.relative_to(PROJECT_DIR)}"
    )


def read_hcahps() -> pd.DataFrame:
    if not HCAHPS_FILE.exists():
        download_hcahps()

    last_error = None

    for encoding in [
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]:
        try:
            df = pd.read_csv(
                HCAHPS_FILE,
                dtype=str,
                low_memory=False,
                encoding=encoding,
            )
            print(
                f"HCAHPS decoded with encoding={encoding}"
            )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace(
            {
                "": "Unknown",
                "nan": "Unknown",
                "<NA>": "Unknown",
            }
        )
    )


def select_measure(
    hcahps: pd.DataFrame,
    measure_col: str,
    label_col: str | None,
    answer_col: str,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    preferred_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Find a patient-experience percentage measure using measure IDs/labels.

    Supports exclusions so categories such as "definitely no" are not
    accidentally selected when we want "definitely yes".
    """
    measure_text = (
        hcahps[measure_col]
        .fillna("")
        .astype(str)
    )

    if label_col is not None:
        combined = (
            measure_text
            + " "
            + hcahps[label_col]
            .fillna("")
            .astype(str)
        )
    else:
        combined = measure_text

    mask = pd.Series(
        False,
        index=hcahps.index,
    )

    for pat in include_patterns:
        mask |= combined.str.contains(
            pat,
            case=False,
            regex=True,
            na=False,
        )

    if exclude_patterns:
        for pat in exclude_patterns:
            mask &= ~combined.str.contains(
                pat,
                case=False,
                regex=True,
                na=False,
            )

    subset = hcahps.loc[mask].copy()

    if subset.empty:
        raise RuntimeError(
            "No HCAHPS measure matched requested patterns."
        )

    subset["outcome_value"] = pd.to_numeric(
        subset[answer_col],
        errors="coerce",
    )

    subset = subset.loc[
        subset["outcome_value"].between(
            0,
            100,
        )
    ].copy()

    if subset.empty:
        raise RuntimeError(
            "Matched measure rows but no numeric 0-100 percentages."
        )

    available_ids = set(
        subset[measure_col]
        .dropna()
        .astype(str)
    )

    chosen_measure = None

    if preferred_ids:
        for preferred in preferred_ids:
            if preferred in available_ids:
                chosen_measure = preferred
                break

    if chosen_measure is None:
        counts = (
            subset.groupby(
                measure_col
            )["outcome_value"]
            .count()
            .sort_values(
                ascending=False
            )
        )

        chosen_measure = counts.index[0]

    selected = subset.loc[
        subset[measure_col]
        .astype(str)
        == str(chosen_measure)
    ].copy()

    return selected, str(chosen_measure)


def fit_clustered(
    formula: str,
    data: pd.DataFrame,
):
    model = smf.ols(
        formula=formula,
        data=data,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data[
                "cluster_id"
            ]
        },
    )

    return result


def extract_terms(
    model,
    model_name: str,
    outcome_name: str,
    terms: list[str],
) -> list[dict]:
    rows = []
    ci = model.conf_int()

    for term in terms:
        if term not in model.params.index:
            continue

        rows.append(
            {
                "outcome": outcome_name,
                "model": model_name,
                "term": term,
                "coefficient": float(
                    model.params[term]
                ),
                "clustered_se": float(
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
                "aic": float(
                    model.aic
                ),
            }
        )

    return rows


def run_model_family(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_label: str,
) -> tuple[list[dict], list[dict]]:
    controls = (
        "rural + log_cluster_size "
        "+ C(hospital_type_clean) "
        "+ C(state_clean)"
    )

    formulas = {
        "H1_HHI_only": (
            f"{outcome_col} ~ hhi_count_scaled + "
            + controls
        ),
        "H2_Spatial_only": (
            f"{outcome_col} ~ log_nearest_cluster_km + "
            + controls
        ),
        "H3_HHI_plus_spatial": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
        "H4_With_interaction": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ log_nearest_cluster_km "
            "+ rural "
            "+ log_nearest_cluster_km:rural "
            "+ log_cluster_size "
            "+ C(hospital_type_clean) "
            "+ C(state_clean)"
        ),
        "H5_Bed_HHI_plus_spatial": (
            f"{outcome_col} ~ hhi_beds_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
    }

    coef_rows = []
    comp_rows = []

    for name, formula in formulas.items():
        needed = [
            outcome_col,
            "rural",
            "log_cluster_size",
            "cluster_id",
        ]

        if "hhi_count_scaled" in formula:
            needed.append(
                "hhi_count_scaled"
            )

        if "hhi_beds_scaled" in formula:
            needed.append(
                "hhi_beds_scaled"
            )

        if "log_nearest_cluster_km" in formula:
            needed.append(
                "log_nearest_cluster_km"
            )

        d = df.dropna(
            subset=list(
                dict.fromkeys(
                    needed
                )
            )
        ).copy()

        result = fit_clustered(
            formula,
            d,
        )

        coef_rows.extend(
            extract_terms(
                result,
                name,
                outcome_label,
                [
                    "hhi_count_scaled",
                    "hhi_beds_scaled",
                    "log_nearest_cluster_km",
                    "rural",
                    "log_nearest_cluster_km:rural",
                ],
            )
        )

        spatial_beta = (
            float(
                result.params[
                    "log_nearest_cluster_km"
                ]
            )
            if "log_nearest_cluster_km"
            in result.params.index
            else np.nan
        )

        comp_rows.append(
            {
                "outcome": outcome_label,
                "model": name,
                "n": int(
                    result.nobs
                ),
                "r_squared": float(
                    result.rsquared
                ),
                "adj_r_squared": float(
                    result.rsquared_adj
                ),
                "aic": float(
                    result.aic
                ),
                "outcome_change_if_distance_doubles": (
                    spatial_beta * np.log(2)
                    if pd.notna(spatial_beta)
                    else np.nan
                ),
            }
        )

    return coef_rows, comp_rows


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        HOSPITAL_PANEL,
        CLUSTER_HHI_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    hospital = pd.read_csv(
        HOSPITAL_PANEL,
        low_memory=False,
    )

    cluster_hhi = pd.read_csv(
        CLUSTER_HHI_FILE,
        low_memory=False,
    )

    hospital["year"] = pd.to_numeric(
        hospital["year"],
        errors="coerce",
    )

    hospital = hospital.loc[
        hospital["year"] == TARGET_YEAR
    ].drop_duplicates(
        [
            "facility_id",
            "year",
        ]
    ).copy()

    print("=" * 76)
    print("2023 HCAHPS OUTCOME MODELS")
    print("=" * 76)
    print(
        f"2023 spatial hospital rows: "
        f"{len(hospital):,}"
    )

    hcahps = read_hcahps()

    print(
        f"HCAHPS rows: {len(hcahps):,}"
    )

    facility_col = find_col(
        hcahps,
        [
            "Facility ID",
            "Provider ID",
        ],
    )

    measure_col = find_col(
        hcahps,
        [
            "HCAHPS Measure ID",
            "Measure ID",
        ],
    )

    answer_col = find_col(
        hcahps,
        [
            "HCAHPS Answer Percent",
            "Patients Survey Answer Percent",
            "Score",
        ],
    )

    try:
        label_col = find_col(
            hcahps,
            [
                "HCAHPS Question",
                "Measure Name",
            ],
        )
    except KeyError:
        label_col = None

    # Rating 9-10
    rating_rows, rating_measure = select_measure(
        hcahps,
        measure_col,
        label_col,
        answer_col,
        include_patterns=[
            r"RATING.*9.*10",
            r"9.*10.*RATING",
            r"HSP_RATING_9_10",
        ],
        preferred_ids=[
            "H_HSP_RATING_9_10",
        ],
    )

    # Definitely recommend
    recommend_rows, recommend_measure = select_measure(
        hcahps,
        measure_col,
        label_col,
        answer_col,
        include_patterns=[
            r"RECMND",
            r"RECOMMEND",
        ],
        exclude_patterns=[
            r"RECMND_DN",
            r"DEFINITELY NO",
            r"PROBABLY",
        ],
        preferred_ids=[
            "H_RECMND_DY",
        ],
    )

    print(
        f"Selected rating measure: {rating_measure}"
    )
    print(
        f"Selected recommend measure: {recommend_measure}"
    )

    rating_rows["ccn_key"] = normalize_ccn(
        rating_rows[
            facility_col
        ]
    )

    recommend_rows["ccn_key"] = normalize_ccn(
        recommend_rows[
            facility_col
        ]
    )

    rating_map = (
        rating_rows[
            [
                "ccn_key",
                "outcome_value",
            ]
        ]
        .drop_duplicates(
            "ccn_key"
        )
        .rename(
            columns={
                "outcome_value": (
                    "pct_rating_9_10"
                )
            }
        )
    )

    recommend_map = (
        recommend_rows[
            [
                "ccn_key",
                "outcome_value",
            ]
        ]
        .drop_duplicates(
            "ccn_key"
        )
        .rename(
            columns={
                "outcome_value": (
                    "pct_definitely_recommend"
                )
            }
        )
    )

    hospital["ccn_key"] = normalize_ccn(
        hospital["facility_id"]
    )

    hosp_dupes = hospital.duplicated(
        "ccn_key",
        keep=False,
    )

    rating_dupes = rating_map.duplicated(
        "ccn_key",
        keep=False,
    )

    recommend_dupes = recommend_map.duplicated(
        "ccn_key",
        keep=False,
    )

    print(
        f"Duplicate hospital CCN keys after normalization: "
        f"{int(hosp_dupes.sum()):,}"
    )
    print(
        f"Duplicate rating-map CCN keys: "
        f"{int(rating_dupes.sum()):,}"
    )
    print(
        f"Duplicate recommend-map CCN keys: "
        f"{int(recommend_dupes.sum()):,}"
    )

    if hosp_dupes.any():
        print(
            "WARNING: repeated provider IDs remain in the spatial hospital "
            "panel; keeping one row per CCN for HCAHPS analysis because "
            "HCAHPS outcomes are provider-level."
        )
        hospital = (
            hospital.sort_values(
                ["ccn_key", "facility_id"]
            )
            .drop_duplicates(
                "ccn_key",
                keep="first",
            )
            .copy()
        )

    df = (
        hospital.merge(
            rating_map,
            on="ccn_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            recommend_map,
            on="ccn_key",
            how="left",
            validate="one_to_one",
        )
    )

    hhi_keep = [
        c for c in [
            "cluster_id",
            "hhi_count_10000",
            "hhi_beds_10000",
            "cluster_size",
        ]
        if c in cluster_hhi.columns
    ]

    df = df.merge(
        cluster_hhi[
            hhi_keep
        ].drop_duplicates(
            "cluster_id"
        ),
        on="cluster_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_hhi"),
    )

    # Numeric model variables
    df["rural"] = pd.to_numeric(
        df["rural"],
        errors="coerce",
    )

    df["nearest_competing_cluster_km"] = (
        pd.to_numeric(
            df[
                "nearest_competing_cluster_km"
            ],
            errors="coerce",
        )
    )

    df["log_nearest_cluster_km"] = np.where(
        df[
            "nearest_competing_cluster_km"
        ] > 0,
        np.log(
            df[
                "nearest_competing_cluster_km"
            ]
        ),
        np.nan,
    )

    size_col = (
        "cluster_size_hhi"
        if "cluster_size_hhi" in df.columns
        else "cluster_size"
    )

    df["cluster_size_model"] = pd.to_numeric(
        df[size_col],
        errors="coerce",
    )

    df["log_cluster_size"] = np.where(
        df["cluster_size_model"] > 0,
        np.log(
            df["cluster_size_model"]
        ),
        np.nan,
    )

    df["hhi_count_scaled"] = (
        pd.to_numeric(
            df["hhi_count_10000"],
            errors="coerce",
        )
        / 1000.0
    )

    df["hhi_beds_scaled"] = (
        pd.to_numeric(
            df["hhi_beds_10000"],
            errors="coerce",
        )
        / 1000.0
    )

    df["hospital_type_clean"] = clean_category(
        df["hospital_type"]
    )

    df["state_clean"] = clean_category(
        df["state"]
    )

    # Descriptives
    desc_rows = []

    for outcome in [
        "pct_rating_9_10",
        "pct_definitely_recommend",
    ]:
        for rural_val, g in df.loc[
            df[outcome].notna()
        ].groupby("rural"):

            desc_rows.append(
                {
                    "outcome": outcome,
                    "rural": rural_val,
                    "rural_label": (
                        "rural"
                        if rural_val == 1
                        else "metro"
                    ),
                    "hospitals": len(g),
                    "mean": g[outcome].mean(),
                    "median": g[outcome].median(),
                    "median_nearest_cluster_km": (
                        g[
                            "nearest_competing_cluster_km"
                        ].median()
                    ),
                }
            )

    descriptives = pd.DataFrame(
        desc_rows
    )

    descriptives.to_csv(
        DESC_OUTPUT,
        index=False,
    )

    coef_rows = []
    comparison_rows = []

    for outcome_col, outcome_label in [
        (
            "pct_rating_9_10",
            "HCAHPS % rating 9-10",
        ),
        (
            "pct_definitely_recommend",
            "HCAHPS % definitely recommend",
        ),
    ]:
        c, m = run_model_family(
            df,
            outcome_col,
            outcome_label,
        )

        coef_rows.extend(c)
        comparison_rows.extend(m)

    coefs = pd.DataFrame(
        coef_rows
    )

    compare = pd.DataFrame(
        comparison_rows
    )

    coefs.to_csv(
        COEF_OUTPUT,
        index=False,
    )

    compare.to_csv(
        COMPARE_OUTPUT,
        index=False,
    )

    lines = [
        "2023 HCAHPS PATIENT EXPERIENCE MODELS",
        "=" * 76,
        "",
        f"Selected rating measure: {rating_measure}",
        f"Selected recommend measure: {recommend_measure}",
        "",
        "Descriptives:",
        descriptives.to_string(index=False),
        "",
        "Model comparison:",
        compare.to_string(index=False),
        "",
        "Key coefficients:",
        coefs.to_string(index=False),
        "",
        (
            "Caution: these are observational cross-sectional associations, "
            "not causal effects."
        ),
    ]

    SUMMARY_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("STEP 11 COMPLETE")
    print("=" * 76)

    print("\nPatient-experience descriptives:")
    print(
        descriptives.round(3).to_string(
            index=False
        )
    )

    print()
    print("Model comparison:")
    print(
        compare[
            [
                "outcome",
                "model",
                "n",
                "r_squared",
                "adj_r_squared",
                "aic",
                "outcome_change_if_distance_doubles",
            ]
        ]
        .round(
            {
                "r_squared": 4,
                "adj_r_squared": 4,
                "aic": 1,
                "outcome_change_if_distance_doubles": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Key coefficients:")
    print(
        coefs.loc[
            coefs["term"].isin(
                [
                    "hhi_count_scaled",
                    "hhi_beds_scaled",
                    "log_nearest_cluster_km",
                    "rural",
                    "log_nearest_cluster_km:rural",
                ]
            ),
            [
                "outcome",
                "model",
                "term",
                "coefficient",
                "clustered_se",
                "p_value",
                "n",
            ],
        ]
        .round(
            {
                "coefficient": 4,
                "clustered_se": 4,
                "p_value": 5,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
