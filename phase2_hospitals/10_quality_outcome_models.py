#!/usr/bin/env python3
"""
10_quality_outcome_models.py

Test whether spatial isolation adds explanatory power beyond ownership
concentration for a hospital quality outcome: CMS Hospital Overall Rating.

Place directly inside:
    phase2_hospitals/

Expected inputs:
    data/spatial/hospital_cluster_rucc_panel.csv
    data/spatial/cluster_hhi_2023.csv

Creates:
    data/results/quality_models_2023.csv
    data/results/quality_model_comparison_2023.csv
    data/results/quality_descriptives_2023.csv
    data/results/quality_model_summary_2023.txt

Main outcome:
    hospital_overall_rating (1-5), 2023

Models:
    Q1 HHI-only:
        rating ~ HHI + rural + cluster size + hospital controls

    Q2 Spatial-only:
        rating ~ log(nearest competing cluster distance)
               + rural + cluster size + hospital controls

    Q3 Combined:
        rating ~ HHI + log(nearest competing cluster distance)
               + rural + cluster size + hospital controls

    Q4 Combined + interaction:
        rating ~ HHI + log(distance) + rural
               + rural*log(distance) + cluster size + controls

Robustness:
    bed-weighted HHI version of Q3.

Standard errors:
    clustered by spatial cluster_id.

Important:
    This is observational cross-sectional evidence, not causal identification.

Run:
    pip3 install pandas numpy statsmodels
    python3 10_quality_outcome_models.py
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
RESULTS_DIR = PROJECT_DIR / "data" / "results"

HOSPITAL_PANEL = (
    SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
)

CLUSTER_HHI_FILE = (
    SPATIAL_DIR / "cluster_hhi_2023.csv"
)

COEF_OUTPUT = (
    RESULTS_DIR / "quality_models_2023.csv"
)

COMPARE_OUTPUT = (
    RESULTS_DIR / "quality_model_comparison_2023.csv"
)

DESC_OUTPUT = (
    RESULTS_DIR / "quality_descriptives_2023.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR / "quality_model_summary_2023.txt"
)

TARGET_YEAR = 2023


def parse_rating(series: pd.Series) -> pd.Series:
    """
    Convert CMS rating values to numeric 1-5 and discard nonnumeric labels.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def clean_category(series: pd.Series) -> pd.Series:
    s = (
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
    return s


def extract_terms(
    model,
    model_name: str,
    outcome: str,
    terms: list[str],
) -> list[dict]:
    rows = []
    ci = model.conf_int()

    for term in terms:
        if term not in model.params.index:
            continue

        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
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
                "bic": float(
                    model.bic
                ),
            }
        )

    return rows


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
    ].copy()

    print("=" * 76)
    print("2023 QUALITY OUTCOME MODELS")
    print("=" * 76)
    print(
        f"2023 hospital rows: {len(hospital):,}"
    )

    # Keep one row per 2023 hospital.
    hospital = hospital.drop_duplicates(
        [
            "facility_id",
            "year",
        ]
    ).copy()

    # Merge cluster-level concentration measures.
    hhi_keep = [
        c for c in [
            "cluster_id",
            "hhi_count_10000",
            "hhi_beds_10000",
            "hhi_parent_count_10000",
            "hhi_parent_beds_10000",
            "top_system_share_count",
            "top_system_share_beds",
            "cluster_size",
        ]
        if c in cluster_hhi.columns
    ]

    df = hospital.merge(
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

    if "hospital_overall_rating" not in df.columns:
        raise KeyError(
            "hospital_overall_rating not found in hospital panel."
        )

    df["rating"] = parse_rating(
        df["hospital_overall_rating"]
    )

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
        df["nearest_competing_cluster_km"] > 0,
        np.log(
            df["nearest_competing_cluster_km"]
        ),
        np.nan,
    )

    cluster_size_col = (
        "cluster_size_hhi"
        if "cluster_size_hhi" in df.columns
        else "cluster_size"
    )

    df["cluster_size_model"] = pd.to_numeric(
        df[cluster_size_col],
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

    # Hospital controls
    df["hospital_type_clean"] = clean_category(
        df["hospital_type"]
    )

    df["ownership_clean"] = clean_category(
        df["hospital_ownership"]
    )

    # Limit very tiny categories from exploding the design matrix.
    for col in [
        "hospital_type_clean",
        "ownership_clean",
    ]:
        counts = df[col].value_counts()

        rare = set(
            counts.loc[
                counts < 20
            ].index
        )

        df.loc[
            df[col].isin(rare),
            col,
        ] = "Other"

    model_vars = [
        "rating",
        "hhi_count_scaled",
        "hhi_beds_scaled",
        "log_nearest_cluster_km",
        "rural",
        "log_cluster_size",
        "cluster_id",
        "hospital_type_clean",
        "ownership_clean",
    ]

    sample = df.dropna(
        subset=[
            "rating",
            "rural",
            "log_cluster_size",
            "cluster_id",
        ]
    ).copy()

    print(
        f"Hospitals with usable CMS rating: "
        f"{sample['rating'].notna().sum():,}"
    )

    print()
    print("Rating distribution:")
    print(
        sample["rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -------------------------------------------------------------
    # DESCRIPTIVES
    # -------------------------------------------------------------

    descriptives = (
        sample.groupby(
            "rural",
            dropna=False,
        )
        .agg(
            hospitals=(
                "facility_id",
                "size",
            ),
            mean_rating=(
                "rating",
                "mean",
            ),
            median_rating=(
                "rating",
                "median",
            ),
            median_hhi_count=(
                "hhi_count_10000",
                "median",
            ),
            median_nearest_cluster_km=(
                "nearest_competing_cluster_km",
                "median",
            ),
        )
        .reset_index()
    )

    descriptives[
        "rural_label"
    ] = np.where(
        descriptives["rural"] == 1,
        "rural",
        "metro",
    )

    descriptives.to_csv(
        DESC_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------
    # MODEL FORMULAS
    # -------------------------------------------------------------

    controls = (
        "rural + log_cluster_size "
        "+ C(hospital_type_clean) "
        "+ C(ownership_clean)"
    )

    formulas = {
        "Q1_HHI_only": (
            "rating ~ hhi_count_scaled + "
            + controls
        ),
        "Q2_Spatial_only": (
            "rating ~ log_nearest_cluster_km + "
            + controls
        ),
        "Q3_HHI_plus_spatial": (
            "rating ~ hhi_count_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
        "Q4_With_interaction": (
            "rating ~ hhi_count_scaled "
            "+ log_nearest_cluster_km "
            "+ rural "
            "+ log_nearest_cluster_km:rural "
            "+ log_cluster_size "
            "+ C(hospital_type_clean) "
            "+ C(ownership_clean)"
        ),
        "Q5_Bed_HHI_plus_spatial": (
            "rating ~ hhi_beds_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
    }

    coefficient_rows = []
    comparison_rows = []
    fitted = {}

    for name, formula in formulas.items():

        needed = [
            "rating",
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

        d = sample.dropna(
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

        fitted[name] = result

        terms = [
            "hhi_count_scaled",
            "hhi_beds_scaled",
            "log_nearest_cluster_km",
            "rural",
            "log_nearest_cluster_km:rural",
            "log_cluster_size",
        ]

        coefficient_rows.extend(
            extract_terms(
                result,
                name,
                "hospital_overall_rating",
                terms,
            )
        )

        comparison_rows.append(
            {
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
                "bic": float(
                    result.bic
                ),
            }
        )

    coefs = pd.DataFrame(
        coefficient_rows
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

    # -------------------------------------------------------------
    # TEXT SUMMARY
    # -------------------------------------------------------------

    lines = []

    lines.append(
        "2023 CMS HOSPITAL OVERALL RATING MODELS"
    )
    lines.append(
        "=" * 76
    )
    lines.append("")
    lines.append(
        "Descriptives:"
    )
    lines.append(
        descriptives.to_string(
            index=False
        )
    )
    lines.append("")
    lines.append(
        "Model comparison:"
    )
    lines.append(
        compare.to_string(
            index=False
        )
    )
    lines.append("")
    lines.append(
        "Key coefficients:"
    )
    lines.append(
        coefs.to_string(
            index=False
        )
    )
    lines.append("")
    lines.append(
        "Interpretation caution: these are observational 2023 "
        "cross-sectional associations. They do not identify causal "
        "effects of concentration or spatial isolation on quality."
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # TERMINAL
    # -------------------------------------------------------------

    print()
    print("=" * 76)
    print("STEP 10 COMPLETE")
    print("=" * 76)

    print("\nCreated:")
    for path in [
        COEF_OUTPUT,
        COMPARE_OUTPUT,
        DESC_OUTPUT,
        SUMMARY_OUTPUT,
    ]:
        print(
            f"  {path.relative_to(PROJECT_DIR)}"
        )

    print()
    print("Quality descriptives:")
    print(
        descriptives[
            [
                "rural_label",
                "hospitals",
                "mean_rating",
                "median_rating",
                "median_hhi_count",
                "median_nearest_cluster_km",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print("Model comparison:")
    print(
        compare[
            [
                "model",
                "n",
                "r_squared",
                "adj_r_squared",
                "aic",
            ]
        ]
        .round(
            {
                "r_squared": 4,
                "adj_r_squared": 4,
                "aic": 1,
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

    print()
    print(
        "Compare Q1, Q2, and Q3 first. "
        "If spatial isolation remains associated with rating in Q3 "
        "and model fit improves over Q1, it adds information beyond HHI."
    )


if __name__ == "__main__":
    main()
