#!/usr/bin/env python3
"""
12_robustness_cluster_radius.py

Robustness battery for the hospital spatial-competition project.

Purpose
-------
Test whether the main quality/patient-experience findings survive:
1) Alternative spatial cluster radii: 5 km, 10 km, 15 km
2) Alternative distance functional forms:
   - log(nearest competing cluster distance)
   - nearest competing cluster distance in 10-km units
3) Count-HHI and bed-HHI specifications

Outcomes
--------
A) CMS Hospital Overall Rating (1-5)
B) HCAHPS % rating hospital 9-10
C) HCAHPS % definitely recommend

Inputs
------
data/spatial/hospital_clusters_all_eps.csv
data/spatial/cluster_competition_all_eps.csv
data/spatial/hospital_cluster_rucc_panel.csv
data/spatial/hospital_spatial_panel.csv
data/spatial/cluster_hhi_2023.csv
data/raw/hcahps_2023.csv

Outputs
-------
data/results/robustness_radius_models_2023.csv
data/results/robustness_radius_summary_2023.csv
data/results/robustness_radius_matrix_2023.csv
data/results/robustness_radius_summary_2023.txt

Run
---
pip3 install pandas numpy statsmodels
python3 12_robustness_cluster_radius.py
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "data" / "results"

ASSIGNMENTS_FILE = SPATIAL_DIR / "hospital_clusters_all_eps.csv"
COMPETITION_FILE = SPATIAL_DIR / "cluster_competition_all_eps.csv"
RUCC_PANEL_FILE = SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
FULL_PANEL_FILE = SPATIAL_DIR / "hospital_spatial_panel.csv"
CLUSTER_HHI_2023_FILE = SPATIAL_DIR / "cluster_hhi_2023.csv"
HCAHPS_FILE = RAW_DIR / "hcahps_2023.csv"

COEF_OUTPUT = RESULTS_DIR / "robustness_radius_models_2023.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "robustness_radius_summary_2023.csv"
MATRIX_OUTPUT = RESULTS_DIR / "robustness_radius_matrix_2023.csv"
TEXT_OUTPUT = RESULTS_DIR / "robustness_radius_summary_2023.txt"

TARGET_YEAR = 2023
EPS_VALUES = [5, 10, 15]


def normalize_ccn(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    numeric_only = s.str.fullmatch(r"\d+")
    s.loc[numeric_only] = s.loc[numeric_only].str.zfill(6)
    return s


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "<NA>": "Unknown"})
    )


def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in lookup:
            return lookup[key]
    raise KeyError(
        "Could not find any of: "
        + ", ".join(candidates)
        + "\nAvailable:\n"
        + "\n".join(f"  {c}" for c in df.columns)
    )


def build_rucc_map() -> pd.DataFrame:
    rucc = pd.read_csv(RUCC_PANEL_FILE, low_memory=False)
    rucc["year"] = pd.to_numeric(rucc["year"], errors="coerce")

    keep = [
        c for c in [
            "facility_id",
            "year",
            "rural",
            "metro_status",
            "rucc_2023",
            "county_fips",
        ]
        if c in rucc.columns
    ]

    out = (
        rucc[keep]
        .drop_duplicates(["facility_id", "year"])
        .copy()
    )

    out["rural"] = pd.to_numeric(out["rural"], errors="coerce")
    return out


def build_outcome_panel() -> pd.DataFrame:
    full = pd.read_csv(FULL_PANEL_FILE, low_memory=False)
    full["year"] = pd.to_numeric(full["year"], errors="coerce")

    full = full.loc[
        full["year"] == TARGET_YEAR
    ].copy()

    keep = [
        c for c in [
            "facility_id",
            "year",
            "snapshot",
            "state",
            "hospital_type",
            "hospital_ownership",
            "hospital_overall_rating",
        ]
        if c in full.columns
    ]

    full = (
        full[keep]
        .drop_duplicates(["facility_id", "year"])
        .copy()
    )

    full["ccn_key"] = normalize_ccn(full["facility_id"])
    full["rating"] = pd.to_numeric(
        full["hospital_overall_rating"],
        errors="coerce",
    )

    # HCAHPS
    hc = pd.read_csv(
        HCAHPS_FILE,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )

    facility_col = find_col(hc, ["Facility ID", "Provider ID"])
    measure_col = find_col(hc, ["HCAHPS Measure ID", "Measure ID"])
    answer_col = find_col(
        hc,
        ["HCAHPS Answer Percent", "Patients Survey Answer Percent", "Score"],
    )

    hc["ccn_key"] = normalize_ccn(hc[facility_col])
    hc["answer_num"] = pd.to_numeric(hc[answer_col], errors="coerce")

    rating_map = (
        hc.loc[
            (hc[measure_col] == "H_HSP_RATING_9_10")
            & hc["answer_num"].between(0, 100),
            ["ccn_key", "answer_num"],
        ]
        .drop_duplicates("ccn_key")
        .rename(columns={"answer_num": "pct_rating_9_10"})
    )

    recommend_map = (
        hc.loc[
            (hc[measure_col] == "H_RECMND_DY")
            & hc["answer_num"].between(0, 100),
            ["ccn_key", "answer_num"],
        ]
        .drop_duplicates("ccn_key")
        .rename(columns={"answer_num": "pct_definitely_recommend"})
    )

    full = (
        full.merge(
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

    full["hospital_type_clean"] = clean_category(full["hospital_type"])
    full["state_clean"] = clean_category(full["state"])

    return full


def build_eps_panel(eps: int, outcomes: pd.DataFrame, rucc_map: pd.DataFrame) -> pd.DataFrame:
    assignments = pd.read_csv(ASSIGNMENTS_FILE, low_memory=False)
    competition = pd.read_csv(COMPETITION_FILE, low_memory=False)

    assignments["year"] = pd.to_numeric(assignments["year"], errors="coerce")
    assignments["eps_km"] = pd.to_numeric(assignments["eps_km"], errors="coerce")

    competition["year"] = pd.to_numeric(competition["year"], errors="coerce")
    competition["eps_km"] = pd.to_numeric(competition["eps_km"], errors="coerce")

    a = assignments.loc[
        (assignments["year"] == TARGET_YEAR)
        & (assignments["eps_km"] == eps)
    ].copy()

    c = competition.loc[
        (competition["year"] == TARGET_YEAR)
        & (competition["eps_km"] == eps)
    ].copy()

    # One row per facility-year at the requested eps.
    a = a.drop_duplicates(["facility_id", "year"]).copy()

    # Bring cluster-level spatial metrics to hospitals.
    c_keep = [
        c2 for c2 in [
            "cluster_id",
            "cluster_size",
            "nearest_competing_cluster_km",
            "within_cluster_mean_km",
            "isolation_ratio",
        ]
        if c2 in c.columns
    ]

    a = a.merge(
        c[c_keep].drop_duplicates("cluster_id"),
        on="cluster_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_cluster"),
    )

    # Rurality + outcomes.
    a = a.merge(
        rucc_map,
        on=["facility_id", "year"],
        how="left",
        validate="many_to_one",
    )

    a = a.merge(
        outcomes,
        on=["facility_id", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_out"),
    )

    # 2023 cluster HHI exists only for the 10-km cluster definition.
    # For radius robustness, rebuild an approximate system concentration
    # measure from the 2023 hospital-system linkage file generated in Step 09.
    system_file = SPATIAL_DIR / "hospital_2023_system_linkage.csv"

    if not system_file.exists():
        raise FileNotFoundError(
            "Missing hospital_2023_system_linkage.csv. "
            "Run Step 09 first."
        )

    sys = pd.read_csv(system_file, low_memory=False)
    sys["year"] = pd.to_numeric(sys["year"], errors="coerce")

    sys_keep = [
        c2 for c2 in [
            "facility_id",
            "year",
            "firm_id",
            "hos_beds",
        ]
        if c2 in sys.columns
    ]

    sys = (
        sys[sys_keep]
        .drop_duplicates(["facility_id", "year"])
        .copy()
    )

    a = a.merge(
        sys,
        on=["facility_id", "year"],
        how="left",
        validate="one_to_one",
    )

    # Independent hospitals each get unique firm identity if needed.
    missing_firm = a["firm_id"].isna()

    a.loc[
        missing_firm,
        "firm_id",
    ] = (
        "INDEPENDENT_"
        + a.loc[missing_firm, "facility_id"].astype(str)
    )

    a["hos_beds_num"] = pd.to_numeric(a["hos_beds"], errors="coerce")

    # Rebuild HHI for THIS eps definition.
    count_hhi = []
    bed_hhi = []

    for cluster_id, g in a.groupby("cluster_id", dropna=False):
        counts = g.groupby("firm_id").size().astype(float)
        shares = counts / counts.sum()
        hhi_c = float(10000 * np.square(shares).sum())

        valid_beds = g.loc[
            g["hos_beds_num"].notna()
            & (g["hos_beds_num"] > 0)
        ].copy()

        if len(valid_beds):
            beds = valid_beds.groupby("firm_id")["hos_beds_num"].sum()
            bed_shares = beds / beds.sum()
            hhi_b = float(10000 * np.square(bed_shares).sum())
        else:
            hhi_b = np.nan

        count_hhi.append((cluster_id, hhi_c))
        bed_hhi.append((cluster_id, hhi_b))

    hhi_df = pd.DataFrame(
        count_hhi,
        columns=["cluster_id", "hhi_count_10000"],
    )

    bed_df = pd.DataFrame(
        bed_hhi,
        columns=["cluster_id", "hhi_beds_10000"],
    )

    a = (
        a.merge(
            hhi_df,
            on="cluster_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            bed_df,
            on="cluster_id",
            how="left",
            validate="many_to_one",
        )
    )

    # Model transforms.
    a["rural"] = pd.to_numeric(a["rural"], errors="coerce")

    a["nearest_competing_cluster_km"] = pd.to_numeric(
        a["nearest_competing_cluster_km"],
        errors="coerce",
    )

    a["log_nearest_cluster_km"] = np.where(
        a["nearest_competing_cluster_km"] > 0,
        np.log(a["nearest_competing_cluster_km"]),
        np.nan,
    )

    a["distance_10km"] = (
        a["nearest_competing_cluster_km"] / 10.0
    )

    a["cluster_size_model"] = pd.to_numeric(
        a["cluster_size_cluster"]
        if "cluster_size_cluster" in a.columns
        else a["cluster_size"],
        errors="coerce",
    )

    a["log_cluster_size"] = np.where(
        a["cluster_size_model"] > 0,
        np.log(a["cluster_size_model"]),
        np.nan,
    )

    a["hhi_count_scaled"] = (
        pd.to_numeric(a["hhi_count_10000"], errors="coerce")
        / 1000.0
    )

    a["hhi_beds_scaled"] = (
        pd.to_numeric(a["hhi_beds_10000"], errors="coerce")
        / 1000.0
    )

    a["hospital_type_clean"] = clean_category(a["hospital_type"])
    a["state_clean"] = clean_category(a["state"])

    return a


def fit_model(data: pd.DataFrame, formula: str):
    return smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data["cluster_id"]
        },
    )


def model_rows_for_outcome(
    df: pd.DataFrame,
    eps: int,
    outcome_col: str,
    outcome_label: str,
) -> list[dict]:

    controls = (
        "rural + log_cluster_size "
        "+ C(hospital_type_clean) "
        "+ C(state_clean)"
    )

    specs = {
        "R1_log_HHI_plus_spatial": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
        "R2_level_HHI_plus_spatial": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ distance_10km + "
            + controls
        ),
        "R3_log_bedHHI_plus_spatial": (
            f"{outcome_col} ~ hhi_beds_scaled "
            "+ log_nearest_cluster_km + "
            + controls
        ),
    }

    rows = []

    for spec, formula in specs.items():
        needed = [
            outcome_col,
            "rural",
            "log_cluster_size",
            "cluster_id",
        ]

        if "hhi_count_scaled" in formula:
            needed.append("hhi_count_scaled")
        if "hhi_beds_scaled" in formula:
            needed.append("hhi_beds_scaled")
        if "log_nearest_cluster_km" in formula:
            needed.append("log_nearest_cluster_km")
        if "distance_10km" in formula:
            needed.append("distance_10km")

        d = df.dropna(
            subset=list(dict.fromkeys(needed))
        ).copy()

        result = fit_model(d, formula)

        spatial_term = (
            "log_nearest_cluster_km"
            if "log_nearest_cluster_km" in result.params.index
            else "distance_10km"
        )

        hhi_term = (
            "hhi_beds_scaled"
            if "hhi_beds_scaled" in result.params.index
            else "hhi_count_scaled"
        )

        for term in [spatial_term, hhi_term, "rural"]:
            if term not in result.params.index:
                continue

            ci = result.conf_int()

            rows.append(
                {
                    "eps_km": eps,
                    "outcome": outcome_label,
                    "specification": spec,
                    "term": term,
                    "coefficient": float(result.params[term]),
                    "clustered_se": float(result.bse[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_lower": float(ci.loc[term, 0]),
                    "ci_upper": float(ci.loc[term, 1]),
                    "n": int(result.nobs),
                    "r_squared": float(result.rsquared),
                    "adj_r_squared": float(result.rsquared_adj),
                    "aic": float(result.aic),
                }
            )

    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        ASSIGNMENTS_FILE,
        COMPETITION_FILE,
        RUCC_PANEL_FILE,
        FULL_PANEL_FILE,
        HCAHPS_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input:\n{path}")

    print("=" * 80)
    print("STEP 12: CLUSTER-RADIUS ROBUSTNESS")
    print("=" * 80)

    rucc_map = build_rucc_map()
    outcomes = build_outcome_panel()

    all_rows = []
    summary_rows = []

    outcome_defs = [
        ("rating", "CMS overall rating"),
        ("pct_rating_9_10", "HCAHPS % rating 9-10"),
        ("pct_definitely_recommend", "HCAHPS % definitely recommend"),
    ]

    for eps in EPS_VALUES:
        print()
        print(f"Building 2023 analysis panel for eps={eps} km...")

        df = build_eps_panel(
            eps,
            outcomes,
            rucc_map,
        )

        print(
            f"  hospital rows: {len(df):,}"
        )

        for outcome_col, outcome_label in outcome_defs:
            rows = model_rows_for_outcome(
                df,
                eps,
                outcome_col,
                outcome_label,
            )
            all_rows.extend(rows)

    results = pd.DataFrame(all_rows)

    results.to_csv(
        COEF_OUTPUT,
        index=False,
    )

    # Compact summary: spatial coefficient only.
    spatial = results.loc[
        results["term"].isin(
            [
                "log_nearest_cluster_km",
                "distance_10km",
            ]
        )
    ].copy()

    spatial["significant_5pct"] = (
        spatial["p_value"] < 0.05
    )

    spatial["same_expected_sign"] = (
        spatial["coefficient"] < 0
    )

    spatial.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # Matrix for quick reviewer-style comparison.
    matrix = (
        spatial.pivot_table(
            index=[
                "outcome",
                "specification",
            ],
            columns="eps_km",
            values=[
                "coefficient",
                "p_value",
            ],
            aggfunc="first",
        )
    )

    matrix.columns = [
        f"{metric}_eps{int(eps)}"
        for metric, eps in matrix.columns
    ]

    matrix = matrix.reset_index()

    matrix.to_csv(
        MATRIX_OUTPUT,
        index=False,
    )

    lines = [
        "STEP 12: CLUSTER-RADIUS ROBUSTNESS",
        "=" * 80,
        "",
        "Spatial coefficient summary:",
        spatial.to_string(index=False),
        "",
        "Compact matrix:",
        matrix.to_string(index=False),
        "",
        (
            "Interpretation rule: a robust spatial result should retain the "
            "expected negative sign across 5/10/15 km and preferably remain "
            "statistically distinguishable from zero under both log-distance "
            "and level-distance specifications."
        ),
    ]

    TEXT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 12 COMPLETE")
    print("=" * 80)

    print("\nSpatial robustness table:")
    print(
        spatial[
            [
                "eps_km",
                "outcome",
                "specification",
                "term",
                "coefficient",
                "clustered_se",
                "p_value",
                "n",
                "significant_5pct",
            ]
        ]
        .round(
            {
                "coefficient": 4,
                "clustered_se": 4,
                "p_value": 5,
            }
        )
        .to_string(index=False)
    )

    print()
    print("Reviewer-style pass/fail summary:")
    pass_summary = (
        spatial.groupby(
            [
                "outcome",
                "specification",
            ]
        )
        .agg(
            all_negative=("same_expected_sign", "all"),
            all_significant_5pct=("significant_5pct", "all"),
            min_n=("n", "min"),
            max_p=("p_value", "max"),
        )
        .reset_index()
    )

    print(
        pass_summary
        .round({"max_p": 5})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
