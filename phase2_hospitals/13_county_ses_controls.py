#!/usr/bin/env python3
"""
13_county_ses_controls.py

Reviewer-style omitted-variable robustness test.

Adds 2023 county socioeconomic controls to the baseline 10-km hospital
competition models and asks whether the spatial-isolation coefficient survives.

County controls
---------------
- log population density
- log median household income
- poverty rate
- median age
- unemployment rate

Official sources
----------------
1) U.S. Census Bureau, 2023 ACS 5-year county estimates
2) U.S. Census Bureau, 2023 National Counties Gazetteer (land area)

Outcomes
--------
A) CMS Hospital Overall Rating
B) HCAHPS % rating hospital 9-10
C) HCAHPS % definitely recommend

Main comparison
---------------
S1 baseline:
    outcome ~ HHI + spatial isolation + rural + cluster size
              + hospital type FE + state FE

S2 + county SES:
    baseline + density + income + poverty + age + unemployment

S3 bed-HHI + county SES:
    same as S2, replacing count-HHI with bed-HHI

Standard errors clustered by 10-km service cluster.

Run:
    pip3 install pandas numpy requests statsmodels
    python3 13_county_ses_controls.py
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"
RESULTS_DIR = PROJECT_DIR / "data" / "results"

RUCC_PANEL = SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
FULL_PANEL = SPATIAL_DIR / "hospital_spatial_panel.csv"
CLUSTER_HHI = SPATIAL_DIR / "cluster_hhi_2023.csv"
HCAHPS_FILE = RAW_DIR / "hcahps_2023.csv"

ACS_CACHE = RAW_DIR / "acs_county_controls_2023.csv"
GAZ_CACHE = RAW_DIR / "census_county_gazetteer_2023.csv"

COEF_OUTPUT = RESULTS_DIR / "ses_control_models_2023.csv"
COMPARE_OUTPUT = RESULTS_DIR / "ses_control_model_comparison_2023.csv"
DESC_OUTPUT = RESULTS_DIR / "ses_control_descriptives_2023.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "ses_control_summary_2023.txt"

TARGET_YEAR = 2023

ACS_URL = "https://api.census.gov/data/2023/acs/acs5"
GAZ_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)

ACS_VARIABLES = {
    "population": "B01003_001E",
    "median_household_income": "B19013_001E",
    "poverty_universe": "B17001_001E",
    "poverty_count": "B17001_002E",
    "median_age": "B01002_001E",
    "labor_force": "B23025_003E",
    "unemployed": "B23025_005E",
}


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


def normalize_fips(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    return s.str.zfill(5)


def download_acs() -> pd.DataFrame:
    if ACS_CACHE.exists():
        print(f"Using cached ACS controls: {ACS_CACHE.relative_to(PROJECT_DIR)}")
        return pd.read_csv(ACS_CACHE, dtype={"county_fips": str})

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    get_vars = ["NAME"] + list(ACS_VARIABLES.values())

    params = {
        "get": ",".join(get_vars),
        "for": "county:*",
        "in": "state:*",
    }

    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key

    print("Downloading 2023 ACS 5-year county controls...")

    response = requests.get(
        ACS_URL,
        params=params,
        headers={
            "User-Agent": "hospital-spatial-competition-research/1.0"
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Census ACS request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:1000]}\n\n"
            "If Census now requires an API key on your connection, create "
            "one and run:\n"
            "export CENSUS_API_KEY='YOUR_KEY'\n"
            "python3 13_county_ses_controls.py"
        )

    data = response.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)

    df["county_fips"] = (
        df["state"].astype(str).str.zfill(2)
        + df["county"].astype(str).str.zfill(3)
    )

    rename = {
        v: k
        for k, v in ACS_VARIABLES.items()
    }
    df = df.rename(columns=rename)

    numeric_cols = list(ACS_VARIABLES.keys())

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )
        # ACS sentinel values are large negatives.
        df.loc[df[col] < 0, col] = np.nan

    df["poverty_rate"] = np.where(
        df["poverty_universe"] > 0,
        100.0 * df["poverty_count"] / df["poverty_universe"],
        np.nan,
    )

    df["unemployment_rate"] = np.where(
        df["labor_force"] > 0,
        100.0 * df["unemployed"] / df["labor_force"],
        np.nan,
    )

    keep = [
        "county_fips",
        "NAME",
        "population",
        "median_household_income",
        "poverty_rate",
        "median_age",
        "unemployment_rate",
    ]

    out = df[keep].copy()
    out.to_csv(ACS_CACHE, index=False)

    print(
        f"Saved ACS controls: {ACS_CACHE.relative_to(PROJECT_DIR)} "
        f"({len(out):,} counties)"
    )

    return out


def download_gazetteer() -> pd.DataFrame:
    if GAZ_CACHE.exists():
        print(
            f"Using cached Gazetteer: "
            f"{GAZ_CACHE.relative_to(PROJECT_DIR)}"
        )
        return pd.read_csv(GAZ_CACHE, dtype={"county_fips": str})

    print("Downloading 2023 Census county Gazetteer...")

    response = requests.get(
        GAZ_URL,
        headers={
            "User-Agent": "hospital-spatial-competition-research/1.0"
        },
        timeout=120,
    )
    response.raise_for_status()

    zf = zipfile.ZipFile(BytesIO(response.content))
    txt_names = [
        n for n in zf.namelist()
        if n.lower().endswith(".txt")
    ]

    if not txt_names:
        raise RuntimeError("No text file found inside Gazetteer zip.")

    with zf.open(txt_names[0]) as f:
        gaz = pd.read_csv(
            f,
            sep="\t",
            dtype=str,
            low_memory=False,
        )

    gaz.columns = [c.strip() for c in gaz.columns]

    geoid_col = find_col(gaz, ["GEOID"])
    aland_col = find_col(gaz, ["ALAND_SQMI"])

    out = pd.DataFrame(
        {
            "county_fips": normalize_fips(gaz[geoid_col]),
            "land_sq_mi": pd.to_numeric(
                gaz[aland_col],
                errors="coerce",
            ),
        }
    )

    out.to_csv(GAZ_CACHE, index=False)

    print(
        f"Saved Gazetteer: {GAZ_CACHE.relative_to(PROJECT_DIR)} "
        f"({len(out):,} counties)"
    )

    return out


def build_county_controls() -> pd.DataFrame:
    acs = download_acs()
    gaz = download_gazetteer()

    controls = acs.merge(
        gaz,
        on="county_fips",
        how="left",
        validate="one_to_one",
    )

    controls["population_density"] = np.where(
        controls["land_sq_mi"] > 0,
        controls["population"] / controls["land_sq_mi"],
        np.nan,
    )

    controls["log_population_density"] = np.where(
        controls["population_density"] > 0,
        np.log(controls["population_density"]),
        np.nan,
    )

    controls["log_median_household_income"] = np.where(
        controls["median_household_income"] > 0,
        np.log(controls["median_household_income"]),
        np.nan,
    )

    return controls


def build_hcahps_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    rating = (
        hc.loc[
            (hc[measure_col] == "H_HSP_RATING_9_10")
            & hc["answer_num"].between(0, 100),
            ["ccn_key", "answer_num"],
        ]
        .drop_duplicates("ccn_key")
        .rename(columns={"answer_num": "pct_rating_9_10"})
    )

    recommend = (
        hc.loc[
            (hc[measure_col] == "H_RECMND_DY")
            & hc["answer_num"].between(0, 100),
            ["ccn_key", "answer_num"],
        ]
        .drop_duplicates("ccn_key")
        .rename(columns={"answer_num": "pct_definitely_recommend"})
    )

    return rating, recommend


def build_analysis_panel(county_controls: pd.DataFrame) -> pd.DataFrame:
    spatial = pd.read_csv(RUCC_PANEL, low_memory=False)
    full = pd.read_csv(FULL_PANEL, low_memory=False)
    hhi = pd.read_csv(CLUSTER_HHI, low_memory=False)

    spatial["year"] = pd.to_numeric(spatial["year"], errors="coerce")
    full["year"] = pd.to_numeric(full["year"], errors="coerce")

    spatial = (
        spatial.loc[spatial["year"] == TARGET_YEAR]
        .drop_duplicates(["facility_id", "year"])
        .copy()
    )

    full = (
        full.loc[full["year"] == TARGET_YEAR]
        .drop_duplicates(["facility_id", "year"])
        .copy()
    )

    full_keep = [
        c for c in [
            "facility_id",
            "year",
            "hospital_overall_rating",
            "hospital_type",
            "hospital_ownership",
            "state",
        ]
        if c in full.columns
    ]

    df = spatial.merge(
        full[full_keep],
        on=["facility_id", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_full"),
    )

    hhi_keep = [
        c for c in [
            "cluster_id",
            "hhi_count_10000",
            "hhi_beds_10000",
            "cluster_size",
        ]
        if c in hhi.columns
    ]

    df = df.merge(
        hhi[hhi_keep].drop_duplicates("cluster_id"),
        on="cluster_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_hhi"),
    )

    rating_map, recommend_map = build_hcahps_maps()

    df["ccn_key"] = normalize_ccn(df["facility_id"])

    df = (
        df.merge(
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

    # Find county FIPS in the RUCC-linked panel.
    county_candidates = [
        "county_fips",
        "fips",
        "county_fips_2023",
        "rucc_fips",
    ]

    county_col = None
    for candidate in county_candidates:
        if candidate in df.columns:
            county_col = candidate
            break

    if county_col is None:
        # Search heuristically.
        plausible = [
            c for c in df.columns
            if "fips" in c.lower()
            and "county" in c.lower()
        ]
        if plausible:
            county_col = plausible[0]

    if county_col is None:
        raise KeyError(
            "Could not locate county FIPS in hospital_cluster_rucc_panel.csv.\n"
            "Columns containing 'fips':\n"
            + "\n".join(
                f"  {c}" for c in df.columns
                if "fips" in c.lower()
            )
        )

    print(f"Using county FIPS column: {county_col}")

    df["county_fips_model"] = normalize_fips(df[county_col])

    df = df.merge(
        county_controls,
        left_on="county_fips_model",
        right_on="county_fips",
        how="left",
        validate="many_to_one",
    )

    df["rating"] = pd.to_numeric(
        df["hospital_overall_rating"],
        errors="coerce",
    )

    df["rural"] = pd.to_numeric(df["rural"], errors="coerce")

    df["nearest_competing_cluster_km"] = pd.to_numeric(
        df["nearest_competing_cluster_km"],
        errors="coerce",
    )

    df["log_nearest_cluster_km"] = np.where(
        df["nearest_competing_cluster_km"] > 0,
        np.log(df["nearest_competing_cluster_km"]),
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
        np.log(df["cluster_size_model"]),
        np.nan,
    )

    df["hhi_count_scaled"] = (
        pd.to_numeric(df["hhi_count_10000"], errors="coerce")
        / 1000.0
    )

    df["hhi_beds_scaled"] = (
        pd.to_numeric(df["hhi_beds_10000"], errors="coerce")
        / 1000.0
    )

    df["hospital_type_clean"] = clean_category(df["hospital_type"])
    df["state_clean"] = clean_category(df["state"])

    return df


def fit_clustered(formula: str, data: pd.DataFrame):
    model = smf.ols(formula=formula, data=data)

    exog = np.asarray(model.exog)
    rank = int(np.linalg.matrix_rank(exog))
    cols = int(exog.shape[1])

    if rank != cols:
        print(
            f"WARNING: rank deficient design "
            f"({rank}/{cols})"
        )

    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cluster_id"]},
    )


def extract_rows(
    model,
    outcome_label: str,
    specification: str,
    terms: list[str],
) -> list[dict]:
    ci = model.conf_int()
    rows = []

    for term in terms:
        if term not in model.params.index:
            continue

        rows.append(
            {
                "outcome": outcome_label,
                "specification": specification,
                "term": term,
                "coefficient": float(model.params[term]),
                "clustered_se": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_lower": float(ci.loc[term, 0]),
                "ci_upper": float(ci.loc[term, 1]),
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
            }
        )

    return rows


def run_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_label: str,
) -> tuple[list[dict], list[dict]]:

    baseline_controls = (
        "rural + log_cluster_size "
        "+ C(hospital_type_clean) "
        "+ C(state_clean)"
    )

    ses_controls = (
        "log_population_density "
        "+ log_median_household_income "
        "+ poverty_rate "
        "+ median_age "
        "+ unemployment_rate"
    )

    formulas = {
        "S1_baseline": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ log_nearest_cluster_km + "
            + baseline_controls
        ),
        "S2_plus_county_SES": (
            f"{outcome_col} ~ hhi_count_scaled "
            "+ log_nearest_cluster_km + "
            + baseline_controls
            + " + "
            + ses_controls
        ),
        "S3_bedHHI_plus_county_SES": (
            f"{outcome_col} ~ hhi_beds_scaled "
            "+ log_nearest_cluster_km + "
            + baseline_controls
            + " + "
            + ses_controls
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
            "log_nearest_cluster_km",
        ]

        if "hhi_count_scaled" in formula:
            needed.append("hhi_count_scaled")

        if "hhi_beds_scaled" in formula:
            needed.append("hhi_beds_scaled")

        if "county_SES" in name:
            needed += [
                "log_population_density",
                "log_median_household_income",
                "poverty_rate",
                "median_age",
                "unemployment_rate",
            ]

        d = df.dropna(
            subset=list(dict.fromkeys(needed))
        ).copy()

        result = fit_clustered(formula, d)

        terms = [
            "hhi_count_scaled",
            "hhi_beds_scaled",
            "log_nearest_cluster_km",
            "rural",
            "log_population_density",
            "log_median_household_income",
            "poverty_rate",
            "median_age",
            "unemployment_rate",
        ]

        coef_rows.extend(
            extract_rows(
                result,
                outcome_label,
                name,
                terms,
            )
        )

        spatial_beta = (
            float(result.params["log_nearest_cluster_km"])
            if "log_nearest_cluster_km" in result.params.index
            else np.nan
        )

        comp_rows.append(
            {
                "outcome": outcome_label,
                "specification": name,
                "n": int(result.nobs),
                "r_squared": float(result.rsquared),
                "adj_r_squared": float(result.rsquared_adj),
                "aic": float(result.aic),
                "spatial_beta": spatial_beta,
                "spatial_p_value": (
                    float(result.pvalues["log_nearest_cluster_km"])
                    if "log_nearest_cluster_km" in result.pvalues.index
                    else np.nan
                ),
                "effect_if_distance_doubles": (
                    spatial_beta * np.log(2)
                    if pd.notna(spatial_beta)
                    else np.nan
                ),
            }
        )

    return coef_rows, comp_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        RUCC_PANEL,
        FULL_PANEL,
        CLUSTER_HHI,
        HCAHPS_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    print("=" * 80)
    print("STEP 13: COUNTY SES CONTROL ROBUSTNESS")
    print("=" * 80)

    county_controls = build_county_controls()
    df = build_analysis_panel(county_controls)

    ses_complete = df[
        [
            "log_population_density",
            "log_median_household_income",
            "poverty_rate",
            "median_age",
            "unemployment_rate",
        ]
    ].notna().all(axis=1)

    print(
        f"Hospital rows with complete county SES controls: "
        f"{int(ses_complete.sum()):,} / {len(df):,} "
        f"({100 * ses_complete.mean():.2f}%)"
    )

    descriptives = (
        df.loc[ses_complete]
        .groupby("rural")
        .agg(
            hospitals=("facility_id", "size"),
            median_population_density=(
                "population_density",
                "median",
            ),
            median_household_income=(
                "median_household_income",
                "median",
            ),
            mean_poverty_rate=(
                "poverty_rate",
                "mean",
            ),
            median_age=(
                "median_age",
                "median",
            ),
            mean_unemployment_rate=(
                "unemployment_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    descriptives["rural_label"] = np.where(
        descriptives["rural"] == 1,
        "rural",
        "metro",
    )

    descriptives.to_csv(DESC_OUTPUT, index=False)

    all_coef = []
    all_comp = []

    outcomes = [
        ("rating", "CMS overall rating"),
        ("pct_rating_9_10", "HCAHPS % rating 9-10"),
        (
            "pct_definitely_recommend",
            "HCAHPS % definitely recommend",
        ),
    ]

    for outcome_col, outcome_label in outcomes:
        coef, comp = run_outcome(
            df,
            outcome_col,
            outcome_label,
        )
        all_coef.extend(coef)
        all_comp.extend(comp)

    coefs = pd.DataFrame(all_coef)
    compare = pd.DataFrame(all_comp)

    coefs.to_csv(COEF_OUTPUT, index=False)
    compare.to_csv(COMPARE_OUTPUT, index=False)

    lines = [
        "STEP 13: COUNTY SES CONTROL ROBUSTNESS",
        "=" * 80,
        "",
        "County control descriptives:",
        descriptives.to_string(index=False),
        "",
        "Model comparison:",
        compare.to_string(index=False),
        "",
        "Key coefficients:",
        coefs.to_string(index=False),
        "",
        (
            "Interpretation caution: county controls reduce an important "
            "source of observable confounding but do not establish causal "
            "identification."
        ),
    ]

    SUMMARY_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 13 COMPLETE")
    print("=" * 80)

    print("\nCounty-control descriptives:")
    print(
        descriptives[
            [
                "rural_label",
                "hospitals",
                "median_population_density",
                "median_household_income",
                "mean_poverty_rate",
                "median_age",
                "mean_unemployment_rate",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Spatial coefficient comparison:")
    print(
        compare[
            [
                "outcome",
                "specification",
                "n",
                "spatial_beta",
                "spatial_p_value",
                "effect_if_distance_doubles",
                "adj_r_squared",
                "aic",
            ]
        ]
        .round(
            {
                "spatial_beta": 4,
                "spatial_p_value": 5,
                "effect_if_distance_doubles": 3,
                "adj_r_squared": 4,
                "aic": 1,
            }
        )
        .to_string(index=False)
    )

    print()
    print("County-SES coefficients:")
    ses_terms = [
        "log_population_density",
        "log_median_household_income",
        "poverty_rate",
        "median_age",
        "unemployment_rate",
    ]

    print(
        coefs.loc[
            coefs["term"].isin(ses_terms),
            [
                "outcome",
                "specification",
                "term",
                "coefficient",
                "clustered_se",
                "p_value",
            ],
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


if __name__ == "__main__":
    main()
