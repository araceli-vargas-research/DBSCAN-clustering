#!/usr/bin/env python3
"""
08_spatial_rural_tests.py

Formal tests of whether rural hospital service clusters are more spatially
isolated than metro hospital service clusters.

Place directly inside:
    phase2_hospitals/

Expected inputs:
    data/spatial/hospital_clusters_all_eps.csv
    data/spatial/cluster_competition_all_eps.csv
    data/spatial/hospital_cluster_rucc_panel.csv

Creates:
    data/spatial/all_eps_cluster_rucc.csv
    data/results/spatial_rural_regressions.csv
    data/results/spatial_rural_regression_summary.txt
    data/results/spatial_rural_descriptives.csv

Main models, run separately for eps = 5, 10, 15, 20, 25 km:

1) All metro-only and rural-only service clusters:
       log(nearest competing cluster distance)
           ~ rural + log(cluster size) + year fixed effects

2) Multi-hospital clusters only:
       log(isolation ratio)
           ~ rural + log(cluster size) + year fixed effects

3) Multi-hospital clusters only:
       log(within-cluster mean dispersion)
           ~ rural + log(cluster size) + year fixed effects

Interpretation:
    rural = 1 for rural-only clusters
    rural = 0 for metro-only clusters

Mixed and unmatched clusters are excluded from regression samples.
HC1 heteroskedasticity-robust standard errors are reported.

Run:
    pip3 install pandas numpy statsmodels
    python3 08_spatial_rural_tests.py
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

ASSIGNMENTS_FILE = (
    SPATIAL_DIR / "hospital_clusters_all_eps.csv"
)

COMPETITION_FILE = (
    SPATIAL_DIR / "cluster_competition_all_eps.csv"
)

RUCC_PANEL_FILE = (
    SPATIAL_DIR / "hospital_cluster_rucc_panel.csv"
)

ALL_EPS_CLUSTER_RUCC_FILE = (
    SPATIAL_DIR / "all_eps_cluster_rucc.csv"
)

REGRESSION_FILE = (
    RESULTS_DIR / "spatial_rural_regressions.csv"
)

DESCRIPTIVES_FILE = (
    RESULTS_DIR / "spatial_rural_descriptives.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR / "spatial_rural_regression_summary.txt"
)

EPS_VALUES = [5, 10, 15, 20, 25]


def safe_log(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(
        series,
        errors="coerce",
    )
    return np.where(
        x > 0,
        np.log(x),
        np.nan,
    )


def classify_cluster(group: pd.DataFrame) -> pd.Series:
    """
    Classify a cluster using matched hospital RUCC labels.
    """
    matched = group.loc[
        group["rural"].notna()
    ].copy()

    n_total = len(group)
    n_matched = len(matched)

    if n_matched == 0:
        return pd.Series(
            {
                "member_hospitals": n_total,
                "rucc_matched_hospitals": 0,
                "rural_hospitals": 0,
                "metro_hospitals": 0,
                "rural_share": np.nan,
                "cluster_rurality": "unmatched",
                "rural_only": np.nan,
            }
        )

    rural_count = int(
        (matched["rural"] == 1).sum()
    )
    metro_count = int(
        (matched["rural"] == 0).sum()
    )

    rural_share = (
        rural_count / n_matched
    )

    if rural_count == n_matched:
        rurality = "rural_only"
        rural_only = 1
    elif metro_count == n_matched:
        rurality = "metro_only"
        rural_only = 0
    else:
        rurality = "mixed"
        rural_only = np.nan

    return pd.Series(
        {
            "member_hospitals": n_total,
            "rucc_matched_hospitals": n_matched,
            "rural_hospitals": rural_count,
            "metro_hospitals": metro_count,
            "rural_share": rural_share,
            "cluster_rurality": rurality,
            "rural_only": rural_only,
        }
    )


def get_term_result(
    model,
    term: str,
    outcome: str,
    eps_km: int,
    specification: str,
    sample_label: str,
) -> dict:
    """
    Extract one coefficient row from a statsmodels result.
    """
    params = model.params
    bse = model.bse
    pvalues = model.pvalues
    conf = model.conf_int()

    if term not in params.index:
        raise KeyError(
            f"Term {term} not found in model."
        )

    coef = float(params[term])
    se = float(bse[term])
    p = float(pvalues[term])
    ci_low = float(conf.loc[term, 0])
    ci_high = float(conf.loc[term, 1])

    # For log outcomes and a 0/1 rural regressor, exp(beta)-1
    # is the exact proportional difference implied by the model.
    pct_effect = (
        100.0 * (math.exp(coef) - 1.0)
    )

    return {
        "eps_km": eps_km,
        "specification": specification,
        "sample": sample_label,
        "outcome": outcome,
        "term": term,
        "coefficient": coef,
        "robust_se_hc1": se,
        "p_value": p,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "implied_percent_difference": pct_effect,
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(
            model.rsquared_adj
        ),
    }


def fit_model(
    data: pd.DataFrame,
    formula: str,
):
    """
    Fit OLS with HC1 robust standard errors.
    """
    return smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="HC1"
    )


def build_all_eps_rucc() -> pd.DataFrame:
    """
    Carry hospital-level RUCC labels from the successful Step 07
    merge onto every DBSCAN eps specification, then classify each
    service cluster as rural-only, metro-only, mixed, or unmatched.
    """

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        low_memory=False,
    )

    competition = pd.read_csv(
        COMPETITION_FILE,
        low_memory=False,
    )

    rucc_panel = pd.read_csv(
        RUCC_PANEL_FILE,
        low_memory=False,
    )

    # Facility-year RUCC labels do not depend on eps.
    rucc_map_cols = [
        c for c in [
            "facility_id",
            "year",
            "rural",
            "metro_status",
            "rucc_2023",
            "county_fips",
        ]
        if c in rucc_panel.columns
    ]

    rucc_map = (
        rucc_panel[
            rucc_map_cols
        ]
        .drop_duplicates(
            ["facility_id", "year"]
        )
        .copy()
    )

    for frame in [
        assignments,
        competition,
        rucc_map,
    ]:
        if "year" in frame.columns:
            frame["year"] = pd.to_numeric(
                frame["year"],
                errors="coerce",
            )
        if "eps_km" in frame.columns:
            frame["eps_km"] = pd.to_numeric(
                frame["eps_km"],
                errors="coerce",
            )

    rucc_map["rural"] = pd.to_numeric(
        rucc_map["rural"],
        errors="coerce",
    )

    assignment_rucc = assignments.merge(
        rucc_map,
        on=[
            "facility_id",
            "year",
        ],
        how="left",
        validate="many_to_one",
    )

    cluster_labels = (
        assignment_rucc.groupby(
            [
                "year",
                "snapshot",
                "eps_km",
                "projection_region",
                "cluster_id",
            ],
            dropna=False,
        )
        .apply(
            classify_cluster,
            include_groups=False,
        )
        .reset_index()
    )

    # competition already has one row per cluster.
    cluster_data = competition.merge(
        cluster_labels,
        on=[
            "year",
            "snapshot",
            "eps_km",
            "projection_region",
            "cluster_id",
        ],
        how="left",
        validate="one_to_one",
    )

    cluster_data.to_csv(
        ALL_EPS_CLUSTER_RUCC_FILE,
        index=False,
    )

    return cluster_data


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        ASSIGNMENTS_FILE,
        COMPETITION_FILE,
        RUCC_PANEL_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input:\n{path}"
            )

    print("=" * 76)
    print("FORMAL RURAL SPATIAL-COMPETITION TESTS")
    print("=" * 76)

    df = build_all_eps_rucc()

    print(
        f"Cluster-year-eps observations: "
        f"{len(df):,}"
    )

    # Numeric variables
    numeric_cols = [
        "eps_km",
        "year",
        "cluster_size",
        "nearest_competing_cluster_km",
        "within_cluster_mean_km",
        "within_cluster_max_km",
        "isolation_ratio",
        "rural_only",
        "multi_hospital_cluster",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Transformations
    df["log_cluster_size"] = safe_log(
        df["cluster_size"]
    )

    df["log_nearest_cluster_km"] = safe_log(
        df["nearest_competing_cluster_km"]
    )

    df["log_isolation_ratio"] = safe_log(
        df["isolation_ratio"]
    )

    df["log_within_cluster_mean_km"] = safe_log(
        df["within_cluster_mean_km"]
    )

    # Main regression universe: only pure metro or pure rural clusters.
    pure = df.loc[
        df["cluster_rurality"].isin(
            [
                "metro_only",
                "rural_only",
            ]
        )
    ].copy()

    print(
        f"Pure rural/metro cluster observations: "
        f"{len(pure):,}"
    )

    descriptive_rows = []
    regression_rows = []
    text_lines = []

    for eps in EPS_VALUES:
        d = pure.loc[
            pure["eps_km"] == eps
        ].copy()

        text_lines.append(
            "=" * 76
        )
        text_lines.append(
            f"EPS = {eps} km"
        )
        text_lines.append(
            "=" * 76
        )

        # ---------------------------------------------------------
        # DESCRIPTIVES
        # ---------------------------------------------------------
        for rurality, g in d.groupby(
            "cluster_rurality"
        ):
            descriptive_rows.append(
                {
                    "eps_km": eps,
                    "cluster_rurality": rurality,
                    "clusters": len(g),
                    "median_cluster_size": (
                        g["cluster_size"].median()
                    ),
                    "median_nearest_competing_cluster_km": (
                        g[
                            "nearest_competing_cluster_km"
                        ].median()
                    ),
                    "mean_nearest_competing_cluster_km": (
                        g[
                            "nearest_competing_cluster_km"
                        ].mean()
                    ),
                    "median_isolation_ratio": (
                        g["isolation_ratio"].median()
                    ),
                    "median_within_cluster_mean_km": (
                        g[
                            "within_cluster_mean_km"
                        ].replace(
                            0,
                            np.nan,
                        ).median()
                    ),
                }
            )

        # ---------------------------------------------------------
        # MODEL 1:
        # log distance to nearest competing cluster
        # all pure metro/rural clusters
        # ---------------------------------------------------------
        m1 = d.dropna(
            subset=[
                "log_nearest_cluster_km",
                "rural_only",
                "log_cluster_size",
                "year",
            ]
        ).copy()

        formula1 = (
            "log_nearest_cluster_km "
            "~ rural_only + log_cluster_size + C(year)"
        )

        model1 = fit_model(
            m1,
            formula1,
        )

        regression_rows.append(
            get_term_result(
                model1,
                "rural_only",
                "log_nearest_cluster_km",
                eps,
                "M1",
                "all pure metro/rural clusters",
            )
        )

        # ---------------------------------------------------------
        # MODEL 2:
        # log isolation ratio
        # multi-hospital pure clusters only
        # ---------------------------------------------------------
        m2 = d.loc[
            d["multi_hospital_cluster"] == 1
        ].dropna(
            subset=[
                "log_isolation_ratio",
                "rural_only",
                "log_cluster_size",
                "year",
            ]
        ).copy()

        formula2 = (
            "log_isolation_ratio "
            "~ rural_only + log_cluster_size + C(year)"
        )

        model2 = fit_model(
            m2,
            formula2,
        )

        regression_rows.append(
            get_term_result(
                model2,
                "rural_only",
                "log_isolation_ratio",
                eps,
                "M2",
                "multi-hospital pure clusters",
            )
        )

        # ---------------------------------------------------------
        # MODEL 3:
        # log within-cluster dispersion
        # multi-hospital pure clusters only
        # ---------------------------------------------------------
        m3 = d.loc[
            d["multi_hospital_cluster"] == 1
        ].dropna(
            subset=[
                "log_within_cluster_mean_km",
                "rural_only",
                "log_cluster_size",
                "year",
            ]
        ).copy()

        formula3 = (
            "log_within_cluster_mean_km "
            "~ rural_only + log_cluster_size + C(year)"
        )

        model3 = fit_model(
            m3,
            formula3,
        )

        regression_rows.append(
            get_term_result(
                model3,
                "rural_only",
                "log_within_cluster_mean_km",
                eps,
                "M3",
                "multi-hospital pure clusters",
            )
        )

        # Terminal / txt summary
        r1 = regression_rows[-3]
        r2 = regression_rows[-2]
        r3 = regression_rows[-1]

        text_lines.append(
            "M1: log nearest competing-cluster distance"
        )
        text_lines.append(
            f"  rural coefficient = {r1['coefficient']:.4f}"
            f"  HC1 SE = {r1['robust_se_hc1']:.4f}"
            f"  p = {r1['p_value']:.4g}"
            f"  implied difference = "
            f"{r1['implied_percent_difference']:.2f}%"
            f"  N = {r1['n']:,}"
        )

        text_lines.append(
            "M2: log isolation ratio, multi-hospital clusters"
        )
        text_lines.append(
            f"  rural coefficient = {r2['coefficient']:.4f}"
            f"  HC1 SE = {r2['robust_se_hc1']:.4f}"
            f"  p = {r2['p_value']:.4g}"
            f"  implied difference = "
            f"{r2['implied_percent_difference']:.2f}%"
            f"  N = {r2['n']:,}"
        )

        text_lines.append(
            "M3: log within-cluster mean dispersion, "
            "multi-hospital clusters"
        )
        text_lines.append(
            f"  rural coefficient = {r3['coefficient']:.4f}"
            f"  HC1 SE = {r3['robust_se_hc1']:.4f}"
            f"  p = {r3['p_value']:.4g}"
            f"  implied difference = "
            f"{r3['implied_percent_difference']:.2f}%"
            f"  N = {r3['n']:,}"
        )

        text_lines.append("")

    regressions = pd.DataFrame(
        regression_rows
    )

    descriptives = pd.DataFrame(
        descriptive_rows
    )

    regressions.to_csv(
        REGRESSION_FILE,
        index=False,
    )

    descriptives.to_csv(
        DESCRIPTIVES_FILE,
        index=False,
    )

    SUMMARY_FILE.write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("RURAL SPATIAL TESTS COMPLETE")
    print("=" * 76)

    print("\nCreated:")
    for path in [
        ALL_EPS_CLUSTER_RUCC_FILE,
        REGRESSION_FILE,
        DESCRIPTIVES_FILE,
        SUMMARY_FILE,
    ]:
        print(
            f"  {path.relative_to(PROJECT_DIR)}"
        )

    print()
    print("Rural coefficient robustness table:")
    print(
        regressions[
            [
                "eps_km",
                "specification",
                "outcome",
                "coefficient",
                "robust_se_hc1",
                "p_value",
                "implied_percent_difference",
                "n",
            ]
        ]
        .round(
            {
                "coefficient": 4,
                "robust_se_hc1": 4,
                "p_value": 5,
                "implied_percent_difference": 2,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Read M1 first: a positive rural coefficient means rural "
        "clusters are farther from the nearest competing cluster, "
        "conditional on cluster size and year fixed effects."
    )


if __name__ == "__main__":
    main()
