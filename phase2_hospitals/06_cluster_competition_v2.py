#!/usr/bin/env python3
"""
06_cluster_competition.py

Build cluster-level spatial competition measures from the hospital
DBSCAN assignments created in Step 05.

Place directly inside:
    phase2_hospitals/

Expected inputs:
    data/spatial/hospital_clusters_all_eps.csv
    data/spatial/cluster_centroids_all_eps.csv

Creates:
    data/spatial/cluster_competition_all_eps.csv
    data/spatial/cluster_competition_eps10.csv
    data/spatial/hospital_cluster_competition_panel.csv
    data/spatial/cluster_competition_summary.csv

Primary specification:
    eps = 10 km

Robustness specifications:
    eps = 5, 15, 20, 25 km

Key measures:
    within_cluster_mean_km
        Mean hospital distance to own cluster centroid.

    within_cluster_max_km
        Maximum hospital distance to own cluster centroid.

    nearest_competing_cluster_km
        Distance from a cluster centroid to the nearest OTHER cluster
        centroid within the same year and projection region.

    isolation_ratio
        nearest_competing_cluster_km / within_cluster_mean_km

    isolation_ratio_max
        nearest_competing_cluster_km / within_cluster_max_km

Singleton clusters:
    Their within-cluster dispersion is exactly zero, so the isolation
    ratios are undefined (NaN). Their nearest competing-cluster distance
    is still retained.

Run:
    pip3 install pandas numpy scikit-learn
    python3 06_cluster_competition.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"

ASSIGNMENTS_FILE = (
    SPATIAL_DIR / "hospital_clusters_all_eps.csv"
)

CENTROIDS_FILE = (
    SPATIAL_DIR / "cluster_centroids_all_eps.csv"
)

OUTPUT_ALL = (
    SPATIAL_DIR / "cluster_competition_all_eps.csv"
)

OUTPUT_EPS10 = (
    SPATIAL_DIR / "cluster_competition_eps10.csv"
)

OUTPUT_HOSPITAL_PANEL = (
    SPATIAL_DIR / "hospital_cluster_competition_panel.csv"
)

OUTPUT_SUMMARY = (
    SPATIAL_DIR / "cluster_competition_summary.csv"
)

PRIMARY_EPS = 10


def build_nearest_cluster_distances(
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For one year / projection region / eps solution, compute
    nearest competing cluster centroid distance.

    Each row in cluster_df represents one service cluster.
    """

    g = cluster_df.copy().reset_index(drop=True)

    n = len(g)

    if n == 0:
        return g

    if n == 1:
        g["nearest_cluster_id"] = pd.NA
        g["nearest_competing_cluster_m"] = np.nan
        g["nearest_competing_cluster_km"] = np.nan
        g["nearest_competing_cluster_miles"] = np.nan
        return g

    coords = g[
        ["centroid_x_m", "centroid_y_m"]
    ].to_numpy(dtype=float)

    # Ask for two neighbors: self + nearest other cluster.
    model = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean",
    )
    model.fit(coords)

    distances, indices = model.kneighbors(coords)

    nearest_ids = []
    nearest_m = []

    for i in range(n):
        # Usually first result is self at distance 0.
        candidate_dist = None
        candidate_idx = None

        for dist, idx in zip(
            distances[i],
            indices[i],
        ):
            if idx == i:
                continue
            candidate_dist = float(dist)
            candidate_idx = int(idx)
            break

        if candidate_idx is None:
            nearest_ids.append(pd.NA)
            nearest_m.append(np.nan)
        else:
            nearest_ids.append(
                g.loc[
                    candidate_idx,
                    "cluster_id",
                ]
            )
            nearest_m.append(
                candidate_dist
            )

    g["nearest_cluster_id"] = nearest_ids
    g["nearest_competing_cluster_m"] = nearest_m
    g["nearest_competing_cluster_km"] = (
        g["nearest_competing_cluster_m"]
        / 1000.0
    )
    g["nearest_competing_cluster_miles"] = (
        g["nearest_competing_cluster_m"]
        / 1609.344
    )

    return g


def add_isolation_measures(
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add ratios comparing between-cluster separation to
    within-cluster dispersion.
    """

    g = cluster_df.copy()

    mean_disp = pd.to_numeric(
        g["mean_distance_to_centroid_km"],
        errors="coerce",
    )

    max_disp = pd.to_numeric(
        g["max_distance_to_centroid_km"],
        errors="coerce",
    )

    nearest = pd.to_numeric(
        g["nearest_competing_cluster_km"],
        errors="coerce",
    )

    g["within_cluster_mean_km"] = mean_disp
    g["within_cluster_max_km"] = max_disp

    g["isolation_ratio"] = np.where(
        mean_disp > 0,
        nearest / mean_disp,
        np.nan,
    )

    g["isolation_ratio_max"] = np.where(
        max_disp > 0,
        nearest / max_disp,
        np.nan,
    )

    g["multi_hospital_cluster"] = (
        pd.to_numeric(
            g["cluster_size"],
            errors="coerce",
        ) >= 2
    ).astype(int)

    g["singleton_cluster"] = (
        pd.to_numeric(
            g["cluster_size"],
            errors="coerce",
        ) == 1
    ).astype(int)

    return g


def summarize_solution(
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize cluster competition measures by year and eps.
    """

    d = cluster_df.copy()

    # Overall cluster statistics.
    summary = (
        d.groupby(
            [
                "year",
                "snapshot",
                "eps_km",
            ],
            dropna=False,
        )
        .agg(
            service_clusters=(
                "cluster_id",
                "nunique",
            ),
            multi_hospital_clusters=(
                "multi_hospital_cluster",
                "sum",
            ),
            singleton_clusters=(
                "singleton_cluster",
                "sum",
            ),
            mean_cluster_size=(
                "cluster_size",
                "mean",
            ),
            median_cluster_size=(
                "cluster_size",
                "median",
            ),
            median_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                "median",
            ),
            mean_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                "mean",
            ),
            p25_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                lambda s: s.quantile(0.25),
            ),
            p75_nearest_competing_cluster_km=(
                "nearest_competing_cluster_km",
                lambda s: s.quantile(0.75),
            ),
        )
        .reset_index()
    )

    # Dispersion and isolation ratios only make sense for
    # multi-hospital clusters. Singleton clusters have zero
    # within-cluster dispersion by construction.
    multi = d.loc[
        d["multi_hospital_cluster"] == 1
    ].copy()

    multi_summary = (
        multi.groupby(
            [
                "year",
                "snapshot",
                "eps_km",
            ],
            dropna=False,
        )
        .agg(
            median_within_cluster_mean_km_multi=(
                "within_cluster_mean_km",
                "median",
            ),
            mean_within_cluster_mean_km_multi=(
                "within_cluster_mean_km",
                "mean",
            ),
            median_within_cluster_max_km_multi=(
                "within_cluster_max_km",
                "median",
            ),
            median_nearest_competing_cluster_km_multi=(
                "nearest_competing_cluster_km",
                "median",
            ),
            median_isolation_ratio_multi=(
                "isolation_ratio",
                "median",
            ),
            mean_isolation_ratio_multi=(
                "isolation_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        multi_summary,
        on=[
            "year",
            "snapshot",
            "eps_km",
        ],
        how="left",
        validate="one_to_one",
    )

    round_cols = [
        "mean_cluster_size",
        "median_cluster_size",
        "median_nearest_competing_cluster_km",
        "mean_nearest_competing_cluster_km",
        "p25_nearest_competing_cluster_km",
        "p75_nearest_competing_cluster_km",
        "median_within_cluster_mean_km_multi",
        "mean_within_cluster_mean_km_multi",
        "median_within_cluster_max_km_multi",
        "median_nearest_competing_cluster_km_multi",
        "median_isolation_ratio_multi",
        "mean_isolation_ratio_multi",
    ]

    for col in round_cols:
        summary[col] = (
            pd.to_numeric(
                summary[col],
                errors="coerce",
            )
            .round(3)
        )

    return summary


def main() -> None:

    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{ASSIGNMENTS_FILE}\n\n"
            "Run python3 05_identify_spatial_clusters.py first."
        )

    if not CENTROIDS_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{CENTROIDS_FILE}\n\n"
            "Run python3 05_identify_spatial_clusters.py first."
        )

    assignments = pd.read_csv(
        ASSIGNMENTS_FILE,
        low_memory=False,
    )

    centroids = pd.read_csv(
        CENTROIDS_FILE,
        low_memory=False,
    )

    numeric_cols = [
        "year",
        "eps_km",
        "cluster_size",
        "centroid_x_m",
        "centroid_y_m",
        "mean_distance_to_centroid_km",
        "max_distance_to_centroid_km",
    ]

    for col in numeric_cols:
        if col in centroids.columns:
            centroids[col] = pd.to_numeric(
                centroids[col],
                errors="coerce",
            )

    print("=" * 72)
    print("BUILD CLUSTER COMPETITION MEASURES")
    print("=" * 72)
    print(
        f"Cluster rows: {len(centroids):,}"
    )
    print(
        f"Hospital assignment rows: "
        f"{len(assignments):,}"
    )

    competition_parts = []

    group_cols = [
        "year",
        "projection_region",
        "eps_km",
    ]

    for (
        year,
        region,
        eps_km,
    ), group in centroids.groupby(
        group_cols,
        dropna=False,
    ):

        print(
            f"year={int(year)} "
            f"region={region} "
            f"eps={int(eps_km)} km "
            f"clusters={len(group):,}"
        )

        result = (
            build_nearest_cluster_distances(
                group
            )
        )

        result = add_isolation_measures(
            result
        )

        competition_parts.append(
            result
        )

    competition = pd.concat(
        competition_parts,
        ignore_index=True,
        sort=False,
    )

    competition.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    eps10 = competition.loc[
        competition["eps_km"]
        == PRIMARY_EPS
    ].copy()

    eps10.to_csv(
        OUTPUT_EPS10,
        index=False,
    )

    # -------------------------------------------------------------
    # MERGE PRIMARY 10-KM COMPETITION METRICS BACK TO HOSPITALS
    # -------------------------------------------------------------

    assignments["eps_km"] = pd.to_numeric(
        assignments["eps_km"],
        errors="coerce",
    )

    primary_assignments = assignments.loc[
        assignments["eps_km"]
        == PRIMARY_EPS
    ].copy()

    cluster_keep = [
        "cluster_id",
        "year",
        "snapshot",
        "projection_region",
        "eps_km",
        "cluster_size",
        "centroid_x_m",
        "centroid_y_m",
        "within_cluster_mean_km",
        "within_cluster_max_km",
        "nearest_cluster_id",
        "nearest_competing_cluster_km",
        "nearest_competing_cluster_miles",
        "isolation_ratio",
        "isolation_ratio_max",
        "multi_hospital_cluster",
        "singleton_cluster",
    ]

    cluster_keep = [
        c for c in cluster_keep
        if c in eps10.columns
    ]

    merged = primary_assignments.merge(
        eps10[cluster_keep],
        on=[
            "cluster_id",
            "year",
            "snapshot",
            "projection_region",
            "eps_km",
        ],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cluster"),
    )

    merged.to_csv(
        OUTPUT_HOSPITAL_PANEL,
        index=False,
    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    summary = summarize_solution(
        competition
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    latest_year = (
        pd.to_numeric(
            summary["year"],
            errors="coerce",
        )
        .max()
    )

    latest_primary = summary.loc[
        (summary["year"] == latest_year)
        & (summary["eps_km"] == PRIMARY_EPS)
    ].copy()

    print()
    print("=" * 72)
    print("CLUSTER COMPETITION COMPLETE")
    print("=" * 72)

    print("\nCreated:")
    for path in [
        OUTPUT_ALL,
        OUTPUT_EPS10,
        OUTPUT_HOSPITAL_PANEL,
        OUTPUT_SUMMARY,
    ]:
        print(
            f"  {path.relative_to(PROJECT_DIR)}"
        )

    print()
    print(
        f"Primary specification "
        f"({PRIMARY_EPS} km), latest year:"
    )

    if len(latest_primary):
        cols = [
            "year",
            "eps_km",
            "service_clusters",
            "multi_hospital_clusters",
            "singleton_clusters",
            "mean_cluster_size",
            "median_within_cluster_mean_km_multi",
            "median_nearest_competing_cluster_km",
            "median_nearest_competing_cluster_km_multi",
            "p25_nearest_competing_cluster_km",
            "p75_nearest_competing_cluster_km",
            "median_isolation_ratio_multi",
        ]

        print(
            latest_primary[
                cols
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No latest-year primary row found."
        )

    print()
    print(
        "Next step: merge rurality / RUCC and compare "
        "cluster isolation across rural and urban hospitals."
    )


if __name__ == "__main__":
    main()
