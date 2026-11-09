#!/usr/bin/env python3
"""
05_identify_spatial_clusters.py

Identify hospital spatial clusters using DBSCAN at multiple distance
thresholds, year by year and projection region by projection region.

Place directly inside:
    phase2_hospitals/

Expected input:
    data/spatial/hospital_spatial_points.csv

Creates:
    data/spatial/hospital_clusters_all_eps.csv
    data/spatial/cluster_diagnostics.csv
    data/spatial/cluster_summary_by_year.csv
    data/spatial/hospital_clusters_eps10.csv
    data/spatial/hospital_clusters_eps15.csv
    data/spatial/hospital_clusters_eps20.csv

Run:
    pip3 install pandas numpy scikit-learn
    python3 05_identify_spatial_clusters.py

Method:
- DBSCAN is run separately for each year and projection region.
- eps is interpreted in kilometers because x/y coordinates are meters.
- min_samples=2 means a "cluster" requires at least two hospitals.
- DBSCAN noise points are NOT discarded; each noise hospital is treated
  as its own singleton service cluster for downstream competition work.
- Multiple eps values are saved so cluster definitions can be tested
  for robustness rather than chosen arbitrarily.

Primary thresholds:
    5, 10, 15, 20, 25 km
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"

INPUT_FILE = SPATIAL_DIR / "hospital_spatial_points.csv"

ALL_ASSIGNMENTS_FILE = (
    SPATIAL_DIR / "hospital_clusters_all_eps.csv"
)

DIAGNOSTICS_FILE = (
    SPATIAL_DIR / "cluster_diagnostics.csv"
)

SUMMARY_FILE = (
    SPATIAL_DIR / "cluster_summary_by_year.csv"
)

EPS_VALUES_KM = [5, 10, 15, 20, 25]
MIN_SAMPLES = 2


def dbscan_clusters(
    group: pd.DataFrame,
    eps_km: float,
) -> pd.DataFrame:
    """
    Run DBSCAN for one year-region group.

    Noise observations from DBSCAN (-1) are converted into unique
    singleton service clusters so every hospital remains usable.
    """

    out = group.copy().reset_index(drop=True)

    coords = out[
        ["x_m", "y_m"]
    ].to_numpy(dtype=float)

    eps_m = eps_km * 1000.0

    model = DBSCAN(
        eps=eps_m,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    )

    labels = model.fit_predict(coords)

    out["dbscan_label"] = labels

    # Number the real DBSCAN clusters first.
    valid_labels = sorted(
        x for x in set(labels)
        if x >= 0
    )

    label_map = {
        old: new
        for new, old in enumerate(
            valid_labels,
            start=1,
        )
    }

    service_cluster = []
    next_singleton = len(valid_labels) + 1

    for label in labels:
        if label >= 0:
            service_cluster.append(
                label_map[label]
            )
        else:
            service_cluster.append(
                next_singleton
            )
            next_singleton += 1

    out["service_cluster_local"] = service_cluster

    # Make globally unique cluster IDs.
    year = str(out["year"].iloc[0])
    region = str(
        out["projection_region"].iloc[0]
    )

    out["cluster_id"] = [
        f"{year}_{region}_eps{int(eps_km)}_c{c}"
        for c in out["service_cluster_local"]
    ]

    out["eps_km"] = eps_km

    # Cluster size.
    size_map = (
        out.groupby("cluster_id")
        .size()
        .to_dict()
    )

    out["cluster_size"] = (
        out["cluster_id"]
        .map(size_map)
        .astype(int)
    )

    out["singleton_cluster"] = (
        out["cluster_size"] == 1
    ).astype(int)

    return out


def cluster_centroid_stats(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cluster-level centroid and dispersion statistics.
    """

    rows = []

    for cluster_id, g in assignments.groupby(
        "cluster_id",
        dropna=False,
    ):
        x_bar = g["x_m"].mean()
        y_bar = g["y_m"].mean()

        dist_to_centroid = np.sqrt(
            (g["x_m"] - x_bar) ** 2
            + (g["y_m"] - y_bar) ** 2
        )

        rows.append(
            {
                "year": g["year"].iloc[0],
                "snapshot": g["snapshot"].iloc[0],
                "projection_region": (
                    g["projection_region"].iloc[0]
                ),
                "eps_km": g["eps_km"].iloc[0],
                "cluster_id": cluster_id,
                "cluster_size": len(g),
                "centroid_x_m": x_bar,
                "centroid_y_m": y_bar,
                "mean_distance_to_centroid_km": (
                    dist_to_centroid.mean() / 1000.0
                ),
                "max_distance_to_centroid_km": (
                    dist_to_centroid.max() / 1000.0
                ),
            }
        )

    return pd.DataFrame(rows)


def diagnostics_for_assignment(
    assignments: pd.DataFrame,
) -> dict:
    """
    Summarize one year-region-eps cluster solution.
    """

    clusters = (
        assignments[
            [
                "cluster_id",
                "cluster_size",
            ]
        ]
        .drop_duplicates()
    )

    n_hospitals = len(assignments)
    n_clusters = len(clusters)

    singleton_clusters = int(
        (clusters["cluster_size"] == 1).sum()
    )

    multi_clusters = int(
        (clusters["cluster_size"] >= 2).sum()
    )

    hospitals_in_multi = int(
        assignments.loc[
            assignments["cluster_size"] >= 2
        ].shape[0]
    )

    largest_cluster = int(
        clusters["cluster_size"].max()
    ) if n_clusters else 0

    median_cluster_size = float(
        clusters["cluster_size"].median()
    ) if n_clusters else np.nan

    return {
        "year": assignments["year"].iloc[0],
        "snapshot": assignments["snapshot"].iloc[0],
        "projection_region": (
            assignments[
                "projection_region"
            ].iloc[0]
        ),
        "eps_km": assignments["eps_km"].iloc[0],
        "hospitals": n_hospitals,
        "service_clusters": n_clusters,
        "multi_hospital_clusters": multi_clusters,
        "singleton_clusters": singleton_clusters,
        "hospitals_in_multi_clusters": (
            hospitals_in_multi
        ),
        "pct_hospitals_in_multi_clusters": (
            100 * hospitals_in_multi
            / n_hospitals
        ) if n_hospitals else np.nan,
        "largest_cluster_size": largest_cluster,
        "median_cluster_size": (
            median_cluster_size
        ),
    }


def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}\n\n"
            "Run python3 04_build_euclidean_distances.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    for col in [
        "x_m",
        "y_m",
        "year",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    usable = df.loc[
        df["x_m"].notna()
        & df["y_m"].notna()
        & df["year"].notna()
    ].copy()

    print("=" * 72)
    print("IDENTIFY HOSPITAL SPATIAL CLUSTERS")
    print("=" * 72)
    print(
        f"Projectable hospital-year rows: "
        f"{len(usable):,}"
    )
    print(
        f"DBSCAN min_samples: {MIN_SAMPLES}"
    )
    print(
        "Thresholds: "
        + ", ".join(
            f"{x} km"
            for x in EPS_VALUES_KM
        )
    )

    assignment_parts = []
    diagnostic_rows = []
    centroid_parts = []

    group_cols = [
        "year",
        "projection_region",
    ]

    for eps_km in EPS_VALUES_KM:

        print()
        print("-" * 72)
        print(
            f"DBSCAN EPS = {eps_km} km"
        )
        print("-" * 72)

        for (
            year,
            region,
        ), group in usable.groupby(
            group_cols,
            dropna=False,
        ):

            if len(group) == 0:
                continue

            print(
                f"year={int(year)} "
                f"region={region} "
                f"N={len(group):,}"
            )

            assignments = (
                dbscan_clusters(
                    group,
                    eps_km,
                )
            )

            assignment_parts.append(
                assignments
            )

            diagnostic_rows.append(
                diagnostics_for_assignment(
                    assignments
                )
            )

            centroid_parts.append(
                cluster_centroid_stats(
                    assignments
                )
            )

    assignments_all = pd.concat(
        assignment_parts,
        ignore_index=True,
        sort=False,
    )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    centroids = pd.concat(
        centroid_parts,
        ignore_index=True,
        sort=False,
    )

    assignments_all.to_csv(
        ALL_ASSIGNMENTS_FILE,
        index=False,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # YEAR-LEVEL SUMMARY ACROSS REGIONS
    # -------------------------------------------------------------

    year_summary = (
        diagnostics.groupby(
            [
                "year",
                "snapshot",
                "eps_km",
            ],
            dropna=False,
        )
        .agg(
            hospitals=(
                "hospitals",
                "sum",
            ),
            service_clusters=(
                "service_clusters",
                "sum",
            ),
            multi_hospital_clusters=(
                "multi_hospital_clusters",
                "sum",
            ),
            singleton_clusters=(
                "singleton_clusters",
                "sum",
            ),
            hospitals_in_multi_clusters=(
                "hospitals_in_multi_clusters",
                "sum",
            ),
            largest_cluster_size=(
                "largest_cluster_size",
                "max",
            ),
        )
        .reset_index()
    )

    year_summary[
        "pct_hospitals_in_multi_clusters"
    ] = (
        100
        * year_summary[
            "hospitals_in_multi_clusters"
        ]
        / year_summary["hospitals"]
    ).round(2)

    year_summary[
        "mean_hospitals_per_service_cluster"
    ] = (
        year_summary["hospitals"]
        / year_summary["service_clusters"]
    ).round(3)

    year_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE PRIMARY THRESHOLD FILES
    # -------------------------------------------------------------

    for eps_km in [10, 15, 20]:
        out = assignments_all.loc[
            assignments_all["eps_km"]
            == eps_km
        ].copy()

        out_file = (
            SPATIAL_DIR
            / f"hospital_clusters_eps{eps_km}.csv"
        )

        out.to_csv(
            out_file,
            index=False,
        )

    # Save centroid data too.
    CENTROIDS_FILE = (
        SPATIAL_DIR /
        "cluster_centroids_all_eps.csv"
    )

    centroids.to_csv(
        CENTROIDS_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("CLUSTERING COMPLETE")
    print("=" * 72)

    print("\nCreated:")
    print(
        f"  {ALL_ASSIGNMENTS_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {DIAGNOSTICS_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {SUMMARY_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        f"  {CENTROIDS_FILE.relative_to(PROJECT_DIR)}"
    )
    print(
        "  data/spatial/hospital_clusters_eps10.csv"
    )
    print(
        "  data/spatial/hospital_clusters_eps15.csv"
    )
    print(
        "  data/spatial/hospital_clusters_eps20.csv"
    )

    # Compact comparison for the latest year.
    latest_year = (
        year_summary["year"]
        .max()
    )

    latest = year_summary.loc[
        year_summary["year"]
        == latest_year
    ].sort_values("eps_km")

    print()
    print(
        f"Latest-year threshold comparison "
        f"({int(latest_year)}):"
    )

    print(
        latest[
            [
                "eps_km",
                "hospitals",
                "service_clusters",
                "multi_hospital_clusters",
                "singleton_clusters",
                "pct_hospitals_in_multi_clusters",
                "largest_cluster_size",
                "mean_hospitals_per_service_cluster",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
