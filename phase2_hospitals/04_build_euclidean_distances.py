#!/usr/bin/env python3
"""
04_build_euclidean_distances.py

Create projected Euclidean hospital coordinates and nearest-neighbor
distance measures from the geocoded hospital panel.

Place directly inside:
    phase2_hospitals/

Expected input:
    data/spatial/hospital_panel_geocoded.csv

Creates:
    data/spatial/hospital_spatial_points.csv
    data/spatial/hospital_nearest_neighbors.csv
    data/spatial/hospital_spatial_panel.csv
    data/spatial/spatial_distance_summary.csv

Run:
    pip3 install pandas numpy pyproj scikit-learn
    python3 04_build_euclidean_distances.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors


PROJECT_DIR = Path(__file__).resolve().parent
SPATIAL_DIR = PROJECT_DIR / "data" / "spatial"

INPUT_FILE = SPATIAL_DIR / "hospital_panel_geocoded.csv"
POINTS_FILE = SPATIAL_DIR / "hospital_spatial_points.csv"
NEIGHBORS_FILE = SPATIAL_DIR / "hospital_nearest_neighbors.csv"
PANEL_FILE = SPATIAL_DIR / "hospital_spatial_panel.csv"
SUMMARY_FILE = SPATIAL_DIR / "spatial_distance_summary.csv"


REGION_CRS = {
    "CONUS": "EPSG:5070",
    "AK": "EPSG:3338",
    "HI": "EPSG:3759",
    "PR_VI": "EPSG:32161",
}

OTHER_TERRITORIES = {"GU", "MP", "AS"}


def projection_region(state: str) -> str:
    s = str(state).strip().upper()

    if s == "AK":
        return "AK"
    if s == "HI":
        return "HI"
    if s in {"PR", "VI"}:
        return "PR_VI"
    if s in OTHER_TERRITORIES:
        return "UNPROJECTED_TERRITORY"
    return "CONUS"


def project_group(df: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()

    if region not in REGION_CRS:
        out["projected_crs"] = pd.NA
        out["x_m"] = np.nan
        out["y_m"] = np.nan
        return out

    crs = REGION_CRS[region]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        out["longitude"].to_numpy(),
        out["latitude"].to_numpy(),
    )

    out["projected_crs"] = crs
    out["x_m"] = x
    out["y_m"] = y

    return out


def build_neighbors_for_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy().reset_index(drop=True)

    n = len(group)
    if n < 2:
        return pd.DataFrame()

    coords = group[["x_m", "y_m"]].to_numpy(dtype=float)

    k = min(4, n)

    model = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    )
    model.fit(coords)

    distances, indices = model.kneighbors(coords)

    rows = []

    for i in range(n):
        source = group.iloc[i]
        candidates = []

        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            target = group.iloc[idx]
            candidates.append((float(dist), target))

        row = {
            "facility_id": source.get("facility_id"),
            "facility_name": source.get("facility_name"),
            "year": source.get("year"),
            "snapshot": source.get("snapshot"),
            "state": source.get("state"),
            "county": source.get("county"),
            "projection_region": source.get("projection_region"),
            "projected_crs": source.get("projected_crs"),
            "x_m": source.get("x_m"),
            "y_m": source.get("y_m"),
            "latitude": source.get("latitude"),
            "longitude": source.get("longitude"),
        }

        for rank in range(1, 4):
            if rank <= len(candidates):
                dist_m, target = candidates[rank - 1]

                row[f"nn{rank}_facility_id"] = target.get("facility_id")
                row[f"nn{rank}_facility_name"] = target.get("facility_name")
                row[f"nn{rank}_distance_m"] = dist_m
                row[f"nn{rank}_distance_km"] = dist_m / 1000.0
                row[f"nn{rank}_distance_miles"] = dist_m / 1609.344
            else:
                row[f"nn{rank}_facility_id"] = pd.NA
                row[f"nn{rank}_facility_name"] = pd.NA
                row[f"nn{rank}_distance_m"] = np.nan
                row[f"nn{rank}_distance_km"] = np.nan
                row[f"nn{rank}_distance_miles"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}\n"
            "Run the geocoding steps first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    print("=" * 72)
    print("BUILD EUCLIDEAN HOSPITAL DISTANCES")
    print("=" * 72)
    print(f"Hospital-year rows: {len(df):,}")

    valid = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df["spatial_sample"] = valid.astype(int)

    matched = df.loc[valid].copy()
    unmatched = df.loc[~valid].copy()

    print(f"Rows with coordinates:    {len(matched):,}")
    print(f"Rows without coordinates: {len(unmatched):,}")
    print(
        f"Spatial row coverage:     "
        f"{100 * len(matched) / len(df):.2f}%"
    )

    matched["projection_region"] = (
        matched["state"]
        .fillna("")
        .map(projection_region)
    )

    projected_parts = []

    for region, group in matched.groupby(
        "projection_region",
        dropna=False,
    ):
        projected_parts.append(
            project_group(group, region)
        )

    points = pd.concat(
        projected_parts,
        ignore_index=True,
        sort=False,
    )

    points["projectable"] = (
        points["x_m"].notna()
        & points["y_m"].notna()
    )

    print("\nProjection-region counts:")
    print(
        points["projection_region"]
        .value_counts(dropna=False)
        .to_string()
    )

    point_cols = [
        "facility_id",
        "facility_name",
        "year",
        "snapshot",
        "address",
        "city",
        "state",
        "zip_code",
        "county",
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
        "latitude",
        "longitude",
        "projection_region",
        "projected_crs",
        "x_m",
        "y_m",
        "geocode_method",
        "spatial_sample",
    ]

    point_cols = [c for c in point_cols if c in points.columns]

    points[point_cols].to_csv(
        POINTS_FILE,
        index=False,
    )

    neighbor_parts = []

    projectable = points.loc[
        points["projectable"]
    ].copy()

    for (year, region), group in projectable.groupby(
        ["year", "projection_region"],
        dropna=False,
    ):
        print(
            f"Computing neighbors: "
            f"year={year}, region={region}, N={len(group):,}"
        )

        result = build_neighbors_for_group(group)

        if not result.empty:
            neighbor_parts.append(result)

    if neighbor_parts:
        neighbors = pd.concat(
            neighbor_parts,
            ignore_index=True,
            sort=False,
        )
    else:
        neighbors = pd.DataFrame()

    neighbors.to_csv(
        NEIGHBORS_FILE,
        index=False,
    )

    key_cols = [
        "facility_id",
        "year",
        "snapshot",
    ]

    distance_cols = [
        c for c in neighbors.columns
        if c.startswith("nn")
    ]

    if not neighbors.empty:
        distance_merge = neighbors[
            key_cols + distance_cols
        ].copy()

        full = df.merge(
            distance_merge,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    else:
        full = df.copy()

    xy = points[
        [
            "facility_id",
            "year",
            "snapshot",
            "projection_region",
            "projected_crs",
            "x_m",
            "y_m",
        ]
    ].copy()

    full = full.merge(
        xy,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    full.to_csv(
        PANEL_FILE,
        index=False,
    )

    full["nn1_distance_km"] = pd.to_numeric(
        full.get("nn1_distance_km"),
        errors="coerce",
    )

    summary = (
        full.groupby(
            ["year", "snapshot"],
            dropna=False,
        )
        .agg(
            hospital_rows=("facility_id", "size"),
            spatial_rows=("spatial_sample", "sum"),
            hospitals_with_nn1=(
                "nn1_distance_km",
                lambda s: s.notna().sum(),
            ),
            median_nn1_km=("nn1_distance_km", "median"),
            mean_nn1_km=("nn1_distance_km", "mean"),
            p25_nn1_km=(
                "nn1_distance_km",
                lambda s: s.quantile(0.25),
            ),
            p75_nn1_km=(
                "nn1_distance_km",
                lambda s: s.quantile(0.75),
            ),
        )
        .reset_index()
    )

    summary["spatial_coverage_pct"] = (
        100
        * summary["spatial_rows"]
        / summary["hospital_rows"]
    ).round(2)

    for col in [
        "median_nn1_km",
        "mean_nn1_km",
        "p25_nn1_km",
        "p75_nn1_km",
    ]:
        summary[col] = summary[col].round(3)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("EUCLIDEAN DISTANCE BUILD COMPLETE")
    print("=" * 72)

    print("\nCreated:")
    for path in [
        POINTS_FILE,
        NEIGHBORS_FILE,
        PANEL_FILE,
        SUMMARY_FILE,
    ]:
        print(f"  {path.relative_to(PROJECT_DIR)}")

    print("\nYear summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
