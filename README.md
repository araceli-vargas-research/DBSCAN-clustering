# Rural Hospital Spatial Competition and Consolidation Research

This repository contains the current research pipeline for a national analysis of rural hospital competition, geographic isolation, ownership concentration, and consolidation in the United States.

The project builds a 2019–2026 hospital panel from CMS data, geocodes hospital locations, converts latitude/longitude coordinates into projected coordinates using region-specific coordinate systems, and measures hospital proximity using Euclidean nearest-neighbour distances. Hospitals are then grouped into spatial service clusters using DBSCAN across alternative distance thresholds.

The analysis combines these spatial measures with USDA Rural-Urban Continuum Codes, health-system and corporate-parent ownership information, conventional Herfindahl-Hirschman Index measures, CMS hospital ratings, and HCAHPS patient-experience outcomes.

## Main Results Thus Far

The national hospital panel contains 42,415 hospital-year observations and 5,922 unique hospitals from 2019 through 2026.

Using a 10 km baseline DBSCAN specification, rural hospital clusters are substantially more geographically isolated than metropolitan clusters. In 2026, the median distance to the nearest competing cluster was approximately 31.3 km for rural-only clusters compared with 21.8 km for metro-only clusters.

Regression estimates show that rural hospital clusters remain significantly farther from competing clusters after controlling for cluster size and year effects. At the 10 km specification, rural clusters were associated with approximately 42% greater distance to the nearest competing service cluster.

Spatial isolation and conventional ownership concentration appear to capture different dimensions of hospital competition. In the 2023 cross-section, correlations between HHI and distance-based isolation measures were small, while rural hospitals remained substantially more geographically isolated after controlling for HHI.

Greater distance to the nearest competing hospital cluster was also associated with lower CMS overall hospital ratings and lower HCAHPS patient-experience measures. These relationships remained negative and statistically significant across 5 km, 10 km, and 15 km clustering thresholds, alternative functional forms of distance, and both count- and bed-weighted HHI measures.

These results should currently be interpreted as associations rather than causal effects.

## Current Research Direction

The next stage extends the spatial competition framework into a rural hospital consolidation atlas. The project is identifying verified ownership transitions and acquisition events, measuring how those events change the distance to independently owned competitors, and evaluating whether post-acquisition outcomes differ according to pre-merger geographic isolation and the availability of meaningful outside options.

Future extensions include road-network travel time, patient-origin flows, predicted hospital diversion, and event-study or difference-in-differences designs for hospital acquisitions and mergers.

# DBSCAN-clustering
Converts geographic coordinates into projected coordinates using pyproj. This uses  separate coordinate systems for the contiguous U.S., Alaska, Hawaii, and Puerto Rico/U.S. Virgin Islands, and then applies Euclidean nearest-neighbour calculations.
