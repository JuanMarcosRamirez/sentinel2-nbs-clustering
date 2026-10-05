#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end pipeline for Sentinel-2 rule-based semantic clustering.

This script reproduces the results reported in:

    Ramírez et al. (2026). "Sentinel-2 Rule-based Semantic Clustering
    for Zone Identification Supporting Nature-Based Solutions in
    Valencia, Spain."

Usage
-----
Edit the ``# --- USER CONFIGURATION ---`` block below to point to your
Sentinel-2 GeoTIFF and DEM files, then run:

    python run_pipeline.py

All intermediate and final outputs are displayed as Matplotlib figures.
Area statistics are printed to stdout.

Data requirements
-----------------
Sentinel-2 GeoTIFF (``S2_PATH``):
    Multi-band GeoTIFF at 20-m resolution containing 10 layers in the
    band order documented in ``src/features.py``.

DEM GeoTIFF (``DEM_PATH``):
    Single-band GeoTIFF co-registered with the Sentinel-2 grid,
    containing elevation values in metres.
"""

import numpy as np
import tifffile

from src.features          import build_feature_matrix, FEATURE_NAMES
from src.clustering        import (preprocess, run_kmeans,
                                   apply_cloud_mask,
                                   compute_cluster_statistics)
from src.semantic_labeling import (compute_otsu_thresholds,
                                   label_clusters,
                                   build_semantic_map,
                                   area_summary)
from src.visualization     import (plot_rgb_composite,
                                   plot_cluster_map,
                                   plot_semantic_map,
                                   plot_elevation_stratified)

# ---------------------------------------------------------------------------
# --- USER CONFIGURATION ---
# ---------------------------------------------------------------------------

# Path to the Sentinel-2 Level-2A multi-band GeoTIFF (20 m, 10 bands)
S2_PATH  = "data/sample/S2_20260419.tif"

# Path to the co-registered DEM GeoTIFF (elevation in metres)
DEM_PATH = "data/sample/DEM_model.tif"

# Acquisition date label (used in figure titles)
DATE_LABEL = "April 19, 2026"

# Number of k-means clusters
# k = 9 was selected based on cluster validity analysis (see paper §4.1)
K = 9

# Elevation threshold separating valley from upland zones (metres)
ELEVATION_THRESHOLD = 300

# ---------------------------------------------------------------------------
# Step 0 – Load data
# ---------------------------------------------------------------------------

print(f"[1/6] Loading Sentinel-2 image: {S2_PATH}")
raw = tifffile.imread(S2_PATH)         # shape (H, W, 10)

print(f"      Scene size: {raw.shape[0]} × {raw.shape[1]} pixels")

print(f"[2/6] Loading DEM: {DEM_PATH}")
dem = tifffile.imread(DEM_PATH)        # shape (H, W)

# ---------------------------------------------------------------------------
# Step 1 – Feature extraction
# ---------------------------------------------------------------------------

print("[3/6] Extracting spectral bands and computing indices …")
X_raw, scl, (H, W) = build_feature_matrix(raw)

# Display the true-color composite as a visual reference
plot_rgb_composite(raw, title=f"True-color composite — {DATE_LABEL}")

# ---------------------------------------------------------------------------
# Step 2 – Preprocessing and k-means clustering
# ---------------------------------------------------------------------------

print("[4/6] Preprocessing (median imputation + RobustScaler) …")
X_scaled = preprocess(X_raw)

print(f"      Running k-means with k = {K} …")
cluster_labels = run_kmeans(X_scaled, k=K)

# Display the raw (unsupervised) cluster map
plot_cluster_map(cluster_labels, (H, W),
                 title=f"k-means clusters (k={K}) — {DATE_LABEL}")

# ---------------------------------------------------------------------------
# Step 3 – Cloud masking
# ---------------------------------------------------------------------------

print("[5/6] Applying SCL-based cloud mask …")
cluster_labels_masked = apply_cloud_mask(cluster_labels, scl)

plot_cluster_map(cluster_labels_masked, (H, W),
                 title=f"Cloud-corrected clusters — {DATE_LABEL}")

# ---------------------------------------------------------------------------
# Step 4 – Cluster statistics and Otsu thresholding
# ---------------------------------------------------------------------------

print("      Computing cluster-level median statistics …")
cluster_stats = compute_cluster_statistics(X_raw, cluster_labels_masked,
                                           FEATURE_NAMES)

# Add the SCL median to cluster_stats so the cloud rule can be evaluated
import pandas as pd
scl_df = pd.DataFrame({"scl": scl, "cluster": cluster_labels_masked})
scl_stats = (scl_df[scl_df["cluster"] != 36]
             .groupby("cluster")["scl"]
             .median())
cluster_stats["scl"] = scl_stats

print("      Computing Otsu thresholds …")
# Build a pixel-level DataFrame of non-cloud index values for thresholding
non_cloud_mask = cluster_labels_masked != 36
df_pixels = pd.DataFrame(X_raw[non_cloud_mask], columns=FEATURE_NAMES)
thresholds = compute_otsu_thresholds(df_pixels, (H, W))

print("\n      Otsu thresholds:")
for idx, vals in thresholds.items():
    formatted = ", ".join(f"{v:.4f}" for v in vals)
    print(f"        {idx:8s}: [{formatted}]")

# ---------------------------------------------------------------------------
# Step 5 – Semantic labeling
# ---------------------------------------------------------------------------

print("\n[6/6] Assigning semantic labels …")
cluster_to_label = label_clusters(cluster_stats, thresholds)

print("\n      Cluster → semantic label mapping:")
for cid in sorted(cluster_to_label):
    print(f"        Cluster {cid:2d} → {cluster_to_label[cid]}")

semantic_labels = build_semantic_map(cluster_labels_masked,
                                     cluster_to_label)

# ---------------------------------------------------------------------------
# Step 6 – Outputs
# ---------------------------------------------------------------------------

# Full-scene semantic map
plot_semantic_map(semantic_labels, (H, W),
                  title=f"Semantic classification — {DATE_LABEL}")

# Elevation-stratified maps
plot_elevation_stratified(
    semantic_labels.reshape(H, W),
    dem,
    elevation_threshold=ELEVATION_THRESHOLD,
    title_valley=f"Valley zone (z < {ELEVATION_THRESHOLD} m) — {DATE_LABEL}",
    title_upland=f"Upland zone (z ≥ {ELEVATION_THRESHOLD} m) — {DATE_LABEL}",
)

# Area statistics
print("\n      Class area summary:")
print(area_summary(semantic_labels).to_string(index=False))

print("\nDone.")
