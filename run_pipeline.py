#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end pipeline for Sentinel-2 rule-based semantic clustering.

Reproduces the results reported in:

    Ramírez et al. (2026). "Sentinel-2 Rule-based Semantic Clustering
    for Zone Identification Supporting Nature-Based Solutions in
    Valencia, Spain."

Usage
-----
Edit the USER CONFIGURATION block below to point to your GeoTIFF files,
then run:

    python run_pipeline.py

Output figures (in order)
--------------------------
1. k-means cluster map (before cloud masking)          → Sentinel2d palette
2. k-means cluster map (after SCL cloud masking)       → Sentinel2d palette
3. Semantic classification map (full scene)            → Sentinel2b palette
4. SCL classification map                              → scl palette
5. Semantic map — valley zone  (DEM ≤ 300 m)           → Sentinel2b palette
6. Semantic map — upland zone  (DEM ≥ 300 m)           → Sentinel2b palette

Area statistics are printed to stdout.

Data requirements
-----------------
S2_PATH  : Multi-band Sentinel-2 Level-2A GeoTIFF at 20 m (13 layers).
           Band order is documented in src/features.py.
DEM_PATH : Single-band GeoTIFF co-registered with the Sentinel-2 grid,
           elevation values in metres.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from skimage import exposure

from src.palettes          import label2color
from src.features          import build_features, to_dataframe, preprocess
from src.clustering        import run_kmeans, apply_cloud_mask, compute_cluster_stats
from src.semantic_labeling import (compute_otsu_thresholds, label_clusters,
                                   build_semantic_map, area_summary)

# ===========================================================================
# USER CONFIGURATION
# ===========================================================================

S2_PATH    = "data/2026-04-19_B1B2B3B4.tif"   # Sentinel-2 GeoTIFF
DEM_PATH   = "data/DEM_model.tif"              # DEM GeoTIFF

DATE_LABEL = "April 19, 2026"    # displayed in figure titles
K          = 9                   # number of k-means clusters
DEM_THRESH = 300                 # elevation boundary (metres)

# ===========================================================================
# Step 0 — Load data
# ===========================================================================

print(f"Loading Sentinel-2 image : {S2_PATH}")
data = tifffile.imread(S2_PATH)

print(f"Loading DEM              : {DEM_PATH}")
dem = tifffile.imread(DEM_PATH)

# ===========================================================================
# Step 1 — Feature extraction
# ===========================================================================

print("Extracting features …")
feat = build_features(data)
nx, ny, nz = feat.shape            # spatial dims and number of features

df = to_dataframe(feat)            # shape (nx*ny, 17), columns = ALL_COLS

# ===========================================================================
# Step 2 — Preprocessing and k-means clustering
# ===========================================================================

print("Preprocessing (median imputation + RobustScaler) …")
X = preprocess(df)                 # shape (nx*ny, 16), SCL excluded

print(f"Running k-means (k={K}) …")
df["cluster"] = run_kmeans(X, k=K)

# Figure 1 — raw cluster map (before cloud masking)
labelsx = df["cluster"].to_numpy()
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(label2color(labelsx.reshape(nx, ny), 'Sentinel2d'))
ax.axis("off")
ax.set_title(DATE_LABEL)
plt.tight_layout()
plt.show()

# ===========================================================================
# Step 3 — SCL cloud masking
# ===========================================================================

print("Applying SCL cloud mask …")
df = apply_cloud_mask(df)          # sets cluster=36 for SCL {3,8,9,10}

# Figure 2 — cloud-corrected cluster map
labelsx = df["cluster"].to_numpy()
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(label2color(labelsx.reshape(nx, ny), 'Sentinel2d'))
ax.axis("off")
ax.set_title(DATE_LABEL)
plt.tight_layout()
plt.show()

# ===========================================================================
# Step 4 — Cluster statistics and Otsu thresholds
# ===========================================================================

print("Computing cluster statistics …")
cluster_stats = compute_cluster_stats(df)

print("Computing Otsu thresholds …")
t = compute_otsu_thresholds(df, nx, ny)

print("\nOtsu thresholds:")
for name, vals in t.items():
    print(f"  {name:10s}: {[round(v, 4) for v in vals]}")

# ===========================================================================
# Step 5 — Semantic labeling
# ===========================================================================

print("\nAssigning semantic labels …")
cluster_to_label = label_clusters(cluster_stats, t)

print("\nCluster → semantic label:")
for cid in sorted(cluster_to_label):
    print(f"  Cluster {cid:2d} → {cluster_to_label[cid]}")

labels1 = df['cluster'].to_numpy()
labels  = build_semantic_map(labels1, cluster_to_label)

# ===========================================================================
# Step 6 — Outputs
# ===========================================================================

# Figure 3 — full-scene semantic map
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(label2color(labels.reshape(nx, ny), 'Sentinel2b'))
ax.axis("off")
ax.set_title(DATE_LABEL)
plt.tight_layout()
plt.show()

# Figure 4 — SCL map
scl_map = label2color(df['scl'].to_numpy().reshape(nx, ny), 'scl')
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(scl_map)
ax.axis("off")
ax.set_title(DATE_LABEL)
plt.tight_layout()
plt.show()

# --- DEM-stratified maps ---
lab = labels.reshape(nx, ny)

# Valley zone: pixels with DEM <= threshold keep their semantic label;
# upland pixels are set to 7 (black in Sentinel2b palette).
vl_model = np.zeros(dem.shape, dtype=int)
vl_model[dem >  DEM_THRESH] = 7
vl_model[dem <= DEM_THRESH] = lab[dem <= DEM_THRESH]

# Figure 5 — valley semantic map
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label2color(vl_model, 'Sentinel2b'))
ax.axis("off")
plt.tight_layout()
plt.show()

# Upland zone: pixels with DEM >= threshold keep their semantic label;
# valley pixels are set to 7 (black).
mo_model = np.zeros(dem.shape, dtype=int)
mo_model[dem <  DEM_THRESH] = 7
mo_model[dem >= DEM_THRESH] = lab[dem >= DEM_THRESH]

# Figure 6 — upland semantic map
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label2color(mo_model, 'Sentinel2b'))
ax.axis("off")
plt.tight_layout()
plt.show()

# --- Area statistics ---
print("\nClass area summary:")
print(area_summary(labels).to_string(index=False))

# --- Valley area (hectares) ---
ha_valley = int(np.sum(dem < DEM_THRESH) * 4)
print(f"\nValley area (DEM < {DEM_THRESH} m): {ha_valley:,} ha")

print("\nDone.")
