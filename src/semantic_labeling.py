"""
semantic_labeling.py
====================
Data-adaptive semantic labeling of k-means clusters using multi-class
Otsu thresholding.

Workflow
--------
1. For each spectral index, compute scene-level Otsu thresholds from
   the pixel-level index distribution (excluding cloud pixels).
2. Evaluate the cluster-level median statistics of each cluster against
   the sequential decision rules in :func:`assign_label`.
3. Map cluster IDs to semantic class IDs in the output label array.

Semantic classes and integer encodings
---------------------------------------
ID  Class name          Color in figures
--  ------------------  ----------------
 0  Sparse vegetation   Light green
 1  Bare soil / Rock    Light brown
 2  Urban               Red
 3  Water               Blue
 4  Dense vegetation    Dark green
 5  Uncertain           Magenta
 6  Clouds              White

The precedence order of the rules is: clouds → water → dense vegetation
→ sparse vegetation → urban → bare soil/rock → uncertain.
Each cluster is assigned the label of the first rule whose conditions
are satisfied.

References
----------
Otsu (1979) – threshold_multiotsu.
van der Walt et al. (2014) – scikit-image.
"""

import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu


# ---------------------------------------------------------------------------
# Class encoding
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Sparse vegetation",
    1: "Bare soil / Rock",
    2: "Urban",
    3: "Water",
    4: "Dense vegetation",
    5: "Uncertain",
    6: "Clouds",
}

# Pixel area at 20-m spatial resolution (km²)
PIXEL_AREA_KM2: float = (20 ** 2) / 1e6   # = 0.0004 km²


# ---------------------------------------------------------------------------
# Step 1 – Otsu threshold computation
# ---------------------------------------------------------------------------

def compute_otsu_thresholds(df: pd.DataFrame,
                             spatial_shape: tuple) -> dict:
    """
    Compute multi-class Otsu thresholds for each spectral index.

    Each index is partitioned into m classes, yielding m-1 thresholds
    that maximise the inter-class variance of the scene-level index
    distribution.  The number of classes per index was chosen to match
    the number of distinguishable spectral regimes each index typically
    exhibits in a heterogeneous Mediterranean landscape.

    Index     Classes  Thresholds returned
    -------   -------  -------------------
    NDVI        4       θ1 < θ2 < θ3
    SWIR1       2       θ1
    MNDWI       2       θ1
    NDRE        2       θ1
    BSI         2       θ1
    NDBI        3       θ1 < θ2
    UI          3       θ1 < θ2
    SVC         3       θ1 < θ2

    Parameters
    ----------
    df : pd.DataFrame
        Pixel-level DataFrame containing at least the columns:
        'ndvi', 'swir1', 'mndwi', 'ndre', 'bsi', 'ndbi', 'ui', 'svc'.
    spatial_shape : tuple (H, W)
        Spatial dimensions of the scene (required by threshold_multiotsu
        which operates on 2-D arrays).

    Returns
    -------
    thresholds : dict
        Maps index name to a 1-D NumPy array of threshold values.
    """
    H, W = spatial_shape

    def _otsu(col: str, classes: int) -> np.ndarray:
        img = df[col].to_numpy().reshape(H, W)
        return threshold_multiotsu(img, classes=classes)

    return {
        "ndvi":  _otsu("ndvi",  classes=4),   # θ1, θ2, θ3
        "swir1": _otsu("swir1", classes=2),   # θ1
        "mndwi": _otsu("mndwi", classes=2),   # θ1
        "ndre":  _otsu("ndre",  classes=2),   # θ1
        "bsi":   _otsu("bsi",   classes=2),   # θ1
        "ndbi":  _otsu("ndbi",  classes=3),   # θ1, θ2
        "ui":    _otsu("ui",    classes=3),    # θ1, θ2
        "svc":   _otsu("svc",   classes=3),   # θ1, θ2
    }


# ---------------------------------------------------------------------------
# Step 2 – Per-cluster semantic labeling
# ---------------------------------------------------------------------------

def assign_label(row: pd.Series, t: dict) -> str:
    """
    Apply the sequential decision rules to a single cluster's median
    spectral statistics.

    Parameters
    ----------
    row : pd.Series
        Cluster-level median values.  Required keys match the columns
        listed in :func:`compute_otsu_thresholds`.
    t : dict
        Otsu threshold dictionary returned by
        :func:`compute_otsu_thresholds`.

    Returns
    -------
    str
        One of: 'clouds', 'water', 'dense_vegetation',
        'sparse_vegetation', 'urban', 'bare_soil_rock', 'uncertain'.
    """
    # Rule 1 – Clouds (SCL-based; evaluated first)
    if 8.0 <= row["scl"] <= 10.0:
        return "clouds"

    # Rule 2 – Water
    # High MNDWI, low NDVI, low SWIR1
    if (row["mndwi"] > t["mndwi"][0] and
            row["ndvi"]  < t["ndvi"][0]  and
            row["swir1"] < t["swir1"][0]):
        return "water"

    # Rule 3 – Dense vegetation
    # High NDVI (above upper threshold), active chlorophyll (NDRE),
    # low built-up index (NDBI)
    if (row["ndvi"]  > t["ndvi"][2]  and
            row["ndre"]  > t["ndre"][0]  and
            row["ndbi"]  < t["ndbi"][0]):
        return "dense_vegetation"

    # Rule 4 – Sparse vegetation
    # Intermediate NDVI range
    if t["ndvi"][1] <= row["ndvi"] <= t["ndvi"][2]:
        return "sparse_vegetation"

    # Rule 5 – Urban / impervious
    # Low NDVI, elevated BSI and UI, suppressed SVC and MNDWI
    if (row["ndvi"]  < t["ndvi"][1]  and
            row["bsi"]   > t["bsi"][0]   and
            row["ui"]    > t["ui"][1]    and
            row["svc"]   < t["svc"][1]   and
            row["mndwi"] < t["mndwi"][0]):
        return "urban"

    # Rule 6 – Bare soil / Rock
    # Low NDVI, elevated BSI, negative MNDWI
    if (row["ndvi"]  < t["ndvi"][1]  and
            row["bsi"]   > t["bsi"][0]   and
            row["mndwi"] < t["mndwi"][0]):
        return "bare_soil_rock"

    # Rule 7 – Uncertain (fallback)
    return "uncertain"


def label_clusters(cluster_stats: pd.DataFrame,
                   thresholds: dict) -> dict:
    """
    Apply :func:`assign_label` to every row of the cluster statistics
    table and return a cluster-ID → semantic-label mapping.

    Parameters
    ----------
    cluster_stats : pd.DataFrame
        Output of :func:`~clustering.compute_cluster_statistics`.
    thresholds : dict
        Output of :func:`compute_otsu_thresholds`.

    Returns
    -------
    cluster_to_label : dict
        Maps integer cluster ID to a semantic label string.
    """
    return {
        cid: assign_label(row, thresholds)
        for cid, row in cluster_stats.iterrows()
    }


# ---------------------------------------------------------------------------
# Step 3 – Map cluster labels to integer class IDs
# ---------------------------------------------------------------------------

_LABEL_TO_ID = {
    "sparse_vegetation": 0,
    "bare_soil_rock":    1,
    "urban":             2,
    "water":             3,
    "dense_vegetation":  4,
    "uncertain":         5,
    "clouds":            6,
}


def build_semantic_map(cluster_labels: np.ndarray,
                       cluster_to_label: dict) -> np.ndarray:
    """
    Replace cluster IDs with integer semantic class IDs.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (N,)
        Flat array of cluster IDs (cloud-masked).
    cluster_to_label : dict
        Output of :func:`label_clusters`.

    Returns
    -------
    semantic_labels : np.ndarray, shape (N,), dtype uint8
        Semantic class IDs aligned with :data:`CLASS_NAMES`.
    """
    semantic_labels = np.full(cluster_labels.shape, 5, dtype=np.uint8)

    for cid, name in cluster_to_label.items():
        class_id = _LABEL_TO_ID[name]
        semantic_labels[cluster_labels == cid] = class_id

    # Cloud pixels carry CLOUD_LABEL (36); remap to class ID 6
    from clustering import CLOUD_LABEL
    semantic_labels[cluster_labels == CLOUD_LABEL] = 6

    return semantic_labels


# ---------------------------------------------------------------------------
# Utility – area summary
# ---------------------------------------------------------------------------

def area_summary(semantic_labels: np.ndarray) -> pd.DataFrame:
    """
    Compute the area (km²) covered by each semantic class.

    Parameters
    ----------
    semantic_labels : np.ndarray, shape (N,)
        Output of :func:`build_semantic_map`.

    Returns
    -------
    pd.DataFrame
        Columns: 'Class', 'Pixels', 'Area (km²)'.
        Includes a TOTAL row at the bottom.
    """
    unique, counts = np.unique(semantic_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))

    rows = [
        {
            "Class":      CLASS_NAMES[cid],
            "Pixels":     count_dict.get(cid, 0),
            "Area (km²)": round(count_dict.get(cid, 0) * PIXEL_AREA_KM2, 4),
        }
        for cid in sorted(CLASS_NAMES)
    ]

    df = pd.DataFrame(rows)
    total = pd.DataFrame([{
        "Class":      "TOTAL",
        "Pixels":     df["Pixels"].sum(),
        "Area (km²)": round(df["Area (km²)"].sum(), 4),
    }])
    return pd.concat([df, total], ignore_index=True)
