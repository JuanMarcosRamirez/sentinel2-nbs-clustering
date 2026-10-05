"""
semantic_labeling.py
====================
Data-adaptive semantic labeling of k-means clusters using multi-class
Otsu thresholding.

Otsu threshold computation
--------------------------
For each spectral index, :func:`threshold_multiotsu` is applied to the
2-D spatial image of that index (not to a 1-D histogram) to compute
thresholds that maximise inter-class variance.  The number of classes
per index matches the original code exactly:

    Index    Classes  Variable name   Thresholds
    -------  -------  --------------  --------------------
    NDVI       4      ``ndvi``        ndvi[0], ndvi[1], ndvi[2]
    SWIR1      2      ``swir1``       swir1[0]
    MNDWI      2      ``mndwi``       mndwi[0]
    NDRE_B5    2      ``ndre_b5``     ndre_b5[0]
    NDBI       3      ``ndbi``        ndbi[0], ndbi[1]
    BSI        2      ``bsi``         bsi[0]
    UI         3      ``ui``          ui[0], ui[1]
    SVC        3      ``svc``         svc[0], svc[1]

Important: in the original code ``bsi`` thresholds are computed from the
``ndbi`` column (``threshold_multiotsu(df['ndbi']..., classes=2)``).
This is reproduced exactly here and noted explicitly.

Decision rules (precedence order)
----------------------------------
Rules are evaluated sequentially.  The first rule that fires wins.

    1. Clouds        scl in {8, 9, 10}  (note: SCL 3 handled by mask)
    2. Water         mndwi > mndwi[0] AND ndvi < ndvi[0] AND swir1 < swir1[0]
    3. Dense veg.    ndvi > ndvi[2]  AND ndre_b5 > ndre_b5[0] AND ndbi < ndbi[0]
    4. Sparse veg.   ndvi[1] <= ndvi <= ndvi[2]
    5. Urban         ndvi < ndvi[1]  AND bsi > bsi[0] AND ui > ui[1]
                     AND svc < svc[1] AND mndwi < mndwi[0]
    6. Bare soil     ndvi < ndvi[1]  AND bsi > bsi[0] AND mndwi < mndwi[0]
    7. Uncertain     (fallback)

Semantic class integer encoding (Sentinel2b palette indices)
------------------------------------------------------------
    0  sparse_vegetation
    1  bare_soil_rock
    2  urban
    3  water
    4  dense_vegetation
    5  uncertain
    6  clouds

References
----------
Otsu (1979)                  – threshold_multiotsu
van der Walt et al. (2014)   – scikit-image
"""

import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu


# ---------------------------------------------------------------------------
# Class encoding
# ---------------------------------------------------------------------------

#: Maps semantic label strings to integer IDs (Sentinel2b palette indices).
LABEL_TO_ID = {
    'sparse_vegetation': 0,
    'bare_soil_rock':    1,
    'urban':             2,
    'water':             3,
    'dense_vegetation':  4,
    'uncertain':         5,
    'clouds':            6,
}

#: Human-readable class names keyed by integer ID.
LABEL_NAMES = {
    0: "Sparse vegetation",
    1: "Bare soil / Rock",
    2: "Urban",
    3: "Water",
    4: "Dense vegetation",
    5: "Uncertain",
    6: "Clouds",
}

#: Pixel area at 20-m spatial resolution in km².
PIXEL_AREA_KM2: float = (20 ** 2) / 1e6   # = 0.0004 km²


# ---------------------------------------------------------------------------
# Step 1 — Otsu threshold computation
# ---------------------------------------------------------------------------

def compute_otsu_thresholds(df: pd.DataFrame,
                             nx: int, ny: int) -> dict:
    """
    Compute multi-class Otsu thresholds for each spectral index by
    applying :func:`~skimage.filters.threshold_multiotsu` to the 2-D
    spatial image of each index.

    Parameters
    ----------
    df : pd.DataFrame
        Full pixel DataFrame (including all pixels, cloud and non-cloud).
        Must contain columns: 'ndvi', 'swir1', 'mndwi', 'ndre_b5',
        'ndbi', 'ui', 'svc'.
    nx : int
        Number of rows (image height).
    ny : int
        Number of columns (image width).

    Returns
    -------
    thresholds : dict
        Maps index name to a 1-D NumPy array of threshold values.
        Keys: 'ndvi', 'swir1', 'mndwi', 'ndre_b5', 'ndbi', 'bsi',
              'ui', 'svc'.

    Notes
    -----
    ``bsi`` thresholds are computed from the ``ndbi`` column — this
    reproduces the original implementation exactly.
    """
    def _otsu(col: str, classes: int) -> np.ndarray:
        img = df[col].to_numpy().reshape(nx, ny)
        return threshold_multiotsu(img, classes=classes)

    return {
        'ndvi':    _otsu('ndvi',    classes=4),   # → [θ1, θ2, θ3]
        'swir1':   _otsu('swir1',   classes=2),   # → [θ1]
        'mndwi':   _otsu('mndwi',   classes=2),   # → [θ1]
        'ndre_b5': _otsu('ndre_b5', classes=2),   # → [θ1]
        'ndbi':    _otsu('ndbi',    classes=3),   # → [θ1, θ2]
        'bsi':     _otsu('ndbi',    classes=2),   # → [θ1]  (uses ndbi column)
        'ui':      _otsu('ui',      classes=3),   # → [θ1, θ2]
        'svc':     _otsu('svc',     classes=3),   # → [θ1, θ2]
    }


# ---------------------------------------------------------------------------
# Step 2 — Decision rules applied to cluster-level medians
# ---------------------------------------------------------------------------

def _is_cloud(row: pd.Series, t: dict) -> bool:
    """SCL class in {8, 9, 10} (cloud shadow class 3 handled by mask)."""
    return 8.0 <= row['scl'] <= 10.0


def _is_water(row: pd.Series, t: dict) -> bool:
    """High MNDWI, low NDVI, low SWIR1."""
    return (row['mndwi'] > t['mndwi'][0] and
            row['ndvi']  < t['ndvi'][0]  and
            row['swir1'] < t['swir1'][0])


def _is_dense_vegetation(row: pd.Series, t: dict) -> bool:
    """High NDVI (above upper threshold), active NDRE, low NDBI."""
    return (row['ndvi']    > t['ndvi'][2]    and
            row['ndre_b5'] > t['ndre_b5'][0] and
            row['ndbi']    < t['ndbi'][0])


def _is_sparse_vegetation(row: pd.Series, t: dict) -> bool:
    """Intermediate NDVI range."""
    return t['ndvi'][1] <= row['ndvi'] <= t['ndvi'][2]


def _is_urban(row: pd.Series, t: dict) -> bool:
    """Low NDVI, elevated BSI and UI, suppressed SVC and MNDWI."""
    return (row['ndvi']  < t['ndvi'][1]  and
            row['bsi']   > t['bsi'][0]   and
            row['ui']    > t['ui'][1]    and
            row['svc']   < t['svc'][1]   and
            row['mndwi'] < t['mndwi'][0])


def _is_bare_soil_rock(row: pd.Series, t: dict) -> bool:
    """Low NDVI, elevated BSI, dry (negative MNDWI)."""
    return (row['ndvi']  < t['ndvi'][1]  and
            row['bsi']   > t['bsi'][0]   and
            row['mndwi'] < t['mndwi'][0])


def assign_semantic_label(row: pd.Series, t: dict) -> str:
    """
    Apply the sequential decision rules to one cluster's median statistics.

    Parameters
    ----------
    row : pd.Series
        Cluster-level median values (one row of cluster statistics).
    t : dict
        Otsu threshold dictionary from :func:`compute_otsu_thresholds`.

    Returns
    -------
    str
        One of: 'clouds', 'water', 'dense_vegetation',
        'sparse_vegetation', 'urban', 'bare_soil_rock', 'uncertain'.
    """
    if _is_cloud(row, t):             return 'clouds'
    if _is_water(row, t):             return 'water'
    if _is_dense_vegetation(row, t):  return 'dense_vegetation'
    if _is_sparse_vegetation(row, t): return 'sparse_vegetation'
    if _is_urban(row, t):             return 'urban'
    if _is_bare_soil_rock(row, t):    return 'bare_soil_rock'
    return 'uncertain'


def label_clusters(cluster_stats: pd.DataFrame, t: dict) -> dict:
    """
    Apply :func:`assign_semantic_label` to every cluster and return a
    cluster-ID → label-string mapping.

    Parameters
    ----------
    cluster_stats : pd.DataFrame
        Output of :func:`~clustering.compute_cluster_stats`.
    t : dict
        Otsu threshold dictionary from :func:`compute_otsu_thresholds`.

    Returns
    -------
    cluster_to_label : dict
        Maps integer cluster ID → semantic label string.
    """
    return {
        cid: assign_semantic_label(row, t)
        for cid, row in cluster_stats.iterrows()
    }


# ---------------------------------------------------------------------------
# Step 3 — Map cluster IDs to semantic integer IDs
# ---------------------------------------------------------------------------

def build_semantic_map(labels1: np.ndarray,
                       cluster_to_label: dict) -> np.ndarray:
    """
    Replace cluster IDs with integer semantic class IDs.

    The mapping is applied exactly as in the original code: each semantic
    class is written into a fresh copy of the label array using
    ``np.isin`` masking, so the order of writes matches the original.

    Parameters
    ----------
    labels1 : np.ndarray, shape (N,)
        Flat cluster label array (cloud-masked, with CLOUD_LABEL = 36
        for cloud pixels).
    cluster_to_label : dict
        Output of :func:`label_clusters`.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype same as labels1
        Semantic class IDs.
    """
    labels = np.copy(labels1)

    # Write order matches original to preserve identical output.
    for sem_name, class_id in [
        ('urban',             2),
        ('dense_vegetation',  4),
        ('sparse_vegetation', 0),
        ('water',             3),
        ('bare_soil_rock',    1),
        ('clouds',            6),
        ('uncertain',         5),
    ]:
        cluster_ids = [k for k, v in cluster_to_label.items()
                       if v == sem_name]
        if cluster_ids:
            labels[np.isin(labels1, cluster_ids)] = class_id

    return labels


# ---------------------------------------------------------------------------
# Utility — area summary
# ---------------------------------------------------------------------------

def area_summary(labels: np.ndarray) -> pd.DataFrame:
    """
    Compute the number of pixels and area (km²) for each semantic class.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Flat semantic label array from :func:`build_semantic_map`.

    Returns
    -------
    pd.DataFrame
        Columns: 'Label ID', 'Class', 'Pixels', 'Area (km²)'.
        A TOTAL row is appended at the bottom.
    """
    unique_labels, pixel_counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique_labels, pixel_counts))

    rows = []
    for label_id, label_name in LABEL_NAMES.items():
        n_pixels = count_dict.get(label_id, 0)
        rows.append({
            "Label ID":   label_id,
            "Class":      label_name,
            "Pixels":     n_pixels,
            "Area (km²)": round(n_pixels * PIXEL_AREA_KM2, 4),
        })

    area_df = pd.DataFrame(rows)
    total = pd.DataFrame([{
        "Label ID":   "",
        "Class":      "TOTAL",
        "Pixels":     area_df["Pixels"].sum(),
        "Area (km²)": round(area_df["Area (km²)"].sum(), 4),
    }])
    return pd.concat([area_df, total], ignore_index=True)
