"""
clustering.py
=============
K-means clustering and cloud-pixel masking for Sentinel-2 imagery.

The clustering pipeline follows three steps:

1. Preprocess  – replace invalid values with feature-wise medians and
                 apply robust normalization (RobustScaler).
2. Cluster     – fit k-means and assign every pixel to a cluster.
3. Cloud mask  – reassign cloud/shadow pixels (SCL classes 3, 8, 9, 10)
                 to a dedicated label that is distinct from all k-means
                 cluster IDs.

References
----------
Pedregosa et al. (2011) – scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


# Label assigned to cloud-contaminated pixels after clustering.
# Chosen to be outside the range [0, k-1] produced by k-means.
CLOUD_LABEL: int = 36

# SCL class codes treated as cloud-contaminated.
# 3  = cloud shadow
# 8  = medium-probability cloud
# 9  = high-probability cloud
# 10 = thin cirrus
CLOUD_SCL_CLASSES: tuple = (3, 8, 9, 10)


def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Replace invalid values and apply robust normalization.

    Invalid values (±inf, NaN) are imputed with the column-wise median
    before scaling.  :class:`~sklearn.preprocessing.RobustScaler`
    centers each feature at its median and scales by its inter-quartile
    range (IQR), making normalization resistant to outliers that are
    common in multispectral index distributions.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Raw feature matrix (float32).

    Returns
    -------
    X_scaled : np.ndarray, shape (N, D)
        Normalized feature matrix ready for k-means.
    """
    # Convert to DataFrame for convenient column-wise median imputation
    df = pd.DataFrame(X)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(axis=0), inplace=True)

    scaler = RobustScaler()
    return scaler.fit_transform(df.values)


def run_kmeans(X_scaled: np.ndarray, k: int = 9,
               random_state: int = 0) -> np.ndarray:
    """
    Fit k-means on the normalized feature matrix.

    Cluster labels are reordered by ascending centroid L2-norm so that
    the mapping from cluster ID to spectral brightness is deterministic
    and reproducible regardless of the random initialization.

    Parameters
    ----------
    X_scaled : np.ndarray, shape (N, D)
        Output of :func:`preprocess`.
    k : int, default 9
        Number of clusters.  The value k = 9 was selected based on a
        cluster validity analysis using Silhouette, CH-index, DB-index,
        ARI, and NMI (see paper Section 4.1).
    random_state : int, default 0
        Seed for the k-means centroid initialization, ensuring that
        results are fully reproducible.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype uint8
        Cluster identifier for each pixel, in [0, k-1].
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    raw_labels = kmeans.fit_predict(X_scaled)

    # Reorder labels by ascending centroid norm for deterministic output
    centroid_norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    order   = np.argsort(centroid_norms)
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))

    return mapping[raw_labels].astype(np.uint8)


def apply_cloud_mask(labels: np.ndarray,
                     scl: np.ndarray) -> np.ndarray:
    """
    Reassign cloud-contaminated pixels to :data:`CLOUD_LABEL`.

    Pixels whose SCL value belongs to :data:`CLOUD_SCL_CLASSES` are
    overwritten after clustering.  This prevents cloud and cloud-shadow
    reflectance from being interpreted as a meaningful land-surface
    class during semantic labeling.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Cluster labels from :func:`run_kmeans`.
    scl : np.ndarray, shape (N,), dtype uint8
        Flat Scene Classification Layer values.

    Returns
    -------
    labels_masked : np.ndarray, shape (N,)
        Updated label array with cloud pixels set to
        :data:`CLOUD_LABEL`.
    """
    labels_masked = labels.copy()
    cloud_pixels  = np.isin(scl, CLOUD_SCL_CLASSES)
    labels_masked[cloud_pixels] = CLOUD_LABEL
    return labels_masked


def compute_cluster_statistics(X_raw: np.ndarray,
                                labels: np.ndarray,
                                feature_names: list) -> pd.DataFrame:
    """
    Compute cluster-level median statistics from the original
    (non-normalized) feature values.

    The median is used instead of the mean to reduce sensitivity to
    atypical pixels and outliers within each cluster.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, D)
        Original feature matrix before normalization.
    labels : np.ndarray, shape (N,)
        Cluster labels (cloud-masked pixels are excluded from
        statistics because their label equals :data:`CLOUD_LABEL`).
    feature_names : list of str, length D
        Column names matching the feature order in ``X_raw``.

    Returns
    -------
    stats : pd.DataFrame, shape (k, D)
        Median value of each feature for each cluster.
        Index is the cluster ID.
    """
    # Exclude cloud pixels from cluster statistics
    mask = labels != CLOUD_LABEL
    df   = pd.DataFrame(X_raw[mask], columns=feature_names)
    df["cluster"] = labels[mask]
    return df.groupby("cluster").median(numeric_only=True)
