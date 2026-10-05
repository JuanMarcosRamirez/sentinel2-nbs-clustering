"""
clustering.py
=============
K-means clustering and SCL-based cloud masking.

The two steps implemented here follow the original code exactly:

1. **K-means** — fit with ``random_state=0`` and ``n_init="auto"``, then
   reorder cluster labels by ascending centroid L2-norm so the label
   assignment is deterministic regardless of initialization order.

2. **Cloud mask** — pixels whose SCL value belongs to
   {3, 8, 9, 10} are reassigned to label 36 *after* clustering.
   Label 36 lies outside the range [0, k-1] produced by k-means and is
   therefore unambiguously distinct from all spectral clusters.
   SCL classes masked:
       3  = cloud shadow
       8  = medium-probability cloud
       9  = high-probability cloud
       10 = thin cirrus

References
----------
Pedregosa et al. (2011) – scikit-learn KMeans.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Integer label assigned to cloud/shadow pixels after clustering.
# Must be outside [0, k-1]; chosen to match the original code.
CLOUD_LABEL: int = 36

# SCL class codes treated as cloud-contaminated.
CLOUD_SCL_CLASSES: tuple = (3, 8, 9, 10)


def run_kmeans(X_scaled: np.ndarray, k: int = 9,
               random_state: int = 0) -> np.ndarray:
    """
    Fit k-means on the normalized feature matrix and return pixel labels
    reordered by ascending centroid L2-norm.

    Reordering makes the cluster-ID → color mapping deterministic: the
    cluster whose centroid has the smallest norm always receives ID 0,
    so the qualitative cluster map looks the same across runs with
    different random seeds.

    Parameters
    ----------
    X_scaled : np.ndarray, shape (N, 16)
        Normalized feature matrix from
        :func:`~features.preprocess`.
    k : int, default 9
        Number of clusters.  k = 9 was selected via cluster validity
        analysis (Silhouette, CH, DB, ARI, NMI); see paper Section 4.1.
    random_state : int, default 0
        Seed for centroid initialization, ensuring full reproducibility.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype uint8
        Reordered cluster IDs in [0, k-1].
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state,
                    n_init="auto")
    raw_labels = kmeans.fit_predict(X_scaled)

    # Reorder labels so cluster 0 = smallest centroid norm, etc.
    centroid_norms = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    order   = np.argsort(centroid_norms)
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))

    return mapping[raw_labels].astype(np.uint8)


def apply_cloud_mask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reassign cloud-contaminated pixels to :data:`CLOUD_LABEL` in the
    ``cluster`` column of the pixel DataFrame.

    The mask is applied *in place on a copy* so the original DataFrame
    is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``'scl'`` and ``'cluster'``.

    Returns
    -------
    df_masked : pd.DataFrame
        Copy of ``df`` with cloud pixels set to :data:`CLOUD_LABEL`.
    """
    df_masked = df.copy()
    cloud_mask = (
        (df_masked['scl'] == 3)  |
        (df_masked['scl'] == 8)  |
        (df_masked['scl'] == 9)  |
        (df_masked['scl'] == 10)
    )
    df_masked.loc[cloud_mask, 'cluster'] = CLOUD_LABEL
    return df_masked


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cluster-level median of every feature column.

    The median is used instead of the mean to reduce sensitivity to
    atypical pixels and residual outliers within each cluster.
    Cloud pixels (cluster == :data:`CLOUD_LABEL`) are included so the
    cloud rule in semantic labeling can fire on them if needed, but in
    practice the SCL cloud mask handles them before this stage.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel DataFrame with a ``'cluster'`` column (post-cloud-mask).

    Returns
    -------
    stats : pd.DataFrame
        Shape (n_clusters, n_features), indexed by cluster ID.
    """
    return df.groupby("cluster").median(numeric_only=True)
