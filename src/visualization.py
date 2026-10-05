"""
visualization.py
================
Color mapping and figure generation for semantic clustering results.

All color palettes are stored as NumPy arrays indexed by integer label,
allowing O(1) vectorized lookup over large Sentinel-2 scenes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import exposure


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Semantic class palette (index = class ID defined in semantic_labeling.py)
# ID  Class               Color
#  0  Sparse vegetation   Light green
#  1  Bare soil / Rock    Light brown (tan)
#  2  Urban               Red
#  3  Water               Blue
#  4  Dense vegetation    Dark green
#  5  Uncertain           Magenta
#  6  Clouds              White
SEMANTIC_PALETTE = np.array([
    [209, 255, 189],   # 0 – Sparse vegetation  (light green)
    [196, 164, 132],   # 1 – Bare soil / Rock   (tan)
    [255,   0,   0],   # 2 – Urban              (red)
    [  0, 150, 255],   # 3 – Water              (blue)
    [  0, 100,   0],   # 4 – Dense vegetation   (dark green)
    [255,   0, 255],   # 5 – Uncertain          (magenta)
    [255, 255, 255],   # 6 – Clouds             (white)
], dtype=np.uint8)


# SCL palette: indices match Sentinel-2 SCL class codes (0–11 used)
# Codes used in paper: 3 (shadow), 4 (vegetation), 5 (bare soil),
#                      6 (water), 8–10 (clouds), 11 (snow/ice).
SCL_PALETTE = np.array([
    [  0,   0,   0],   #  0 – No data         (black)
    [  0,   0,   0],   #  1 – Saturated        (black)
    [  0,   0,   0],   #  2 – Dark area        (black)
    [100, 100, 100],   #  3 – Cloud shadow     (dark grey)
    [  0, 100,   0],   #  4 – Vegetation       (dark green)
    [196, 164, 132],   #  5 – Bare soil        (tan)
    [  0, 150, 255],   #  6 – Water            (blue)
    [  0,   0,   0],   #  7 – Unclassified     (black)
    [255, 225, 255],   #  8 – Cloud med. prob. (light pink)
    [255, 200, 255],   #  9 – Cloud high prob. (pink)
    [220, 180, 220],   # 10 – Thin cirrus      (lilac)
    [255, 255, 255],   # 11 – Snow / ice       (white)
], dtype=np.uint8)


# Qualitative palette for unsupervised cluster maps (up to 40 clusters)
CLUSTER_PALETTE = np.array([
    [230,  25,  75], [60,  180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128],
    [255, 255, 255], [255,   0,   0], [  0, 255,   0], [  0,   0, 255],
    [255, 255,   0], [255,   0, 255], [  0, 255, 255], [128,   0, 255],
    [255, 128,   0], [  0, 100,   0], [255, 105, 180], [139,  69,  19],
    [ 75,   0, 130], [154, 205,  50], [  0, 191, 255], [255, 255, 255],
    [112, 128, 144], [255, 160, 122], [ 46, 139,  87], [  0,   0,   0],
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def labels_to_rgb(label_img: np.ndarray,
                  palette: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D integer label image to an RGB image using a palette.

    This vectorised lookup replaces any nested loop and runs in O(N)
    time for an N-pixel image.

    Parameters
    ----------
    label_img : np.ndarray, shape (H, W)
        Integer label image.  Values must be valid palette indices.
    palette : np.ndarray, shape (M, 3), dtype uint8
        RGB color table indexed by label value.

    Returns
    -------
    rgb : np.ndarray, shape (H, W, 3), dtype uint8
    """
    return palette[label_img.astype(np.int32)]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_rgb_composite(raw: np.ndarray,
                       title: str = "",
                       band_indices: tuple = (7, 8, 9)) -> None:
    """
    Display a true-color composite (B4/B3/B2 by default).

    Parameters
    ----------
    raw : np.ndarray, shape (H, W, C)
        Raw Sentinel-2 stack (integer, scaled by 10,000).
    title : str
        Figure title (e.g. acquisition date).
    band_indices : tuple of int
        Indices of the red, green, and blue bands within ``raw``.
    """
    img = raw[:, :, list(band_indices)].astype(np.float32)
    p2, p98 = np.percentile(img[img > 0], (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98),
                                     out_range=(0.0, 1.0))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_semantic_map(semantic_labels: np.ndarray,
                      spatial_shape: tuple,
                      title: str = "") -> None:
    """
    Display the semantic classification map with a labeled legend.

    Parameters
    ----------
    semantic_labels : np.ndarray, shape (N,)
        Flat semantic label array.
    spatial_shape : tuple (H, W)
        Original spatial dimensions for reshaping.
    title : str
        Figure title (e.g. acquisition date).
    """
    from semantic_labeling import CLASS_NAMES

    rgb = labels_to_rgb(semantic_labels.reshape(spatial_shape),
                        SEMANTIC_PALETTE)

    legend_handles = [
        mpatches.Patch(
            facecolor=SEMANTIC_PALETTE[cid] / 255.0,
            edgecolor="grey",
            linewidth=0.5,
            label=name,
        )
        for cid, name in CLASS_NAMES.items()
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=7, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def plot_cluster_map(cluster_labels: np.ndarray,
                     spatial_shape: tuple,
                     title: str = "") -> None:
    """
    Display the raw (unsupervised) k-means cluster map.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (N,)
        Flat cluster label array.
    spatial_shape : tuple (H, W)
        Original spatial dimensions.
    title : str
        Figure title.
    """
    rgb = labels_to_rgb(cluster_labels.reshape(spatial_shape),
                        CLUSTER_PALETTE)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_elevation_stratified(semantic_labels: np.ndarray,
                               dem: np.ndarray,
                               elevation_threshold: int = 300,
                               title_valley: str = "Valley (z < 300 m)",
                               title_upland: str = "Upland (z ≥ 300 m)"
                               ) -> None:
    """
    Display semantic maps masked to valley and upland elevation zones.

    Parameters
    ----------
    semantic_labels : np.ndarray, shape (H, W)
        2-D semantic label array (already reshaped).
    dem : np.ndarray, shape (H, W)
        Co-registered DEM in metres.
    elevation_threshold : int, default 300
        Elevation boundary separating valley from upland zones (m).
    title_valley : str
        Title for the valley-zone panel.
    title_upland : str
        Title for the upland-zone panel.
    """
    MASK_ID = 7   # Label value used for the masked (out-of-zone) pixels

    # Valley: keep pixels below threshold; mask upland pixels
    valley = semantic_labels.copy()
    valley[dem >= elevation_threshold] = MASK_ID

    # Upland: keep pixels at or above threshold; mask valley pixels
    upland = semantic_labels.copy()
    upland[dem < elevation_threshold] = MASK_ID

    # Extend palette with a black entry for masked pixels (index 7)
    extended = np.vstack([SEMANTIC_PALETTE,
                          np.array([[0, 0, 0]], dtype=np.uint8)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, title in zip(axes,
                               [valley, upland],
                               [title_valley, title_upland]):
        ax.imshow(extended[img.astype(np.int32)])
        ax.axis("off")
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
