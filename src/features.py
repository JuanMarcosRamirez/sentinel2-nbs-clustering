"""
features.py
===========
Sentinel-2 Level-2A band extraction and spectral index computation.

All reflectance bands are expected to be stored in a single multi-band
GeoTIFF at 20-m spatial resolution in the following band order:

    Index  Band      Wavelength  Region
    -----  --------  ----------  -------------------
      0    SCL       —           Scene Classification Layer
      1    B12       2190 nm     SWIR-2
      2    B11       1610 nm     SWIR-1
      3    B8A        865 nm     Narrow NIR
      4    B7         783 nm     Red-Edge 3
      5    B6         740 nm     Red-Edge 2
      6    B5         705 nm     Red-Edge 1
      7    B4         665 nm     Red (resampled from 10 m)
      8    B3         560 nm     Green (resampled from 10 m)
      9    B2         490 nm     Blue (resampled from 10 m)

Sentinel-2 Level-2A surface reflectance values are stored as integers
scaled by 10,000.  The functions below divide by 10,000 to work in
physical reflectance units in the range [0, 1].

References
----------
Rouse et al. (1974)  – NDVI
Tucker (1979)        – NDVI
Barnes et al. (2000) – NDRE
Kawamura et al. (1996) – UI
Zha et al. (2003)    – NDBI
Rikimaru et al. (2002) – BSI
Xu (2006)            – MNDWI
"""

import numpy as np

# Small constant added to all index denominators to prevent
# division by zero or near-zero reflectance values.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a normalised difference (a - b) / (a + b + EPS)."""
    return (a - b) / (a + b + _EPS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bands(raw: np.ndarray) -> dict:
    """
    Convert a raw Sentinel-2 multi-band array to a dictionary of named
    reflectance bands and the Scene Classification Layer (SCL).

    Parameters
    ----------
    raw : np.ndarray, shape (H, W, 10)
        Raw Sentinel-2 stack with integer reflectance values scaled by
        10,000. Band order must follow the convention described in the
        module docstring.

    Returns
    -------
    dict
        Keys: 'blue', 'green', 'red', 'red_edge1', 'red_edge2',
              'red_edge3', 'nir', 'swir1', 'swir2', 'scl'.
        Reflectance arrays are float32 in [0, 1]; 'scl' is uint8.
    """
    refl = raw.astype(np.float32) / 10_000.0

    return {
        # Visible bands (originally 10 m, resampled to 20 m)
        "blue":       refl[:, :, 9],   # B2
        "green":      refl[:, :, 8],   # B3
        "red":        refl[:, :, 7],   # B4
        # Red-edge bands (native 20 m)
        "red_edge1":  refl[:, :, 6],   # B5
        "red_edge2":  refl[:, :, 5],   # B6
        "red_edge3":  refl[:, :, 4],   # B7
        # Near-infrared (native 20 m)
        "nir":        refl[:, :, 3],   # B8A
        # Short-wave infrared (native 20 m)
        "swir1":      refl[:, :, 2],   # B11
        "swir2":      refl[:, :, 1],   # B12
        # Scene Classification Layer (integer class codes)
        "scl":        (raw[:, :, 0]).astype(np.uint8),
    }


def compute_indices(bands: dict) -> dict:
    """
    Compute the seven spectral indices used for semantic labeling.

    Parameters
    ----------
    bands : dict
        Output of :func:`extract_bands`.

    Returns
    -------
    dict
        Keys: 'ndvi', 'ndbi', 'ui', 'bsi', 'mndwi', 'ndre', 'svc'.
        All arrays are float32.

    Notes
    -----
    Index definitions (see paper Table 2):

    NDVI  = (NIR - Red)  / (NIR + Red  + ε)
    NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR + ε)
    UI    = (SWIR2 - NIR) / (SWIR2 + NIR + ε)
    BSI   = [(SWIR1 + Red) - (NIR + Blue)] /
            [(SWIR1 + Red) + (NIR + Blue) + ε]
    MNDWI = (Green - SWIR1) / (Green + SWIR1 + ε)
    NDRE  = (NIR - RedEdge1) / (NIR + RedEdge1 + ε)
    SVC   = (SWIR1 - mean_visible) / (SWIR1 + mean_visible + ε)
    """
    b   = bands["blue"]
    g   = bands["green"]
    r   = bands["red"]
    re1 = bands["red_edge1"]
    nir = bands["nir"]
    s1  = bands["swir1"]
    s2  = bands["swir2"]

    mean_visible = (r + g + b) / 3.0

    return {
        "ndvi":  _norm_diff(nir, r),
        "ndbi":  _norm_diff(s1,  nir),
        "ui":    _norm_diff(s2,  nir),
        "bsi":   _norm_diff((s1 + r) - (nir + b),
                            (s1 + r) + (nir + b)),
        "mndwi": _norm_diff(g,   s1),
        "ndre":  _norm_diff(nir, re1),
        "svc":   _norm_diff(s1,  mean_visible),
    }


def build_feature_matrix(raw: np.ndarray) -> tuple:
    """
    Build the full (H × W, D) feature matrix used as input to k-means.

    Combines nine reflectance bands and seven spectral indices into a
    single 2-D array suitable for :class:`sklearn.cluster.KMeans`.
    The Scene Classification Layer (SCL) is returned separately because
    it is used only for cloud masking, not as a clustering feature.

    Parameters
    ----------
    raw : np.ndarray, shape (H, W, 10)
        Raw Sentinel-2 stack (see :func:`extract_bands`).

    Returns
    -------
    feature_matrix : np.ndarray, shape (H * W, 16)
        Float32 array of nine bands + seven indices, in the order:
        [red, green, blue, nir, red_edge1, red_edge2, red_edge3,
         swir1, swir2, ndvi, ndbi, ui, bsi, mndwi, ndre, svc].
    scl_flat : np.ndarray, shape (H * W,)
        Flat SCL array (uint8) aligned with ``feature_matrix``.
    shape : tuple (H, W)
        Original spatial dimensions, needed to reshape results back.
    """
    bands   = extract_bands(raw)
    indices = compute_indices(bands)

    H, W = raw.shape[:2]

    # Stack bands (9) and indices (7) in documented order
    band_stack = np.dstack([
        bands["red"],   bands["green"],  bands["blue"],
        bands["nir"],
        bands["red_edge1"], bands["red_edge2"], bands["red_edge3"],
        bands["swir1"], bands["swir2"],
        indices["ndvi"],  indices["ndbi"], indices["ui"],
        indices["bsi"],   indices["mndwi"],
        indices["ndre"],  indices["svc"],
    ]).astype(np.float32)

    feature_matrix = band_stack.reshape(H * W, -1)
    scl_flat       = bands["scl"].reshape(-1)

    return feature_matrix, scl_flat, (H, W)


# Column names aligned with build_feature_matrix output order
FEATURE_NAMES = [
    "red", "green", "blue",
    "nir",
    "red_edge1", "red_edge2", "red_edge3",
    "swir1", "swir2",
    "ndvi", "ndbi", "ui", "bsi", "mndwi", "ndre", "svc",
]

# Subset used as clustering features (all 16 above)
CLUSTERING_FEATURES = FEATURE_NAMES.copy()
