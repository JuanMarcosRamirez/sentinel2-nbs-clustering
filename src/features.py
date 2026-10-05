"""
features.py
===========
Sentinel-2 Level-2A band extraction and spectral index computation.

Input GeoTIFF band order (20-m stack, 13 layers)
-------------------------------------------------
The pipeline expects a single multi-band GeoTIFF in which the layers are
ordered as follows.  This is the order produced by the data-preparation
step described in the paper.

    Layer  Band   Wavelength  Region
    -----  -----  ----------  -------------------
      0    SCL    —           Scene Classification Layer  (integer codes)
      1    B12    2190 nm     SWIR-2
      2    B11    1610 nm     SWIR-1
      3    B8A     865 nm     Narrow NIR
      4    B7      783 nm     Red-Edge 3
      5    B6      740 nm     Red-Edge 2
      6    B5      705 nm     Red-Edge 1
      7    B4      665 nm     Red   (resampled from 10 m)
      8    B3      560 nm     Green (resampled from 10 m)
      9    B2      490 nm     Blue  (resampled from 10 m)
     10    —       —          (padding / unused)
     11    —       —          (padding / unused)
     12    —       —          (padding / unused)

Reflectance values are stored as integers scaled by 10,000.

Feature matrix column order (17 columns)
-----------------------------------------
After calling :func:`build_features` and constructing the DataFrame,
the columns are:

    Index  Name         Source
    -----  -----------  ----------------------------------------
      0    red          B4  (layer 9 of RGB block = data[:,  :, 9])
      1    green        B3  (layer 10)
      2    blue         B2  (layer 11)
      3    nir          B8A (layer 5)
      4    red_edge3    B7  (layer 4, first slice of RED_EDGE block)
      5    red_edge2    B6  (layer 5, second slice)
      6    red_edge1    B5  (layer 6, third slice)
      7    swir1        B11 (layer 4 of raw → SWIR1 = data[:,:,4])
      8    swir2        B12 (layer 3 of raw → SWIR2 = data[:,:,3])
      9    ndvi         (NIR - Red)  / (NIR + Red  + ε)
     10    ndbi         (SWIR1 - NIR) / (SWIR1 + NIR + ε)
     11    ui           (SWIR2 - NIR) / (SWIR2 + NIR + ε)
     12    bsi          [(SWIR1+Red)-(NIR+Blue)] / [(SWIR1+Red)+(NIR+Blue)+ε]
     13    mndwi        (Green - SWIR1) / (Green + SWIR1 + ε)
     14    ndre_b5      (NIR - RED_EDGE[:,2]) / (NIR + RED_EDGE[:,2] + ε)
                        RED_EDGE[:,2] = layer 8 of raw = B7 (Red-Edge 3)
     15    svc          (SWIR1 - mean_RGB) / (SWIR1 + mean_RGB + ε)
     16    scl          Scene Classification Layer (uint8 class codes)

Note: ``ndre_b5`` uses RED_EDGE[:, :, 2], which is the *third* slice of
``data[:, :, 6:9]``, corresponding to layer index 8 and band B7
(Red-Edge 3, 783 nm).  This matches the original implementation exactly.

``col_names`` (the 16 features passed to RobustScaler and KMeans) excludes
``scl``.  ``all_cols`` includes ``scl`` as the last column for cloud
masking and semantic labeling.

References
----------
Rouse et al. (1974), Tucker (1979)       – NDVI
Barnes et al. (2000)                     – NDRE
Kawamura et al. (1996), Zha et al. (2003)– NDBI, UI
Rikimaru et al. (2002)                   – BSI
Xu (2006)                                – MNDWI
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Small constant added to all index denominators to prevent division by zero.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Column name lists — must match build_features() output order exactly
# ---------------------------------------------------------------------------

#: All 17 column names (bands + indices + SCL).
ALL_COLS = [
    'red', 'green', 'blue',
    'nir',
    'red_edge3', 'red_edge2', 'red_edge1',
    'swir1', 'swir2',
    'ndvi', 'ndbi', 'ui', 'bsi', 'mndwi', 'ndre_b5', 'svc',
    'scl',
]

#: The 16 features used as input to RobustScaler and KMeans (excludes SCL).
COL_NAMES = [
    'red', 'green', 'blue',
    'nir',
    'red_edge3', 'red_edge2', 'red_edge1',
    'swir1', 'swir2',
    'ndvi', 'ndbi', 'ui', 'bsi', 'mndwi', 'ndre_b5', 'svc',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_index(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Normalised difference (num - den) / (num + den + EPS)."""
    return num / (den + _EPS)


# ---------------------------------------------------------------------------
# Main feature-extraction function
# ---------------------------------------------------------------------------

def build_features(data: np.ndarray) -> np.ndarray:
    """
    Extract reflectance bands and compute spectral indices from a raw
    Sentinel-2 Level-2A GeoTIFF array.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, C)
        Raw multi-band array with integer reflectance scaled by 10,000.
        Band order must follow the convention in the module docstring.

    Returns
    -------
    feat : np.ndarray, shape (H, W, 17), dtype float32
        Feature stack in the column order defined by :data:`ALL_COLS`.
        The SCL layer is cast to float32 for consistent stacking but
        retains integer class codes (it is never passed to the scaler).
    """
    # Convert reflectance bands to physical units [0, 1].
    # SCL is recovered separately before the division.
    data = data.astype(np.float32) / 10_000.0

    # --- Band extraction (layer indices match the GeoTIFF band order) ---
    RGB      = data[:, :, 9:12]   # layers 9, 10, 11 → B4 (Red), B3 (Green), B2 (Blue)
    RED_EDGE = data[:, :, 6:9]    # layers 6, 7, 8  → B5, B6, B7  (Red-Edge 1–3)
    NIR      = data[:, :, 5]      # layer 5          → B8A (Narrow NIR)
    SWIR1    = data[:, :, 4]      # layer 4          → B11 (SWIR-1)
    SWIR2    = data[:, :, 3]      # layer 3          → B12 (SWIR-2)
    SCL      = (data[:, :, 0] * 10_000).astype(np.uint8)  # recover integer SCL codes

    red        = RGB[:, :, 0]
    green      = RGB[:, :, 1]
    blue       = RGB[:, :, 2]
    color_mean = np.mean(RGB, axis=2)  # mean of Red, Green, Blue

    # --- Spectral indices ---
    ndvi    = _safe_index(NIR - red,              NIR + red)
    ndbi    = _safe_index(SWIR1 - NIR,            SWIR1 + NIR)
    ui      = _safe_index(SWIR2 - NIR,            SWIR2 + NIR)
    bsi     = _safe_index((SWIR1 + red) - (NIR + blue),
                          (SWIR1 + red) + (NIR + blue))
    mndwi   = _safe_index(green - SWIR1,          green + SWIR1)
    # ndre_b5 uses RED_EDGE[:, :, 2] = third slice of data[:,:,6:9]
    # = layer index 8 = B7 (Red-Edge 3, 783 nm)
    ndre_b5 = _safe_index(NIR - RED_EDGE[:, :, 2], NIR + RED_EDGE[:, :, 2])
    svc     = _safe_index(SWIR1 - color_mean,     SWIR1 + color_mean)

    # --- Stack in ALL_COLS order ---
    feat = np.dstack([
        RGB,        # red, green, blue          (cols 0–2)
        NIR,        # nir                       (col  3)
        RED_EDGE,   # red_edge3, red_edge2, red_edge1  (cols 4–6)
        SWIR1,      # swir1                     (col  7)
        SWIR2,      # swir2                     (col  8)
        ndvi,       # ndvi                      (col  9)
        ndbi,       # ndbi                      (col 10)
        ui,         # ui                        (col 11)
        bsi,        # bsi                       (col 12)
        mndwi,      # mndwi                     (col 13)
        ndre_b5,    # ndre_b5                   (col 14)
        svc,        # svc                       (col 15)
        SCL,        # scl                       (col 16)
    ]).astype(np.float32)

    return feat


def to_dataframe(feat: np.ndarray) -> pd.DataFrame:
    """
    Reshape the (H, W, 17) feature array into a (H*W, 17) DataFrame
    with columns :data:`ALL_COLS`.

    Parameters
    ----------
    feat : np.ndarray, shape (H, W, 17)
        Output of :func:`build_features`.

    Returns
    -------
    df : pd.DataFrame, shape (H * W, 17)
    """
    H, W, nz = feat.shape
    return pd.DataFrame(feat.reshape(H * W, nz), columns=ALL_COLS)


def preprocess(df: pd.DataFrame) -> np.ndarray:
    """
    Replace invalid values and apply robust normalization to the 16
    clustering features (all columns except ``scl``).

    Invalid values (±inf, NaN) are imputed with the column-wise median.
    :class:`~sklearn.preprocessing.RobustScaler` then centers each
    feature at its median and scales by its interquartile range (IQR),
    making normalization robust to the outliers common in multispectral
    index distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`to_dataframe`.

    Returns
    -------
    X_scaled : np.ndarray, shape (N, 16)
        Normalized feature matrix ready for k-means.
    """
    X = df[COL_NAMES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(axis=0))
    scaler = RobustScaler()
    return scaler.fit_transform(X)
