"""
palettes.py
===========
Color palettes and label-to-RGB conversion for Sentinel-2 cluster maps.

Two palettes are used in the pipeline:

``Sentinel2d``
    Qualitative palette for unsupervised k-means cluster maps (up to 40
    clusters).  Index 36 is white, used for cloud-masked pixels.

``Sentinel2b``
    Semantic class palette indexed by the integer class IDs defined in
    ``semantic_labeling.py``:

        0  Sparse vegetation   [209, 255, 189]  light green
        1  Bare soil / Rock    [196, 164, 132]  tan
        2  Urban               [255,   0,   0]  red
        3  Water               [  0, 150, 255]  blue
        4  Dense vegetation    [  0, 100,   0]  dark green
        5  Uncertain           [255,   0, 255]  magenta
        6  Clouds              [255, 255, 255]  white
        7  Masked (DEM)        [  0,   0,   0]  black

``scl``
    Palette for the Sentinel-2 Scene Classification Layer (SCL), indexed
    by SCL class code (0–11 in practice):

        3   Cloud shadow        [255, 225, 255]
        4   Vegetation          [  0, 100,   0]
        5   Bare soil           [196, 164, 132]
        6   Water               [  0, 150, 255]
        8   Medium-prob. cloud  [255, 225, 255]
        9   High-prob. cloud    [255, 225, 255]
        10  Thin cirrus         [255, 225, 255]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Palette definitions
# These are reproduced exactly from the original implementation so that
# all output figures are pixel-identical to the original code.
# ---------------------------------------------------------------------------

PALETTES = {

    # ------------------------------------------------------------------
    # Sentinel2d: qualitative palette for k-means cluster maps.
    # Index = cluster ID (0–35); index 36 = cloud label (white).
    # ------------------------------------------------------------------
    "Sentinel2d": np.array([
        [  0,   0,   0],   #  0
        [230,  25,  75],   #  1
        [ 60, 180,  75],   #  2
        [255, 225,  25],   #  3
        [  0, 130, 200],   #  4
        [245, 130,  48],   #  5
        [145,  30, 180],   #  6
        [ 70, 240, 240],   #  7
        [240,  50, 230],   #  8
        [210, 245,  60],   #  9
        [250, 190, 212],   # 10
        [  0, 128, 128],   # 11
        [220, 190, 255],   # 12
        [170, 110,  40],   # 13
        [255, 250, 200],   # 14
        [128,   0,   0],   # 15
        [170, 255, 195],   # 16
        [128, 128,   0],   # 17
        [255, 215, 180],   # 18
        [  0,   0, 128],   # 19
        [128, 128, 128],   # 20
        [255, 255, 255],   # 21
        [255,   0,   0],   # 22
        [  0, 255,   0],   # 23
        [  0,   0, 255],   # 24
        [255, 255,   0],   # 25
        [255,   0, 255],   # 26
        [  0, 255, 255],   # 27
        [128,   0, 255],   # 28
        [255, 128,   0],   # 29
        [  0, 100,   0],   # 30
        [255, 105, 180],   # 31
        [139,  69,  19],   # 32
        [ 75,   0, 130],   # 33
        [154, 205,  50],   # 34
        [  0, 191, 255],   # 35
        [255, 255, 255],   # 36  ← cloud label
        [112, 128, 144],   # 37
        [255, 160, 122],   # 38
        [ 46, 139,  87],   # 39
    ], dtype=np.uint8),

    # ------------------------------------------------------------------
    # Sentinel2b: semantic class palette.
    # Index = semantic class ID (see module docstring).
    # Index 7 (black) is used for DEM-masked pixels.
    # ------------------------------------------------------------------
    "Sentinel2b": np.array([
        [209, 255, 189],   # 0  Sparse vegetation
        [196, 164, 132],   # 1  Bare soil / Rock
        [255,   0,   0],   # 2  Urban
        [  0, 150, 255],   # 3  Water
        [  0, 100,   0],   # 4  Dense vegetation
        [255,   0, 255],   # 5  Uncertain
        [255, 255, 255],   # 6  Clouds
        [  0,   0,   0],   # 7  Masked (DEM zone boundary)
        [156, 218, 255],   # 8  (spare)
        [255, 156, 255],   # 9  (spare)
        [218, 112, 214],   # 10 (spare)
        [ 74,  44,   0],   # 11 (spare)
    ], dtype=np.uint8),

    # ------------------------------------------------------------------
    # scl: palette for the Sentinel-2 Scene Classification Layer.
    # Index = SCL class code (0–11 used; higher indices kept for safety).
    # ------------------------------------------------------------------
    "scl": np.array([
        [  0,   0,   0],   #  0  No data
        [  0,   0,   0],   #  1  Saturated / defective
        [  0,   0,   0],   #  2  Dark area pixels
        [255, 225, 255],   #  3  Cloud shadow
        [  0, 100,   0],   #  4  Vegetation
        [196, 164, 132],   #  5  Bare soil / desert
        [  0, 150, 255],   #  6  Water
        [  0,   0,   0],   #  7  Unclassified
        [255, 225, 255],   #  8  Cloud medium probability
        [255, 225, 255],   #  9  Cloud high probability
        [255, 225, 255],   # 10  Thin cirrus
        [255,   0, 255],   # 11  Snow / ice
        [220, 190, 255],   # 12
        [170, 110,  40],   # 13
        [255, 250, 200],   # 14
        [128,   0,   0],   # 15
        [170, 255, 195],   # 16
        [128, 128,   0],   # 17
        [255, 215, 180],   # 18
        [  0,   0, 128],   # 19
        [128, 128, 128],   # 20
        [255, 255, 255],   # 21
        [255,   0,   0],   # 22
        [  0, 255,   0],   # 23
        [  0,   0, 255],   # 24
        [255, 255,   0],   # 25
        [255,   0, 255],   # 26
        [  0, 255, 255],   # 27
        [128,   0, 255],   # 28
        [255, 128,   0],   # 29
        [  0, 100,   0],   # 30
        [255, 105, 180],   # 31
        [139,  69,  19],   # 32
        [ 75,   0, 130],   # 33
        [154, 205,  50],   # 34
        [  0, 191, 255],   # 35
        [255, 255, 255],   # 36
        [112, 128, 144],   # 37
        [255, 160, 122],   # 38
        [ 46, 139,  87],   # 39
    ], dtype=np.uint8),
}


def label2color(label_img: np.ndarray, palette_name: str) -> np.ndarray:
    """
    Convert an integer label image to an RGB image using a named palette.

    This vectorised lookup runs in O(N) time and is orders of magnitude
    faster than a pixel-level loop over large Sentinel-2 scenes.

    Parameters
    ----------
    label_img : np.ndarray
        Integer array of any shape whose values are valid palette indices.
    palette_name : str
        Key into :data:`PALETTES` ('Sentinel2d', 'Sentinel2b', or 'scl').

    Returns
    -------
    np.ndarray, dtype uint8
        RGB image with the same leading dimensions as ``label_img`` and
        a trailing dimension of 3.
    """
    colors = PALETTES[palette_name]
    return colors[label_img.astype(np.int32)]
