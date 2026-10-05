# Sentinel-2 Rule-based Semantic Clustering for NbS Zone Identification

Reproducible research repository for:

> Ramírez, J. M., Aguilar, J., Paredes, M., & Fernández-Anta, A. (2026).
> *Sentinel-2 Rule-based Semantic Clustering for Zone Identification
> Supporting Nature-Based Solutions in Valencia, Spain.*

Funded by the Urban-Swarm Project (MICIU/AEI/EU, Grant PCI2025-167066-2).

---

## Overview

This repository implements a pixel-wise unsupervised semantic land-surface
classification pipeline for Sentinel-2 Level-2A multispectral imagery.
The method combines k-means clustering with data-adaptive multi-class Otsu
thresholding to produce interpretable land-surface maps for nature-based
solution (NbS) planning.

**Seven semantic classes are produced:**

| ID | Class             | Color       |
|----|-------------------|-------------|
|  0 | Sparse vegetation | Light green |
|  1 | Bare soil / Rock  | Light brown |
|  2 | Urban             | Red         |
|  3 | Water             | Blue        |
|  4 | Dense vegetation  | Dark green  |
|  5 | Uncertain         | Magenta     |
|  6 | Clouds            | White       |

---

## Repository structure

```
.
├── run_pipeline.py          # Main end-to-end script (single entry point)
├── src/
│   ├── __init__.py
│   ├── palettes.py          # Color palettes and label-to-RGB conversion
│   ├── features.py          # Band extraction, spectral indices, preprocessing
│   ├── clustering.py        # K-means, centroid reordering, SCL cloud mask
│   └── semantic_labeling.py # Otsu thresholds, decision rules, class encoding
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/JuanMarcosRamirez/sentinel2-nbs-clustering.git
cd sentinel2-nbs-clustering
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data preparation

### Sentinel-2 Level-2A GeoTIFF

Download Level-2A products (< 30% cloud cover) from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu).

Stack all bands into a single multi-band GeoTIFF at 20-m resolution in
the following layer order (bands B2/B3/B4 resampled from 10 m;
B1/B9 excluded):

| Layer | Band | Wavelength | Region            |
|-------|------|-----------|-------------------|
|   0   | SCL  | —         | Scene Class Layer |
|   1   | B12  | 2190 nm   | SWIR-2            |
|   2   | B11  | 1610 nm   | SWIR-1            |
|   3   | B8A  |  865 nm   | Narrow NIR        |
|   4   | B7   |  783 nm   | Red-Edge 3        |
|   5   | B6   |  740 nm   | Red-Edge 2        |
|   6   | B5   |  705 nm   | Red-Edge 1        |
|   7   | B4   |  665 nm   | Red               |
|   8   | B3   |  560 nm   | Green             |
|   9   | B2   |  490 nm   | Blue              |

### DEM GeoTIFF

Any elevation raster in metres stored as a single-band
GeoTIFF.  The paper uses TanDEM-X at 90 m resampled to 20 m.

---

## Running the pipeline

1. Update `S2_PATH` and `DEM_PATH` in `run_pipeline.py`.
2. Run:

```bash
python run_pipeline.py
```

The script produces 6 figures and prints Otsu thresholds, the
cluster-to-label mapping, and a class area summary to stdout.

---

## Methodology notes

### K-means — k = 9
Selected by evaluating five complementary metrics (Silhouette,
Calinski–Harabász, Davies–Bouldin, ARI, NMI) over k ∈ {6, …, 12}.
k = 9 achieves the highest ARI (0.93 ± 0.12) and NMI (0.93 ± 0.11)
while retaining competitive geometric cluster quality (paper §4.1).

### Otsu thresholds
`threshold_multiotsu` from scikit-image is applied to the 2-D spatial
image of each index.  The number of classes per index:

| Index   | Classes | Thresholds        |
|---------|---------|-------------------|
| NDVI    |    4    | θ₁, θ₂, θ₃        |
| SWIR1   |    2    | θ₁                |
| MNDWI   |    2    | θ₁                |
| NDRE_B5 |    2    | θ₁                |
| NDBI    |    3    | θ₁, θ₂            |
| BSI     |    2    | θ₁ (from NDBI col)|
| UI      |    3    | θ₁, θ₂            |
| SVC     |    3    | θ₁, θ₂            |

### Cloud masking
SCL classes 3 (shadow), 8, 9, 10 (clouds/cirrus) → label 36.

### NDRE band
`ndre_b5` is computed as `(NIR − RED_EDGE[:,:,2]) / (NIR + RED_EDGE[:,:,2] + ε)`,
where `RED_EDGE[:,:,2]` is the third slice of `data[:,:,6:9]` = layer
index 8 = B7 (Red-Edge 3, 783 nm).

---

<!-- ## Citation

```bibtex
@article{ramirez2026sentinel2,
  author  = {Ram{\'i}rez, Juan Marcos and Aguilar, Jose and
             Paredes, Maylen and Fern{\'a}ndez-Anta, Antonio},
  title   = {Sentinel-2 Rule-based Semantic Clustering for Zone
             Identification Supporting Nature-Based Solutions
             in Valencia, Spain},
  year    = {2026},
}
```

---
-->

## License

MIT License — see `LICENSE`.
