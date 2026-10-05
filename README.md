# Sentinel-2 Rule-based Semantic Clustering for NbS Zone Identification

Reproducible research repository for:

> Ramírez, J. M., Aguilar, J., Paredes, M., & Fernández-Anta, A. (2026).
> *Sentinel-2 Rule-based Semantic Clustering for Zone Identification
> Supporting Nature-Based Solutions in Valencia, Spain.*

Funded by the Urban-Swarm Project (MICIU/AEI/EU, Grant PCI2025-167066-2).

---

## Overview

This repository implements a pixel-wise, unsupervised semantic land-surface
classification pipeline for Sentinel-2 Level-2A multispectral imagery.
The method combines k-means clustering with data-adaptive Otsu thresholding
to produce interpretable land-surface maps relevant to nature-based solution
(NbS) planning in flood-prone Mediterranean landscapes.

The pipeline produces seven semantic classes:

| ID | Class             | Color in figures |
|----|-------------------|-----------------|
|  0 | Sparse vegetation | Light green      |
|  1 | Bare soil / Rock  | Light brown      |
|  2 | Urban             | Red              |
|  3 | Water             | Blue             |
|  4 | Dense vegetation  | Dark green       |
|  5 | Uncertain         | Magenta          |
|  6 | Clouds            | White            |

---

## Repository structure

```
.
├── run_pipeline.py          # Main end-to-end script
├── src/
│   ├── features.py          # Band extraction and spectral index computation
│   ├── clustering.py        # Preprocessing, k-means, cloud masking
│   ├── semantic_labeling.py # Otsu thresholds and decision rules
│   └── visualization.py     # Color palettes and figure helpers
├── data/
│   └── sample/              # Place your GeoTIFF files here
├── notebooks/
│   └── walkthrough.ipynb    # Step-by-step Jupyter walkthrough (optional)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/sentinel2-nbs-clustering.git
cd sentinel2-nbs-clustering

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data preparation

### Sentinel-2 Level-2A imagery
Download cloud-free Level-2A products (< 30 % cloud cover) from the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu).

Resample all bands to 20 m and stack them into a single multi-band
GeoTIFF with the following band order:

| Layer | Band | Wavelength | Region            |
|-------|------|------------|-------------------|
|   0   | SCL  | —          | Scene Class Layer |
|   1   | B12  | 2190 nm    | SWIR-2            |
|   2   | B11  | 1610 nm    | SWIR-1            |
|   3   | B8A  |  865 nm    | Narrow NIR        |
|   4   | B7   |  783 nm    | Red-Edge 3        |
|   5   | B6   |  740 nm    | Red-Edge 2        |
|   6   | B5   |  705 nm    | Red-Edge 1        |
|   7   | B4   |  665 nm    | Red               |
|   8   | B3   |  560 nm    | Green             |
|   9   | B2   |  490 nm    | Blue              |

Bands B2, B3, B4 (native 10 m) must be resampled to 20 m before
stacking. Bands B1 and B9 (60 m) are excluded.

### Digital Elevation Model (DEM)
The paper uses TanDEM-X (90 m), resampled to 20 m and co-registered
with the Sentinel-2 grid. Any co-registered elevation raster in metres
stored as a single-band GeoTIFF is accepted.

---

## Running the pipeline

1. Place your GeoTIFF files in `data/sample/` (or update the paths in
   `run_pipeline.py`).
2. Edit the `USER CONFIGURATION` block at the top of `run_pipeline.py`.
3. Run:

```bash
python run_pipeline.py
```

The script prints Otsu thresholds, the cluster-to-label mapping, and a
class area summary (km²), and displays six Matplotlib figures:

1. True-color RGB composite
2. Raw k-means cluster map
3. Cloud-corrected cluster map
4. Full-scene semantic classification map
5. Valley-zone semantic map (z < 300 m)
6. Upland-zone semantic map (z ≥ 300 m)

---

## Key methodology notes

**k = 9 clusters** was selected by evaluating five complementary metrics
(Silhouette, Calinski–Harabász, Davies–Bouldin, ARI, NMI) across k ∈
{6, …, 12}. k = 9 achieves the highest ARI (0.93 ± 0.12) and NMI
(0.93 ± 0.11) while retaining competitive geometric quality.

**Otsu thresholds** are computed independently for each spectral index
from its scene-level pixel distribution, making the semantic rules
self-calibrating across acquisition dates, seasons, and atmospheric
conditions. The number of Otsu classes per index is:

| Index  | Classes | Thresholds |
|--------|---------|------------|
| NDVI   |    4    | θ1, θ2, θ3 |
| SWIR1  |    2    | θ1         |
| MNDWI  |    2    | θ1         |
| NDRE   |    2    | θ1         |
| BSI    |    2    | θ1         |
| NDBI   |    3    | θ1, θ2     |
| UI     |    3    | θ1, θ2     |
| SVC    |    3    | θ1, θ2     |

**Cloud masking** uses SCL classes 3 (cloud shadow), 8 (medium-probability
cloud), 9 (high-probability cloud), and 10 (thin cirrus).

---

## Requirements

See `requirements.txt`. Core dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-image`
- `tifffile`
- `matplotlib`

---

## Citation

```bibtex
@article{ramirez2026sentinel2,
  author    = {Ram{\'i}rez, Juan Marcos and Aguilar, Jose and
               Paredes, Maylen and Fern{\'a}ndez-Anta, Antonio},
  title     = {Sentinel-2 Rule-based Semantic Clustering for Zone
               Identification Supporting Nature-Based Solutions
               in Valencia, Spain},
  year      = {2026},
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
