# 🔥 Satellite Wildfire Detection with U-Net

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
[![CI](https://github.com/eternalrot/satellite-wildfire-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/eternalrot/satellite-wildfire-unet/actions)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange)

Pixel-level wildfire segmentation from **Sentinel-2** multispectral imagery using a **U-Net**, with an interactive web map that overlays the detections on **NASA FIRMS** hotspots.

> 🚧 **Work in progress.** The sections below describe the planned pipeline. Results will be added once training is complete.

---

## Overview

Given a Sentinel-2 scene of a region, the model outputs a binary mask marking active fires and burned areas. A FastAPI service exposes this as GeoJSON, and a Folium/Leaflet map visualizes it.

```
 NASA FIRMS ──┐
              ├─► match by location + date (±1–2 days) ─► 256×256 patches + masks
 Sentinel-2 ──┘                                                   │
                                                                   ▼
          Folium map ◄── FastAPI (bbox + date → GeoJSON) ◄── U-Net (PyTorch)
```

## Why ML instead of a threshold?

Simple index thresholds (e.g. NBR) are easily fooled by clouds, smoke, sun glint, and hot bare ground. A U-Net learns spectral **and** spatial patterns across bands, which helps separate real fires from these false positives. A plain NBR threshold is used as the **baseline** for comparison.

## Pipeline

| Step | Description |
|------|-------------|
| **1. Collect** | FIRMS API (lat/lon, date, confidence, FRP) + Sentinel-2 from Copernicus Data Space. SWIR bands B11/B12 are the most informative for heat and burn scars. |
| **2. Build dataset** | Match hotspots to scenes (±1–2 days), cut 256×256 patches, generate masks from buffered hotspots and from NBR = (B8 − B12) / (B8 + B12). Split **by region**, not randomly, to avoid leakage. |
| **3. Train** | U-Net, input bands B2, B3, B4, B8, B11, B12 → binary mask. Loss: Dice + BCE. Augmentations: flips, rotations, brightness shifts. |
| **4. Evaluate** | IoU, Dice/F1, precision, recall on held-out regions, compared to the NBR baseline. |
| **5. Serve** | FastAPI endpoint + Folium/Leaflet map. |

## Tech Stack

| Area | Tools |
|------|-------|
| Deep learning | PyTorch, segmentation-models-pytorch |
| Geodata | rasterio, geopandas, shapely, pyproj |
| Data handling | numpy, pandas, scikit-learn |
| Data sources | FIRMS API, Copernicus Data Space (sentinelsat / openEO) |
| Backend | FastAPI |
| Visualization | Folium, Matplotlib |
| Experiment tracking | Weights & Biases or MLflow |
| DevOps | Docker, GitHub Actions, pytest |

## Project Structure

```
satellite-wildfire-unet/
├── data/                 # raw and processed data (git-ignored)
├── notebooks/            # exploration and visualization
├── src/
│   ├── data/             # FIRMS / Sentinel-2 download, matching, patching
│   ├── models/           # U-Net, losses
│   ├── train.py
│   ├── evaluate.py
│   └── baseline_nbr.py
├── api/                  # FastAPI app
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/eternalrot/satellite-wildfire-unet.git
cd satellite-wildfire-unet

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your API credentials:

```bash
export FIRMS_MAP_KEY="your_firms_key"
export CDSE_USER="your_copernicus_username"
export CDSE_PASSWORD="your_copernicus_password"
```

## Usage

**1. Download and prepare data**

```bash
python -m src.data.download --region portugal --start 2025-07-01 --end 2025-09-30
python -m src.data.build_dataset --patch-size 256 --buffer-m 500
```

**2. Train**

```bash
python -m src.train --encoder resnet34 --epochs 50 --batch-size 16 --loss dice_bce
```

**3. Evaluate against the NBR baseline**

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --split test
python -m src.baseline_nbr --split test
```

**4. Run the API**

```bash
uvicorn api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bbox": [-8.2, 39.5, -7.8, 39.9], "date": "2025-08-14"}'
```

Example response:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"confidence": 0.93},
      "geometry": {"type": "Polygon", "coordinates": [[[-8.01, 39.71], [-8.00, 39.71], [-8.00, 39.72], [-8.01, 39.72], [-8.01, 39.71]]]}
    }
  ]
}
```

**5. Docker**

```bash
docker build -t wildfire-unet .
docker run -p 8000:8000 wildfire-unet
```

## Model

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",   # optional transfer learning
    in_channels=6,                # B2, B3, B4, B8, B11, B12
    classes=1,
)
```

## Results

> To be added after training.

| Method | IoU | Dice / F1 | Precision | Recall |
|--------|-----|-----------|-----------|--------|
| NBR threshold (baseline) | – | – | – | – |
| U-Net | – | – | – | – |

## Challenges

- **Class imbalance**: fire is under 1% of pixels → Dice + BCE loss.
- **Noisy labels**: FIRMS accuracy is ~375 m–1 km vs. 10–20 m Sentinel-2 pixels (weak supervision).
- **Clouds and date mismatch** between hotspots and acquisitions.
- **Data size**: the study is limited to one or two regions (e.g. Portugal, Greece, or California fire seasons).

## Roadmap

- [ ] Data download scripts (FIRMS + Sentinel-2)
- [ ] Dataset builder (patches, masks, region-based split)
- [ ] NBR baseline
- [ ] U-Net training with Dice + BCE
- [ ] Evaluation report
- [ ] FastAPI endpoint
- [ ] Folium map with FIRMS overlay
- [ ] Docker + GitHub Actions CI
- [ ] Pretrained encoder comparison (ResNet / EfficientNet)
- [ ] U-Net vs. patch-level classifier

## Data Sources

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) (Sentinel-2)

## License

This project is licensed under the MIT License.

## Author

**Pavlo Budko** — [@eternalrot](https://github.com/eternalrot)
