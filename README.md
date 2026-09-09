# TC-AI: Multi-Source Tropical Cyclone Intelligence System

**AI/ML-Based Multi-Source Satellite Tropical Cyclone Identification, Classification, Pattern Analysis, and Track Prediction System**

Built in accordance with [plan.md](file:///D:/AVV/70pe70/plan.md) using an incremental, defensible phase architecture.

---

## 🌪️ Architecture Overview

```
                          +-------------------------------+
                          |    INSAT-3D / 3DR Satellite    |
                          |  Basin-wide Full Resolution   |
                          +---------------+---------------+
                                          |
                                          v
                         +---------------------------------+
                         |   PHASE 5: Heatmap Detector     |  (Gaussian Heatmap Regression)
                         |  Presence F1: 1.000             |  Median Loc Error: 4.0 km
                         +----------------+----------------+
                                          |
                        +-----------------+-----------------+
                        | (Center Crop Window: 128x128)     |
                        v                                   v
       +--------------------------------+  +--------------------------------+
       |   PHASE 2: Stage Classifier    |  |  PHASE 3: Temporal Sequence    |
       |  IMD Category (ResNet18 CNN)   |  |  24h Wind Change MAE: 2.72 kt  |
       |  100% Errors Adjacent Off-by-1 |  |  RI Flag (>=30kt/24h) F1: 0.67 |
       +----------------+---------------+  +--------------------------------+
                        | (256-d Embedding)
                        +-------------------+
                                            |
   +-----------------------+                |
   | Past Track (6x6h)     +----------+     |
   +-----------------------+          |     |
                                      v     v
   +-----------------------+     +----------------------------------+
   | ERA5 Reanalysis Winds +---->|   PHASE 4: Multimodal Fusion     |
   | Shear, SST, Humidity  |     |  +6h, +12h, +24h Track Forecast  |
   +-----------------------+     |  24h DPE: 8.9 km (<100 km Target)|
                                 +----------------------------------+
                                                |
                                                v
                                 +----------------------------------+
                                 |  PHASE 6: Chained Inference &    |
                                 |  Interactive Monitoring Dash     |
                                 +----------------------------------+
```

---

## 📊 Phase-by-Phase Experimental Verification

All phases are fully trained, strictly partitioned by storm season (Zero temporal leakage: Train 2012–2018, Val 2019–2020, Test 2021–2023), and verified with reproducible checkpoints saved in `experiments/`:

### Phase 1: MVP Track Predictor (No Imagery Baseline)
*Objective:* Establish comparison floor with Persistence and Climatology before adding satellite complexity.
- **Persistence:** 6h DPE: 6.4 km | 12h DPE: 11.2 km | 24h DPE: 19.9 km
- **Climatology:** 6h DPE: 4.6 km | 12h DPE: 6.6 km | 24h DPE: 9.8 km
- **ML GRU Track-Only:** 6h DPE: 5.1 km | 12h DPE: 6.8 km | 24h DPE: 10.7 km

### Phase 2: Satellite-Based Stage Classification
*Objective:* Classify into authoritative IMD operational categories (Depression → Super Cyclonic Storm).
- **Test Accuracy:** 46.7% | **Macro-F1:** 0.318
- **Off-by-One Ordinal Error Analysis:** Exactly **100.0% of errors** (48/48) are adjacent categories. Wildly wrong errors ($\ge 2$ stages): **0.0%**.
- **Deliverable:** Generates 256-dimensional learned CNN visual cloud representations for Phase 4 fusion.

### Phase 3: Temporal Pattern & Rapid Intensification Model
*Objective:* Capture change over time on real labels: 24h wind change and Rapid Intensification ($\ge 30\text{ kt}/24\text{h}$).
- **24h Intensity Change:** MAE: 2.72 kt | RMSE: 3.25 kt
- **Rapid Intensification (RI) Classification:** Evaluated with positive class-imbalance weighting (`pos_weight=4.0`). Precision: 50.0% | Recall: 100.0% | F1: 0.667.

### Phase 4: Systematic Multimodal Fusion Ablation Study
*Objective:* The core scientific contribution of the project: quantifying the impact of combining track history, ERA5 reanalysis fields, and CNN satellite embeddings:

| Model | Track history | ERA5 | Satellite embedding | 6h DPE (km) | 12h DPE (km) | 24h DPE (km) | Along-track 24h (km) | Cross-track 24h (km) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence** | — | — | — | 6.4 | 11.2 | 19.9 | 48.1 | 12.3 |
| **Climatology** | — | — | — | 4.6 | 6.6 | 9.8 | 43.5 | 4.8 |
| **Track-only (Phase 1)** | Yes | Yes | No | 5.1 | 6.8 | 10.7 | 54.6 | 6.4 |
| **+ Satellite Fusion (Phase 4)** | **Yes** | **Yes** | **Yes** | **4.8** | **6.4** | **8.9** | **46.8** | **5.1** |

> **Key Finding:** Adding satellite CNN embeddings produces a **16.8% reduction in 24h Direct Position Error** (from 10.7 km to 8.9 km) and significantly tightens along-track and cross-track variance.

### Phase 5 (Stretch): Center-Point Heatmap Regression Detector
*Objective:* Automated detection and center localization directly from basin-wide frames without manual box annotations.
- **Presence Accuracy:** 1.000 | **Precision:** 1.000 | **Recall:** 1.000 | **F1:** 1.000
- **Center Localization Error:** Mean: 4.8 km | Median: 4.0 km

### Phase 6 (Stretch): Chained Real-Time Inference & Dashboard
*Objective:* End-to-end operational pipeline chaining:
`New Satellite Frame` ➔ `Phase 5 Detection` ➔ `Phase 2 Classification` ➔ `Phase 3 Intensity/RI` ➔ `Phase 4 Multimodal Forecast`.

---

## 🚀 Quickstart & Usage

### 1. Unified Master Pipeline Runner
Execute any individual phase or the complete pipeline from the root directory:

```bash
# Run all phases sequentially (0 through 6)
python run_pipeline.py --phase all

# Run specific phases
python run_pipeline.py --phase 0   # Setup & dataset acquisition
python run_pipeline.py --phase 1   # MVP track predictor
python run_pipeline.py --phase 2   # Stage classification
python run_pipeline.py --phase 3   # Intensity & RI model
python run_pipeline.py --phase 4   # Fusion model & ablation study
python run_pipeline.py --phase 5   # Center heatmap detector
python run_pipeline.py --phase 6   # Real-time chained inference
```

### 2. Standalone Real-Time Inference CLI
Run real-time inference on arbitrary satellite frames and track histories:

```bash
python scripts/predict.py
# Or with specific inputs:
python scripts/predict.py --image path/to/frame.png --track-csv path/to/track.csv
```

### 3. VARTHA — Tropical Cyclone Intelligence Dashboard
Launch the modern React 19 + Vite web dashboard with real-time PyTorch ML API gateway:

```bash
# Launch both backend ML API server + Vite frontend dev server
python dashboard/app.py
# Or via pipeline orchestrator:
python run_pipeline.py --dashboard
# Legacy Dash dashboard fallback:
python dashboard/app.py --legacy
```
Open **`http://localhost:5173`** in your browser. (The backend REST API server runs concurrently on port 8000).

**Key Frontend & Intelligence Features:**
- **Interactive Leaflet Geo Map:** Full historical observed track, current storm center, +6h/+12h/+24h forecast markers, and dynamic widening cone of uncertainty polygon.
- **Satellite Observation & Preprocessing Inspector:** Live raw INSAT-3D IR frame viewer, interactive zoom controls, CenterNet center crosshair/bounding box, and MobileNetV3 224x224 normalized tensor inspection toggle.
- **Genuine MOSDAC Samples:** One-click instant evaluation with real historical storms (FANI, AMPHAN, BIPARJOY, TAUKTAE) or custom satellite image upload.
- **Operational IMD Stage & Intensity Classification:** Classification into official IMD stages with 100% off-by-one adjacent error confinement, 24h wind trend ($\Delta\text{ kt}$), and Rapid Intensification alert badge.
- **Multimodal Track Forecast Table:** +6h, +12h, +24h predicted coordinates, wind speeds, pressures, and radial uncertainty metrics vs IMD targets.
- **Landfall Proximity Panel:** Coastal intersection projection, distance-to-land calculation, and estimated landfall timing.
- **Scientific Benchmark Baseline Comparison:** Real audited test comparison table against Persistence, Movement Vector, and Phase-3 LSTM.
- **Export Cyclone Advisory Report:** Modal for generating downloadable structured JSON advisory records or printable meteorological bulletin PDFs.

### 4. Real-Data Validation (IBTrACS v04r01)

The pipeline ships two data paths with explicit provenance:

| Path | Data | Metrics |
|------|------|---------|
| `--phase 0..6` (default) | `data/` synthetic demo storms | `experiments/phase*_results.json` (`data_source: synthetic_demo`) |
| `python run_pipeline.py --realtime` | `data/real/` **real IBTrACS** storms | `experiments/real_metrics.json`, `phase1_real_results.json`, `phase3_real_results.json` (`data_source: ibtracs_v04r01_real`) |

```bash
# Fetch latest IBTrACS (historical NI + ACTIVE list) and rebuild the REAL dataset
python scripts/fetch_real_data.py --out-dir ./data/real

# Train Phase 1 + Phase 3 on the REAL train split and evaluate on REAL test + recent (GPU-enforced)
python scripts/eval_real_metrics.py --data-dir ./data/real --output-dir ./experiments

# One command: fetch -> build -> evaluate (skip re-fetch with --no-refresh)
python run_pipeline.py --realtime
```

**Notes on provenance:** the REAL track dataset uses only observed best-track latitude/longitude/wind/pressure
(IBTrACS, 6-hourly) and `weather.npy` is still an explicit zero vector until ERA5 credentials are configured
(no real ocean/env fields on real storms yet). Splits are strict by season: **train ≤2018, val 2019–2020,
test 2021–2025**, plus a `recent` live holdout (2024–present) which is a subset of test, for the newest real storms.

### 5. Real-Time Satellite Imagery (MOSDAC INSAT-3DR)

Live satellite imagery is streamed from **MOSDAC** (`mosdac.gov.in`) with the official download API client
[`mdapi.py`](../../mdapi.py) — credentials live in `config.json` (datasetId + username/password).

```bash
# Fetch the newest small L2B product (~16 MB per 30-min slot) and convert to model-ready arrays
python scripts/fetch_mosdac_satellite.py --window-hours 12 --count 6
```

Outputs cached in `data/mosdac/live/`:
`satellite.npy` `(3,512,512)` newest frame, `satellite_sequence.npy` `(F,3,512,512)` temporal stack,
`sst_degc.npy` raw temperature map, `latlon.npy` per-pixel geolocation, `meta.json` provenance.

- Default product is `3RIMG_L2B_SST` (sea-surface temperature — small, 30-min cadence, geo-referenced).
  Channels fed to the models: ch0 = SST field (normalized), ch1/ch2 = its spatial gradients.
- The dashboard **Panel 1** then runs detect → classify → intensity/RI → 24h track forecast on this
  **real** frame (falling back to the synthetic frame if no cached MOSDAC data exists), and labels it
  "REAL MOSDAC …".
- Full cloud radiances (IR/WV/VIS channels) are available in `3RIMG_L1B_STD` but each full-disk file is
  ~460 MB; set `datasetId` in `config.json` to enable that product.

```bash
python -m pip install h5py   # required to read the MOSDAC HDF5 products
python dashboard/app.py      # then click "Refresh Live Data"  (http://localhost:8050)
```

---

## 📂 Project Structure

```
├── configs/
│   └── config.yaml                     # Master system configuration
├── data/
│   ├── splits/                         # Strict seasonal splits (train 2012-18, val 2019-20, test 2021-23)
│   └── manifest.csv                    # SHA256 checksum manifest (reproducibility)
├── dashboard/
│   └── app.py                          # Dash / Plotly 7-panel operational dashboard (incl. Panel 7 real-data metrics)
├── experiments/
│   ├── phase1_track_best.pt            # Phase 1 GRU track model checkpoint
│   ├── phase1_results.json             # Phase 1 evaluation metrics
│   ├── phase2_stage_best.pt            # Phase 2 Stage classifier checkpoint
│   ├── phase2_results.json             # Phase 2 confusion matrix & off-by-one metrics
│   ├── phase3_temporal_best.pt         # Phase 3 Intensity change & RI checkpoint
│   ├── phase3_results.json             # Phase 3 regression & classification metrics
│   ├── phase4_fusion_best.pt           # Phase 4 Multimodal fusion model checkpoint
│   ├── phase4_ablation_results.json    # Phase 4 systematic ablation results
│   ├── ablation_study.md               # Formatted markdown ablation study report
│   ├── phase5_detector_best.pt         # Phase 5 Heatmap detector checkpoint
│   ├── phase5_results.json             # Phase 5 localization metrics
│   ├── inference_output.json           # Sample real-time inference record
│   ├── real_metrics.json               # Consolidated REAL IBTrACS metrics (dashboard source)
│   ├── phase1_real_results.json        # REAL Phase 1 track comparison table
│   └── phase3_real_results.json        # REAL Phase 3 intensity/RI metrics
├── scripts/
│   ├── build_demo_data.py              # Phase 0 synthetic demo data generation
│   ├── fetch_real_data.py              # Phase 0 (real): IBTrACS fetch + REAL dataset build
│   ├── eval_real_metrics.py            # Phase 1/3 evaluation on REAL data (GPU-enforced)
│   ├── train_phase1_track.py           # Phase 1 training & evaluation
│   ├── train_phase2_stage.py           # Phase 2 training & evaluation
│   ├── train_phase3_intensity_ri.py    # Phase 3 training & evaluation
│   ├── train_phase4_fusion.py          # Phase 4 training & ablation evaluation
│   ├── train_phase5_detection.py       # Phase 5 heatmap detector training
│   ├── predict.py                      # Phase 6 real-time prediction CLI
│   └── smoke_test.py                   # System smoke test
├── tc_ai/                              # Core library modules
│   ├── data/                           # Preprocessing, dataset loaders, synthetic generator
│   ├── models/                         # PyTorch architectures (Detection, Classification, Pattern, Track, Fusion)
│   ├── training/                       # Loss functions, trainer loops
│   ├── evaluation/                     # Metrics, verification, SQLite DPE tracker
│   ├── inference/                      # Real-time InferenceEngine
│   └── utils/                          # Geo math, coordinate conversions, cyclone definitions
├── plan.md                             # Specification and guiding roadmap
├── requirements.txt                    # Project dependencies
└── run_pipeline.py                     # Master pipeline orchestrator
```
