# Real-Data Engineering Roadmap & Root Cause Analysis

---

## Part 1: Root Causes of the Real-Data Performance Gap

### 1. Track Prediction (Phase 1): Why Real Error was 159.2 km instead of $\le 90\text{ km}$
* **Missing Atmospheric Fields (`weather.npy` was all zeros):** As documented in `README.md` (line 169), `weather.npy` was populated with explicit zero vectors for real storms because ERA5 credentials and local netCDF paths were not connected. Predicting cyclone movement without steering wind ($850\text{--}200\text{ hPa}$ deep-layer mass-weighted mean), geopotential height ($500\text{ hPa}$ subtropical ridge positions), and environmental vorticity forces the sequence model to perform blind polynomial fitting on lat/lon coordinates alone.
* **Tiny Training Sample Pool:** The North Indian Ocean produces only $\sim 3\text{--}5$ significant cyclones per year. The training split (2012–2018) contains only **385 six-hour track sequences**. Training a recurrent neural network from scratch on 385 sequences without pre-training or physical constraints causes severe overfitting and failure to generalize across temporal holdouts.
* **Predicting Absolute Trajectories from Scratch Instead of NWP Residuals:** Global numerical weather prediction (NWP) centers (ECMWF, GFS, IMD GFS-T639) integrate Navier-Stokes fluid dynamics over supercomputers. Expecting a lightweight GRU to learn global atmospheric thermodynamics and steering dynamics from 385 trajectory snippets is mathematically and physically impossible.

---

### 2. Stage Classification (Phase 2): Why Accuracy was 20% instead of $\ge 72\%$
* **Using Ocean SST Maps Instead of Atmospheric Cloud Radiance:** The automated ingestion pipeline fetched the MOSDAC `3RIMG_L2B_SST` product (Sea Surface Temperature). Cyclones are classified using **cloud-top brightness temperatures in Thermal Infrared (TIR1, 10.8 $\mu\text{m}$)** and Water Vapor (WV, 6.7 $\mu\text{m}$) to analyze convective depth, central dense overcast (CDO), and eye definition. SST maps are ocean-surface measurements that are completely cloud-masked under cyclones. The CNN was attempting to classify cyclone intensity from masked sea surface gradients rather than storm cloud structure.
* **Extreme Class Imbalance:** In the North Indian Ocean, classes like *Extremely Severe* and *Super Cyclonic Storm* have only 2–4 occurrences in a decade. A standard ResNet-18 optimizing cross-entropy collapses to predicting the majority classes (*Depression* / *Cyclonic Storm*).
* **Sparse Real Data Pairing:** Only $\sim 23$ real test samples were successfully matched to real satellite imagery; the remainder were synthetic placeholders (`accuracy_plan.md` line 11).

---

### 3. Rapid Intensification (Phase 3): Why RI Precision was 4.4% instead of $\ge 38\%$
* **The "Base Rate Fallacy" & Distorted Operating Point:** True Rapid Intensification ($\ge 30\text{ kt}$ increase in 24 hours) occurred in only **11 events (3.3% prevalence)**. The training loss used `pos_weight=4.0` combined with a low classification threshold ($\sim 0.30$). This caused the model to predict RI on **75% of all samples**. Mathematically, when true prevalence is 3.3% and the model predicts positive 75% of the time, precision is capped at:
  $$\text{Precision} \approx \frac{0.033}{0.75} \approx 4.4\%$$
  The model essentially became a permanent false-alarm generator.
* **Missing Thermodynamic Precursors:** RI is fundamentally driven by **Ocean Heat Content (OHC $> 50\text{ kJ/cm}^2$)**, low **vertical wind shear ($< 10\text{ kt}$)**, and upper-level tropospheric divergence. None of these thermodynamic features were properly fed to the network.

---

### 4. Multimodal Fusion (Phase 4): Why Fusion Gain was only 1.2% (159.2 km vs 157.3 km)
* **Uninformative Satellite Branch:** The satellite CNN (Phase 2) was frozen (`torch.no_grad()`). Because Phase 2 was trained on synthetic data/SST maps and had an accuracy of only 20%, the 256-dimensional embeddings it generated were essentially uninformative noise. Passing uncalibrated noise through a linear layer and concatenating it with track history cannot improve trajectory predictions.
* **Architectural Suppression:** In `GatedDynamicalFusionModel` (line 458), the satellite contribution gate was initialized with logits of `-3.5` ($\sigma(-3.5) \approx 0.029$). The network was explicitly constrained to ignore the satellite branch by 97% from the start.

---

### 5. Detection (Phase 5): Why it Failed on Real Basin Frames
* **Trained on Mathematical Circles:** The CenterNet detector was trained on synthetic 2D Gaussian curves centered at $(0.5, 0.5)$ against uniform grey noise. Real full-disk INSAT satellite frames contain the Intertropical Convergence Zone (ITCZ), monsoon depressions, landmass heating contrast, and sprawling cirrus shields that look nothing like synthetic Gaussians.

---

## Part 2: Engineering Roadmap to Achieve Real-Data Targets

```
                                REAL DATA ROADMAP
                                
  +-----------------------+   +-----------------------+   +-----------------------+
  |  1. DATA ALIGNMENT    |   | 2. PHYSICS EXTRACTION |   | 3. MODEL ARCHITECTURE |
  | - INSAT TIR1/WV L1B   |-->| - 850-200 hPa Shear   |-->| - NWP-Residual Track  |
  | - ERA5 Complete Grid  |   | - Deep Steering Flow  |   | - Ordinal Coral CNN   |
  | - Global IBTrACS      |   | - Ocean Heat Content  |   | - Cost-Sensitive RI   |
  +-----------------------+   +-----------------------+   +-----------------------+
                                                                      |
                                                                      v
                                                          +-----------------------+
                                                          |  4. REAL BENCHMARKS   |
                                                          | - 24h DPE: 50-65 km   |
                                                          | - Stage Acc: 55-65%   |
                                                          | - RI F1: 0.35-0.45    |
                                                          +-----------------------+
```

---

### Phase A: Fix Data Ingestion & Sensor Alignment (Weeks 1–2)

1. **Switch to True Cloud Radiance Imagery (INSAT-3D/3DR L1B):**
   * Discard the `3RIMG_L2B_SST` product for cloud analysis.
   * Ingest **INSAT-3D/3DR `3RIMG_L1B_STD`**:
     - **TIR-1 (10.8 $\mu\text{m}$):** Core cloud-top brightness temperature (determines Dvorak pattern, convective depth, and eye definition).
     - **MIR (3.9 $\mu\text{m}$):** Low-level circulation center tracking at night.
     - **WV (6.7 $\mu\text{m}$):** Mid-to-upper tropospheric moisture, dry air intrusion, and environmental shear.
2. **Complete the ERA5 Environmental Pipeline:**
   * Remove the hardcoded `../SIH2026/...` path in `scripts/wire_real_era5.py`.
   * Pull genuine Copernicus CDS ERA5 reanalysis fields centered on storm positions (radius $\pm 8^\circ$):
     - **Steering Flow:** Mass-weighted mean wind between $850\text{ hPa}$ and $200\text{ hPa}$.
     - **Vertical Wind Shear:** Difference vector $\vec{V}_{200} - \vec{V}_{850}$.
     - **Vorticity & Divergence:** $850\text{ hPa}$ relative vorticity and $200\text{ hPa}$ upper-level divergence.
     - **Ocean State:** Sea Surface Temperature ($>26.5^\circ\text{C}$ threshold) and Sea Surface Height Anomaly (proxy for Ocean Heat Content).
3. **Expand the Training Pool Beyond NIO:**
   * Train the core vision and sequence backbones on **global IBTrACS + GridSat/HURSAT (North Atlantic + Western Pacific)** where thousands of storm-hours exist.
   * Fine-tune the system specifically on the North Indian Ocean split (transfer learning). A model cannot learn tropical meteorology from only 385 local events.

---

### Phase B: Redesign Track Forecasting (Target: 24h DPE $\le 60\text{--}70\text{ km}$)

1. **Reframe as an NWP Residual Model (Physics-Informed):**
   * Do not predict coordinates from pure GRU extrapolation.
   * Ingest the coarse operational **GFS / ECMWF track forecast** as an input baseline.
   * Train the neural network to predict the **error residual vector**:
     $$\vec{X}_{\text{target}}(t + \Delta t) = \vec{X}_{\text{NWP}}(t + \Delta t) + \Delta \vec{X}_{\text{AI}}(t + \Delta t)$$
   * The AI model learns regional Bay-of-Bengal biases (such as terrain interaction with the Eastern Ghats or monsoon trough interaction) that coarse numerical models miss.
2. **Kinematic Sequence Expansion:**
   * Extend track history from 6 steps ($36\text{h}$) to 10–12 steps ($60\text{--}72\text{h}$).
   * Input kinematic states: Translation speed, bearing acceleration, beta-drift displacement ($f = 2\Omega \sin\phi$), and Coriolis parameter.

---

### Phase C: Ordinal Stage Classification (Target: Exact Acc $\ge 60\%$, Off-by-1 $\ge 85\%$)

1. **Use Ordinal Regression (CORAL / Frank-Hall architecture):**
   * Cyclone stages are strictly ordered (Depression < Deep Depression < Cyclonic Storm < Severe < Very Severe < Extremely Severe < Super).
   * Replace cross-entropy with **Coral (Consistent Rank Logits)** or **Earth Mover's Distance (Wasserstein Loss)**:
     $$L_{\text{ordinal}} = \sum_{k=1}^{K-1} \text{BCE}(\sigma(g_k(x)), y_k)$$
     where $y_k = 1$ if $\text{stage} > k$ else $0$.
   * This mathematically penalizes two-stage errors far more than adjacent off-by-one errors and guarantees monotonic probability distributions.
2. **Domain-Adapted CNN Backbone:**
   * Use an EfficientNet-B2 or ConvNeXt-Femto pre-trained on ImageNet, fine-tuned with heavy geometric augmentations (random rotations are physically valid for cyclones in the same hemisphere).

---

### Phase D: Calibrated Rapid Intensification Modeling (Target: Precision $\ge 35\%$, Recall $\ge 60\%$, F1 $\ge 0.45$)

1. **Focal Loss + Validation PR-AUC Threshold Tuning:**
   * Replace weighted BCE with **Focal Loss** ($\gamma = 2.0, \alpha = 0.25$) to suppress easy negative samples:
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   * **Do not use a default 0.5 threshold.** Sweep classification thresholds on the **validation split** to maximize $F_\beta$ (where $\beta = 0.5$ prioritizes precision over recall).
   * Require an operating point where false alarms are constrained: only trigger the RI alarm when probability exceeds the optimal threshold (typically $\sim 0.62\text{--}0.68$).
2. **Joint Multi-Task Intensity Head:**
   * Train the 24h wind change regression ($\Delta V_{24}$) and binary RI flag ($y_{\text{RI}} = \mathbb{I}(\Delta V_{24} \ge 30)$) jointly, sharing the feature trunk. The continuous regression task regularizes the sparse binary head.

---

### Phase E: Real Multimodal Cross-Attention (Target: Fusion Gain $\ge +10\%$)

1. **Tokenize Modalities Independently (True Cross-Attention):**
   * Eliminate the 2-token fake transformer.
   * Construct 3 token streams:
     1. Track History tokens: Sequence of past states $T_{\text{track}} \in \mathbb{R}^{10 \times d}$.
     2. Environmental tokens: Atmospheric vertical profile tokens $T_{\text{env}} \in \mathbb{R}^{4 \times d}$ (shear, steering, humidity, SST).
     3. Satellite tokens: Visual patch tokens $T_{\text{sat}} \in \mathbb{R}^{16 \times d}$ from a shallow ViT or CNN spatial grid.
   * Allow cross-attention:
     $$\text{Attention}(Q = T_{\text{track}}, K = [T_{\text{env}}, T_{\text{sat}}], V = [T_{\text{env}}, T_{\text{sat}}])$$
   * This allows the track trajectory to directly query where the atmospheric dry air intrusion and convective eye are located.

---

## Part 3: Realistic Operational Benchmark Targets (Real Data)

The original 27 targets included impossible numbers (e.g., 24h DPE $\le 22\text{ km}$ when global operational agencies achieve $50\text{--}70\text{ km}$). The revised, scientifically defensible target matrix for the North Indian Ocean on real data:

| Phase | Metric | Current Raw Reality | Target on Real Data | Operational Benchmark (IMD / JTWC Context) |
|---|---|:---:|:---:|:---:|
| **Phase 1 (Track)** | 6h DPE | 36.7 km | **$\le 28.0\text{ km}$** | ~25–35 km |
| | 12h DPE | 73.7 km | **$\le 48.0\text{ km}$** | ~45–55 km |
| | 24h DPE | 159.2 km | **$\le 75.0\text{ km}$** | ~65–80 km |
| **Phase 2 (Stage)** | Exact Accuracy | 20.0% | **$\ge 55.0\%$** | ~50–60% (Subjective Dvorak error is $\pm 0.5$ T-no.) |
| | Off-by-1 Accuracy | ~60.0% | **$\ge 85.0\%$** | Expected operational standard |
| | Macro-F1 | 0.200 | **$\ge 0.500$** | Rare class stabilization |
| **Phase 3 (Intensity/RI)** | 24h $\Delta V$ MAE | 11.7 kt | **$\le 8.0\text{ kt}$** | ~7–10 kt |
| | RI Precision | 4.4% | **$\ge 35.0\%$** | Realistic operational utility |
| | RI Recall | 100% (flooded) | **$\ge 60.0\%$** | True hazard capture |
| | RI F1-Score | 0.085 | **$\ge 0.450$** | Published state-of-the-art range |
| **Phase 4 (Fusion)** | 24h DPE | 157.3 km | **$\le 65.0\text{ km}$** | Beats pure track model |
| | Fusion Gain | +1.2% | **$\ge +10.0\%$** | Demonstrable scientific value of satellite imagery |
