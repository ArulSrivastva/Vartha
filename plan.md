# TC-AI: Final Roadmap
### AI/ML-Based Multi-Source Satellite Tropical Cyclone Identification, Classification, Pattern Analysis and Track Prediction System

---

## 0. Guiding principle

Build in **phases that each produce a complete, working, evaluable system** — not one giant architecture that only works when every module is finished. Each phase below stands on its own as a demonstrable result. Later phases extend, never block, earlier ones.

The long-term research vision (full multimodal fusion of satellite + atmosphere + ocean + NWP for detection, classification, pattern recognition, and forecasting) remains the north star and final report framing — but it is reached incrementally.

---

## 1. Phase overview (at a glance)

| Phase | Goal | Core Inputs | Core Output | Realistic Timeframe |
|---|---|---|---|---|
| 0 | Setup & data acquisition | — | Clean, versioned datasets | 1–2 weeks |
| 1 | MVP track predictor | IBTrACS + ERA5 | 6/12/24h track forecast + DPE | 2–3 weeks |
| 2 | Satellite-based stage classification | INSAT IR/WV + IMD labels | Cyclone category classifier | 2–3 weeks |
| 3 | Temporal intensity / pattern model | Satellite sequences + track history | Intensity trend + RI probability | 2–3 weeks |
| 4 | Fusion track model | Phase 1 + Phase 2/3 features | Improved track + intensity forecast | 2 weeks |
| 5 (stretch) | Detection/localization | Raw satellite frames | Cyclone presence + center localization | 2–3 weeks |
| 6 (stretch) | Dashboard & real-time inference | All above | Live monitoring UI | 1–2 weeks |

Total for Phases 0–4 (the credible, completable core): **~10–13 weeks**. Phases 5–6 are additive if time/compute allow.

---

## 2. Phase 0 — Setup & Data Acquisition

**Objective:** Get clean, aligned, version-controlled data before writing any model code.

**Tasks:**
- Register and pull:
  - **IBTrACS** (NOAA) — global historical best-track data: lat/lon, wind, pressure, category, every 6h, per storm. This is your backbone dataset.
  - **IMD/RSMC New Delhi** best-track bulletins — for North Indian Ocean storms, as the authoritative regional label source (use where it overlaps/refines IBTrACS).
  - **ERA5 reanalysis** (ECMWF/Copernicus) — atmospheric fields: wind (u/v at multiple levels), humidity, temperature, geopotential, at 6-hourly resolution, subset spatially around each storm's track.
  - **INSAT-3D/3DR imagery** via MOSDAC — IR, water vapor, visible channels (register early; approval can take time).
- Build a **storm-centric data structure**: for each storm ID and timestamp, one record containing track truth + matched ERA5 fields + matched satellite frame (nearest in time, spatially cropped around the storm center).
- Define the **train/val/test split by storm season**, not by random frame (e.g., train 2005–2018, validate 2019–2020, test 2021–2023). This is non-negotiable — frame-level random splits will leak and invalidate every downstream result.
- Set up a lightweight data versioning approach (even a simple manifest CSV + checksums is fine) so every experiment is reproducible.

**Deliverable:** A documented, reproducible data pipeline and a storm-indexed dataset table.

---

## 3. Phase 1 — MVP Track Predictor (no imagery yet)

**Objective:** Prove the core forecasting pipeline works, establish your baseline, before adding satellite complexity.

**Inputs:** Sequence of past track states (lat, lon, wind, pressure, heading, translation speed) + co-located ERA5 atmospheric features (steering wind, vertical shear, humidity).

**Model:** Sequence model — start simple:
- Baseline: **persistence** (assume storm continues on current heading/speed) and **climatology** (average behavior for that basin/season) — these are your comparison floor, not optional.
- ML model: **GRU or LSTM** over the last 4–8 track states → predicts lat/lon offset at +6h, +12h, +24h. A small Transformer encoder is a reasonable upgrade once the GRU baseline works.

**Metrics:**
- Direct Position Error (DPE) at 6h/12h/24h, compared against persistence/climatology baselines and, where available, published IMD/JTWC operational verification figures for context.
- Along-track and cross-track error breakdown.

**Deliverable:** A working, evaluated track-forecasting model with a results table (Model vs. Persistence vs. Climatology). **This alone is a complete, defensible project checkpoint.**

---

## 4. Phase 2 — Satellite-Based Stage Classification

**Objective:** Add satellite imagery, but scope classification to labels that actually exist (IMD/IBTrACS intensity category), not invented structural taxonomies.

**Inputs:** INSAT IR (primary) + WV channel, cropped to a fixed window around the storm center (from Phase 0 track truth).

**Labels:** IMD/IBTrACS operational category at that timestamp — Depression → Deep Depression → Cyclonic Storm → Severe → Very Severe → Extremely Severe (use the real IMD category boundaries, don't invent new ones).

**Model:** CNN classifier (ResNet-style or a small custom CNN — don't reach for ViT until the CNN baseline works and you have enough data to justify it). Multi-class classification, optionally ordinal-aware loss since categories are ordered.

**Metrics:** Accuracy, macro-F1 (categories are imbalanced — few extremely severe cases), confusion matrix (check whether errors are "off by one category" — expected — vs. wildly wrong).

**Deliverable:** A trained classifier with per-category performance and a discussion of where it confuses adjacent categories (this is expected and worth analyzing, not hiding).

---

## 5. Phase 3 — Temporal Pattern / Intensity-Change Model

**Objective:** Capture *change over time*, scoped to what's actually labeled: intensity trend and rapid intensification (RI), rather than hand-crafted structural labels like "eye forming" or "sheared" (which have no ground-truth source without manual annotation).

**Inputs:** Sequence of the last 4–6 satellite frames (from Phase 2's crop pipeline) + track history.

**Labels (derived, not invented):**
- Intensity change over next 24h (regression, from IBTrACS wind speed deltas) — this **is** available in your existing data.
- RI flag: binary label using a defined operational threshold (e.g., a documented ≥30kt/24h criterion) — derivable directly from IBTrACS wind speed sequences.

**Model:** ConvLSTM or a CNN feature extractor per frame + GRU over the sequence (simpler and more trainable than a full spatio-temporal Transformer at this stage).

**Metrics:** MAE/RMSE on intensity-change regression; precision/recall/F1 on RI detection (expect low precision initially — RI is rare and hard; report it honestly).

**Note on structural pattern labels (eye, shear, asymmetry):** treat as a **stretch addition** only if time allows — either via weak/proxy labels derived from ERA5 shear fields, or a small manually-annotated subset. Do not present these as core deliverables unless labeled.

**Deliverable:** An intensity-trend/RI model with honestly-reported metrics, including a discussion of class imbalance handling (RI events are rare).

---

## 6. Phase 4 — Fusion Model

**Objective:** Combine Phase 1's track model with Phase 2/3's satellite-derived features to see if satellite information improves track/intensity forecasts over the track-only baseline.

**Approach:** Take the Phase 2/3 CNN's learned embedding (not raw pixels) as an additional input feature to the Phase 1 GRU/Transformer. Retrain.

**This is where your ablation study lives** — the single most important scientific result of the project:

| Model | Track history | ERA5 | Satellite embedding | 6h DPE | 12h DPE | 24h DPE |
|---|---|---|---|---|---|---|
| Persistence | — | — | — | | | |
| Track-only (Phase 1) | ✓ | ✓ | ✗ | | | |
| + Satellite (Phase 4) | ✓ | ✓ | ✓ | | | |

This table is what demonstrates the actual contribution of multi-source fusion — it's more convincing than any architecture diagram.

**Deliverable:** Final fusion model + ablation table + written analysis of whether/where satellite data helped.

---

## 7. Phase 5 (Stretch) — Detection / Localization

**Objective:** Automate cyclone presence detection and center localization directly from a full satellite frame (rather than assuming a pre-cropped, pre-centered image as earlier phases do).

**Reality check first:** IBTrACS/IMD gives you a center point per 6h, not a bounding box. Two viable approaches:
- **Center-point heatmap regression** (predict a Gaussian-peaked heatmap centered on the true storm location) — easier to derive ground truth for than boxes, and standard in this kind of geophysical detection task.
- **Box-based detection (YOLO)** — only if you're willing to manually annotate a subset of frames, since no public box-labeled dataset exists for this.

**Recommendation:** Do heatmap regression unless someone on the team has bandwidth to hand-label boxes.

**Deliverable:** A detector that takes a raw basin-wide satellite frame and outputs storm presence + estimated center, validated against IBTrACS-known storm positions and dates.

---

## 8. Phase 6 (Stretch) — Real-Time Inference & Dashboard

**Objective:** Wrap the trained pipeline in something demonstrable.

**Components:**
- Inference script: new satellite frame → detection (if Phase 5 built) → classification → intensity/RI → track forecast, chained together.
- Simple dashboard (a web app, not necessarily elaborate) showing: current classification, forecast track with uncertainty, and RI probability.
- **Do not build uncertainty quantification, live data feeds, or a polished production dashboard before Phases 1–4 are solid** — this is presentation layer, not the research contribution.

**Deliverable:** A demo-able end-to-end pipeline, even if running on historical "replay" data rather than truly live feeds (live feed integration is its own project).

---

## 9. Evaluation summary (what to report at the end regardless of how many phases you complete)

- **Track:** DPE at 6/12/24h vs. persistence/climatology baselines, along-track/cross-track breakdown.
- **Classification:** Accuracy, macro-F1, confusion matrix, off-by-one-category analysis.
- **Intensity/RI:** MAE/RMSE on intensity change, precision/recall on RI flag.
- **Ablation table:** contribution of each data source (this is your strongest scientific claim).
- **Honest limitations section:** data scarcity for severe/RI events, satellite coverage gaps, temporal-split methodology, and what wasn't attempted (e.g., structural pattern labels, live SAR fusion) and why.

---

## 10. What was deliberately cut from the original architecture (and why)

| Original element | Why deferred/cut |
|---|---|
| Simultaneous multi-satellite fusion (INSAT + Himawari + GOES + SAR + scatterometer) at Phase 1 | Each source adds significant preprocessing/access overhead; start with INSAT only, add sources only after the pipeline works |
| Hand-crafted structural pattern taxonomy (eye forming, sheared, asymmetric, etc.) as a core module | No existing ground-truth labels; would require manual annotation effort not currently budgeted |
| End-to-end joint multi-task model (detection+classification+pattern+track+intensity+uncertainty) from the start | Very heavy compute/data requirement; staged single-task models are more debuggable and each produces standalone results |
| Full uncertainty cones / production dashboard early | Presentation layer; valuable only once the underlying forecasts are validated |
| Vision Transformer as first choice for imagery | CNNs are more data-efficient at the dataset sizes typically available here; ViT is a reasonable later upgrade, not a starting point |

---

## 11. Suggested team/resource allocation (if working in a group)

- **1–2 people:** Data pipeline (Phase 0) + track model (Phase 1) — this is the most schedule-critical path.
- **1–2 people:** Satellite classification (Phase 2) — can start in parallel once Phase 0's satellite-crop pipeline is ready.
- **1 person:** Evaluation framework + ablation study design (Phase 4) — should start early so metrics are defined before models exist, not after.
- Stretch phases (5–6) only staffed once Phases 1–4 are stable.

---

## 12. Final framing for your report

> A staged, multimodal AI framework for tropical cyclone track and intensity forecasting, beginning with a track-history and reanalysis-based baseline, extended with INSAT satellite-derived classification and intensity-trend features, and evaluated through a systematic ablation study quantifying the contribution of each data source — with detection/localization and real-time dashboarding as extensions beyond the validated core system.

This framing lets you claim the full multi-source vision as context/future work while presenting Phases 0–4 as your actual, defensible, evaluated contribution./