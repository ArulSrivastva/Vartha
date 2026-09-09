# Accuracy Maximization Plan (with per-phase targets)

> Primary goal: **maximize measurable accuracy at every phase**, evaluated on the **seasonally split test and recent (2024–25) holdout sets**, with all results recorded in `experiments/` (JSON + metric tables).
>
> Hard constraints from `AGENTS.md`: all training on **CUDA** (raise `RuntimeError` if unavailable), seasonal train/val/test splits only.

---

## 0. The single biggest lever: real data, not models

Every phase is currently trained/evaluated on **mostly synthetic satellite frames** (only ~23/331 real test samples matched to INSAT imagery) and partially-zero ERA5 vectors. Model improvements top out quickly under synthetic inputs; measured multipliers of impact across all phases:

1. **Real INSAT IR/WV frames matched per sample** (biggest single win for Phases 2, 3, 4, 5) — via `scripts/populate_train_val_satellite.py` + MOSDAC fetch (`scripts/fetch_mosdac_satellite.py`).
2. **Complete ERA5 features** (no zero-fallback rows) — via `scripts/wire_real_era5.py`.
3. **More storm-years** (lower `--min-season`, add IMD/RSMC bulletins to extend IBTrACS) to grow the rare-category pool (RI, Extremely Severe).
4. Then and only then: model architecture / loss / hyperparameter optimization.

**Rule:** never report a phase as "done" on synthetic inputs. Always require a real-matched-subset evaluation.

---

## 1. Cross-cutting protocol (applies to every phase)

- Evaluate on **both** `test` and `recent` splits (see `eval_real_metrics.py`); report both in `experiments/*_results.json`.
- Lock the split manifest, seed, and environment; checkpoints + JSON summaries committed under `experiments/`.
- Early-stopping/model-selection on **validation only** (never test); tune thresholds on val.
- Track a **burn-in baseline first** (persistence / climatology / majority-class), then improve on it. A model that beats the floor is publishable; one that doesn't is a bug, not "the model."
- Confirm reproducibility of every number via the SHA256 manifest in Phase 0.

---

## 2. Phase 1 — Track prediction (DPE at 6/12/24 h)

**Current (test / recent, ML model):** DPE 36.7 / 73.7 / 159.2 km  (recent: 36.9 / 75.3 / 166.7)
**Baselines:** persistence 44.4 / 86.5 / 183.9; climatology 64.7 / 127.4 / 254.5

| Tier | 6h DPE | 12h DPE | 24h DPE | 6h hit<25 | 12h hit<50 | 24h hit<100 |
|---|---|---|---|---|---|---|
| Minimum (beat baselines everywhere) | ≤ 36 | ≤ 72 | ≤ 155 | ≥ 0.38 | ≥ 0.40 | ≥ 0.40 |
| **Target** | **≤ 32** | **≤ 65** | **≤ 140** | **≥ 0.45** | **≥ 0.45** | **≥ 0.45** |
| Stretch | ≤ 30 | ≤ 60 | ≤ 130 | ≥ 0.50 | ≥ 0.50 | ≥ 0.50 |

Note: hit-rate targets use the code's operational windows (<25/50/100 km). The stretch tier approaches published NWP/ML verification in NIO.

### Impact-ranked actions
1. **Complete ERA5 wiring** — the 64-dim vectors are currently partial (many zero-fallback). Real steering wind + shear + SST gradients is the single strongest physics signal. Verify no `fallback` rows remain after `wire_real_era5.py`.
2. **Physics features** are already available (`DynamicalTrackModel`, 72-dim): add deep-layer mean steering (from 200–850 hPa real fields), vertical wind shear, and β-drift consistently; the model already has them — feed them real values.
3. **Multi-horizon losses:** train with a decaying weight per horizon (6h > 12h > 24h) and heteroscedastic NLL (`gaussian_nll`), already supported. Prefer NLL over plain MSE so the 24h head can model spread.
4. **Deeper history (8–10 points, 48–60 h)** before forecasting; add a Transformer encoder only after GRU saturates.
5. **NWP residual head** (``NWPResidualTrackModel`) — correct an AIFS/IFS AI forecast (already coded) instead of predicting the track from scratch.
6. **Augmentation for kinematic inputs:** heading/speed jitter on history (torch aug, train only) to regularize against overfit on 385 training samples.

---

## 3. Phase 2 — Satellite stage classification (6 IMD classes)

**Current:** exact accuracy 20% / macro-F1 0.20 (test); real-matched subset accuracy 22%, mean stage distance 1.87.
Extremely Severe class has near-zero support; model rarely emits it.

| Tier | Exact accuracy | Macro-F1 | Mean stage distance | Off-by-one correctness |
|---|---|---|---|---|
| Minimum (real subset) | ≥ 0.40 | ≥ 0.35 | ≤ 1.2 | ≥ 0.60 |
| **Target (real subset)** | **≥ 0.55** | **≥ 0.50** | **≤ 0.9** | **≥ 0.75** |
| Stretch | ≥ 0.65 | ≥ 0.60 | ≤ 0.7 | ≥ 0.85 |

### Impact-ranked actions
1. **Real satellite frames for all samples** (not synthetic). Synthetic IR patterns are generated deterministically from the same wind speed used to derive the label — the classifier currently learns the generator, not storms. This is the largest gap.
2. **Only report real-matched subset** as "classification accuracy"; keep synthetic-set numbers in a footnote labeled as synthetic.
3. **Ordinal training is already on** (`OrdinalStageLoss`); keep it — it is the single best structural fix for adjacent-stage confusion.
4. **Class weighting + WeightedRandomSampler** already present; tune support so Severe/Very Severe/Extremely Severe aren't crushed by prevalence.
5. **Two-stage reframe:** predict wind category from satellite (regression to wind kt) and map to stage — smooths ordinal errors and doubles as Phase 3 input.
6. **Backbone upgrade:** ResNet-18 → EfficientNet/B4 or ResNet-50 **only after real data** is in; with ~300–600 real frames a heavy backbone will overfit first.
7. **Analyze the confusion matrix each run:** target “errors are ±1 category and rarely ≥ 2” — that's the acceptance criterion, alongside raw accuracy.

---

## 4. Phase 3 — Intensity change + Rapid Intensification (RI)

**Current:** ΔV24h MAE 11.7 kt / RMSE 15.9 (recent: 8.3 / 11.4); RI precision 0.044 / recall 1.0 / F1 0.085 (threshold 0.3 — effectively predicts “RI” for 75% of samples). RI ground-truth prevalence 3.3% (11 events).

| Tier | ΔV24h MAE | ΔV24h RMSE | Bias | RI F1 | RI precision | RI recall |
|---|---|---|---|---|---|---|
| Minimum | ≤ 10 | ≤ 14 | \|·\| ≤ 2 | ≥ 0.15 | ≥ 0.12 | ≥ 0.50 |
| **Target** | **≤ 8** | **≤ 12** | **\|·\| ≤ 2** | **≥ 0.30** | **≥ 0.25** | **≥ 0.50** |
| Stretch | ≤ 6 | ≤ 10 | \|·\| ≤ 1 | ≥ 0.40 | ≥ 0.35 | ≥ 0.60 |

### Impact-ranked actions
1. **Fix the RI calibration problem first** — an F1 of 0.08 with 3.3% prevalence means the operating point is wrong, not just the model. Use the existing val-threshold grid search, and require **precision ≥ recall/2** (no more all-positive flooding).
2. **Use real temporal satellite sequences** (Phase 2 crop pipeline, 6 frames) as model inputs; current best checkpoint is satellite-free (`TrackIntensityModel`). Real imagery is where RI signal lives (eye/eyewall evolution).
3. **Balance RI in the loss:** `pos_weight` 4→8 is already available; escalate to focal loss for the binary head if RI recall collapses.
4. **Feature stack for RI:** add SST / ocean heat content and vertical wind shear (both already loadable) — these are the two strongest physical RI precursors.
5. **Output both heads jointly** (already the architecture): share the encoder between ΔV regression and RI binary → the regression acts as a regularizer.
6. **Report RI honestly at low prevalence:** if the pool stays 11 events, treat F1 ≥ 0.30 as exceptional and document the tiny sample; add IMD bulletins to grow the RI pool if possible.

---

## 5. Phase 4 — Fusion track model + ablation

**Current:** 6h 36.6 / 12h 70.4 / 24h 157.3 (test) vs Phase 1 track-only 36.7 / 73.7 / 159.2 — satellite adds **~1%**, i.e., no demonstrated value yet.

The scientific deliverable of this phase is the **ablation table**, and the target is a *meaningful, statistically defensible* improvement from satellite — not mere noise.

| Tier | 6h DPE | 12h DPE | 24h DPE | Δ vs track-only (24h) |
|---|---|---|---|---|
| Minimum (any real improvement) | ≤ 35 | ≤ 70 | ≤ 155 | ≥ 3% |
| **Target** | **≤ 33** | **≤ 62** | **≤ 140** | **≥ 8–10%** |
| Stretch | ≤ 30 | ≤ 55 | ≤ 130 | ≥ 15% |

### Impact-ranked actions
1. **Use a phase-locked satellite encoder** — the Phase 2 classifier is being trained on synthetic frames; its embeddings carry synthetic fingerprints. This is *the* reason fusion shows no gain. Re-train the encoder on real frames, then freeze and attach to the track GRU.
2. **Jointly fine-tune the satellite encoder with a small LR** (already coded in `eval_real_metrics.py`, `sat_lr=1e-4`) — frozen-embedding fusion underperforms joint fine-tuning here.
3. **Ablate each modality cleanly:** track-only → +ERA5 → +satellite, plus satellite-only head, same seed, same split. Keep exactly the table format already in `ablation_study.md`.
4. **Add an attention-based fusion gate** (the `MultimodalTransformer` exists) instead of naive concat, so ERA5 vs satellite contributions are weighted per sample type.
5. **Only claim "satellite helped" if the Δ exceeds the run-to-run noise** (≥ 3–5% and consistent across test *and* recent).

---

## 6. Phase 5 — Detection / center localization (stretch)

**Current:** presence accuracy 100% (trivial — positive-only eval), center localization error **724 km** (broken / off-center heatmap).

| Tier | Presence F1 | Mean center error | Median center error |
|---|---|---|---|
| Minimum | ≥ 0.99 | ≤ 100 km | ≤ 60 km |
| **Target** | **≥ 0.995** | **≤ 40 km** | **≤ 25 km** |
| Stretch | ≥ 0.999 | ≤ 25 km | ≤ 15 km |

### Impact-ranked actions
1. **Fix the evaluation set first:** include true negative (no-TC) full-basin frames and real, non-centered cyclone positions. 100% presence accuracy on an all-positive set is meaningless; the 724 km center error signals an eval/ground-truth mismatch (synthetic frames are center-truth labeled at 0.5,0.5) — verify center labeling and `km_per_pixel` before any model work.
2. **Real basin-wide frames + IBTrACS centers** as ground truth (available in Phase 0 data), heatmap regression with sigma ≈ 1–2% of image width.
3. **Skip-connection U-Net style head** (already in `HeatmapCenterDetector`) with Gaussian targets; loss weighting 10:1 heatmap:presence (already coded).
4. **Multi-peak NMS** for multi-storm basins; validate on recent storms with known positions.
5. Only then consider YOLO/Faster-RCNN boxes if a labeled subset can be produced.

---

## 7. Phase 6 — Dashboard (stretch)

No accuracy target — correctness of the *chain* is the target: the same inputs must reproduce the same JSON across reruns and the chained output must match the standalone phase checkpoints. Gate on Phases 1–4 being validated first.

---

## 8. Summary target table

| Phase | Metric | Current | Target |
|---|---|---|---|
| 1 Track | 6h / 12h / 24h DPE (test) | 36.7 / 73.7 / 159.2 km | ≤ 32 / ≤ 65 / ≤ 140 km |
| 1 Track | hit rate 6/12/24 | 0.35 / 0.38 / 0.38 | ≥ 0.45 all horizons |
| 2 Stage | exact accuracy (real) | 0.22 | ≥ 0.55 |
| 2 Stage | macro-F1 / mean stage dist | 0.20 / 1.87 | ≥ 0.50 / ≤ 0.9 |
| 3 Intensity | ΔV24h MAE / RMSE / bias | 11.7 / 15.9 / +1.2 kt | ≤ 8 / ≤ 12 / \|·\| ≤ 2 |
| 3 RI | F1 / precision / recall | 0.085 / 0.044 / 1.0 | ≥ 0.30 / ≥ 0.25 / ≥ 0.50 |
| 4 Fusion | 24h DPE & Δ vs track-only | 157.3 km (+1%) | ≤ 140 km, ≥ 8–10% |
| 5 Detect | presence F1 / center err | 1.0 / 724 km | ≥ 0.995 / ≤ 40 km |

## 9. Execution order (to hit all targets fastest)

1. Finish **real data wiring** (ERA5 completeness + real INSAT crops for all splits) — unblocks Phases 1–5 simultaneously.
2. Re-run Phase 1, Phase 2, Phase 3 with complete real inputs; record new baselines.
3. Re-train the Phase 2 encoder on real frames → re-run Phase 4 fusion with joint fine-tuning.
4. Implement eval fixes in Phase 5 (real centers + negatives), then train heatmap head.
5. Continuous-verify every run with `scripts/eval_real_metrics.py`; update `experiments/real_metrics.json` + tables after each improvement.

> If any phase target is not reachable within the compute/time budget, the fallback is to **report the achieved numbers honestly** against the Table in §8 with a written gap analysis — a defensible partial result beats an unverifiable one.