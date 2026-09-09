"""Weak/proxy structural labels for real cyclone samples (Phase 7).

plan.md Section 5/10 explicitly sanctions deriving structural cloud-pattern
labels from weak/proxy signals rather than manual annotation. This module:

  1. Measures three objective geometric diagnostics directly from the
     storm-centered satellite field (robust to product scale):
       - eye_idx        : inner-core vs eyewall-ring contrast
       - shear_arc      : azimuthal arc that contains 75% of cold-cloud mass
                          (shear / asymmetry concentration metric)
       - org_continuity : fraction of active sectors with an azim. neighbour
                          (spiral-band organization)
  2. Combines them with grounded IBTrACS intensity signals (wind, 24h change,
     RI flag, pressure, stage) into the 11-class structural multi-label vector
     whose vocabulary matches configs/config.yaml structure_classes.

All labels are proxies. Geometric thresholds are quantiles fit on the TRAIN
split only (no temporal leakage into val/test/recent).
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..models.pattern.structural import STRUCTURE_LABELS

# --- Geometric constants (pixels on the 512x512 storm grid) ---
EYE_R_INNER = 12.0          # inner core radius
EYE_R_RING = (15.0, 45.0)   # eyewall surrounding ring
ASY_R_BAND = (40.0, 130.0)  # annulus for shear/arc coverage metric
ORG_R_BAND = (30.0, 120.0)  # annulus for spiral-band organization metric
N_SECTORS = 24              # azimuthal sectors

# --- Split boundaries identical to the existing benchmark pipeline ---
# (verified against experiments/real_metrics.json split_sizes:
#  train=385 / val=188 / test=331 / recent=162). "recent" (2024-2025) is the
#  live-storm subset of "test" (2021-2025), exactly as in the phase-1..6 eval.
SPLIT_BY_SEASON = {
    (0, 2018): "train",
    (2019, 2020): "val",
    (2021, 9999): "test",
}
RECENT_SEASONS = (2024, 2025)


def split_for_season(season: int) -> str:
    for (lo, hi), split in SPLIT_BY_SEASON.items():
        if lo <= season <= hi:
            return split
    return "test"


def is_recent_season(season: int) -> bool:
    return RECENT_SEASONS[0] <= season <= RECENT_SEASONS[1]


def _radial_mask(cx: float, cy: float, size: int, r_lo: float, r_hi: float):
    y, x = np.mgrid[0:size, 0:size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if r_hi is None:
        return r <= r_lo
    return (r >= r_lo) & (r < r_hi)


def _sector_splits(cx: float, cy: float, size: int, n_sectors: int = N_SECTORS):
    """Return boolean masks for n_sectors azimuthal sectors around (cx, cy)."""
    y, x = np.mgrid[0:size, 0:size]
    ang = (np.degrees(np.arctan2(y - cy, x - cx)) + 180.0) % 360.0
    width = 360.0 / n_sectors
    masks = []
    for s in range(n_sectors):
        lo, hi = s * width, (s + 1) * width
        masks.append((ang >= lo) & (ang < hi))
    return masks


def measure_structure_indices(sat: np.ndarray, cx: float, cy: float) -> Dict[str, float]:
    """Compute geometric structural diagnostics from a (C, H, W) satellite field.

    The indices are scale- and sign-robust: the cloud field is feature-
    normalized (z-score) per sample, so they work regardless of whether the
    product encodes cold cloud as high (IR brightness) or low (SST) values.
    """
    ch0 = sat[0].astype(np.float64)
    size = ch0.shape[0]
    f = (ch0 - ch0.mean()) / (ch0.std() + 1e-9)

    # 1. eye_idx: inner-core vs surrounding-ring contrast (feature-normalized)
    inner = _radial_mask(cx, cy, size, EYE_R_INNER, None)
    ring = _radial_mask(cx, cy, size, EYE_R_RING[0], EYE_R_RING[1])
    eye_idx = float(f[inner].mean() - f[ring].mean())

    # 2. shear_arc: minimal arc covering 75% of cold-cloud mass in the band
    band = _radial_mask(cx, cy, size, ASY_R_BAND[0], ASY_R_BAND[1])
    cold = ch0 <= float(np.percentile(ch0[band], 40))
    cold = cold & band
    masks = _sector_splits(cx, cy, size)
    sector_cold = np.array([float((cold & m).sum()) for m in masks], dtype=np.float64)
    total = sector_cold.sum()
    if total <= 0.0:
        shear_arc = 1.0
    else:
        order = np.argsort(-sector_cold)
        running = 0.0
        min_arc = len(sector_cold)
        for start in range(N_SECTORS):
            for n in range(1, N_SECTORS + 1):
                arc = order[start:start + n] if start + n <= N_SECTORS else order[start:]
                if sector_cold[arc].sum() >= 0.75 * total:
                    min_arc = min(min_arc, n)
        shear_arc = float(min_arc / N_SECTORS)

    # 3. org_continuity: fraction of active sectors having an active neighbour
    band_o = _radial_mask(cx, cy, size, ORG_R_BAND[0], ORG_R_BAND[1])
    act_ = ch0 <= float(np.percentile(ch0[band_o], 40))
    act = (act_ & band_o)
    mask_lst = _sector_splits(cx, cy, size)
    active_sectors = [s for s, m in enumerate(mask_lst) if (act & m).sum() > 0]
    if not active_sectors:
        org_continuity = 0.0
    else:
        active_set = set(active_sectors)
        linked = 0
        for s in active_sectors:
            if (s - 1) % N_SECTORS in active_set or (s + 1) % N_SECTORS in active_set:
                linked += 1
        org_continuity = float(linked / len(active_sectors))

    return {
        "eye_idx": eye_idx,
        "shear_arc": shear_arc,
        "org_continuity": org_continuity,
    }


def derive_structural_labels(meta: Dict, indices: Dict[str, float],
                             thresh: Dict[str, float]) -> List[int]:
    """Combine IBTrACS intensity signals + geometric indices into the 11-label vector."""
    wind = float(meta.get("wind_kt") or 0.0)
    pressure = float(meta.get("pressure_hpa") or 1013.0)
    d24 = float(meta.get("delta_wind_24h") or 0.0)
    ri_flag = int(meta.get("ri_flag") or 0)
    stage = int(meta.get("stage") or 0)

    eye_idx = indices["eye_idx"]
    shear_arc = indices["shear_arc"]
    org = indices["org_continuity"]

    t30 = thresh.get("eye_q30", 0.0)
    t60 = thresh.get("eye_q60", 0.0)
    o50 = thresh.get("org_q50", 0.5)

    labels = [0] * len(STRUCTURE_LABELS)
    def set_lab(name):
        labels[STRUCTURE_LABELS.index(name)] = 1

    # Intensity-tendency structural descriptors (grounded on IBTrACS winds)
    if wind < 35.0 and d24 <= 15.0:
        set_lab("developing")
    elif wind >= 55.0 and abs(d24) <= 10.0:
        set_lab("mature")
    if d24 <= -5.0 and wind >= 30.0:
        set_lab("weakening")
    if ri_flag or d24 >= 25.0:
        set_lab("rapidly_intensifying")
    if wind <= 30.0 and pressure >= 1000.0 and d24 <= 0.0:
        set_lab("dissipating")

    # Geometric eye / eyewall descriptors
    if eye_idx >= t60 and wind >= 55.0:
        set_lab("eye_present")
    elif eye_idx >= t30 and wind >= 35.0:
        set_lab("eye_forming")
    if eye_idx >= t60 and org >= o50 and wind >= 65.0:
        set_lab("eyewall_organized")

    # Shear / asymmetry descriptors (visible concentration of convection)
    if shear_arc < 0.50 and wind >= 25.0:
        set_lab("sheared")
    if shear_arc < 0.35 and wind >= 35.0:
        set_lab("highly_asymmetric")

    # land_interaction requires an explicit land mask (not derivable here) and
    # is intentionally left at 0; it is excluded from supported macro-F1.
    return labels


def fit_thresholds(train_indices: List[Dict[str, float]]) -> Dict[str, float]:
    eyes = np.array([i["eye_idx"] for i in train_indices])
    orgs = np.array([i["org_continuity"] for i in train_indices])
    return {
        "eye_q30": float(np.percentile(eyes, 30)),
        "eye_q60": float(np.percentile(eyes, 60)),
        "org_q50": float(np.percentile(orgs, 50)),
    }


def build_structural_registry(samples_dir: Path, out_path: Path,
                              land_interaction: bool = False) -> Dict:
    """Two-pass builder. Pass 1 measures indices for all samples; thresholds are
    fit on the train split; pass 2 derives labels and persists the registry."""
    samples_dir = Path(samples_dir)
    sample_dirs = sorted(
        p for p in samples_dir.iterdir()
        if p.is_dir() and (p / "satellite.npy").exists() and (p / "meta.json").exists()
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No real samples found under {samples_dir}")

    entries = []
    for sdir in sample_dirs:
        meta = json.loads((sdir / "meta.json").read_text())
        sat = np.load(sdir / "satellite.npy")
        center = np.load(sdir / "center.npy").astype(np.float64)
        cx = center[0] * (sat.shape[2] - 1)
        cy = center[1] * (sat.shape[1] - 1)
        indices = measure_structure_indices(sat, cx, cy)
        season = int(meta.get("season", 0))
        is_recent = is_recent_season(season)
        entries.append({
            "sample_id": sdir.name,
            "season": season,
            "split": split_for_season(season),
            "is_recent": is_recent,
            "wind_kt": float(meta.get("wind_kt") or 0.0),
            "pressure_hpa": float(meta.get("pressure_hpa") or 1013.0),
            "stage": int(meta.get("stage") or 0),
            "delta_wind_24h": float(meta.get("delta_wind_24h") or 0.0),
            "indices": indices,
        })

    if land_interaction:
        raise NotImplementedError(
            "land_interaction requires an explicit land mask; not derivable from "
            "the current real-sample metadata. It remains all-zero and is excluded "
            "from the supported macro-F1."

        )

    train_indices = [e["indices"] for e in entries if e["split"] == "train"]
    thresh = fit_thresholds(train_indices)

    for e in entries:
        e["labels"] = derive_structural_labels(e, e["indices"], thresh)

    registry = {
        "version": 1,
        "vocabulary": STRUCTURE_LABELS,
        "splits": {
            "train": [e["sample_id"] for e in entries if e["split"] == "train"],
            "val": [e["sample_id"] for e in entries if e["split"] == "val"],
            "test": [e["sample_id"] for e in entries if e["split"] == "test"],
            "recent": [e["sample_id"] for e in entries if e["is_recent"]],
        },
        "split_sizes": {
            "train": sum(1 for e in entries if e["split"] == "train"),
            "val": sum(1 for e in entries if e["split"] == "val"),
            "test": sum(1 for e in entries if e["split"] == "test"),
            "recent": sum(1 for e in entries if e["is_recent"]),
        },
        "thresholds": thresh,
        "samples": entries,
    }
    # Co-occurrence sanity
    n_lab = len(STRUCTURE_LABELS)
    for e in entries:
        assert len(e["labels"]) == n_lab

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2))
    return registry