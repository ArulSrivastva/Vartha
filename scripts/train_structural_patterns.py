"""Train & evaluate Phase 7 Structural Cloud-Pattern Recognition (additive).

Trains StructuralPatternModel on storm-centered satellite crops with the weak
structural labels from data/structural/structural_registry.json. GPU training
is mandatory (AGENTS.md): raises RuntimeError when CUDA is unavailable.

Outputs (NEW files only; nothing existing is touched):
  - experiments/phase7_structural_best.pt
  - experiments/phase7_structural_results.json
  - experiments/phase7_structural_report.md

Split boundaries are identical to the phase-1..6 benchmark pipeline:
  train = seasons <= 2018 | val = 2019-2020
  test = 2021-2025 | recent = 2024-2025 (live-storm subset of test)

Usage:
  python scripts/train_structural_patterns.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_ai.models.pattern.structural import StructuralPatternModel, STRUCTURE_LABELS
from tc_ai.utils.common import get_device

REGISTRY_PATH = PROJECT_ROOT / "data" / "structural" / "structural_registry.json"
OUT_PT = PROJECT_ROOT / "experiments" / "phase7_structural_best.pt"
OUT_JSON = PROJECT_ROOT / "experiments" / "phase7_structural_results.json"
OUT_MD = PROJECT_ROOT / "experiments" / "phase7_structural_report.md"

SEED = 42
EPOCHS = 80
BATCH = 64
LR = 1e-3
CROP_HALF = 128          # half-size of the storm-centered crop (256x256 at 512 grid)
IMG_SIZE = 128


def _resize(x: np.ndarray):
    try:
        import cv2
        return cv2.resize(x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    except Exception:
        t = torch.from_numpy(x).unsqueeze(0)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return t[0].numpy()


def load_crop(samples_dir: Path, sample_id: str) -> np.ndarray:
    sat = np.load(samples_dir / sample_id / "satellite.npy").astype(np.float32)
    center = np.load(samples_dir / sample_id / "center.npy").astype(np.float64)
    h, w = sat.shape[1], sat.shape[2]
    cx = min(max(center[0] * (w - 1), 0.0), w - 1)
    cy = min(max(center[1] * (h - 1), 0.0), h - 1)
    x0 = int(cx - CROP_HALF)
    y0 = int(cy - CROP_HALF)
    pad_l = max(0, -x0); pad_r = max(0, x0 + 2 * CROP_HALF - w)
    pad_t = max(0, -y0); pad_b = max(0, y0 + 2 * CROP_HALF - h)
    crop = sat[:, max(y0, 0):min(y0 + 2 * CROP_HALF, h), max(x0, 0):min(x0 + 2 * CROP_HALF, w)]
    if pad_t or pad_b or pad_l or pad_r:
        crop = np.pad(crop, ((0, 0), (pad_t, pad_b), (pad_l, pad_r)), mode="edge")
    return np.ascontiguousarray(_resize(crop))


def per_label_f1(y_true: np.ndarray, y_pred: np.ndarray, names):
    """y_true/y_pred: (N, K) binary arrays. Returns (per-label dict, supported macro-F1)."""
    eps = 1e-9
    tp = (y_pred * y_true).sum(0)
    fp = (y_pred * (1 - y_true)).sum(0)
    fn = ((1 - y_pred) * y_true).sum(0)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    per_label = {names[i]: float(f1[i]) for i in range(len(names))}
    supported = [i for i in range(len(names)) if y_true[:, i].sum() > 0]
    macro = float(f1[supported].mean()) if supported else 0.0
    return per_label, macro, supported


def tune_thresholds(P: np.ndarray, Y: np.ndarray, names, min_pos: int = 3):
    """Per-label decision thresholds maximising F1 on the validation split."""
    thresh = [0.5] * len(names)
    for i in range(len(names)):
        if Y[:, i].sum() < min_pos:
            continue
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.20, 0.85, 0.05):
            yp = (P[:, i] >= t).astype(np.float32)
            _, macro, _ = per_label_f1(Y[:, [i]], yp[:, None], ["x"])
            if macro > best_f:
                best_t, best_f = float(t), macro
        thresh[i] = best_t
    return thresh


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def predict(model, X, device, bs: int = 256):
    probs, ee, aa, oo = [], [], [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out = model(xb)
        probs.append(torch.sigmoid(out["structure_logits"]).cpu().numpy())
        ee.append(out["eye_idx"].cpu().numpy().ravel())
        aa.append(out["asym_idx"].cpu().numpy().ravel())
        oo.append(out["org_idx"].cpu().numpy().ravel())
    return (np.concatenate(probs), np.concatenate(ee), np.concatenate(aa), np.concatenate(oo))


def evaluate(model, X, Y, eye_t, asym_t, org_t, device, names, thresh=None):
    P, eye_p, asym_p, org_p = predict(model, X, device)
    thresh = [0.5] * len(names) if thresh is None else thresh
    y_pred = (P >= np.array(thresh, dtype=np.float32)).astype(np.float32)
    y_true = Y[:len(P)]
    per_label, macro, supported = per_label_f1(y_true, y_pred, names)
    bce = float(F.binary_cross_entropy(torch.from_numpy(P), torch.from_numpy(y_true), reduction="mean").item())
    return {
        "bce": bce,
        "macro_f1": macro,
        "supported_labels": [names[i] for i in supported],
        "per_label_f1": per_label,
        "eye_mae": float(np.abs(eye_p - eye_t).mean()),
        "eye_pearson": pearson(eye_p, eye_t),
        "asym_mae": float(np.abs(asym_p - asym_t).mean()),
        "asym_pearson": pearson(asym_p, asym_t),
        "org_mae": float(np.abs(org_p - org_t).mean()),
        "org_pearson": pearson(org_p, org_t),
    }


def main():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry not found at {REGISTRY_PATH}. Run scripts/build_structural_dataset.py first.")
    device = get_device(require_gpu=True)
    print(f"[phase7] training on {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    registry = json.loads(REGISTRY_PATH.read_text())
    vocab = registry["vocabulary"]
    assert vocab == STRUCTURE_LABELS, f"Registry vocabulary mismatch: {vocab}"
    samples_dir = PROJECT_ROOT / "data" / "real" / "samples"

    splits = registry["splits"]
    print(f"[phase7] split sizes: {registry['split_sizes']}")

    # Load crops + targets into memory (904 x (3,128,128) float32)
    X, Y, SPLIT, IS_RECENT, IDs = [], [], [], [], []
    eye_raw, asym_raw, org_raw = [], [], []
    for e in registry["samples"]:
        crop = load_crop(samples_dir, e["sample_id"])
        X.append(crop)
        eye_raw.append(e["indices"]["eye_idx"])
        asym_raw.append(e["indices"]["shear_arc"])
        org_raw.append(e["indices"]["org_continuity"])
        Y.append(np.array(e["labels"], dtype=np.float32))
        SPLIT.append(e["split"])
        IS_RECENT.append(bool(e.get("is_recent", False)))
        IDs.append(e["sample_id"])
    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    eye_raw = np.array(eye_raw, dtype=np.float32)
    asym_raw = np.array(asym_raw, dtype=np.float32)
    org_raw = np.array(org_raw, dtype=np.float32)
    SPLIT = np.array(SPLIT)
    IS_RECENT = np.array(IS_RECENT)

    # Normalise eye target with TRAIN stats only
    tr_mask = SPLIT == "train"
    eye_mean, eye_std = float(eye_raw[tr_mask].mean()), float(eye_raw[tr_mask].std()) + 1e-6
    eye_z = (eye_raw - eye_mean) / eye_std

    idx = {s: np.where(SPLIT == s)[0] for s in ["train", "val", "test"]}
    idx["recent"] = np.where(IS_RECENT)[0]
    Xtr, Ytr = X[idx["train"]], Y[idx["train"]]
    etr = eye_z[idx["train"]]; atr = asym_raw[idx["train"]]; otr = org_raw[idx["train"]]

    model = StructuralPatternModel(input_channels=3, n_structure_classes=len(vocab),
                                   embed_dim=128, image_size=IMG_SIZE).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    # Class imbalance: pos_weight from TRAIN positives only
    pos_count = Ytr.sum(0)
    pw = (Ytr.shape[0] - pos_count) / np.maximum(pos_count, 1.0)
    pw = np.clip(pw, 1.0, 10.0)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw).float().to(device))

    n_tr = len(Xtr)
    best_val = -1.0
    best_state = None
    print(f"[phase7] training on {n_tr} samples | labels: {vocab}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr).numpy()
        tot = 0.0
        for i in range(0, n_tr, BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(Ytr[b]).to(device)
            out = model(xb)
            loss = bce_loss(out["structure_logits"], yb)
            loss = loss + 0.5 * F.mse_loss(out["eye_idx"].squeeze(-1), torch.from_numpy(etr[b]).to(device))
            loss = loss + 0.5 * F.mse_loss(out["asym_idx"].squeeze(-1), torch.from_numpy(atr[b]).to(device))
            loss = loss + 0.5 * F.mse_loss(out["org_idx"].squeeze(-1), torch.from_numpy(otr[b]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()

        val_metrics = evaluate(model, X[idx["val"]], Y[idx["val"]],
                               eye_z[idx["val"]], asym_raw[idx["val"]], org_raw[idx["val"]],
                               device, vocab)
        if val_metrics["macro_f1"] > best_val and epoch >= 10:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"[phase7] epoch {epoch:3d} | train_loss={tot / n_tr:.4f} | "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f}")

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"[phase7] best val macro-F1 = {best_val:.4f}")

    # Val threshold tuning (mirrors Phase-3 RI validation threshold-tuning practice)
    val_P, _, _, _ = predict(model, X[idx["val"]], device)
    tuned_thresh = tune_thresholds(val_P, Y[idx["val"]], vocab)
    val_fixed = evaluate(model, X[idx["val"]], Y[idx["val"]],
                         eye_z[idx["val"]], asym_raw[idx["val"]], org_raw[idx["val"]],
                         device, vocab)
    val_tuned = evaluate(model, X[idx["val"]], Y[idx["val"]],
                         eye_z[idx["val"]], asym_raw[idx["val"]], org_raw[idx["val"]],
                         device, vocab, thresh=tuned_thresh)
    print(f"[phase7] val macro-F1: fixed-0.5={val_fixed['macro_f1']:.4f} | tuned={val_tuned['macro_f1']:.4f}")

    results = {
        "method": "StructuralPatternModel (CNN) on storm-centered satellite crops; "
                  "weak/proxy labels per plan.md Section 5/10",
        "label_vocabulary": vocab,
        "weak_label_notes": (
            "Labels are proxies: intensity-tendency classes from IBTrACS "
            "wind/pressure/RI signals; eye/eyewall/shear/organization classes from "
            "objective geometric indices (eye contrast, 75%-mass shear arc, spiral "
            "continuity) with quantile thresholds fit on TRAIN seasons only. "
            "'land_interaction' needs an explicit land mask and is excluded from "
            "supported macro-F1."),
        "split_sizes": registry["split_sizes"],
        "thresholds": registry["thresholds"],
        "train_eye_stats": {"mean": eye_mean, "std": eye_std},
        "pos_weight": pw.tolist(),
        "decision_thresholds_val_tuned": tuned_thresh,
        "threshold_note": "Per-label decision thresholds tuned on VAL macro-F1; "
                          "reported below.",
        "val_macro_f1_fixed": val_fixed["macro_f1"],
        "val_macro_f1_tuned": val_tuned["macro_f1"],
        "metrics": {},
        "class_distribution": {},
    }

    for split in ["train", "val", "test", "recent"]:
        i = idx[split]
        results["metrics"][split] = evaluate(
            model, X[i], Y[i], eye_z[i], asym_raw[i], org_raw[i], device, vocab,
            thresh=tuned_thresh)
        results["class_distribution"][split] = {
            vocab[j]: int(Y[i][:, j].sum()) for j in range(len(vocab))
        }
        m = results["metrics"][split]
        print(f"[phase7] {split} (n={len(i)}): macro_f1={m['macro_f1']:.4f} bce={m['bce']:.4f}")

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "config": {"n_structure_classes": len(vocab), "embed_dim": 128, "image_size": IMG_SIZE},
        "label_names": vocab,
        "split_sizes": registry["split_sizes"],
        "thresholds": registry["thresholds"],
        "train_eye_stats": {"mean": eye_mean, "std": eye_std},
        "results": results,
    }, OUT_PT)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    _write_report(results)
    print(f"[phase7] wrote {OUT_PT}, {OUT_JSON}, {OUT_MD}")


def _write_report(results):
    lines = [
        "# Phase 7: Structural Cloud-Pattern Recognition",
        "",
        "Additive module closing the Phase-6 audit gap on **\"different tropical cyclone",
        "patterns\"** in the Problem Statement: the system now recognizes structural",
        "cloud organization (eye formation, organized eyewall, shear-induced asymmetry,",
        "spiral banding) in addition to the stage/intensity patterns of Phase 2/3.",
        "",
        "## Approach",
        "",
        "- **Model**: lightweight storm-centered CNN `StructuralPatternModel`",
        "  (`tc_ai/models/pattern/structural.py`) producing an 11-class multi-label",
        "  structure vector plus three diagnostic regressions (eye contrast, shear-arc",
        "  asymmetry, spiral organization).",
        "- **Labels**: weak/proxy by design (sanctioned by plan.md Sections 5/10).",
        "  Intensity-tendency classes derive from grounded IBTrACS wind/pressure/RI",
        "  signals; eye/eyewall/shear/organization classes from objective geometric",
        "  indices with quantile thresholds fit on **train seasons only** (no leakage).",
        "- **Split**: identical to the rest of the pipeline (seasons train",
        "  <= 2018 / val 2019-2020 / test 2021-2025 / recent 2024-2025), verified",
        "  against `experiments/real_metrics.json` split sizes.",
        "",
        f"- Vocabulary ({len(results['label_vocabulary'])} classes): "
        + ", ".join(results["label_vocabulary"]) + ".",
        "",
        "## Splits",
        "",
        "| split | n |",
        "|---|---|",
    ]
    for s, n in results["split_sizes"].items():
        lines.append(f"| {s} | {n} |")

    lines += [
        "",
        "## Multi-label evaluation (test & recent)",
        "",
        "Evaluation on the **temporal out-of-sample** splits. macro-F1 averages only the",
        "supported labels (positive support in that split); `land_interaction` is",
        "excluded (requires an explicit land mask).",
        "",
        "| split | macro-F1 | BCE | eye MAE (z) | asym MAE | org MAE | eye r | asym r | org r |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in ["train", "val", "test", "recent"]:
        m = results["metrics"][s]
        lines.append(
            f"| {s} | {m['macro_f1']:.3f} | {m['bce']:.3f} | {m['eye_mae']:.3f} | "
            f"{m['asym_mae']:.3f} | {m['org_mae']:.3f} | {m['eye_pearson']:.3f} | "
            f"{m['asym_pearson']:.3f} | {m['org_pearson']:.3f} |")

    lines += [
        "",
        "Per-label F1 (test):",
        "",
        "| class | F1 |",
        "|---|---|",
    ]
    for lab, f1 in results["metrics"]["test"]["per_label_f1"].items():
        lines.append(f"| {lab} | {f1:.3f} |")

    lines += [
        "",
        "## Threshold choice",
        "",
        "Per-label decision thresholds are tuned on the VALIDATION split to maximise",
        "macro-F1 (same validation-threshold-tuning practice used for the Phase-3 RI",
        "benchmark). Tuned thresholds (test split, 2021-2025):",
        "",
        "| class | threshold |",
        "|---|---|",
    ]
    for lab, t in zip(results["label_vocabulary"], results["decision_thresholds_val_tuned"]):
        lines.append(f"| {lab} | {t:.2f} |")

    lines += [
        "",
        "## Honest limitations",
        "",
        "- Structural labels are **weak/proxy** (no manual annotation in the dataset).",
        "  Reported F1/Pearson values measure agreement with these proxies.",
        "- `land_interaction` is not derivable without an explicit land mask and",
        "  remains all-zero; it is excluded from supported macro-F1.",
        "- Geometric indices are computed on the real satellite ch0 field and are",
        "  robust to product scale, but remain an approximation of convective",
        "  organization.",
        "",
        "This phase is fully additive: it writes only new files and does not modify",
        "any existing benchmark result, checkpoint, or metric.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()