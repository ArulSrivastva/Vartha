"""Evaluation framework (plan section 19).

Tasks:
  - Identification: Precision/Recall/F1, mAP, IoU
  - Classification: Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC
  - Pattern: Macro/Weighted F1, per-class recall
  - Track: DPE (6/12/24h targets: <25/<50/<100 km), along/cross-track, direction error
  - Intensity: MAE, RMSE (wind, pressure)
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, average_precision_score,
)

from ..utils.geo import (
    direct_positional_error, vectorized_dpe, cross_track_error,
    along_track_error, bearing_angle, direction_error,
)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class DetectionMetrics:
    """Metrics for cyclone identification (Module 1)."""

    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def compute(self, predictions: Dict, targets: Dict) -> Dict[str, float]:
        """predictions: {has_tc_probs, boxes, labels}. targets: {has_tc, boxes}."""
        pred_probs = np.concatenate([p["has_tc"] for p in predictions])
        true_labels = np.concatenate([t["has_tc"] for t in targets])
        pred_labels = (pred_probs > self.conf_threshold).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="binary", zero_division=0)

        metrics = {
            "precision": p, "recall": r, "f1": f,
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

        # ROC-AUC
        if len(np.unique(true_labels)) > 1:
            metrics["roc_auc"] = roc_auc_score(true_labels, pred_probs)
        else:
            metrics["roc_auc"] = float("nan")

        # mAP (approximate: mean average precision over classes)
        try:
            metrics["map"] = average_precision_score(
                true_labels, pred_probs) if len(np.unique(true_labels)) > 1 else float("nan")
        except Exception:
            metrics["map"] = float("nan")

        # IoU for detected boxes
        ious = []
        for i in range(len(predictions)):
            pb = predictions[i].get("boxes")
            tb = targets[i].get("boxes")
            if pb is not None and tb is not None and len(pb) and len(tb):
                iou = self._box_iou(pb, tb)
                ious.append(iou)
        metrics["mean_iou"] = float(np.mean(ious)) if ious else float("nan")
        metrics["center_distance_km"] = self._center_distance(predictions, targets)
        return metrics

    def _box_iou(self, box_a, box_b):
        if box_a.ndim == 1:
            box_a = box_a[None, :]
        if box_b.ndim == 1:
            box_b = box_b[None, :]
        x1 = np.maximum(box_a[:, 0][:, None], box_b[:, 0][None, :])
        y1 = np.maximum(box_a[:, 1][:, None], box_b[:, 1][None, :])
        x2 = np.minimum(box_a[:, 2][:, None], box_b[:, 2][None, :])
        y2 = np.minimum(box_a[:, 3][:, None], box_b[:, 3][None, :])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        ious = inter / np.maximum(union, 1e-6)
        return float(np.max(ious))

    def _center_distance(self, predictions, targets):
        dists = []
        for i in range(len(predictions)):
            pc = predictions[i].get("center")
            tc = targets[i].get("center")
            if pc is not None and tc is not None:
                # Convert normalized center to km using image size (assume 4km/px)
                d = np.linalg.norm(pc - tc) * 4.0 * 512
                dists.append(d)
        return float(np.mean(dists)) if dists else float("nan")


class ClassificationMetrics:
    """Metrics for cyclone classification (Module 2)."""

    def compute(self, all_preds: List[torch.Tensor], all_targets: List[torch.Tensor]) -> Dict[str, float]:
        stage_preds = []
        stage_true = []
        struct_preds = []
        struct_true = []

        for p, t in zip(all_preds, all_targets):
            # p = model output dict {'classification': {...}}
            cls = p.get("classification", p)
            if "stage_logits" in cls:
                logits = _to_numpy(cls["stage_logits"])
                stage_preds.append(logits.argmax(axis=-1))
                stage_true.append(_to_numpy(t["stage"]))
            if "structure_logits" in cls:
                s_logits = cls["structure_logits"]
                if isinstance(s_logits, torch.Tensor):
                    s_preds = (torch.sigmoid(s_logits) > 0.5).float().cpu().numpy()
                else:
                    s_arr = np.asarray(s_logits)
                    s_preds = (1.0 / (1.0 + np.exp(-s_arr)) > 0.5).astype(float)
                struct_preds.append(s_preds)
                struct_true.append(_to_numpy(t["structure"]))

        metrics = {}
        if stage_preds:
            y_pred = np.concatenate(stage_preds)
            y_true = np.concatenate(stage_true)
            metrics["stage_accuracy"] = accuracy_score(y_true, y_pred)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0)
            metrics["stage_precision"] = p
            metrics["stage_recall"] = r
            metrics["stage_f1"] = f
            # Confusion matrix (returned separately for viz)
            metrics["_stage_confusion"] = confusion_matrix(y_true, y_pred)

        if struct_preds:
            y_pred = np.concatenate(struct_preds)
            y_true = np.concatenate(struct_true)
            metrics["structure_f1"] = f1_score(y_true, y_pred, average="macro",
                                                zero_division=0)
            metrics["structure_weighted_f1"] = f1_score(y_true, y_pred, average="weighted",
                                                          zero_division=0)
            metrics["structure_precision"] = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0)[0]
            metrics["structure_recall"] = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0)[1]
            # Per-class recall for rare classes (RI, recurvature, land interaction)
            per_class_r = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                          zero_division=0)[1]
            metrics["structure_per_class_recall"] = per_class_r.tolist()
            metrics["_structure_confusion"] = confusion_matrix(
                y_true.ravel(), y_pred.ravel()) if y_true.size else None
        return metrics


class PatternMetrics:
    """Metrics for pattern recognition (Module 3, multi-label)."""

    def compute(self, all_preds, all_targets) -> Dict[str, float]:
        all_pred_probs = []
        all_true = []
        for p, t in zip(all_preds, all_targets):
            pattern = p.get("pattern", p)
            if isinstance(pattern, dict):
                pattern = pattern.get("pattern", pattern)
            if isinstance(pattern, torch.Tensor):
                probs = torch.sigmoid(pattern).detach().cpu().numpy()
            else:
                p_arr = np.asarray(pattern)
                if np.any(p_arr < 0) or np.any(p_arr > 1):
                    probs = 1.0 / (1.0 + np.exp(-p_arr))
                else:
                    probs = p_arr
            all_pred_probs.append(probs)
            all_true.append(_to_numpy(t["pattern"]))

        if not all_pred_probs:
            return {}
        y_pred_probs = np.concatenate(all_pred_probs, axis=0)
        y_pred = (y_pred_probs > 0.5).astype(int)
        y_true = np.concatenate(all_true, axis=0)

        metrics = {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "per_class_recall": precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0)[1].tolist(),
            "subset_accuracy": np.mean(np.all(y_pred == y_true, axis=1)),
        }
        # ROC-AUC per class
        try:
            aurocs = []
            for c in range(y_true.shape[1]):
                if len(np.unique(y_true[:, c])) > 1:
                    aurocs.append(roc_auc_score(y_true[:, c], y_pred_probs[:, c]))
            if aurocs:
                metrics["mean_auc"] = float(np.mean(aurocs))
        except Exception:
            pass
        return metrics


class TrackMetrics:
    """Track prediction metrics (Module 4) with DPE targets.

    Targets: 6h < 25km, 12h < 50km, 24h < 100km.
    """

    def __init__(self, horizons: List[int] = [6, 12, 24],
                 lat_factor_warning: Optional[Dict[int, float]] = None):
        self.horizons = horizons
        self.dpe_targets = {6: 25.0, 12: 50.0, 24: 100.0}
        if lat_factor_warning:
            self.dpe_targets.update(lat_factor_warning)

    def compute(self, all_preds, all_targets, center_lats: Optional[List[float]] = None) -> Dict[str, float]:
        metrics = {}
        for h in self.horizons:
            dpes = []
            along = []
            cross = []
            dir_errs = []
            hkey = str(h)
            prev_preds = {}

            for i, (p, t) in enumerate(zip(all_preds, all_targets)):
                pred = p.get("prediction", p)
                if hkey not in pred:
                    continue
                pos = _to_numpy(pred[hkey]["position"]).squeeze()
                if hkey in t:
                    true = _to_numpy(t[hkey]).squeeze()
                    dpe = direct_positional_error(pos[0], pos[1], true[0], true[1])
                    dpes.append(dpe)

                    # Direction error (using previous position as origin)
                    if "current_position" in t:
                        cur = _to_numpy(t["current_position"]).squeeze()
                        pred_bear = bearing_angle(cur[0], cur[1], pos[0], pos[1])
                        true_bear = bearing_angle(cur[0], cur[1], true[0], true[1])
                        dir_errs.append(direction_error(pred_bear, true_bear))
                        along.append(along_track_error(pos[0], pos[1], cur[0], cur[1], true[0], true[1]))
                        cross.append(cross_track_error(pos[0], pos[1], cur[0], cur[1], true[0], true[1]))

            if dpes:
                dpe_arr = np.array(dpes)
                metrics[f"dpe_{h}h_mean"] = float(np.mean(dpe_arr))
                metrics[f"dpe_{h}h_median"] = float(np.median(dpe_arr))
                metrics[f"dpe_{h}h_p95"] = float(np.percentile(dpe_arr, 95))
                metrics[f"dpe_{h}h_rmse"] = float(np.sqrt(np.mean(dpe_arr ** 2)))
                # Met target hit rate
                target = self.dpe_targets.get(h)
                if target:
                    metrics[f"dpe_{h}h_hit_rate"] = float(np.mean(dpe_arr <= target))
                # Absolute track error metrics
                if along:
                    metrics[f"along_track_{h}h"] = float(np.mean(along))
                    metrics[f"cross_track_{h}h"] = float(np.mean(cross))
                    metrics[f"direction_error_{h}h"] = float(np.mean(dir_errs))
                # Aggregated for reporting
                metrics["dpe_all_mean"] = metrics.get("dpe_all_mean", 0.0) + metrics[f"dpe_{h}h_mean"]

        # DPE aggregate with targets checked individually
        return metrics

    def summary_table(self, metrics: Dict[str, float], horizons: Optional[List[int]] = None) -> pd.DataFrame:
        rows = []
        for h in (horizons or self.horizons):
            target = self.dpe_targets.get(h, float("nan"))
            rows.append({
                "Horizon": f"{h}h",
                "DPE (km)": metrics.get(f"dpe_{h}h_mean"),
                "Median (km)": metrics.get(f"dpe_{h}h_median"),
                "P95 (km)": metrics.get(f"dpe_{h}h_p95"),
                "RMSE (km)": metrics.get(f"dpe_{h}h_rmse"),
                "Target (km)": target,
                "Hit rate": metrics.get(f"dpe_{h}h_hit_rate"),
                "Along-track (km)": metrics.get(f"along_track_{h}h"),
                "Cross-track (km)": metrics.get(f"cross_track_{h}h"),
                "Direction err (°)": metrics.get(f"direction_error_{h}h"),
            })
        return pd.DataFrame(rows)


class IntensityMetrics:
    """Intensity prediction metrics (wind + minimum pressure)."""

    def compute(self, all_preds, all_targets, horizons: List[int] = [6, 12, 24]) -> Dict[str, float]:
        metrics = {}
        for h in horizons:
            hkey = str(h)
            wind_err, pres_err = [], []
            for p, t in zip(all_preds, all_targets):
                pred = p.get("prediction", p)
                if hkey not in pred or "intensity" not in pred[hkey]:
                    continue
                inten = _to_numpy(pred[hkey]["intensity"]).squeeze()
                tgt = t.get(hkey)
                if tgt is None:
                    continue
                if isinstance(tgt, dict) and "intensity" in tgt:
                    tgt_val = _to_numpy(tgt["intensity"]).squeeze()
                elif hasattr(tgt, "shape") and np.prod(tgt.shape) == 2:
                    tgt_val = _to_numpy(tgt).squeeze()
                else:
                    continue
                wind_err.append(inten[0] - tgt_val[0])
                pres_err.append(inten[1] - tgt_val[1])

            if wind_err:
                we = np.array(wind_err)
                pe = np.array(pres_err)
                metrics[f"mae_wind_{h}h"] = float(np.mean(np.abs(we)))
                metrics[f"rmse_wind_{h}h"] = float(np.sqrt(np.mean(we ** 2)))
                metrics[f"mae_pressure_{h}h"] = float(np.mean(np.abs(pe)))
                metrics[f"rmse_pressure_{h}h"] = float(np.sqrt(np.mean(pe ** 2)))
        return metrics


class AblationEvaluator:
    """Ablation study runner (plan section 20)."""

    MODELS = ["Baseline", "Model A", "Model B", "Model C"]
    FEATURES = {
        "Baseline": (False, False, False),
        "Model A": (True, False, False),
        "Model B": (True, True, False),
        "Model C": (True, True, True),
    }

    def __init__(self, horizons: List[int] = [6, 12, 24]):
        self.metrics_eval = TrackMetrics(horizons)

    @staticmethod
    def table(model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """model_results: {model_name: {metric: value}}"""
        rows = []
        for model, metrics in model_results.items():
            row = {"Model": model}
            for h in [6, 12, 24]:
                row[f"{h}h DPE"] = metrics.get(f"dpe_{h}h_mean")
            rows.append(row)
        return pd.DataFrame(rows)