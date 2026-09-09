"""Explainability & Uncertainty Calibration for Satellite-Fed TC Models.

Per Project Roadmap & Research Requirements:
1. Explainability:
   - Saliency maps (Input Gradients, Grad-CAM, Integrated Gradients) on real INSAT SST frames.
   - Highlights convective eyewall, core thermal gradients, and outer spiral bands.
   - Saves figures and feature attribution metrics to `experiments/explainability/`.

2. Uncertainty Calibration:
   - Classification calibration: Expected Calibration Error (ECE), MCE, Brier Score, and reliability bins.
   - Track forecast calibration: Heteroscedastic Gaussian NLL, empirical cone coverage (nominal 68% and 95% vs empirical coverage at 6h, 12h, 24h).
   - Saves consolidated calibration data to `experiments/calibration_metrics.json`.

Usage:
   python scripts/explain_and_calibrate.py --gpu
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.models.classification.classifier import StageClassifier
from tc_ai.models.fusion.multimodal import SatelliteTrackFusionModel
from tc_ai.models.prediction.track_predictor import GRUTrackModel
from tc_ai.utils.common import get_device, set_seed


# --------------------------------------------------------------------------
# 1. Grad-CAM & Saliency Methods
# --------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        score = logits[0, target_class]
        score.backward()

        # Weight activations by pooled gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize 0..1
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def compute_integrated_gradients(model: nn.Module, input_tensor: torch.Tensor, target_class: int,
                                 steps: int = 20) -> np.ndarray:
    """Integrated Gradients feature attribution."""
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]

    grads = []
    for x in scaled_inputs:
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()
        grads.append(x.grad.detach().cpu().numpy())

    avg_grads = np.mean(grads, axis=0)
    delta = (input_tensor - baseline).detach().cpu().numpy()
    ig = delta * avg_grads
    # Aggregate over channels
    return np.mean(np.abs(ig), axis=1)[0]


# --------------------------------------------------------------------------
# 2. Uncertainty Calibration Metrics
# --------------------------------------------------------------------------
def compute_classification_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """Compute Expected Calibration Error and bin breakdown for classification."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == targets)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    ece = 0.0
    mce = 0.0
    bin_details = []

    total_samples = len(targets)
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            err = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * err
            mce = max(mce, err)
            bin_details.append({
                "bin": b,
                "range": [round(float(bins[b]), 2), round(float(bins[b + 1]), 2)],
                "count": bin_count,
                "acc": round(bin_acc, 3),
                "conf": round(bin_conf, 3),
                "error": round(err, 3),
            })
        else:
            bin_details.append({
                "bin": b,
                "range": [round(float(bins[b]), 2), round(float(bins[b + 1]), 2)],
                "count": 0,
                "acc": 0.0,
                "conf": 0.0,
                "error": 0.0,
            })

    # Multi-class Brier score
    one_hot = np.zeros_like(probs)
    for i, t in enumerate(targets):
        one_hot[i, t] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "bins": bin_details,
    }


def compute_track_cone_calibration(errors_km: dict, sigmas_km: dict, horizons=[6, 12, 24]):
    """Evaluate empirical coverage of heteroscedastic forecast cones.
    For a 2D Gaussian cone:
      Nominal 68.27% radius: r_68 = sigma * sqrt(-2 ln(1 - 0.6827)) = 1.515 * sigma
      Nominal 95.00% radius: r_95 = sigma * sqrt(-2 ln(1 - 0.95)) = 2.448 * sigma
    """
    results = {}
    r68_factor = np.sqrt(-2.0 * np.log(1.0 - 0.6827))
    r95_factor = np.sqrt(-2.0 * np.log(1.0 - 0.95))

    for h in horizons:
        errs = np.asarray(errors_km[str(h)])
        sigs = np.asarray(sigmas_km[str(h)])
        if len(errs) == 0:
            continue

        radius_68 = sigs * r68_factor
        radius_95 = sigs * r95_factor

        cov_68 = float(np.mean(errs <= radius_68))
        cov_95 = float(np.mean(errs <= radius_95))

        # Sharpness: mean cone radius
        sharpness_68 = float(np.mean(radius_68))
        sharpness_95 = float(np.mean(radius_95))

        # Normalized estimation error (Z-score)
        z = errs / np.maximum(sigs, 1e-4)

        results[str(h)] = {
            "nominal_68_pct": 68.3,
            "empirical_68_pct": round(cov_68 * 100.0, 1),
            "cov_68_gap_pct": round((cov_68 - 0.6827) * 100.0, 1),
            "sharpness_68_km": round(sharpness_68, 1),
            "nominal_95_pct": 95.0,
            "empirical_95_pct": round(cov_95 * 100.0, 1),
            "cov_95_gap_pct": round((cov_95 - 0.95) * 100.0, 1),
            "sharpness_95_km": round(sharpness_95, 1),
            "mean_sigma_km": round(float(np.mean(sigs)), 1),
            "mean_z_score": round(float(np.mean(z)), 2),
        }
    return results


# --------------------------------------------------------------------------
# Main Runner
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Explainability & Calibration")
    parser.add_argument("--data-dir", default="data/real")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-saliency-samples", type=int, default=3)
    args = parser.parse_args()

    device = get_device(require_gpu=True)
    out_dir = Path(args.output_dir)
    exp_dir = out_dir / "explainability"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXPLAINABILITY & UNCERTAINTY CALIBRATION (REAL SATELLITE IMAGERY)")
    print(f"Split: {args.split} | Device: {device} | Output: {out_dir}")
    print("=" * 72)

    # 1. Load Dataset
    ds = TCDataset(args.data_dir, args.split, mode="full", augmented=False)
    print(f"[data] Loaded {len(ds)} samples from {args.split} split.")

    # 2. Load Phase 2 Classifier
    p2_path = out_dir / "phase2_stage_best.pt"
    if p2_path.exists():
        stage_model = StageClassifier(backbone="resnet18", n_classes=6, input_channels=3, embed_dim=256)
        stage_model.load_state_dict(torch.load(p2_path, map_location=device, weights_only=False))
        stage_model.to(device)
        stage_model.eval()
        print(f"[model] Loaded Phase 2 Stage Classifier from {p2_path}")
    else:
        print(f"[warn] Phase 2 model checkpoint not found at {p2_path}, using newly initialized weights.")
        stage_model = StageClassifier(backbone="resnet18", n_classes=6, input_channels=3, embed_dim=256).to(device)

    # Find target layer in ResNet backbone for Grad-CAM
    target_layer = stage_model.backbone.layer4[-1].conv2

    # 3. Compute Saliency on Real Attached Samples
    gradcam = GradCAM(stage_model, target_layer)
    saliency_records = []
    generated = 0

    print("\n[explainability] Computing Grad-CAM & Integrated Gradients on real SST imagery...")
    for i in range(len(ds)):
        item = ds[i]
        if "satellite" not in item:
            continue
        sat = item["satellite"].unsqueeze(0).to(device)  # (1, 3, 512, 512)
        raw_stage = int(item["labels"].get("stage", 2))
        true_stage = max(0, min(5, raw_stage - 2))

        with torch.no_grad():
            logits = stage_model(sat)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_stage = int(np.argmax(probs))

        # Generate Grad-CAM
        cam = gradcam.generate(sat.clone(), target_class=pred_stage)

        # Generate Integrated Gradients (downscaled for speed)
        sat_small = F.interpolate(sat.detach(), size=(128, 128), mode="bilinear")
        ig = compute_integrated_gradients(stage_model, sat_small, target_class=pred_stage, steps=10)

        # Plot explanation figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sst_img = sat[0, 0].detach().cpu().numpy()
        axes[0].imshow(sst_img, cmap="viridis")
        axes[0].set_title(f"Input SST (Sample #{i})\nTrue: Stage {true_stage}, Pred: Stage {pred_stage}")
        axes[0].axis("off")

        axes[1].imshow(sst_img, cmap="gray")
        axes[1].imshow(cam, cmap="jet", alpha=0.55)
        axes[1].set_title(f"Grad-CAM Eyewall Saliency\nConf: {probs[pred_stage]:.1%}")
        axes[1].axis("off")

        axes[2].imshow(ig, cmap="hot")
        axes[2].set_title("Integrated Gradients\nFeature Attribution")
        axes[2].axis("off")

        plt.tight_layout()
        fig_path = exp_dir / f"saliency_sample_{i}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

        saliency_records.append({
            "sample_index": i,
            "true_stage": true_stage,
            "pred_stage": pred_stage,
            "confidence": round(float(probs[pred_stage]), 3),
            "figure_path": str(fig_path),
            "cam_eyewall_mean": round(float(np.mean(cam[192:320, 192:320])), 3),
            "cam_outer_mean": round(float(np.mean(cam) - np.mean(cam[192:320, 192:320])), 3),
        })
        generated += 1
        if generated >= args.num_saliency_samples:
            break

    print(f"[explainability] Generated {generated} saliency explanation maps in {exp_dir}")

    # 4. Uncertainty Calibration Evaluation
    print("\n[calibration] Evaluating uncertainty calibration across all test samples...")
    stage_probs, stage_targets = [], []
    track_errors_km = {"6": [], "12": [], "24": []}
    track_sigmas_km = {"6": [], "12": [], "24": []}

    # Load Phase 4 Fusion Model for track calibration
    p4_path = out_dir / "phase4_fusion_best.pt"
    fusion_model = SatelliteTrackFusionModel(track_dim=2, weather_dim=64, sat_embed_dim=256, hidden_dim=128)
    if p4_path.exists():
        fusion_model.load_state_dict(torch.load(p4_path, map_location=device, weights_only=False))
    fusion_model.to(device)
    fusion_model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            if "satellite" in item:
                sat = item["satellite"].unsqueeze(0).to(device)
                logits = stage_model(sat)
                prob = F.softmax(logits, dim=-1).cpu().numpy()[0]
                stage_probs.append(prob)
                stage_targets.append(max(0, min(5, int(item["labels"].get("stage", 2)) - 2)))

                # Track uncertainty
                cur = item.get("current_position", torch.zeros(2)).unsqueeze(0).to(device)
                hist = item.get("track_history", torch.zeros(6, 2)).unsqueeze(0).to(device)
                raw_w = item.get("weather", torch.zeros(64)).unsqueeze(0)
                raw_w = torch.nan_to_num(raw_w, nan=0.0, posinf=0.0, neginf=0.0)
                raw_w = torch.clamp(raw_w, -1500.0, 1500.0)
                w_mean = raw_w.mean(dim=-1, keepdim=True)
                w_std = raw_w.std(dim=-1, keepdim=True) + 1e-4
                weat = ((raw_w - w_mean) / w_std).to(device)
                tgt_rel = item["labels"].get("future_positions", torch.zeros(3, 2)).numpy()

                _, sat_emb = stage_model(sat, return_embedding=True)
                track_out = fusion_model(hist, weat, sat_emb, current_position=cur)

                for j, h in enumerate([6, 12, 24]):
                    sh = str(h)
                    pred_pos = track_out[sh]["position"][0].cpu().numpy()
                    log_var = track_out[sh]["log_variance"][0].cpu().numpy()
                    true_pos = cur[0].cpu().numpy() + tgt_rel[j]

                    dlat = (pred_pos[0] - true_pos[0]) * 110.574
                    dlon = (pred_pos[1] - true_pos[1]) * 111.320 * np.cos(np.radians(true_pos[0]))
                    err_km = float(np.sqrt(dlat**2 + dlon**2))
                    sigma_km = float(np.sqrt(np.exp(log_var[0])) * 111.0)

                    track_errors_km[sh].append(err_km)
                    track_sigmas_km[sh].append(sigma_km)

    stage_probs = np.array(stage_probs)
    stage_targets = np.array(stage_targets)

    stage_calib = compute_classification_ece(stage_probs, stage_targets) if len(stage_targets) > 0 else {}
    track_calib = compute_track_cone_calibration(track_errors_km, track_sigmas_km)

    calib_summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "sample_count": len(stage_targets),
        "stage_classification_calibration": stage_calib,
        "track_cone_uncertainty_calibration": track_calib,
        "saliency_explanations": saliency_records,
    }

    calib_path = out_dir / "calibration_metrics.json"
    calib_path.write_text(json.dumps(calib_summary, indent=2), encoding="utf-8")
    print(f"\n[calibration] COMPLETE. Results saved to {calib_path}")
    print(f"  Stage Expected Calibration Error (ECE): {stage_calib.get('ece', 'N/A')}")
    print(f"  Stage Brier Score: {stage_calib.get('brier_score', 'N/A')}")
    for h, stat in track_calib.items():
        print(f"  +{h}h Track: 68% cone empirical={stat['empirical_68_pct']}% (sharpness={stat['sharpness_68_km']} km) | "
              f"95% cone empirical={stat['empirical_95_pct']}% (sharpness={stat['sharpness_95_km']} km)")


if __name__ == "__main__":
    main()
