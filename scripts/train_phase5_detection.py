"""Phase 5: Detection / Center Localization via Heatmap Regression.

Per plan.md Section 7:
- Inputs: Basin-wide satellite frames.
- Model: Center-point heatmap regression detector (predicts Gaussian-peaked heatmap + presence).
- Metrics: Presence Precision/Recall/F1 + center localization error (km).
- Deliverable: A trained detector that outputs storm presence + estimated center.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.models.detection.detector import (
    HeatmapCenterDetector, generate_gaussian_heatmap, extract_heatmap_center
)
from tc_ai.training.losses import centernet_focal_loss
from tc_ai.utils.common import set_seed, get_device, count_parameters


def collate_phase5(batch):
    """Collate batch with satellite frames, presence labels, and center coordinates."""
    images = []
    has_tcs = []
    centers = []
    heatmaps = []

    for item in batch:
        if "satellite" in item and "has_tc" in item["labels"]:
            img = item["satellite"]
            c, h, w = img.shape
            images.append(img)

            has_tc = item["labels"]["has_tc"].reshape(1)
            has_tcs.append(has_tc)

            center = item["labels"].get("center", torch.tensor([0.5, 0.5])).reshape(2)
            centers.append(center)

            # Generate target heatmap if cyclone present
            if has_tc.item() > 0.5:
                hmap = generate_gaussian_heatmap((center[0].item(), center[1].item()), (h, w), sigma=4.0)
            else:
                hmap = np.zeros((h, w), dtype=np.float32)
            heatmaps.append(torch.from_numpy(hmap).unsqueeze(0))

    if not images:
        return {}

    return {
        "images": torch.stack(images),
        "has_tc": torch.stack(has_tcs).float(),
        "centers": torch.stack(centers).float(),
        "heatmaps": torch.stack(heatmaps).float(),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Center-Point Heatmap Detection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--km-per-pixel", type=float, default=4.0)
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(require_gpu=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 5: CENTER-POINT HEATMAP REGRESSION DETECTOR (STRETCH)")
    print(f"Device: {device} | Resolution: ~{args.km_per_pixel} km/pixel")
    print("=" * 70)

    train_ds = TCDataset(args.data_dir, "train", mode="detection", augmented=False)
    val_ds = TCDataset(args.data_dir, "val", mode="detection", augmented=False)
    test_ds = TCDataset(args.data_dir, "test", mode="detection", augmented=False)

    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_phase5)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase5)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase5)

    model = HeatmapCenterDetector(in_channels=3, base_channels=32).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None

    print("\nTraining heatmap center detector...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if not batch:
                continue
            images = batch["images"].to(device)
            has_tc = batch["has_tc"].to(device)
            centers = batch["centers"].to(device)
            heatmaps = batch["heatmaps"].to(device)

            optimizer.zero_grad()
            out = model(images)
            loss_pres = F.binary_cross_entropy_with_logits(out["presence_logits"], has_tc)
            loss_hmap = centernet_focal_loss(out["heatmap"], heatmaps)

            pos_mask = (has_tc.squeeze(-1) > 0.5)
            if pos_mask.any():
                loss_coord = F.smooth_l1_loss(out["center"][pos_mask], centers[pos_mask])
            else:
                loss_coord = torch.tensor(0.0, device=device)

            total_loss = loss_pres + 2.0 * loss_hmap + 10.0 * loss_coord

            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                images = batch["images"].to(device)
                has_tc = batch["has_tc"].to(device)
                centers = batch["centers"].to(device)
                heatmaps = batch["heatmaps"].to(device)
                out = model(images)
                loss_pres = F.binary_cross_entropy_with_logits(out["presence_logits"], has_tc)
                loss_hmap = centernet_focal_loss(out["heatmap"], heatmaps)
                pos_mask = (has_tc.squeeze(-1) > 0.5)
                if pos_mask.any():
                    loss_coord = F.smooth_l1_loss(out["center"][pos_mask], centers[pos_mask])
                else:
                    loss_coord = torch.tensor(0.0, device=device)
                val_losses.append((loss_pres + 2.0 * loss_hmap + 10.0 * loss_coord).item())

        mean_val = np.mean(val_losses) if val_losses else 0.0
        mean_tr = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {mean_tr:.4f} | Val Loss: {mean_val:.4f}", flush=True)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    pres_true, pres_pred = [], []
    dist_errors_km = []

    with torch.no_grad():
        for batch in test_loader:
            if not batch:
                continue
            images = batch["images"].to(device)
            has_tc = batch["has_tc"].to(device)
            centers = batch["centers"]

            out = model(images)
            probs = torch.sigmoid(out["presence_logits"]).cpu().numpy().flatten()
            h_maps = torch.sigmoid(out["heatmap"]).cpu().numpy()
            pred_centers = out["center"].cpu().numpy()

            for i in range(images.size(0)):
                true_pres = int(has_tc[i].item() > 0.5)
                pred_pres = int(probs[i] >= 0.5)
                pres_true.append(true_pres)
                pres_pred.append(pred_pres)

                if true_pres == 1:
                    # Use spatial soft-argmax continuous coordinates
                    pred_cx = float(np.clip(pred_centers[i, 0], 0.0, 1.0))
                    pred_cy = float(np.clip(pred_centers[i, 1], 0.0, 1.0))
                    true_cx = float(centers[i, 0].item())
                    true_cy = float(centers[i, 1].item())

                    # Euclidean distance in pixels on image (assumed 256x256 or 512x512)
                    h, w = h_maps[i, 0].shape
                    px_err = np.sqrt(((pred_cx - true_cx) * w) ** 2 + ((pred_cy - true_cy) * h) ** 2)
                    km_err = px_err * args.km_per_pixel
                    dist_errors_km.append(km_err)

    acc = float(accuracy_score(pres_true, pres_pred))
    p, r, f1, _ = precision_recall_fscore_support(pres_true, pres_pred, average="binary", zero_division=0)
    mean_loc_err = float(np.mean(dist_errors_km)) if dist_errors_km else 0.0
    median_loc_err = float(np.median(dist_errors_km)) if dist_errors_km else 0.0

    metrics = {
        "presence_accuracy": round(acc, 3),
        "presence_precision": round(float(p), 3),
        "presence_recall": round(float(r), 3),
        "presence_f1": round(float(f1), 3),
        "mean_center_localization_error_km": round(mean_loc_err, 1),
        "median_center_localization_error_km": round(median_loc_err, 1),
        "evaluated_cyclone_centers": len(dist_errors_km),
    }

    # Save model checkpoint
    ckpt_path = out_dir / "phase5_detector_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved best model checkpoint -> {ckpt_path}")

    # Display Deliverable
    print("\n" + "=" * 70)
    print("PHASE 5 DELIVERABLE: DETECTION & CENTER LOCALIZATION RESULTS")
    print("=" * 70)
    print(f"Cyclone Presence Accuracy:    {metrics['presence_accuracy']:.3f}")
    print(f"Cyclone Presence Precision:   {metrics['presence_precision']:.3f}")
    print(f"Cyclone Presence Recall:      {metrics['presence_recall']:.3f}")
    print(f"Cyclone Presence F1:          {metrics['presence_f1']:.3f}")
    print("-" * 70)
    print(f"Center Localization Error (Evaluated over {metrics['evaluated_cyclone_centers']} storms):")
    print(f"  Mean Error:                 {metrics['mean_center_localization_error_km']} km")
    print(f"  Median Error:               {metrics['median_center_localization_error_km']} km")
    print("=" * 70)

    # Save metrics JSON
    out_file = out_dir / "phase5_results.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Phase 5 results -> {out_file}\n")


if __name__ == "__main__":
    main()
