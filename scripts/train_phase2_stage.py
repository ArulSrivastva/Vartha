"""Phase 2: Satellite-Based Stage Classification.

Per plan.md Section 4:
- Inputs: INSAT IR (primary) + WV channels cropped to fixed window around storm center.
- Labels: Real IMD operational category boundaries (Depression -> Super Cyclone).
- Model: CNN classifier (ResNet/EfficientNet).
- Metrics: Accuracy, macro-F1, confusion matrix, off-by-one-category analysis.
- Deliverable: Trained classifier with per-category performance and off-by-one error analysis.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.models.classification.classifier import StageClassifier
from tc_ai.training.losses import OrdinalStageLoss
from tc_ai.utils.common import set_seed, get_device, count_parameters


def collate_phase2(batch):
    """Collate batch for stage classification (satellite images + stage labels)."""
    images = []
    stages = []
    for item in batch:
        if "satellite" in item and "stage" in item["labels"]:
            images.append(item["satellite"])
            stages.append(item["labels"]["stage"])
    if not images:
        return {"satellite": torch.empty(0), "stage": torch.empty(0, dtype=torch.long)}
    return {
        "satellite": torch.stack(images),
        "stage": torch.stack(stages),
    }


def compute_stage_metrics(y_true, y_pred, stage_names):
    """Compute standard metrics + off-by-one error analysis (plan.md Phase 2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = np.abs(y_true - y_pred)
    total = len(y_true)
    errors = diff > 0
    num_errors = int(np.sum(errors))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(stage_names)))).tolist()

    off_by_one_count = int(np.sum(diff == 1))
    wildly_wrong_count = int(np.sum(diff >= 2))
    off_by_one_of_errors = float(off_by_one_count / max(num_errors, 1))
    mean_stage_dist = float(np.mean(diff))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_report = {
        stage_names[i]: {
            "f1": round(per_class_f1[i], 3),
            "support": int(np.sum(y_true == i)),
        }
        for i in range(min(len(stage_names), len(per_class_f1)))
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_stage_distance": round(mean_stage_dist, 3),
        "total_samples": total,
        "total_errors": num_errors,
        "off_by_one_errors": off_by_one_count,
        "wildly_wrong_errors": wildly_wrong_count,
        "off_by_one_fraction_of_errors": round(off_by_one_of_errors, 3),
        "confusion_matrix": cm,
        "per_class_report": per_class_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Satellite-Based Stage Classification")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true", default=False, help="Download pretrained weights")
    parser.add_argument("--ordinal", action="store_true", default=True, help="Use ordinal-aware loss")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(require_gpu=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_classes = cfg.classification.stage_classes
    n_classes = len(stage_classes)

    print("=" * 70)
    print("PHASE 2: SATELLITE-BASED STAGE CLASSIFICATION")
    print(f"Device: {device} | Backbone: {args.backbone} | Stage Classes: {n_classes}")
    print(f"Categories: {stage_classes}")
    print("=" * 70)

    # Datasets
    train_ds = TCDataset(args.data_dir, "train", mode="classification", augmented=True)
    val_ds = TCDataset(args.data_dir, "val", mode="classification", augmented=False)
    test_ds = TCDataset(args.data_dir, "test", mode="classification", augmented=False)

    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_phase2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase2)

    # Model
    model = StageClassifier(
        backbone=args.backbone,
        n_classes=n_classes,
        pretrained=args.pretrained,
        input_channels=3,
        embed_dim=256,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    loss_fn = OrdinalStageLoss(n_classes=n_classes, alpha=0.5) if args.ordinal else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_state = None

    print("\nTraining stage classifier...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if batch["satellite"].size(0) == 0:
                continue
            images = batch["satellite"].to(device)
            stages = batch["stage"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, stages)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Val evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch["satellite"].size(0) == 0:
                    continue
                images = batch["satellite"].to(device)
                logits = model(images)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["stage"].numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        train_loss = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f} | Val Macro-F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch["satellite"].size(0) == 0:
                continue
            images = batch["satellite"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch["stage"].numpy())

    metrics = compute_stage_metrics(test_targets, test_preds, stage_classes)

    # Save model checkpoint
    ckpt_path = out_dir / "phase2_stage_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "stage_classes": stage_classes, "backbone": args.backbone}, ckpt_path)
    print(f"Saved best model checkpoint -> {ckpt_path}")

    # Display Deliverable
    print("\n" + "=" * 70)
    print("PHASE 2 DELIVERABLE: STAGE CLASSIFICATION RESULTS & OFF-BY-ONE ANALYSIS")
    print("=" * 70)
    print(f"Test Accuracy:                {metrics['accuracy']:.3f}")
    print(f"Test Macro-F1:                {metrics['macro_f1']:.3f}")
    print(f"Test Weighted-F1:             {metrics['weighted_f1']:.3f}")
    print(f"Mean Stage Distance:          {metrics['mean_stage_distance']:.3f} stages")
    print(f"Total Errors:                 {metrics['total_errors']} / {metrics['total_samples']}")
    print(f"Off-By-One Errors:            {metrics['off_by_one_errors']} ({metrics['off_by_one_fraction_of_errors']*100:.1f}% of all errors)")
    print(f"Wildly Wrong Errors (>=2):    {metrics['wildly_wrong_errors']}")
    print("-" * 70)
    print("Per-Category Breakdown:")
    for cls_name, info in metrics["per_class_report"].items():
        if info["support"] > 0:
            print(f"  {cls_name:32s}: F1 = {info['f1']:.3f} (support={info['support']})")
    print("=" * 70)

    # Save metrics JSON
    out_file = out_dir / "phase2_results.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Phase 2 results -> {out_file}\n")


if __name__ == "__main__":
    main()
