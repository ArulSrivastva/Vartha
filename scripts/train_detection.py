"""Stage 1-2: Train cycloone detector and classifier (Modules 1-2).

Run: python scripts/train_detection.py [--synthetic]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from configs.config import TCConfig
from tc_ai.utils.common import set_seed, get_device


def collate_fn(batch):
    """Collate TCDataset batch into model inputs/targets."""
    import torch
    result = {}
    for key in ["satellite", "weather", "track_history", "current_position",
                "satellite_sequence", "sequence"]:
        vals = [b.get(key) for b in batch if b.get(key) is not None]
        if vals:
            result[key] = torch.stack(vals)
    # Targets
    labels = {}
    for key in ["has_tc", "bbox", "center", "stage", "structure",
                "pattern", "future_positions", "future_wind", "future_pressure"]:
        vals = []
        for b in batch:
            t = b.get("labels", {})
            if key in b:
                labels[key] = torch.stack([bb[key] for bb in batch]) if all(key in bb for bb in batch) else None
            elif key in t:
                labels[key] = torch.stack([bb["labels"][key] for bb in batch])
    return result, labels


def evaluate(model, loader, device, mode, cfg):
    import torch
    import numpy as np
    from tc_ai.evaluation.metrics import DetectionMetrics, ClassificationMetrics

    model.eval()
    preds_all = []
    targs_all = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = collate_fn(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(**inputs)
            preds_all.append(outputs)
            targs_all.append(targets)

    if mode == "detection":
        det_preds = [{"has_tc": torch.sigmoid(p["detection"]["has_tc"]).cpu().numpy()
                      if "detection" in p else p.get("has_tc", torch.zeros(1)).cpu().numpy(),
                      "boxes": None, "center": None} for p in preds_all]
        det_targs = [{"has_tc": t["has_tc"].cpu().numpy()} for t in targs_all]
        m = DetectionMetrics().compute(det_preds, det_targs)
    else:
        cls_preds = []
        cls_targs = []
        for p, t in zip(preds_all, targs_all):
            cls_preds.append({"classification": p.get("classification", p)})
            cls_targs.append(t)
        m = ClassificationMetrics().compute(cls_preds, cls_targs)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--task", choices=["detection", "classification"], required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    import torch
    from torch.utils.data import DataLoader
    from tc_ai.data.dataset import TCDataset

    mode = args.task
    train_ds = TCDataset(args.data_dir, "train", mode=mode)
    val_ds = TCDataset(args.data_dir, "val", mode=mode)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=2, collate_fn=collate_fn)

    from tc_ai.models.model_factory import build_detector, build_classifier
    from tc_ai.training.trainer import StagedTrainer, train_one_epoch, validate, build_optimizer, build_scheduler
    import torch.nn as nn

    if args.task == "detection":
        model = build_detector(cfg)
    else:
        model = build_classifier(cfg)
    model = model.to(device)

    print(f"Num parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.task == "detection":
        from tc_ai.training.losses import detection_loss
        def loss_fn(outputs, targets):
            if "detection" in outputs:
                outputs = outputs["detection"]
            return {"total": detection_loss(outputs, targets)}
    else:
        from tc_ai.training.losses import classification_loss
        def loss_fn(outputs, targets):
            if "classification" in outputs:
                outputs = outputs["classification"]
            return {"total": classification_loss(outputs, targets)}

    epochs = args.epochs or (cfg.training.stage1_epochs if args.task == "detection"
                             else cfg.training.stage2_epochs)
    optimizer = build_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler, epochs)

    from pathlib import Path as P
    output_dir = P(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                 grad_clip=cfg.training.gradient_clip)
        val_res = validate(model, val_loader, loss_fn, device)
        val_loss = val_res.get("total", float("inf"))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | train loss: {losses['total']:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_loss},
                       output_dir / f"{args.task}_best.pt")

    print(f"Best val loss: {best_loss:.4f}. Saved to {output_dir / f'{args.task}_best.pt'}")

    # Evaluate
    metrics = evaluate(model, val_loader, device, mode, cfg)
    import json
    with open(output_dir / f"{args.task}_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2, default=str)
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()