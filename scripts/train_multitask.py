"""Stage 6/7: Train the joint multi-task multimodal model (final architecture).

Run: python scripts/train_multitask.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.utils.common import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--load-pretrained", type=str, default=None,
                        help="Path to pretrained stage checkpoints")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    import torch
    from torch.utils.data import DataLoader
    from tc_ai.data.dataset import TCDataset
    from tc_ai.models.model_factory import build_multitask_model
    from tc_ai.training.losses import MultiTaskLoss
    from tc_ai.training.trainer import train_one_epoch, validate, build_optimizer, build_scheduler
    from scripts.train_detection import collate_fn

    train_ds = TCDataset(args.data_dir, "train", mode="full")
    val_ds = TCDataset(args.data_dir, "val", mode="full")
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = build_multitask_model(cfg, env_dim=64, temporal_satellite=False).to(device)
    print(f"Multi-task model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_pretrained:
        ckpt = torch.load(args.load_pretrained, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded pretrained weights from {args.load_pretrained}")

    loss_fn = MultiTaskLoss(
        lambda_detection=cfg.training.lambda_detection,
        lambda_classification=cfg.training.lambda_classification,
        lambda_pattern=cfg.training.lambda_pattern,
        lambda_track=cfg.training.lambda_track,
        lambda_intensity=cfg.training.lambda_intensity,
        lambda_uncertainty=cfg.training.lambda_uncertainty,
    )

    epochs = args.epochs or (cfg.training.stage6_epochs + cfg.training.stage7_epochs)
    optimizer = build_optimizer(model, cfg.training.learning_rate / 10 if args.load_pretrained
                                else cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler, epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

        print(f"Epoch {epoch+1}/{epochs} | train: {losses.get('total', 0):.4f} "
              f"| val: {val_loss:.4f} | det: {losses.get('detection', 0):.3f} "
              f"| cls: {losses.get('classification', 0):.3f} "
              f"| pat: {losses.get('pattern', 0):.3f} "
              f"| trk: {losses.get('track', 0):.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_loss},
                       out_dir / "multitask_best.pt")

    print(f"Joint fine-tuning complete. Best val loss: {best_loss:.4f}")

    # Final evaluation across all tasks
    from tc_ai.evaluation.metrics import (
        DetectionMetrics, ClassificationMetrics, PatternMetrics, TrackMetrics,
    )
    from scripts.train_detection import collate_fn
    import numpy as np

    model.eval()
    det_p, det_t, cls_p, cls_t, pat_p, pat_t, trk_p, trk_t = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = collate_fn(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(**inputs)
            if "has_tc" in targets:
                det_p.append({"has_tc": torch.sigmoid(outputs["detection"]["has_tc"]).cpu().numpy(),
                              "boxes": None, "center": None})
                det_t.append({"has_tc": targets["has_tc"].cpu().numpy()})
            if "stage" in targets:
                cls_p.append({"classification": outputs["classification"]})
                cls_t.append(targets)
            if "pattern" in targets:
                pat_p.append({"pattern": outputs["pattern"]})
                pat_t.append({"pattern": targets["pattern"]})
            if "future_positions" in targets:
                pred_out = {}
                for i, h in enumerate(cfg.prediction.forecast_horizons):
                    ph = outputs["prediction"][f"{h}"]
                    pred_out[f"{h}"] = ph
                trk_p.append({"prediction": pred_out})
                t = {f"{h}": targets["future_positions"][:, i]
                     for i, h in enumerate(cfg.prediction.forecast_horizons)}
                t["current_position"] = inputs.get("current_position")
                trk_t.append(t)

    metrics = {}
    if det_p:
        metrics.update(DetectionMetrics().compute(det_p, det_t))
    if cls_p:
        metrics.update(ClassificationMetrics().compute(cls_p, cls_t))
    if pat_p:
        metrics.update(PatternMetrics().compute(pat_p, pat_t))
    if trk_p:
        metrics.update(TrackMetrics(cfg.prediction.forecast_horizons).compute(trk_p, trk_t))

    import json
    with open(out_dir / "multitask_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")},
                  f, indent=2, default=str)
    print("\n=== Final multi-task validation metrics ===")
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()