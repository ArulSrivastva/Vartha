"""Stage 3: Train temporal pattern recognition model (Module 3).

Run: python scripts/train_pattern.py
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
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tc_ai.data.dataset import TCDataset
    from tc_ai.models.model_factory import build_pattern_model
    from tc_ai.training.trainer import train_one_epoch, validate, build_optimizer, build_scheduler

    def collate_fn(batch):
        import torch
        seqs = torch.stack([b["satellite_sequence"] if "satellite_sequence" in b else b["labels"]["satellite_sequence"] for b in batch])
        labels = torch.stack([b["labels"]["pattern"] for b in batch])
        return {"sequence": seqs}, {"pattern": labels}

    train_ds = TCDataset(args.data_dir, "train", mode="pattern")
    val_ds = TCDataset(args.data_dir, "val", mode="pattern")
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_pattern_model(cfg).to(device)

    def loss_fn(outputs, targets):
        if isinstance(outputs, dict) and "pattern" in outputs:
            logits = outputs["pattern"]
        elif isinstance(outputs, dict):
            logits = outputs
        else:
            logits = outputs
        return {"total": torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets["pattern"])}

    epochs = args.epochs or cfg.training.stage3_epochs
    optimizer = build_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
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
        print(f"Epoch {epoch+1}/{epochs} | train: {losses['total']:.4f} | val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_loss},
                       out_dir / "pattern_best.pt")

    print(f"Pattern model complete. Best val loss: {best_loss:.4f}")

    # Metrics on val
    model.eval()
    import numpy as np
    from tc_ai.evaluation.metrics import PatternMetrics
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = collate_fn(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            if "sequence" in inputs:
                out = model(inputs["sequence"])
            out_p = {"pattern": out} if isinstance(out, torch.Tensor) else out
            all_preds.append(out_p)
            all_targs.append(targets)
    metrics = PatternMetrics().compute(all_preds, all_targs)
    import json
    with open(out_dir / "pattern_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, list)},
                  f, indent=2, default=str)
    print(f"Pattern metrics: {metrics}")


if __name__ == "__main__":
    main()