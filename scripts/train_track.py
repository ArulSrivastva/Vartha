"""Stage 4/5: Train track prediction + NWP integration (Module 4).

Run: python scripts/train_track.py  (uses IBTrACS/ERA5-based track data)

Stages 4 (baseline GRU) and 5 (with NWP residual) both supported.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.utils.common import set_seed, get_device


def build_track_dataset(cfg, data_dir: str):
    """Build (or load cached) track samples from TCDataset samples or IBTrACS."""
    import numpy as np
    import pandas as pd
    from pathlib import Path as P

    samples_dir = P(data_dir) / "samples"
    cache_dir = P(data_dir) / "track_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    history_len = cfg.track.history_length
    horizons = cfg.prediction.forecast_horizons
    max_steps = max(horizons) // 6

    rows = []
    for split in ["train", "val", "test"]:
        cache_file = cache_dir / f"{split}_track.npz"
        if cache_file.exists():
            d = np.load(cache_file)
            rows.append(d["rows"])
            continue
        # Build from sample index + meta.json and future positions
        index_path = P(data_dir) / f"{split}_index.csv"
        if not index_path.exists():
            continue
        df = pd.read_csv(index_path)
        split_rows = []
        for sample_id in df["sample_id"]:
            meta_path = samples_dir / sample_id / "meta.json"
            if not meta_path.exists():
                continue
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            if meta.get("negative"):
                continue
            hist_path = samples_dir / sample_id / "track_history.npy"
            fut_path = samples_dir / sample_id / "future_positions.npy"
            cur_path = samples_dir / sample_id / "current_position.npy"
            if not (hist_path.exists() and fut_path.exists()):
                continue
            hist = np.load(hist_path)  # (H, 2) relative
            cur = np.load(cur_path)
            fut = np.load(fut_path)    # (H_hor, 2) relative
            split_rows.append({
                "split": split, "sample_id": sample_id,
                "history": hist, "current": cur, "future": fut,
                "storm_id": meta.get("storm_id", sample_id),
            })
        np.savez(cache_file, rows=np.array(split_rows, dtype=object))
        rows.append(split_rows)
    return rows


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "history": torch.tensor(r["history"], dtype=torch.float32),
            "current_position": torch.tensor(r["current"], dtype=torch.float32),
            "future_positions": torch.tensor(r["future"], dtype=torch.float32),
        }


def collate_track(batch):
    import torch
    history = torch.stack([b["history"] for b in batch])
    current = torch.stack([b["current_position"] for b in batch])
    future = torch.stack([b["future_positions"] for b in batch])
    return {"history": history, "current_position": current}, {"future_positions": future}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model-type", choices=["gru", "transformer"], default=None)
    parser.add_argument("--use-nwp", action="store_true", help="Stage 5: add NWP guidance")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    if args.model_type:
        cfg.prediction.model_type = args.model_type
    set_seed(cfg.seed)
    device = get_device()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tc_ai.models.model_factory import build_track_model
    from tc_ai.training.trainer import train_one_epoch, validate, build_optimizer, build_scheduler
    from tc_ai.training.losses import gaussian_nll_loss

    rows = build_track_dataset(cfg, args.data_dir)
    train_rows = [r for r in rows[0]]
    val_rows = [r for r in rows[1]] if len(rows) > 1 else []
    test_rows = [r for r in rows[2]] if len(rows) > 2 else []

    if not train_rows:
        print("No track samples found. Build dataset first: python scripts/build_demo_data.py")
        return

    train_ds = TrackDataset(train_rows)
    val_ds = TrackDataset(val_rows) if val_rows else None
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_track)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, collate_fn=collate_track) if val_ds else None

    model = build_track_model(cfg, env_dim=64).to(device)
    print(f"Track model ({cfg.prediction.model_type}) params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    def loss_fn(outputs, targets):
        total_loss = torch.tensor(0.0, device=outputs[list(outputs.keys())[0]]["position"].device)
        future = targets["future_positions"]  # (B, H, 2)
        for i, (h, pred) in enumerate(outputs.items()):
            target = future[:, i]
            if "log_variance" in pred and cfg.prediction.uncertainty_estimation:
                total_loss = total_loss + gaussian_nll_loss(
                    pred["position"], pred["log_variance"], target)
            else:
                total_loss = total_loss + nn.functional.mse_loss(pred["position"], target)
        return {"total": total_loss / len(outputs)}

    epochs = args.epochs or (cfg.training.stage5_epochs if args.use_nwp else cfg.training.stage4_epochs)
    optimizer = build_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler, epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(epochs):
        losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                 grad_clip=cfg.training.gradient_clip)
        if val_loader:
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
                           out_dir / "track_best.pt")
        else:
            print(f"Epoch {epoch+1}/{epochs} | train: {losses['total']:.4f}")
            torch.save({"model_state_dict": model.state_dict(), "best_val_loss": losses["total"]},
                       out_dir / "track_best.pt")

    print(f"Track model complete. Best loss: {best_loss:.4f}")

    # Evaluate DPE metrics
    if val_loader:
        from tc_ai.evaluation.metrics import TrackMetrics
        model.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = collate_track(batch)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(history=inputs["history"])
                pred_out = {"prediction": out}
                t = {}
                for i, h in enumerate(cfg.prediction.forecast_horizons):
                    t[f"{h}"] = targets["future_positions"][:, i]
                    t["current_position"] = inputs["current_position"]
                all_preds.append(pred_out)
                all_targs.append(t)
        metrics = TrackMetrics(cfg.prediction.forecast_horizons).compute(all_preds, all_targs)
        import json
        with open(out_dir / "track_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        table = TrackMetrics(cfg.prediction.forecast_horizons).summary_table(metrics)
        print("\n=== DPE verification table (validation) ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    import torch.utils.data
    main()