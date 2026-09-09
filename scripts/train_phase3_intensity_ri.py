"""Phase 3: Temporal Pattern / Intensity-Change & Rapid Intensification Model.

Per plan.md Section 5:
- Inputs: Sequence of last 4-6 satellite frames + track history.
- Derived Labels:
  * 24h Intensity change regression (delta wind in kt over next 24h)
  * Rapid Intensification (RI) binary flag (>=30 kt/24h)
- Model: ConvLSTM or CNN feature extractor per frame + GRU over sequence.
- Metrics: MAE/RMSE on intensity change; Precision/Recall/F1 on RI detection.
- Deliverable: Model with reported metrics and class imbalance handling for rare RI events.
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
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.models.pattern.temporal_model import TemporalIntensityRIModel
from tc_ai.utils.common import set_seed, get_device, count_parameters


def collate_phase3(batch):
    """Collate batch for temporal intensity & RI prediction."""
    sequences = []
    histories = []
    delta_winds = []
    ri_flags = []

    for item in batch:
        if "satellite_sequence" in item:
            sequences.append(item["satellite_sequence"])
        elif "satellite" in item:
            # Repeat frame as pseudo sequence if sequence missing
            s = item["satellite"]
            seq = s.unsqueeze(0).repeat(6, 1, 1, 1)
            sequences.append(seq)
        else:
            continue

        hist = item.get("track_history", torch.zeros(6, 2))
        if hist.shape[-1] > 2:
            hist = hist[:, :2]
        histories.append(hist)

        lbl = item["labels"]
        dw = lbl.get("delta_wind_24h", torch.tensor(0.0))
        ri = lbl.get("ri_flag", torch.tensor(0.0))
        delta_winds.append(dw.reshape(1))
        ri_flags.append(ri.reshape(1))

    if not sequences:
        return {}

    return {
        "sequences": torch.stack(sequences),
        "history": torch.stack(histories),
        "delta_wind": torch.stack(delta_winds).float(),
        "ri_flag": torch.stack(ri_flags).float(),
    }


def compute_phase3_metrics(dw_true, dw_pred, ri_true, ri_probs, ri_threshold=0.5):
    """Compute Phase 3 metrics: MAE, RMSE, RI Precision/Recall/F1."""
    dw_true = np.asarray(dw_true)
    dw_pred = np.asarray(dw_pred)
    ri_true = np.asarray(ri_true).astype(int)
    ri_probs = np.asarray(ri_probs)
    ri_preds = (ri_probs >= ri_threshold).astype(int)

    mae = float(mean_absolute_error(dw_true, dw_pred))
    rmse = float(np.sqrt(mean_squared_error(dw_true, dw_pred)))

    p, r, f1, _ = precision_recall_fscore_support(ri_true, ri_preds, average="binary", zero_division=0)
    total_samples = len(ri_true)
    total_ri = int(np.sum(ri_true))
    pred_ri = int(np.sum(ri_preds))

    return {
        "mae_wind_change_24h_kt": round(mae, 2),
        "rmse_wind_change_24h_kt": round(rmse, 2),
        "ri_precision": round(float(p), 3),
        "ri_recall": round(float(r), 3),
        "ri_f1": round(float(f1), 3),
        "total_samples": total_samples,
        "true_ri_count": total_ri,
        "pred_ri_count": pred_ri,
        "ri_prevalence_pct": round((total_ri / max(total_samples, 1)) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Temporal Intensity & RI Model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pos-weight", type=float, default=4.0, help="Weight for rare positive RI class")
    parser.add_argument("--data-source", default="synthetic_demo",
                        help="Provenance: 'synthetic_demo' or 'ibtracs_v04r01_real'")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(require_gpu=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 3: TEMPORAL PATTERN & INTENSITY CHANGE (RI) MODEL")
    print(f"Device: {device} | RI Threshold: >=30 kt/24h | Pos Weight: {args.pos_weight}")
    print("=" * 70)

    # Datasets
    train_ds = TCDataset(args.data_dir, "train", mode="temporal", augmented=True)
    val_ds = TCDataset(args.data_dir, "val", mode="temporal", augmented=False)
    test_ds = TCDataset(args.data_dir, "test", mode="temporal", augmented=False)

    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_phase3)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase3)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase3)

    model = TemporalIntensityRIModel(
        input_channels=3,
        hidden_dim=128,
        track_history_len=6,
        use_track=True,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    best_state = None

    print("\nTraining temporal intensity & RI model...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if not batch:
                continue
            seq = batch["sequences"].to(device)
            hist = batch["history"].to(device)
            dw_tgt = batch["delta_wind"].to(device)
            ri_tgt = batch["ri_flag"].to(device)

            optimizer.zero_grad()
            out = model(seq, hist)
            loss_dw = F.mse_loss(out["delta_wind_24h"], dw_tgt)
            loss_ri = F.binary_cross_entropy_with_logits(out["ri_logits"], ri_tgt, pos_weight=pos_weight)
            total_loss = loss_dw + 5.0 * loss_ri

            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_dw_true, val_dw_pred = [], []
        val_ri_true, val_ri_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                seq = batch["sequences"].to(device)
                hist = batch["history"].to(device)
                out = model(seq, hist)
                val_dw_pred.extend(out["delta_wind_24h"].cpu().numpy().flatten())
                val_dw_true.extend(batch["delta_wind"].numpy().flatten())
                val_ri_probs.extend(torch.sigmoid(out["ri_logits"]).cpu().numpy().flatten())
                val_ri_true.extend(batch["ri_flag"].numpy().flatten())

        val_metrics = compute_phase3_metrics(val_dw_true, val_dw_pred, val_ri_true, val_ri_probs)
        mean_tr = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {mean_tr:.4f} | Val MAE: {val_metrics['mae_wind_change_24h_kt']:.2f} kt | Val RI F1: {val_metrics['ri_f1']:.3f}")

        if val_metrics["mae_wind_change_24h_kt"] < best_val_mae:
            best_val_mae = val_metrics["mae_wind_change_24h_kt"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    test_dw_true, test_dw_pred = [], []
    test_ri_true, test_ri_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            if not batch:
                continue
            seq = batch["sequences"].to(device)
            hist = batch["history"].to(device)
            out = model(seq, hist)
            test_dw_pred.extend(out["delta_wind_24h"].cpu().numpy().flatten())
            test_dw_true.extend(batch["delta_wind"].numpy().flatten())
            test_ri_probs.extend(torch.sigmoid(out["ri_logits"]).cpu().numpy().flatten())
            test_ri_true.extend(batch["ri_flag"].numpy().flatten())

    metrics = compute_phase3_metrics(test_dw_true, test_dw_pred, test_ri_true, test_ri_probs)

    # Save model checkpoint
    ckpt_path = out_dir / "phase3_temporal_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved best model checkpoint -> {ckpt_path}")

    # Display Deliverable
    print("\n" + "=" * 70)
    print("PHASE 3 DELIVERABLE: TEMPORAL INTENSITY & RI PREDICTION RESULTS")
    print("=" * 70)
    print(f"24h Wind Change MAE:          {metrics['mae_wind_change_24h_kt']:.2f} kt")
    print(f"24h Wind Change RMSE:         {metrics['rmse_wind_change_24h_kt']:.2f} kt")
    print("-" * 70)
    print(f"Rapid Intensification (RI) Metrics (Threshold: >=30 kt / 24h):")
    print(f"  RI Prevalence:              {metrics['true_ri_count']} / {metrics['total_samples']} ({metrics['ri_prevalence_pct']}%)")
    print(f"  RI Precision:               {metrics['ri_precision']:.3f}")
    print(f"  RI Recall:                  {metrics['ri_recall']:.3f}")
    print(f"  RI F1-score:                {metrics['ri_f1']:.3f}")
    print(f"  Predicted RI count:         {metrics['pred_ri_count']}")
    print("=" * 70)

    # Save metrics JSON
    out_file = out_dir / "phase3_results.json"
    with open(out_file, "w") as f:
        json.dump(metrics | {"data_source": args.data_source,
                             "evaluated_at": pd.Timestamp.utcnow().isoformat()}, f, indent=2)
    print(f"Saved Phase 3 results -> {out_file}\n")


if __name__ == "__main__":
    main()
