"""Phase 1: MVP Track Predictor (no satellite imagery).

Per plan.md Section 3:
- Inputs: Sequence of past track states + co-located ERA5 atmospheric features (steering wind, vertical shear, humidity).
- Baselines: Persistence + Climatology comparison floor.
- Model: GRU / Transformer sequence model.
- Metrics: DPE at 6h, 12h, 24h + along-track and cross-track error breakdown.
- Deliverable: Working evaluated track model with comparison table (Model vs Persistence vs Climatology).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.data.dataset import TCDataset
from tc_ai.models.prediction.baselines import PersistenceBaseline, ClimatologyBaseline
from tc_ai.models.prediction.track_predictor import GRUTrackModel, TransformerTrackModel
from tc_ai.training.losses import track_loss
from tc_ai.evaluation.metrics import TrackMetrics
from tc_ai.utils.common import set_seed, get_device, count_parameters


def collate_phase1(batch):
    """Collate batch for track training (track history + weather + targets)."""
    histories = []
    currents = []
    weathers = []
    targets_future = []

    for item in batch:
        if "track_history" in item:
            h = item["track_history"]
        elif "track_states" in item:
            h = item["track_states"][:, :2]
        else:
            h = torch.zeros(6, 2)
        if h.shape[-1] > 2:
            h = h[:, :2]
        histories.append(h)
        currents.append(item.get("current_position", torch.zeros(2)))
        weathers.append(item.get("weather", torch.zeros(64)))
        targets_future.append(item["labels"].get("future_positions", torch.zeros(3, 2)))

    return {
        "history": torch.stack(histories),
        "current": torch.stack(currents),
        "weather": torch.stack(weathers),
        "targets_future": torch.stack(targets_future),
    }


def evaluate_baselines(test_loader, horizons=[6, 12, 24], train_ds=None):
    """Evaluate Persistence and Climatology baselines on test set."""
    persistence = PersistenceBaseline(horizons)
    climatology = ClimatologyBaseline(horizons)

    # Fit climatology if train samples provided
    if train_ds is not None:
        train_samples = [{"future_positions": item["labels"].get("future_positions")}
                         for item in train_ds if "future_positions" in item["labels"]]
        climatology.fit(train_samples)

    pers_preds, pers_targets = [], []
    clim_preds, clim_targets = [], []

    for batch in test_loader:
        hist = batch["history"]
        cur = batch["current"]
        tgt_rel = batch["targets_future"]
        b_sz = hist.shape[0]

        # Targets in absolute coordinates for DPE verification
        # future_positions in targets is relative offset from current: pos_future = cur + delta
        for i in range(b_sz):
            t_dict = {}
            for j, h in enumerate(horizons):
                t_dict[str(h)] = cur[i] + tgt_rel[i, j]
            t_dict["current_position"] = cur[i]

            # Persistence
            p_out = persistence.predict(hist[i, :, :2], cur[i])
            pers_preds.append({"prediction": p_out})
            pers_targets.append(t_dict)

            # Climatology
            c_out = climatology.predict(hist[i, :, :2], cur[i])
            clim_preds.append({"prediction": c_out})
            clim_targets.append(t_dict)

    tm = TrackMetrics(horizons)
    pers_metrics = tm.compute(pers_preds, pers_targets)
    clim_metrics = tm.compute(clim_preds, clim_targets)

    return pers_metrics, clim_metrics


def train_track_model(model, train_loader, val_loader, cfg, epochs=15, device="cpu"):
    """Train ML track model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            hist = batch["history"].to(device)
            cur = batch["current"].to(device)
            weat = batch["weather"].to(device)
            tgt_rel = batch["targets_future"].to(device)

            optimizer.zero_grad()
            # Predict relative positions (current=None) so output aligns with target relative offsets
            preds = model(hist, current=None, features=weat)
            loss_dict_targets = {h: tgt_rel[:, j] for j, h in enumerate(model.horizons)}
            loss = track_loss(preds, loss_dict_targets, use_uncertainty=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["history"].to(device)
                weat = batch["weather"].to(device)
                tgt_rel = batch["targets_future"].to(device)
                preds = model(hist, current=None, features=weat)
                loss_dict_targets = {h: tgt_rel[:, j] for j, h in enumerate(model.horizons)}
                loss = track_loss(preds, loss_dict_targets, use_uncertainty=True)
                val_losses.append(loss.item())

        mean_val = np.mean(val_losses) if val_losses else 0.0
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_ml_model(model, test_loader, horizons=[6, 12, 24], device="cpu"):
    """Evaluate trained ML model on test set in absolute lat/lon coordinates."""
    model.eval()
    ml_preds, ml_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            hist = batch["history"].to(device)
            cur = batch["current"].to(device)
            weat = batch["weather"].to(device)
            tgt_rel = batch["targets_future"].to(device)

            # Pass current so prediction is returned in absolute coordinates
            preds = model(hist, current=cur, features=weat)
            b_sz = hist.shape[0]

            for i in range(b_sz):
                p_dict = {}
                t_dict = {}
                for j, h in enumerate(horizons):
                    h_str = str(h)
                    p_dict[h_str] = {
                        "position": preds[h_str]["position"][i].cpu().numpy(),
                    }
                    if "log_variance" in preds[h_str]:
                        p_dict[h_str]["log_variance"] = preds[h_str]["log_variance"][i].cpu().numpy()
                    t_dict[h_str] = (cur[i] + tgt_rel[i, j]).cpu().numpy()
                t_dict["current_position"] = cur[i].cpu().numpy()

                ml_preds.append({"prediction": p_dict})
                ml_targets.append(t_dict)

    tm = TrackMetrics(horizons)
    return tm.compute(ml_preds, ml_targets)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: MVP Track Predictor")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--model-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-source", default="synthetic_demo",
                        help="Provenance: 'synthetic_demo' or 'ibtracs_v04r01_real'")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(require_gpu=True)
    horizons = cfg.track.forecast_horizons

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 1: MVP TRACK PREDICTOR (Track History + ERA5 Atmosphere)")
    print(f"Device: {device} | Horizons: {horizons} | Model: {args.model_type.upper()}")
    print("=" * 70)

    # Load datasets
    train_ds = TCDataset(args.data_dir, "train", mode="track", augmented=False)
    val_ds = TCDataset(args.data_dir, "val", mode="track", augmented=False)
    test_ds = TCDataset(args.data_dir, "test", mode="track", augmented=False)

    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_phase1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase1)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_phase1)

    # 1. Evaluate Baselines
    print("\n[1/3] Evaluating Comparison Floor: Persistence and Climatology Baselines...")
    pers_metrics, clim_metrics = evaluate_baselines(test_loader, horizons, train_ds)

    # 2. Build and Train ML Track Predictor
    print(f"\n[2/3] Training {args.model_type.upper()} Track Predictor...")
    in_dim = 2
    weather_dim = 64

    if args.model_type == "gru":
        model = GRUTrackModel(
            input_dim=in_dim, weather_dim=weather_dim,
            hidden_dim=128, num_layers=2, horizons=horizons,
            uncertainty_estimation=True,
        )
    else:
        model = TransformerTrackModel(
            input_dim=in_dim, weather_dim=weather_dim,
            hidden_dim=128, num_layers=2, num_heads=4, horizons=horizons,
            uncertainty_estimation=True,
        )

    print(f"Model parameters: {count_parameters(model):,}")
    model = model.to(device)
    model = train_track_model(model, train_loader, val_loader, cfg, epochs=args.epochs, device=device)

    # 3. Evaluate ML Model on Test Set
    print("\n[3/3] Evaluating ML Model on Test Set...")
    ml_metrics = evaluate_ml_model(model, test_loader, horizons, device=device)

    # Save model checkpoint
    ckpt_path = out_dir / "phase1_track_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "horizons": horizons, "in_dim": in_dim}, ckpt_path)
    print(f"Saved best model checkpoint -> {ckpt_path}")

    # Build Comparison Table (Plan.md Section 3 Deliverable)
    results_table = []
    models_dict = {
        "Persistence Baseline": pers_metrics,
        "Climatology Baseline": clim_metrics,
        f"Track ML ({args.model_type.upper()})": ml_metrics,
    }

    for name, m in models_dict.items():
        row = {"Model": name}
        for h in horizons:
            row[f"{h}h DPE (km)"] = f"{m.get(f'dpe_{h}h_mean', 0.0):.1f}"
        row["Along-track 24h (km)"] = f"{m.get('along_track_24h', 0.0):.1f}"
        row["Cross-track 24h (km)"] = f"{m.get('cross_track_24h', 0.0):.1f}"
        row["Dir Error 24h (°)"] = f"{m.get('direction_error_24h', 0.0):.1f}"
        results_table.append(row)

    df_results = pd.DataFrame(results_table)
    print("\n" + "=" * 80)
    print("PHASE 1 DELIVERABLE: RESULTS TABLE (MODEL VS. PERSISTENCE VS. CLIMATOLOGY)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # Save metrics JSON
    out_file = out_dir / "phase1_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "data_source": args.data_source,
            "evaluated_at": pd.Timestamp.utcnow().isoformat(),
            "persistence": pers_metrics,
            "climatology": clim_metrics,
            "ml_track": ml_metrics,
            "table": results_table,
        }, f, indent=2)
    print(f"Saved Phase 1 results -> {out_file}\n")


if __name__ == "__main__":
    main()
