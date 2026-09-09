"""Evaluation script: run ablation study + all module metrics.

Run: python scripts/evaluate.py [--data-dir ./data] [--checkpoint-dir ./experiments]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./experiments")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    import json
    results_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TC-AI Evaluation ===")
    all_metrics = {}

    # Load per-stage metric JSONs if available
    for name in ["detection", "classification", "pattern", "track", "multitask"]:
        mpath = results_dir / f"{name}_metrics.json"
        if mpath.exists():
            with open(mpath) as f:
                all_metrics[name] = json.load(f)
                print(f"\n[{name}]")
                for k, v in all_metrics[name].items():
                    if not k.startswith("_"):
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Track DPE table
    from tc_ai.evaluation.metrics import TrackMetrics, AblationEvaluator
    if "track" in all_metrics:
        tm = TrackMetrics(cfg.prediction.forecast_horizons)
        table = tm.summary_table(all_metrics["track"])
        print("\n=== Track DPE summary (targets: 6h<25km, 12h<50km, 24h<100km) ===")
        print(table.to_string(index=False))

    # Run live evaluation on the val dataset if checkpoints exist
    try:
        import torch
        from torch.utils.data import DataLoader
        from tc_ai.data.dataset import TCDataset
        from tc_ai.models.model_factory import build_multitask_model
        from scripts.train_multitask import collate_fn

        ckpt = results_dir / "multitask_best.pt"
        val_ds = TCDataset(args.data_dir, "val", mode="full")
        if ckpt.exists() and len(val_ds) > 0:
            print("\n=== Running full multi-task evaluation on validation set ===")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_multitask_model(cfg, env_dim=64, temporal_satellite=False).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state), strict=False)
            model.eval()

            from tc_ai.evaluation.metrics import (
                DetectionMetrics, ClassificationMetrics, PatternMetrics, TrackMetrics,
            )
            loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
            det_p, det_t, cls_p, cls_t, pat_p, pat_t, trk_p, trk_t = [], [], [], [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
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
                        pred_out = {f"{h}": outputs["prediction"][f"{h}"]
                                    for h in cfg.prediction.forecast_horizons}
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

            for k, v in metrics.items():
                if not k.startswith("_"):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            with open(results_dir / "final_evaluation.json", "w") as f:
                json.dump({k: v for k, v in metrics.items() if not k.startswith("_")},
                          f, indent=2, default=str)
        else:
            print("\nNo multitask checkpoint found. Run training first.")
    except Exception as e:
        print(f"\nLive evaluation skipped: {e}")

    # Ablation study framework
    print("\n=== Ablation framework ===")
    print("Model | Satellite | ERA5 | NWP | Track history | 6h | 12h | 24h DPE")
    print("Run train_track.py with different data configurations to fill the table.")


if __name__ == "__main__":
    main()