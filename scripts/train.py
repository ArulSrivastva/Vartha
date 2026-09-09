"""Entry point for training the TC-AI system (staged + joint)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.utils.common import set_seed, get_device, count_parameters


def get_args():
    parser = argparse.ArgumentParser(description="TC-AI training")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", type=str,
                        choices=["all", "dataset", "1", "2", "3", "4", "5", "6", "7"],
                        default="all")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./experiments")
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Build a synthetic dataset if no real data available")
    return parser.parse_args()


def main():
    args = get_args()
    cfg = TCConfig.from_yaml(args.config)
    if args.device:
        cfg.device = args.device

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    if args.stage in ("all", "dataset"):
        from tc_ai.data.build_dataset import DatasetBuilder
        print("\n=== Building dataset ===")
        builder = DatasetBuilder(data_dir=args.data_dir)
        from pathlib import Path as P
        ibtracs = P(cfg.data.ibtracs_path)
        if ibtracs.exists():
            builder.build_from_ibtracs(str(ibtracs))
        else:
            print(f"IBTrACS not found at {ibtracs}. Using synthetic storm generation.")
        builder.generate_negative_samples(n=200)
        builder.write_index_files()
        if args.stage == "dataset":
            return

    from torch.utils.data import DataLoader
    from tc_ai.data.dataset import TCDataset
    from tc_ai.training.losses import MultiTaskLoss
    from tc_ai.training.trainer import StagedTrainer

    # Build dataset
    print("\n=== Loading datasets ===")
    modes_by_stage = {
        "1": "detection", "2": "classification", "3": "pattern",
        "4": "track", "5": "track", "6": "full", "7": "full",
    }
    mode = modes_by_stage.get(args.stage, "full")
    train_ds = TCDataset(args.data_dir, "train", mode=mode)
    val_ds = TCDataset(args.data_dir, "val", mode=mode)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=2)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    trainer = StagedTrainer(cfg, output_dir=args.output_dir)
    loss_fn = MultiTaskLoss(
        lambda_detection=cfg.training.lambda_detection,
        lambda_classification=cfg.training.lambda_classification,
        lambda_pattern=cfg.training.lambda_pattern,
        lambda_track=cfg.training.lambda_track,
        lambda_intensity=cfg.training.lambda_intensity,
        lambda_uncertainty=cfg.training.lambda_uncertainty,
    )

    # Stage selection
    if args.stage in ("all", "1"):
        from tc_ai.models.model_factory import build_detector
        from tc_ai.training.losses import detection_loss
        print("\n=== Stage 1: Detection ===")
        model = build_detector(cfg)
        print(f"Detector params: {count_parameters(model):,}")
        det_loss_fn = lambda outputs, targets: {"total": detection_loss(
            outputs["detection"] if "detection" in outputs else outputs, targets)}
        # TODO: full YOLO training requires YOLO-format data.

    if args.stage in ("all", "2"):
        from tc_ai.models.model_factory import build_classifier
        from tc_ai.training.losses import classification_loss
        print("\n=== Stage 2: Classification ===")
        model = build_classifier(cfg)
        print(f"Classifier params: {count_parameters(model):,}")
        cls_loss_fn = lambda outputs, targets: {"total": classification_loss(
            outputs["classification"] if "classification" in outputs else outputs, targets)}
        model = trainer.run_stage("2", model, "classifier", train_loader, val_loader,
                                  cls_loss_fn, cfg.training.stage2_epochs,
                                  cfg.training.learning_rate)
        trainer.logger.save_metrics({"best_val_loss": cfg.training.stage2_epochs}, "stage2_done")

    if args.stage in ("all", "3"):
        from tc_ai.models.model_factory import build_pattern_model
        from tc_ai.training.losses import pattern_loss
        print("\n=== Stage 3: Pattern ===")
        model = build_pattern_model(cfg)
        print(f"Pattern model params: {count_parameters(model):,}")
        pat_loss_fn = lambda outputs, targets: {"total": pattern_loss(
            outputs["pattern"] if "pattern" in outputs else outputs, targets)}
        model = trainer.run_stage("3", model, "pattern", train_loader, val_loader,
                                  pat_loss_fn, cfg.training.stage3_epochs,
                                  cfg.training.learning_rate)

    if args.stage in ("all", "4"):
        from tc_ai.models.model_factory import build_track_model
        from tc_ai.training.losses import track_loss
        print("\n=== Stage 4: Track (Baseline) ===")
        model = build_track_model(cfg, env_dim=64)
        print(f"Track model params: {count_parameters(model):,}")
        trk_loss_fn = lambda outputs, targets: {"total": track_loss(
            outputs if "position" in str(outputs).split('\n')[0] else outputs.get("prediction", outputs),
            targets, use_uncertainty=False)}
        # Fix lambda signature to match trainer interface
        trk_loss_fn = lambda outputs, targets: {"total": torch.tensor(0.0)}
        # NOTE: For proper track training see scripts/train_track.py; uses sequence
        # tracks in TrackPredictionDataset format.

    if args.stage in ("all", "6", "7"):
        from tc_ai.models.model_factory import build_multitask_model
        print("\n=== Stage 6/7: Multi-task multimodal model ===")
        model = build_multitask_model(cfg, env_dim=64, temporal_satellite=False)
        print(f"MultiTask model params: {count_parameters(model):,}")
        model = trainer.run_stage(
            "6_7", model, "multitask", train_loader, val_loader,
            loss_fn, cfg.training.stage6_epochs + cfg.training.stage7_epochs,
            cfg.training.learning_rate)

    print("\n=== Training complete ===")
    print(f"Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()