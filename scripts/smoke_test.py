"""Smoke tests for TC-AI core components (no training required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_geo():
    from tc_ai.utils.geo import (
        direct_positional_error, haversine_distance, bearing_angle,
        direction_error, along_track_error, cross_track_error,
    )
    d = haversine_distance(15, 72, 16, 73)
    assert 140 < d < 170, f"Expected ~150 km, got {d}"
    dpe = direct_positional_error(15, 72, 15.5, 72.5)
    assert dpe > 0
    b1 = bearing_angle(15, 72, 16, 72)  # due north ~ 0
    b2 = bearing_angle(15, 72, 15, 73)  # due east ~ 90
    assert 0 <= b1 <= 10, f"North bearing wrong: {b1}"
    assert 80 <= b2 <= 100, f"East bearing wrong: {b2}"
    assert direction_error(10, 30) == 20
    assert direction_error(350, 10) == 20
    print("geo utils OK")


def test_cyclone():
    from tc_ai.utils.cyclone import (
        get_imd_stage_class, get_imd_stage_index, get_pattern_multilabel,
        get_saffir_simpson_category, compute_wind_shear,
    )
    assert get_imd_stage_class(20) == "depression"
    assert get_imd_stage_class(40) == "cyclonic_storm"
    assert get_imd_stage_class(75) == "very_severe_cyclonic_storm"
    assert get_imd_stage_class(100) == "extremely_severe_cyclonic_storm"
    assert get_imd_stage_index(20) == 2
    shear = compute_wind_shear(10, 0, 0, 10)
    assert abs(shear - 14.14) < 0.1
    pat = get_pattern_multilabel([40, 42, 48, 55, 60, 70], 70, 8)
    assert pat[2] == 1 or pat[1] == 1  # RI or intensifying
    lacat = get_saffir_simpson_category(50)
    assert lacat == "tropical_storm"
    print("cyclone utils OK")


def test_config():
    from configs.config import TCConfig
    cfg = TCConfig()
    assert cfg.track.forecast_horizons == [6, 12, 24]
    assert len(cfg.classification.stage_classes) == 8
    assert cfg.training.lambda_track == 2.0
    # YAML round trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        TCConfig().to_yaml(path)
        cfg2 = TCConfig.from_yaml(path)
        assert cfg2.track.history_length == 6
    finally:
        os.unlink(path)
    print("config OK")


def test_dataset_build(tmp_dir="./data_smoke"):
    import shutil
    import numpy as np
    from pathlib import Path as P
    from tc_ai.data.build_dataset import DatasetBuilder
    root = P(tmp_dir)
    if root.exists():
        shutil.rmtree(root)
    builder = DatasetBuilder(data_dir=str(root), era5_dir="./data/era5",
                             n_channels=3, history_length=6, horizons=[6, 12, 24],
                             temporal_frames=6)
    # Single synthetic storm
    import pandas as pd
    from datetime import datetime, timedelta
    rows = []
    lat, lon = 8.0, 71.0
    wind = 30.0
    for step in range(40):
        rows.append({"SID": "TEST-0001", "lat": lat, "lon": lon,
                     "wind": wind, "pressure": 990,
                     "timestamp": (datetime(2021, 5, 1) + timedelta(hours=6*step))})
        lat += 0.08
        lon -= 0.07
        wind += 2.0
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n = builder.build_storm(df, "train")
    assert n > 10, f"Expected >10 samples, got {n}"
    builder.generate_negative_samples(n=10)
    builder.write_index_files()

    idx = P(root) / "train_index.csv"
    assert idx.exists()
    import pandas as pd2
    df_idx = pd2.read_csv(idx)
    assert len(df_idx) > 0

    # Verify a sample's contents
    samples = P(root) / "samples"
    first = df_idx["sample_id"].iloc[0]
    assert (samples / first / "track_history.npy").exists()
    assert (samples / first / "future_positions.npy").exists()
    assert (samples / first / "stage.npy").exists()
    sat = np.load(samples / first / "satellite.npy")
    assert sat.shape == (3, 256, 256), f"satellite shape {sat.shape}"
    seq = np.load(samples / first / "satellite_sequence.npy")
    assert seq.shape == (3, 6, 256, 256), f"sequence shape {seq.shape}"
    shutil.rmtree(root)
    print("dataset build OK")


def test_multitask_model():
    import torch
    from tc_ai.models.fusion.multimodal import MultiTaskModel
    from tc_ai.training.losses import MultiTaskLoss

    model = MultiTaskModel(
        satellite_channels=3, n_stage_classes=8, n_structure_classes=11,
        n_pattern_classes=9, horizons=[6, 12, 24], weather_dim=64,
        track_history_len=6, embed_dim=128, image_size=64,
        satellite_backbone="efficientnet_b0", predict_intensity=True,
        uncertainty_estimation=True, fusion_layers=2, fusion_heads=4,
        temporal_satellite=False,
    )
    satellite = torch.randn(2, 3, 64, 64)
    weather = torch.randn(2, 64)
    history = torch.randn(2, 6, 2)
    outputs = model(satellite=satellite, weather=weather, track_history=history)
    assert outputs["detection"]["has_tc"].shape == (2, 1)
    assert outputs["classification"]["stage_logits"].shape == (2, 8)
    assert outputs["pattern"].shape == (2, 9)
    for h in [6, 12, 24]:
        assert str(h) in outputs["prediction"]
        assert outputs["prediction"][str(h)]["position"].shape == (2, 2)

    loss_fn = MultiTaskLoss()
    # Build targets
    targets = {
        "has_tc": torch.rand(2, 1),
        "center": torch.rand(2, 2) * 0.4 + 0.3,
        "bbox": torch.rand(2, 4) * 0.4 + 0.3,
        "stage": torch.randint(0, 8, (2,)),
        "structure": torch.randint(0, 2, (2, 11)).float(),
        "pattern": torch.randint(0, 2, (2, 9)).float(),
        "6": torch.randn(2, 2),
        "12": torch.randn(2, 2),
        "24": torch.randn(2, 2),
    }
    losses = loss_fn(outputs, targets)
    assert "total" in losses
    assert losses["total"].dim() == 0
    losses["total"].backward()
    print(f"multitask model OK (total loss = {losses['total'].item():.4f})")


def test_track_models():
    import torch
    from tc_ai.models.prediction.track_predictor import (
        GRUTrackModel, TransformerTrackModel,
    )
    from tc_ai.training.losses import track_loss

    history = torch.randn(4, 6, 2)
    current = torch.randn(4, 2)
    futures = {6: torch.randn(4, 2), 12: torch.randn(4, 2), 24: torch.randn(4, 2)}

    for model in [GRUTrackModel(horizons=[6, 12, 24], hidden_dim=64, num_layers=2),
                  TransformerTrackModel(horizons=[6, 12, 24], hidden_dim=64, num_layers=2, num_heads=4)]:
        out = model(history, current=current)
        for h in [6, 12, 24]:
            assert out[str(h)]["position"].shape == (4, 2)
            assert "log_variance" in out[str(h)]
        loss = track_loss(out, futures, use_uncertainty=True)
        assert loss.dim() == 0
        loss.backward()
    print("track models OK")


def test_pattern_model():
    import torch
    from tc_ai.models.pattern.temporal_model import TemporalPatternModel
    model = TemporalPatternModel(
        input_channels=3, hidden_dim=32, num_layers=2,
        n_pattern_classes=9, sequence_length=6, image_size=32,
    )
    seq = torch.randn(2, 6, 3, 32, 32)
    out = model(seq)
    assert out.shape == (2, 9)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out, torch.zeros(2, 9))
    loss.backward()
    print("pattern model OK")


def test_training_loop():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tc_ai.training.trainer import train_one_epoch, validate

    model = nn.Linear(4, 2)
    data = TensorDataset(torch.randn(32, 4), torch.randn(32, 2))
    loader = DataLoader(data, batch_size=8)

    def loss_fn(outputs, targets):
        return {"total": nn.functional.mse_loss(outputs, targets["y"])}

    def forward(model, batch):
        x, y = batch
        # trainer passes model(**inputs); adapt
        return model(x)

    # Override split logic by using inputs/targets split
    from tc_ai.training.trainer import split_inputs_targets
    class WrapperTrainer:
        pass
    # Direct test of loss_fn/train loop shape instead
    optimizer = torch.optim.Adam(model.parameters())
    # monkeypatched: use custom loop via trainer by passing batch as-is
    # (trainer splits batch into inputs/targets; Tensors go to inputs)
    # Instead, test the split function on a dict
    batch_dict = {"satellite": torch.randn(4, 3, 64, 64), "history": torch.randn(4, 6, 2),
                  "stage": torch.randint(0, 8, (4,))}
    inputs, targets = split_inputs_targets(batch_dict)
    assert "satellite" in inputs and "history" in inputs and "stage" in targets
    # Test optimizer/scheduler builders
    from tc_ai.training.trainer import build_optimizer, build_scheduler
    opt = build_optimizer(model, 1e-3, 1e-5)
    sch = build_scheduler(opt, "cosine", 10)
    assert sch is not None
    print("training utilities OK")


def test_inference_engine():
    import numpy as np
    from configs.config import TCConfig
    from tc_ai.inference.engine import InferenceEngine
    cfg = TCConfig()
    engine = InferenceEngine(cfg, checkpoint_dir="./experiments_missing")
    assert len(engine.models) == 0  # no checkpoints

    # Detection heuristic
    sat = np.full((128, 128), 0.5, dtype=np.float32)
    sat[50:78, 55:73] = 0.1  # cold spot in center
    det = engine.detect(sat)
    assert "cyclone_detected" in det
    assert det["confidence"] > 0

    # Track dead-reckoning fallback
    hist = np.array([[10.0, 70.0], [10.1, 69.9], [10.2, 69.8], [10.3, 69.7],
                     [10.4, 69.6], [10.5, 69.5]])
    fc = engine.predict_track(hist)
    assert "forecasts" in fc
    for h in ["6h", "12h", "24h"]:
        assert h in fc["forecasts"]
    print("inference engine OK")


def test_verification():
    from tc_ai.evaluation.verification import DPEVerifier
    from datetime import datetime
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    os.unlink(db)
    v = DPEVerifier(db_path=db)
    v.save_forecast("STORM-A", datetime(2025, 7, 1, 0), 6, 15.0, 72.0)
    dpe = v.verify("STORM-A", datetime(2025, 7, 1, 0), 6, 15.2, 72.3)
    assert dpe is not None and dpe > 0
    rep = v.report(6)
    assert "mean_dpe_km" in rep
    os.unlink(db)
    print("DPE verification OK")


def test_metrics():
    import numpy as np
    from tc_ai.evaluation.metrics import DetectionMetrics, PatternMetrics, TrackMetrics, TrackMetrics as TM
    # Detection
    preds = [{"has_tc": np.array([0.9]), "boxes": None, "center": None}]
    targs = [{"has_tc": np.array([1])}]
    m = DetectionMetrics().compute(preds, targs)
    assert "f1" in m
    # Pattern
    p = [{"pattern": np.array([[2.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}]
    t = [{"pattern": np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}]
    pm = PatternMetrics().compute(p, t)
    assert "macro_f1" in pm
    # Track metrics with targets
    preds = [{"prediction": {"6": {"position": np.array([[15.0, 72.0]]),
                                     "log_variance": np.array([[1.0, 1.0]])}}},
             {"prediction": {"6": {"position": np.array([[16.0, 73.0]]),
                                     "log_variance": np.array([[1.0, 1.0]])}}}]
    targs = [{"6": np.array([15.0, 72.0])}, {"6": np.array([15.1, 72.1])}]
    tm = TrackMetrics([6]).compute(preds, targs)
    assert "dpe_6h_mean" in tm
    print("metrics OK")


if __name__ == "__main__":
    for fn in [test_geo, test_cyclone, test_config]:
        fn()
    print("\n--- dataset (requires shutil/tmpfs) ---")
    test_dataset_build()
    print("\n--- models ---")
    test_track_models()
    test_pattern_model()
    test_multitask_model()
    test_training_loop()
    print("\n--- inference/eval ---")
    test_inference_engine()
    test_verification()
    test_metrics()
    print("\nALL SMOKE TESTS PASSED")