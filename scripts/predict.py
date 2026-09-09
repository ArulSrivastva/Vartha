"""Real-time prediction script: run inference on new satellite data.

Run: python scripts/predict.py --image path/to/insat.png [--sequence dir/] [--track track.csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import TCConfig
from tc_ai.inference.engine import InferenceEngine
import numpy as np


def load_image(path):
    from PIL import Image
    img = np.array(Image.open(path).convert("L")).astype(np.float32)
    img = img / 255.0 if img.max() > 2 else img
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", default=None, help="Path to satellite image")
    parser.add_argument("--sequence-dir", default=None, help="Dir with temporal frames")
    parser.add_argument("--track-csv", default=None, help="CSV with lat,lon history")
    parser.add_argument("--checkpoint-dir", default="./experiments")
    parser.add_argument("--wind", type=float, default=None, help="Current wind speed (kt)")
    args = parser.parse_args()

    cfg = TCConfig.from_yaml(args.config)
    engine = InferenceEngine(cfg, checkpoint_dir=args.checkpoint_dir)

    print("\n=== TC-AI Real-time Inference ===")
    print("Model status:")
    for k, v in engine.models.items():
        print(f"  {k}: {'loaded' if v is not None else 'missing'}")

    if args.image:
        sat = load_image(args.image)
        print(f"Loaded satellite image: {sat.shape}")
    else:
        # Generate a synthetic active cyclone image
        print("No image provided; generating synthetic satellite frame...")
        from tc_ai.data.build_dataset import DatasetBuilder
        b = DatasetBuilder(data_dir="./data")
        sat = b._synthetic_satellite_channel(512, 512, intensity=1.0, channel=0)
        sat = np.stack([sat] * 3, axis=0).transpose(1, 2, 0) if sat.ndim == 2 else sat

    # Track history
    track_history = None
    if args.track_csv:
        import pandas as pd
        df = pd.read_csv(args.track_csv)
        track_history = df[["lat", "lon"]].values[-6:]
    elif engine.models.get("track") or engine.models.get("multitask"):
        # Generate plausible synthetic track approaching the cyclone
        print("Generating synthetic 36h track history (6 x 6h)")
        cyc_center = np.array([8.0, 71.0])
        steps = 6
        track_history = cyc_center - np.array(
            [0.15 * (steps - i) for i in range(steps)],
        ) - np.array([0.05 * i for i in range(steps)])
        track_history = track_history[::-1]

    result = engine.full_inference(
        satellite=sat,
        track_history=track_history,
        wind_kt=args.wind,
    )

    import json
    print("\n" + "=" * 75)
    print("PHASE 6: CHAINED REAL-TIME INFERENCE RESULT")
    print("=" * 75)
    
    det = result.get("detection", {})
    cls = result.get("classification", {})
    pat = result.get("pattern", {})
    fc = result.get("track_forecast", {})

    print(f"Timestamp:              {result.get('timestamp')}")
    print(f"Pipeline Status:        {result.get('pipeline_status')}")
    print("-" * 75)
    print("[1] DETECTION & LOCALIZATION (Phase 5):")
    print(f"    Cyclone Detected:   {'YES' if det.get('cyclone_detected') else 'NO'}")
    print(f"    Presence Conf:      {det.get('confidence', 0.0) * 100:.1f}%")
    print(f"    Center (lat, lon):  {det.get('center', {}).get('lat', 0.0):.2f}°N, {det.get('center', {}).get('lon', 0.0):.2f}°E")
    print(f"    Bounding Box norm:  {det.get('bbox_normalized')}")

    print("-" * 75)
    print("[2] STAGE CLASSIFICATION (Phase 2):")
    print(f"    Operational Stage:  {cls.get('stage', 'N/A')}")
    print(f"    Stage Confidence:   {cls.get('confidence', 0.0) * 100:.1f}%")

    print("-" * 75)
    print("[3] TEMPORAL INTENSITY & RAPID INTENSIFICATION (Phase 3):")
    print(f"    24h Intensity Trend:{pat.get('delta_wind_24h_kt', 0.0):+.1f} kt ({pat.get('intensity_trend', 'Steady')})")
    print(f"    RI Probability:     {pat.get('ri_probability', 0.0) * 100:.1f}% (>=30kt/24h)")
    print(f"    RI Warning Flag:    {'ALERT: Rapid Intensification Probable' if pat.get('rapid_intensification') else 'No RI detected'}")

    print("-" * 75)
    print("[4] MULTIMODAL FUSION TRACK FORECAST & UNCERTAINTY (Phase 4):")
    curr = fc.get("current", {})
    print(f"    Current Position:   {curr.get('lat', 0.0):.2f}°N, {curr.get('lon', 0.0):.2f}°E")
    print("    Horizon  | Forecast Lat/Lon    | Radius (km) | IMD Operational DPE Target")
    print("    " + "-" * 67)
    
    targets = {"6h": "< 25 km", "12h": "< 50 km", "24h": "< 100 km"}
    for h, f in fc.get("forecasts", {}).items():
        lat_val = f.get("lat", 0.0)
        lon_val = f.get("lon", 0.0)
        rad = f.get("uncertainty_radius_km", 0.0)
        tgt = targets.get(h, "< 50 km")
        print(f"    {h:<8} | {lat_val:6.2f}°N, {lon_val:6.2f}°E | {rad:6.1f} km  | {tgt}")
    print("=" * 75)

    # Save output to json
    out_path = Path(args.checkpoint_dir) / "inference_output.json"
    with open(out_path, "w") as f:
        # Exclude torch tensors
        safe_result = {k: v for k, v in result.items() if k != "classification" or not isinstance(v, dict)}
        safe_result["classification"] = {k: v for k, v in cls.items() if k != "embedding"}
        safe_result["detection"] = {k: v for k, v in det.items() if k != "heatmap"}
        safe_result["pattern"] = {k: v for k, v in pat.items() if k != "temporal_embedding"}
        safe_result["track_forecast"] = fc
        json.dump(safe_result, f, indent=2, default=str)
    print(f"\nSaved inference record -> {out_path}\n")


if __name__ == "__main__":
    main()