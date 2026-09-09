"""Build a demo dataset (synthetic) for training the full pipeline end-to-end.

Run: python scripts/build_demo_data.py [--n-storms 40] [--data-dir ./data]
Generates synthetic cyclone tracks + satellite imagery so the whole system
(modules 1-4, multi-task model, dashboard) can be exercised without external
data downloads.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-storms", type=int, default=60)
    parser.add_argument("--n-negatives", type=int, default=120)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--era5", action="store_true", help="Attach synthetic weather vectors")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    from tc_ai.data.build_dataset import DatasetBuilder

    random.seed(42)
    np.random.seed(42)

    builder = DatasetBuilder(data_dir=args.data_dir,
                             era5_dir="./data/era5" if args.era5 else None,
                             n_channels=3,
                             history_length=6,
                             horizons=[6, 12, 24],
                             temporal_frames=6)

    # Build synthetic storm tracks
    rows = []
    for s in range(args.n_storms):
        storm_id = f"DEMO-{s:04d}"
        # Random start position in the Bay of Bengal / Arabian Sea
        basin = "NIO"
        if s % 3 == 0:
            lat, lon = random.uniform(5, 10), random.uniform(85, 92)
        else:
            lat, lon = random.uniform(5, 12), random.uniform(65, 72)
        wind = random.uniform(25, 45)
        pressure = 1002 - (wind - 25) * 1.5
        # Distribute storm seasons realistically across 2012-2023 (NIO cyclone seasons: May-June & Oct-Nov)
        year = 2012 + (s % 12)
        month = random.choice([5, 6, 10, 11])
        day = random.randint(1, 20)
        t0 = datetime(year, month, day)
        for step in range(40):  # 40 * 6h = 10 days
            rows.append({
                "SID": storm_id,
                "timestamp": (t0 + timedelta(hours=6 * step)).strftime("%Y-%m-%d %H:%M:%S"),
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "wind": round(wind, 1),
                "pressure": round(pressure, 1),
            })
            # Motion: mostly NW with wobble
            lat += random.uniform(0.02, 0.12)
            lon += random.uniform(-0.15, -0.02)
            # Intensity evolution: develop -> peak -> weaken
            wind += random.uniform(-1.5, 2.5)
            if step > 25:
                wind -= random.uniform(1, 2)
            wind = min(wind, 130)
            pressure = max(920, 1002 - (wind - 25) * 1.5)

    from pathlib import Path as P
    data_dir = P(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    track_path = data_dir / "ibtracs_demo.csv"
    pd.DataFrame(rows).to_csv(track_path, index=False)
    print(f"Generated {len(rows)} best-track rows -> {track_path}")

    # Build the dataset
    df = pd.read_csv(track_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build samples for all storms
    total_samples = 0
    storms = df["SID"].unique().tolist()
    for storm in storms:
        storm_df = df[df["SID"] == storm]
        total_samples += builder.build_storm(storm_df, split="all")
    print(f"Total positive samples generated: {total_samples} from {len(storms)} storms")

    builder.generate_negative_samples(n=args.n_negatives)
    builder.write_index_files(split_by_season=True)
    print("Dataset build and seasonal partitioning complete.")
    print("Phase 0 manifest and split indices created successfully.")



if __name__ == "__main__":
    main()