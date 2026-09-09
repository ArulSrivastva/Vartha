"""Phase 0 (real): fetch real IBTrACS data and build the REAL dataset.

Pulls public NOAA IBTrACS v04r01 North Indian Ocean + ACTIVE storm lists,
normalizes to 6-hourly storm-centric tracks, and builds strict seasonal
train/val/test splits (zero temporal leakage) plus a `recent` live holdout
index of the newest real storms.

Usage:
  python scripts/fetch_real_data.py
  python scripts/fetch_real_data.py --force --out-dir ./data/real
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tc_ai.data.real_data import (
    fetch_ibtracs, normalize_ibtracs, consolidate, build_real_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Fetch real IBTrACS and build REAL dataset")
    parser.add_argument("--cache-dir", default="data/external")
    parser.add_argument("--out-dir", default="data/real")
    parser.add_argument("--ibtracs-csv", default=None,
                        help="Optional path to a pre-downloaded NI IBTrACS CSV (skips download)")
    parser.add_argument("--force", action="store_true", help="Re-download raw files")
    parser.add_argument("--min-season", type=int, default=2012)
    parser.add_argument("--recent-since", type=int, default=2024)
    parser.add_argument("--history-length", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ibtracs_csv:
        ni_path = Path(args.ibtracs_csv)
    else:
        paths = fetch_ibtracs(args.cache_dir, force=args.force)
        ni_path = paths["ni"]

    print("=" * 70)
    print("PHASE 0 (REAL): IBTrACS v04r01 REAL DATA ACQUISITION")
    print("=" * 70)

    # 1. Load + normalize historical NI tracks
    ni_df = normalize_ibtracs(str(ni_path), six_hourly=True)
    print(f"[real] historical NI tracks: {len(ni_df):,} rows, "
          f"{ni_df['storm_id'].nunique()} storms, seasons "
          f"{ni_df['season'].min()}-{ni_df['season'].max()}")

    # 2. Merge any currently-active NI storms (real-time freshness)
    active_path = Path(args.cache_dir) / "ibtracs.ACTIVE.list.v04r01.csv"
    active_df = None
    if active_path.exists():
        import pandas as pd
        active_df = pd.read_csv(active_path, low_memory=False)
        active_nis = active_df[active_df["BASIN"].astype(str).str.strip() == "NI"]
        print(f"[real] ACTIVE list: {len(active_df)} rows; "
              f"{len(active_nis)} currently-active NI rows")

    all_df = consolidate(ni_df, active_df)
    all_df = all_df[all_df["season"] >= args.min_season].reset_index(drop=True)

    # 3. Persist the normalized REAL track table
    real_csv = out_dir / "ibtracs_real.csv"
    all_df.to_csv(real_csv, index=False)
    print(f"[real] consolidated real tracks -> {real_csv} "
          f"({len(all_df):,} rows, {all_df['storm_id'].nunique()} storms)")

    # 4. Build storm-centric REAL dataset with strict seasonal splits + manifest
    meta = build_real_dataset(
        str(real_csv),
        data_dir=str(out_dir),
        history_length=args.history_length,
        horizons=[6, 12, 24],
        recent_since=args.recent_since,
        min_season=args.min_season,
    )

    # 5. Summary
    print("\n" + "=" * 70)
    print("REAL DATASET SUMMARY")
    print("=" * 70)
    for split in ["train", "val", "test", "recent"]:
        idx = out_dir / f"{split}_index.csv"
        if idx.exists():
            import pandas as pd
            df = pd.read_csv(idx)
            n_storms = df["storm_id"].nunique()
            seasons = sorted(df["season"].unique())
            print(f"  {split:<7} | samples: {len(df):<5} | storms: {n_storms:<3} | seasons: {seasons}")
    print("=" * 70)
    print(f"Manifest & split indices: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()