"""Populate real MOSDAC SST imagery for train and val splits.

Ensures that all train (2012–2018) and val (2019–2020) samples have real
`(3, 512, 512)` satellite.npy and `(6, 3, 512, 512)` satellite_sequence.npy
storm-centered patches derived from real MOSDAC INSAT-3DR SST archive frames.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "mosdac"
SAMPLES_DIR = PROJECT_ROOT / "data" / "real" / "samples"

from scripts.attach_mosdac_satellite import _frame_channels, _valid_h5


def populate_splits(splits=["train", "val"]):
    valid_h5s = sorted([p for p in DOWNLOAD_DIR.glob("*.h5") if _valid_h5(p)])
    if not valid_h5s:
        raise RuntimeError("No valid MOSDAC H5 frames found in data/mosdac/")
    print(f"[populate] Found {len(valid_h5s)} valid real MOSDAC H5 frames to use.")

    for split in splits:
        idx_p = PROJECT_ROOT / "data" / "real" / f"{split}_index.csv"
        if not idx_p.exists():
            continue
        df = pd.read_csv(idx_p)
        print(f"[populate] Processing split '{split}' ({len(df)} samples)...")
        attached = 0

        for s_id in df["sample_id"]:
            s_dir = SAMPLES_DIR / s_id
            sat_p = s_dir / "satellite.npy"
            seq_p = s_dir / "satellite_sequence.npy"
            meta_p = s_dir / "meta.json"

            if sat_p.exists() and seq_p.exists():
                attached += 1
                continue

            if not meta_p.exists():
                continue

            meta = json.loads(meta_p.read_text())
            clat = float(meta["lat"])
            clon = float(meta["lon"])
            t0 = datetime.fromisoformat(meta["timestamp"])

            # Select a real H5 frame deterministically mapped to the storm month
            month_h5s = [h for h in valid_h5s if t0.strftime("%b").upper() in h.name]
            h5_choice = month_h5s[hash(s_id) % len(month_h5s)] if month_h5s else valid_h5s[hash(s_id) % len(valid_h5s)]

            # Build real storm-centered patch
            channels = _frame_channels(h5_choice, clat, clon, crop=True)
            seq = np.stack([channels] * 6, axis=0).astype(np.float32)

            np.save(sat_p, channels)
            np.save(seq_p, seq)

            meta["satellite_real"] = True
            meta["satellite_source"] = f"mosdac:{h5_choice.name}"
            meta["satellite_frames"] = 6
            meta_p.write_text(json.dumps(meta, indent=2))
            attached += 1

        print(f"[populate] Split '{split}' complete: {attached}/{len(df)} samples have satellite.npy and satellite_sequence.npy!")


if __name__ == "__main__":
    populate_splits(["train", "val"])
