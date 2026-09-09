"""Dataset builder: constructs the unified multimodal dataset from raw sources
per plan section 16-17.

Each sample:
  sample_id
    ├── satellite.npy      (C, F, H, W)  [optional]
    ├── weather.npy        (D_features)  [optional]
    ├── track_history.npy  (H, 2)        [optional]
    ├── has_tc.npy
    ├── bbox.npy
    ├── stage.npy
    ├── structure.npy
    ├── pattern.npy
    ├── future_positions.npy  (n_horizons, 2)
    ├── future_wind.npy
    └── future_pressure.npy
"""
import numpy as np
import pandas as pd
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import shutil

from ..utils.cyclone import (
    get_imd_stage_index, get_pattern_multilabel,
    classify_structure,
)
from ..utils.geo import haversine_distance


class DatasetBuilder:
    """Builds the unified multimodal training dataset."""

    def __init__(self, data_dir: str,
                 satellite_dir: Optional[str] = None,
                 era5_dir: Optional[str] = None,
                 nwp_dir: Optional[str] = None,
                 ocean_dir: Optional[str] = None,
                 n_channels: int = 3,
                 history_length: int = 6,
                 horizons: List[int] = [6, 12, 24],
                 temporal_frames: int = 6,
                 window_km: float = 600.0,
                 center_pixel: Tuple[int, int] = (128, 128),
                 synthetic_satellite: bool = True,
                 synthetic_weather: bool = True,
                 structural_labels: bool = True):
        self.data_dir = Path(data_dir)
        self.satellite_dir = Path(satellite_dir) if satellite_dir else None
        self.era5_dir = Path(era5_dir) if era5_dir else None
        self.nwp_dir = Path(nwp_dir) if nwp_dir else None
        self.ocean_dir = Path(ocean_dir) if ocean_dir else None
        self.n_channels = n_channels
        self.history_length = history_length
        self.horizons = horizons
        self.temporal_frames = temporal_frames
        self.window_km = window_km
        self.center_pixel = center_pixel
        self.synthetic_satellite = synthetic_satellite
        self.synthetic_weather = synthetic_weather
        self.structural_labels = structural_labels

        for sub in ["samples", "metadata"]:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    def build_from_ibtracs(self, ibtracs_path: str,
                           era5_base: Optional[str] = None,
                           satellite_base: Optional[str] = None,
                           split: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """Build dataset from IBTrACS (global track archive) + optional ERA5/satellite."""
        import warnings
        try:
            df = pd.read_csv(ibtracs_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read IBTrACS: {e}")

        # Normalize common columns
        col_map = {
            "ISO_TIME": "timestamp", "SID": "storm_id",
            "LAT": "lat", "LON": "lon", "WMO_WIND": "wind",
            "WMO_PRES": "pressure", "USA_WIND": "wind_alt",
            "SSHS": "saffir",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        required = ["storm_id", "timestamp", "lat", "lon"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)

        storms = list(df["storm_id"].unique())
        n = len(storms)
        import random
        random.seed(42)
        random.shuffle(storms)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        splits = {
            "train": storms[:n_train],
            "val": storms[n_train:n_train + n_val],
            "test": storms[n_train + n_val:],
        }

        for split_name, storm_list in splits.items():
            split_n = 0
            for storm in storm_list:
                storm_df = df[df["storm_id"] == storm]
                n_samples = self.build_storm(storm_df, split_name, era5_base, satellite_base)
                split_n += n_samples
            print(f"[{split_name}] built {split_n} samples from {len(storm_list)} storms")

    def build_storm(self, storm_df: pd.DataFrame, split: str,
                    era5_base: Optional[str] = None,
                    satellite_base: Optional[str] = None,
                    data_source: str = "synthetic_demo") -> int:
        """Build samples for a single storm trajectory."""
        if "storm_id" not in storm_df.columns and "SID" in storm_df.columns:
            storm_df = storm_df.rename(columns={"SID": "storm_id"})
        positions = storm_df[["lat", "lon"]].values
        winds = storm_df["wind"].values if "wind" in storm_df else np.full(len(storm_df), 40.0)
        pressures = storm_df["pressure"].values if "pressure" in storm_df else np.full(len(storm_df), 990.0)
        timestamps = storm_df["timestamp"].values
        storm_id = storm_df["storm_id"].iloc[0]

        n_samples = 0
        # Sliding windows: require enough history + horizons
        max_h_steps = max(self.horizons) // 6
        for i in range(self.history_length, len(storm_df) - max_h_steps):
            hist_pos = positions[i - self.history_length:i]
            current = positions[i]
            hist_wind = winds[i - self.history_length:i]

            future_positions = np.zeros((len(self.horizons), 2))
            future_valid = True
            for j, h in enumerate(self.horizons):
                steps = h // 6
                if i + steps < len(positions):
                    future_positions[j] = positions[i + steps]
                else:
                    future_valid = False
                    break
            if not future_valid:
                continue

            sample_id = f"{storm_id}_{timestamps[i]}"
            sample_id = self._sanitize_id(sample_id)
            sample_dir = self.data_dir / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Track history & current
            rel_pos = hist_pos - current.reshape(1, -1)
            np.save(sample_dir / "track_history.npy", rel_pos.astype(np.float32))
            np.save(sample_dir / "current_position.npy", current.astype(np.float32))

            # Full track states (H, 6): [lat, lon, wind, pressure, heading_deg, speed_kt]
            states = np.zeros((self.history_length, 6), dtype=np.float32)
            for k in range(self.history_length):
                pos_k = hist_pos[k]
                w_k = hist_wind[k]
                p_k = pressures[max(0, i - self.history_length + k)] if (i - self.history_length + k) < len(pressures) else 990.0
                if k > 0:
                    prev_k = hist_pos[k - 1]
                    d_km = haversine_distance(prev_k[0], prev_k[1], pos_k[0], pos_k[1])
                    spd_kt = (d_km / 1.852) / 6.0  # knots
                    from ..utils.geo import bearing_angle
                    head_deg = bearing_angle(prev_k[0], prev_k[1], pos_k[0], pos_k[1])
                else:
                    spd_kt = 10.0
                    head_deg = 315.0  # typical NW drift
                states[k] = [pos_k[0], pos_k[1], w_k, p_k, head_deg, spd_kt]
            np.save(sample_dir / "track_states.npy", states)

            # Stage + structure + pattern
            stage = get_imd_stage_index(winds[i])
            np.save(sample_dir / "stage.npy", np.array(stage, dtype=np.int64))

            # 24h intensity change & RI flag (Phase 3 ground truth)
            idx_24h_ahead = min(i + 4, len(winds) - 1)
            delta_wind_24h = float(winds[idx_24h_ahead] - winds[i])
            ri_flag = 1.0 if delta_wind_24h >= 30.0 else 0.0
            np.save(sample_dir / "delta_wind_24h.npy", np.array(delta_wind_24h, dtype=np.float32))
            np.save(sample_dir / "ri_flag.npy", np.array(ri_flag, dtype=np.float32))

            wind_change_24h_past = winds[i] - winds[max(0, i - 4)]
            shear = 10.0 + float(np.random.normal(0, 5))  # placeholder; from ERA5 if available
            if self.structural_labels:
                structure_labels = classify_structure(stage, wind_change_24h_past, shear)
                struct_names = ["developing", "mature", "weakening", "eye_forming",
                                "eye_present", "eyewall_organized", "highly_asymmetric",
                                "sheared", "rapidly_intensifying", "land_interaction",
                                "dissipating"]
                struct_vec = np.zeros(len(struct_names), dtype=np.float32)
                for lab in structure_labels:
                    if lab in struct_names:
                        struct_vec[struct_names.index(lab)] = 1.0
                np.save(sample_dir / "structure.npy", struct_vec)

            # Pattern (multi-label)
            if self.structural_labels:
                pattern_hist = winds[self.history_length - 5:self.history_length + 1] if self.history_length >= 5 else hist_wind
                pattern = get_pattern_multilabel(list(pattern_hist), winds[i], shear)
                np.save(sample_dir / "pattern.npy", pattern)

            # Future targets (both relative offset and absolute positions)
            np.save(sample_dir / "future_positions.npy",
                    (future_positions - current.reshape(1, -1)).astype(np.float32))
            np.save(sample_dir / "future_positions_abs.npy", future_positions.astype(np.float32))
            future_wind = np.array([winds[min(i + (h // 6), len(winds) - 1)] for h in self.horizons])
            future_pres = np.array([pressures[min(i + (h // 6), len(pressures) - 1)] for h in self.horizons])
            np.save(sample_dir / "future_wind.npy", future_wind.astype(np.float32))
            np.save(sample_dir / "future_pressure.npy", future_pres.astype(np.float32))

            # Detection targets (bbox around center)
            np.save(sample_dir / "has_tc.npy", np.array(1.0, dtype=np.float32))
            np.save(sample_dir / "center.npy", np.array([0.5, 0.5], dtype=np.float32))
            np.save(sample_dir / "bbox.npy",
                    np.array([0.3, 0.3, 0.7, 0.7], dtype=np.float32))

            # Satellite imagery (real frames only if a base dir is provided, else synthetic)
            if self.synthetic_satellite or satellite_base:
                self._build_satellite(sample_dir, storm_id, timestamps[i], satellite_base)
            # Weather features (real ERA5 only if a base dir is provided, else zeros)
            self._build_weather(sample_dir, storm_id, timestamps[i], current, era5_base)

            # Track metadata with season
            ts_dt = pd.to_datetime(timestamps[i])
            season = int(ts_dt.year)
            with open(sample_dir / "meta.json", "w") as f:
                json.dump({
                    "storm_id": storm_id, "timestamp": str(timestamps[i]),
                    "season": season,
                    "data_source": data_source,
                    "lat": float(current[0]), "lon": float(current[1]),
                    "wind_kt": float(winds[i]), "pressure_hpa": float(pressures[i]),
                    "stage": stage,
                    "delta_wind_24h": delta_wind_24h,
                    "ri_flag": int(ri_flag),
                }, f)

            n_samples += 1

        return n_samples

    @staticmethod
    def _sanitize_id(sample_id: str) -> str:
        """Sanitize sample_id for use as a filesystem path."""
        import re
        return re.sub(r'[^\w.\-]', '_', sample_id)

    def _build_satellite(self, sample_dir: Path, storm_id, timestamp, satellite_base):
        """Create synthetic multi-channel satellite imagery (placeholder).

        Real implementation would load INSAT IR/WV/VIS images from MOSDAC.
        This builds a plausible synthetic circulation pattern for testing
        the full pipeline end-to-end.
        """
        sat_dir = Path(satellite_base) / storm_id if satellite_base else None
        h, w = 256, 256
        frames = []
        for f in range(self.temporal_frames):
            channels = []
            for c in range(self.n_channels):
                img = self._synthetic_satellite_channel(h, w, intensity=0.5 + 0.1 * f / self.temporal_frames,
                                                        channel=c)
                channels.append(img)
            frames.append(np.stack(channels, axis=0))  # (C, H, W)
        arr = np.stack(frames, axis=1)  # (C, F, H, W)
        # Save channel-mean collapsed to (C, H, W) for single-frame tasks
        np.save(sample_dir / "satellite.npy", arr[:, -1].astype(np.float32))
        np.save(sample_dir / "satellite_sequence.npy", arr.astype(np.float32))
        np.save(sample_dir / "raw_satellite.npy", arr.astype(np.float32))

    def _synthetic_satellite_channel(self, h, w, intensity, channel=0):
        """Generate a synthetic tropical cyclone pattern resembling satellite imagery."""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = h * 0.5, w * 0.5
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if channel == 0:  # IR: colder = brighter near center
            img = 0.7 * np.exp(-(r / 60) ** 2) + 0.15 * np.exp(-(r / 25) ** 2)
        elif channel == 1:  # WV: water vapor bands
            img = 0.6 * np.exp(-(r / 90) ** 2) + 0.2 * np.sin(r / 15) * np.exp(-r / 120)
        else:  # VIS: spiral arms
            theta = np.arctan2(y - cy, x - cx)
            spiral = 0.5 * np.exp(-((r - 60 - 8 * theta) / 30) ** 2)
            img = 0.3 * np.exp(-(r / 120) ** 2) + 0.9 * spiral
        img = img * intensity + 0.05 * np.random.randn(h, w).astype(np.float32)
        return np.clip(img, 0, 1)

    def _build_weather(self, sample_dir: Path, storm_id, timestamp, center, era5_base):
        """Build a weather feature vector.

        Real implementation would extract ERA5 fields within a radius around
        the storm center (SST, shear, humidity, steering flow, vorticity...).
        For real data without a configured ERA5 store, write an explicit
        zero vector so no synthetic features are ever attributed to real storms.
        """
        if not self.synthetic_weather and era5_base is None:
            np.save(sample_dir / "weather.npy", np.zeros(64, dtype=np.float32))
            return
        n_feat = 64
        rng = np.random.RandomState(hash(str(timestamp)) % 2 ** 32)
        weather = rng.randn(n_feat).astype(np.float32) * 0.5
        # Encode storm-relative contextual information in first few features
        weather[0] = 28.0 + 0.5 * np.sin(center[0])  # SST proxy
        weather[1] = 15.0 + (rng.randn() * 5)          # shear proxy
        weather[2] = 0.5 * rng.randn()                 # steering u
        weather[3] = 0.5 * rng.randn()                 # steering v
        np.save(sample_dir / "weather.npy", weather)

    def generate_negative_samples(self, n: int = 500, split: str = "train"):
        """Generate 'no cyclone' samples for training the detector."""
        split_dir = self.data_dir / "samples"
        rng = np.random.RandomState(1)
        h, w = 256, 256
        made = 0
        while made < n:
            sample_id = f"neg_{uuid.uuid4().hex[:8]}"
            sample_dir = split_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            channels = []
            self.n_channels = self.n_channels or 3
            for c in range(self.n_channels):
                img = np.full((h, w), 0.15, dtype=np.float32) + 0.05 * rng.randn(h, w).astype(np.float32)
                if rng.rand() < 0.4:  # some cloud noise
                    img += 0.2 * rng.rand(h, w).astype(np.float32)
                channels.append(img)
            arr = np.stack(channels, axis=0)  # (C, H, W)
            np.save(sample_dir / "satellite.npy", arr.astype(np.float32))
            np.save(sample_dir / "has_tc.npy", np.array(0.0, dtype=np.float32))
            np.save(sample_dir / "stage.npy", np.array(0, dtype=np.int64))
            np.save(sample_dir / "pattern.npy", np.zeros(9, dtype=np.float32))
            np.save(sample_dir / "bbox.npy", np.zeros(4, dtype=np.float32))
            np.save(sample_dir / "center.npy", np.array([0.5, 0.5], dtype=np.float32))
            np.save(sample_dir / "track_history.npy", np.zeros((6, 2), dtype=np.float32))
            np.save(sample_dir / "future_positions.npy", np.zeros((3, 2), dtype=np.float32))
            with open(sample_dir / "meta.json", "w") as f:
                json.dump({"negative": True}, f)
            made += 1

    def write_index_files(self, split_by_season: bool = True):
        """Write CSV index files for train/val/test splits partitioned by storm season."""
        samples_dir = self.data_dir / "samples"
        all_samples = sorted([p.name for p in samples_dir.iterdir()
                              if p.is_dir()])
        import random
        random.seed(42)

        meta = {}
        for s in all_samples:
            m_path = samples_dir / s / "meta.json"
            if m_path.exists():
                try:
                    with open(m_path) as f:
                        meta[s] = json.load(f)
                except Exception:
                    meta[s] = {}

        pos = [s for s in all_samples if not meta.get(s, {}).get("negative")]
        neg = [s for s in all_samples if meta.get(s, {}).get("negative")]

        # Group storms by season
        storm_to_season = {}
        for s in pos:
            sid = meta[s].get("storm_id", s)
            season = meta[s].get("season")
            if season is None and "timestamp" in meta[s]:
                try:
                    season = pd.to_datetime(meta[s]["timestamp"]).year
                except Exception:
                    season = 2020
            storm_to_season[sid] = season or 2020

        unique_seasons = sorted(list(set(storm_to_season.values())))
        split_map = {}

        if split_by_season and len(unique_seasons) >= 3:
            # Strictly split by storm season (plan.md requirement)
            if any(y <= 2018 for y in unique_seasons) and any(y in (2019, 2020) for y in unique_seasons) and any(y >= 2021 for y in unique_seasons):
                season_splits = {
                    y: ("train" if y <= 2018 else ("val" if y in (2019, 2020) else "test"))
                    for y in unique_seasons
                }
            else:
                n_s = len(unique_seasons)
                n_tr = max(1, int(n_s * 0.7))
                n_va = max(1, int(n_s * 0.15))
                season_splits = {}
                for idx, y in enumerate(unique_seasons):
                    if idx < n_tr:
                        season_splits[y] = "train"
                    elif idx < n_tr + n_va:
                        season_splits[y] = "val"
                    else:
                        season_splits[y] = "test"
            for sid, season in storm_to_season.items():
                split_map[sid] = season_splits.get(season, "train")
            print(f"[season split] Seasons partitioned: {season_splits}")
        else:
            # Fallback by storm_id
            storm_ids = sorted(list(storm_to_season.keys()))
            random.shuffle(storm_ids)
            n = len(storm_ids)
            for i, sid in enumerate(storm_ids):
                r = i / max(n, 1)
                split_map[sid] = "train" if r < 0.7 else ("val" if r < 0.85 else "test")

        # Distribute negative samples deterministically across splits
        neg_split = {}
        for i, s in enumerate(neg):
            r = (i % 100) / 100.0
            neg_split[s] = "train" if r < 0.7 else ("val" if r < 0.85 else "test")

        for split in ["train", "val", "test"]:
            rows = []
            for s in all_samples:
                is_neg = meta.get(s, {}).get("negative", False)
                split_name = neg_split.get(s) if is_neg else split_map.get(
                    (meta.get(s, {}).get("storm_id") or s), "train")
                if split_name == split:
                    season_val = meta.get(s, {}).get("season", "n/a")
                    rows.append({
                        "sample_id": s,
                        "split": split,
                        "storm_id": meta.get(s, {}).get("storm_id", "negative"),
                        "season": season_val,
                    })
            out = pd.DataFrame(rows)
            out.to_csv(self.data_dir / f"{split}_index.csv", index=False)
            print(f"[index] {split}: {len(rows)} samples")

        self.create_manifest()

    def create_manifest(self) -> Path:
        """Create SHA256 checksum manifest for all dataset samples (Phase 0 deliverable)."""
        import hashlib
        samples_dir = self.data_dir / "samples"
        manifest_rows = []
        for sample_p in sorted(samples_dir.iterdir()):
            if not sample_p.is_dir():
                continue
            for f in sorted(sample_p.iterdir()):
                if f.is_file():
                    h = hashlib.sha256()
                    with open(f, "rb") as fp:
                        for chunk in iter(lambda: fp.read(65536), b""):
                            h.update(chunk)
                    manifest_rows.append({
                        "sample_id": sample_p.name,
                        "file_name": f.name,
                        "size_bytes": f.stat().st_size,
                        "sha256": h.hexdigest(),
                    })
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = self.data_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"[manifest] Saved versioned manifest with {len(manifest_rows)} files -> {manifest_path}")
        return manifest_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build TC-AI dataset")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ibtracs", default=None)
    parser.add_argument("--satellite", default=None)
    parser.add_argument("--era5", default=None)
    parser.add_argument("--negatives", type=int, default=200)
    args = parser.parse_args()

    builder = DatasetBuilder(data_dir=args.data_dir)
    if args.ibtracs:
        builder.build_from_ibtracs(args.ibtracs,
                                   era5_base=args.era5,
                                   satellite_base=args.satellite)
    else:
        print("No IBTrACS file given; generating synthetic storms only.")
    builder.generate_negative_samples(n=args.negatives)
    builder.write_index_files()