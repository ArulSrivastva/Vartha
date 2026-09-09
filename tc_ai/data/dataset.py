"""PyTorch dataset classes for the multi-source multimodal TC dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import cv2


class TCDataset(Dataset):
    """Multimodal dataset combining satellite imagery, weather fields, track history.

    Each sample:
      - satellite: (C, F, H, W) multi-channel multi-frame imagery
      - weather: (N_features,) environmental features
      - track_history: (H, 2) past positions [lat, lon]
      - targets: dict with classification, pattern, track, intensity, uncertainty
    """

    def __init__(self, data_dir: str, split: str = "train",
                 mode: str = "full", augmented: bool = True):
        """
        mode: 'detection', 'classification', 'pattern', 'track', 'full'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode
        self.augmented = augmented

        self.samples = self._load_index()

    def _load_index(self) -> List[dict]:
        index_path = self.data_dir / f"{self.split}_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}. "
                                    "Run scripts/build_dataset.py first to create the dataset.")
        df = pd.read_csv(index_path)
        return df.to_dict("records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._load_sample(sample)

    def _load_sample(self, sample: dict) -> dict:
        sample_id = sample["sample_id"]
        base_dir = self.data_dir / "samples" / sample_id

        result = {}

        # Load satellite imagery
        if self.mode in ("detection", "classification", "pattern", "temporal", "full"):
            sat_path = base_dir / "satellite.npy"
            if sat_path.exists():
                sat = np.load(sat_path)
                if self.augmented and self.split == "train":
                    sat = self._augment(sat)
                result["satellite"] = torch.from_numpy(sat).float()

            if self.mode in ("temporal", "full"):
                seq_path = base_dir / "satellite_sequence.npy"
                if seq_path.exists():
                    seq = np.load(seq_path)  # (C, F, H, W)
                    # Convert to (F, C, H, W) for standard temporal models
                    if seq.ndim == 4 and seq.shape[0] < seq.shape[1]:
                        seq = np.transpose(seq, (1, 0, 2, 3))
                    result["satellite_sequence"] = torch.from_numpy(seq).float()

        # Load weather features
        if self.mode in ("track", "prediction", "fusion", "full"):
            weather_path = base_dir / "weather.npy"
            if weather_path.exists():
                weather = np.load(weather_path)
                result["weather"] = torch.from_numpy(weather).float()

        # Load track history & states
        track_path = base_dir / "track_history.npy"
        if track_path.exists():
            track = np.load(track_path)
            result["track_history"] = torch.from_numpy(track).float()

        cur_path = base_dir / "current_position.npy"
        if cur_path.exists():
            cur = np.load(cur_path)
            result["current_position"] = torch.from_numpy(cur).float()

        states_path = base_dir / "track_states.npy"
        if states_path.exists():
            states = np.load(states_path)
            result["track_states"] = torch.from_numpy(states).float()

        # Load targets
        result["labels"] = self._load_targets(base_dir)

        return result

    def _load_targets(self, base_dir: Path) -> dict:
        targets = {}
        if self.mode in ("detection", "full"):
            has_tc_path = base_dir / "has_tc.npy"
            bbox_path = base_dir / "bbox.npy"
            center_path = base_dir / "center.npy"
            if has_tc_path.exists():
                targets["has_tc"] = torch.tensor(np.load(has_tc_path)).float()
            if bbox_path.exists():
                targets["bbox"] = torch.from_numpy(np.load(bbox_path)).float()
            if center_path.exists():
                targets["center"] = torch.from_numpy(np.load(center_path)).float()

        if self.mode in ("classification", "full"):
            stage_path = base_dir / "stage.npy"
            struct_path = base_dir / "structure.npy"
            if stage_path.exists():
                targets["stage"] = torch.tensor(np.load(stage_path)).long()
            if struct_path.exists():
                targets["structure"] = torch.from_numpy(np.load(struct_path)).float()

        if self.mode in ("pattern", "temporal", "full"):
            pattern_path = base_dir / "pattern.npy"
            if pattern_path.exists():
                targets["pattern"] = torch.from_numpy(np.load(pattern_path)).float()
            dw_path = base_dir / "delta_wind_24h.npy"
            if dw_path.exists():
                targets["delta_wind_24h"] = torch.tensor(np.load(dw_path)).float()
            ri_path = base_dir / "ri_flag.npy"
            if ri_path.exists():
                targets["ri_flag"] = torch.tensor(np.load(ri_path)).float()

        if self.mode in ("track", "prediction", "fusion", "full"):
            pos_path = base_dir / "future_positions.npy"
            if pos_path.exists():
                targets["future_positions"] = torch.from_numpy(np.load(pos_path)).float()
            abs_pos_path = base_dir / "future_positions_abs.npy"
            if abs_pos_path.exists():
                targets["future_positions_abs"] = torch.from_numpy(np.load(abs_pos_path)).float()

            if self.mode in ("prediction", "fusion", "full"):
                wind_path = base_dir / "future_wind.npy"
                pres_path = base_dir / "future_pressure.npy"
                if wind_path.exists():
                    targets["future_wind"] = torch.from_numpy(np.load(wind_path)).float()
                if pres_path.exists():
                    targets["future_pressure"] = torch.from_numpy(np.load(pres_path)).float()

        return targets

    def _augment(self, sat: np.ndarray) -> np.ndarray:
        """Random augmentation for satellite imagery (C, F, H, W)."""
        if np.random.rand() < 0.5:
            sat = np.flip(sat, axis=-2)  # vertical flip
        if np.random.rand() < 0.5:
            sat = np.flip(sat, axis=-1)  # horizontal flip
        if np.random.rand() < 0.3:
            # Rotate 90 degrees
            k = np.random.randint(1, 4)
            sat = np.rot90(sat, k, axes=(-2, -1))
        if np.random.rand() < 0.3:
            # Gaussian noise
            noise = np.random.normal(0, 0.02, sat.shape).astype(np.float32)
            sat = sat + noise
        return sat.astype(np.float32)


class TrackPredictionDataset(Dataset):
    """Dataset for track prediction (module 4) using best-track + weather data."""

    def __init__(self, tracks: pd.DataFrame, weather: Optional[Dict],
                 history_length: int = 6, horizons: List[int] = [6, 12, 24]):
        self.tracks = tracks
        self.weather = weather
        self.history_length = history_length
        self.horizons = horizons
        self.samples = self._build_samples()

    def _build_samples(self) -> List[dict]:
        samples = []
        for (storm_id), group in self.tracks.groupby("storm_id"):
            group = group.sort_values("timestamp")
            positions = group[["lat", "lon"]].values
            # Require enough history and horizon coverage
            max_needed = self.history_length + (max(self.horizons) // 6)
            for i in range(self.history_length, len(positions) - 1):
                hist = positions[i - self.history_length:i]
                future = {}
                valid = True
                for h in self.horizons:
                    steps = h // 6
                    if i + steps < len(positions):
                        future[h] = positions[i + steps]
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                samples.append({
                    "history": hist,
                    "future": future,
                    "current": positions[i - 1],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        history = torch.tensor(s["history"], dtype=torch.float32)
        current = torch.tensor(s["current"], dtype=torch.float32)
        # Relative positions (delta from current)
        rel_history = history - current.unsqueeze(0)
        future = {}
        for h, pos in s["future"].items():
            pos = torch.tensor(pos, dtype=torch.float32)
            future[h] = pos - current
        return {
            "history": rel_history,  # (H, 2) relative to current
            "current": current,      # (2,) absolute position
            "future": future,        # {h: (2,) relative}
        }


class SequenceDataset(Dataset):
    """Dataset for temporal sequence models (tropical cyclone pattern)."""

    def __init__(self, sequences_data: List[dict]):
        """sequences_data: list of dicts with 'sequence', 'labels'."""
        self.data = sequences_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "sequence": torch.tensor(d["sequence"], dtype=torch.float32),
            "labels": torch.tensor(d["labels"], dtype=torch.float32),
        }
