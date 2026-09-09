"""Phase 1 Track Baselines: Persistence and Climatology.

These establish the comparison floor for all ML models as required by plan.md.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union


class PersistenceBaseline:
    """Persistence baseline: assumes storm continues on current heading and speed.
    
    Given track history [..., pos_{t-1}, pos_t], translation velocity per 6h step is:
        v = pos_t - pos_{t-1}
    Future position at horizon h hours (steps = h / 6):
        pos_{t+h} = pos_t + steps * v
    """

    def __init__(self, horizons: List[int] = [6, 12, 24]):
        self.horizons = horizons

    def predict(self, track_history: Union[np.ndarray, torch.Tensor],
                current_position: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        track_history: (B, H, 2) or (H, 2)
        current_position: (B, 2) or (2,), optional if history contains absolute positions
        
        Returns: {h_str: {"position": array}}
        """
        if isinstance(track_history, torch.Tensor):
            hist = track_history.detach().cpu().numpy()
        else:
            hist = np.asarray(track_history)

        single = False
        if hist.ndim == 2:
            hist = hist[np.newaxis, ...]  # (1, H, 2)
            single = True

        b, h_len, _ = hist.shape
        if h_len < 2:
            raise ValueError(f"History length must be >= 2, got {h_len}")

        # If history is relative to current (i.e. last point is [0, 0]),
        # then velocity is (hist[:, -1] - hist[:, -2])
        # and current_position must be supplied to return absolute positions.
        v = hist[:, -1] - hist[:, -2]  # change over last 6h step

        if current_position is not None:
            if isinstance(current_position, torch.Tensor):
                cur = current_position.detach().cpu().numpy()
            else:
                cur = np.asarray(current_position)
            if cur.ndim == 1:
                cur = cur[np.newaxis, ...]
        else:
            # Assume history is absolute
            cur = hist[:, -1]

        predictions = {}
        for h in self.horizons:
            steps = h / 6.0
            pred_pos = cur + steps * v
            if single:
                pred_pos = pred_pos.squeeze(0)
            predictions[str(h)] = {"position": pred_pos}

        return predictions


class ClimatologyBaseline:
    """Climatology baseline: predicts movement based on historical average displacements.
    
    Fits the mean displacement vector for each forecast horizon over the training set.
    """

    def __init__(self, horizons: List[int] = [6, 12, 24]):
        self.horizons = horizons
        # Default NIO (North Indian Ocean) empirical 6h, 12h, 24h climatological displacement (lat, lon)
        # Typically cyclones in NIO drift NW: ~ +0.5 deg lat, -0.6 deg lon per 24h
        self.mean_displacements = {
            6: np.array([0.15, -0.18], dtype=np.float32),
            12: np.array([0.30, -0.35], dtype=np.float32),
            24: np.array([0.65, -0.75], dtype=np.float32),
        }

    def fit(self, training_samples: List[dict]):
        """Fit mean displacement vectors from training samples.
        
        training_samples: list of dicts with 'future_positions' (delta or absolute)
        """
        deltas = {h: [] for h in self.horizons}
        for sample in training_samples:
            if "future_positions" in sample:
                fpos = sample["future_positions"]
                if isinstance(fpos, torch.Tensor):
                    fpos = fpos.detach().cpu().numpy()
                for i, h in enumerate(self.horizons):
                    if i < len(fpos):
                        deltas[h].append(fpos[i])
        for h in self.horizons:
            if deltas[h]:
                self.mean_displacements[h] = np.mean(deltas[h], axis=0)

    def predict(self, track_history: Union[np.ndarray, torch.Tensor],
                current_position: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Predict positions using climatological mean displacement."""
        if current_position is not None:
            if isinstance(current_position, torch.Tensor):
                cur = current_position.detach().cpu().numpy()
            else:
                cur = np.asarray(current_position)
        else:
            if isinstance(track_history, torch.Tensor):
                hist = track_history.detach().cpu().numpy()
            else:
                hist = np.asarray(track_history)
            cur = hist[:, -1] if hist.ndim == 3 else hist[-1]

        single = (cur.ndim == 1)
        if single:
            cur = cur[np.newaxis, ...]

        predictions = {}
        for h in self.horizons:
            disp = self.mean_displacements.get(h, np.array([0.0, 0.0]))
            pred_pos = cur + disp[np.newaxis, :]
            if single:
                pred_pos = pred_pos.squeeze(0)
            predictions[str(h)] = {"position": pred_pos}

        return predictions
