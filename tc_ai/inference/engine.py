"""Real-time inference engine (plan section 21/23 - Module 13).

Runtime workflow:
  1. Receive new satellite image
  2. Quality control
  3. Cyclone detector
  4. Feature extraction (multi-source)
  5. Pattern model
  6. Environmental data
  7. NWP/AIFS
  8. Hybrid AI model -> +6h/+12h/+24h
  9. Uncertainty cone
  10. Output for dashboard
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..data.preprocessing import SatellitePreprocessor, GeoProcessor
from ..models.model_factory import (
    build_detector, build_classifier, build_pattern_model,
    build_track_model, build_multitask_model,
)
from ..utils.geo import (
    direct_positional_error, haversine_distance, convert_km_to_degrees_deg,
)


class InferenceEngine:
    """Full real-time inference pipeline."""

    def __init__(self, config=None, checkpoint_dir: str = "./experiments",
                 model_type: str = "fusion"):
        self.cfg = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.device = self._select_device()

        self.sat_preprocessor = SatellitePreprocessor()
        self.geo = GeoProcessor()

        self.models: Dict[str, nn.Module] = {}
        self._load_models()

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_models(self):
        """Load trained phase models from checkpoints."""
        # Phase 5: Heatmap Center Detector
        p5_path = self.checkpoint_dir / "phase5_detector_best.pt"
        if p5_path.exists():
            try:
                from ..models.detection.detector import HeatmapCenterDetector
                model = HeatmapCenterDetector(in_channels=3, base_channels=32)
                ckpt = torch.load(p5_path, map_location=self.device)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                model.to(self.device).eval()
                self.models["detector"] = model
                print("[inference] loaded Phase 5 detector (HeatmapCenterDetector)")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 5 detector: {e}")
        elif (self.checkpoint_dir / "detector_best.pt").exists():
            try:
                model = build_detector(self.cfg) if self.cfg else None
                if model:
                    ckpt = torch.load(self.checkpoint_dir / "detector_best.pt", map_location=self.device)
                    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    model.to(self.device).eval()
                    self.models["detector"] = model
            except Exception as e:
                print(f"[inference] warning: {e}")

        # Phase 2: Stage Classifier
        p2_path = self.checkpoint_dir / "phase2_stage_best.pt"
        if not p2_path.exists():
            p2_path = self.checkpoint_dir / "phase2_stage_aux_best.pt"
        if p2_path.exists():
            try:
                ckpt = torch.load(p2_path, map_location=self.device)
                sd = ckpt.get("model_state_dict", ckpt)
                n_cls = 6
                embed_dim = 256
                if "classifier.weight" in sd:
                    n_cls = sd["classifier.weight"].shape[0]
                    embed_dim = sd["classifier.weight"].shape[1]
                from ..models.classification.classifier import StageClassifier
                model = StageClassifier(backbone="resnet18", n_classes=n_cls, pretrained=False, input_channels=3, embed_dim=embed_dim)
                model.load_state_dict(sd, strict=False)
                model.to(self.device).eval()
                self.models["classifier"] = model
                print(f"[inference] loaded Phase 2 classifier (StageClassifier, n_classes={n_cls}, embed_dim={embed_dim})")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 2 classifier: {e}")
        elif (self.checkpoint_dir / "classifier_best.pt").exists():
            try:
                model = build_classifier(self.cfg) if self.cfg else None
                if model:
                    ckpt = torch.load(self.checkpoint_dir / "classifier_best.pt", map_location=self.device)
                    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                    model.to(self.device).eval()
                    self.models["classifier"] = model
            except Exception as e:
                print(f"[inference] warning: {e}")

        # Phase 3: Temporal Intensity & RI
        p3_path = self.checkpoint_dir / "phase3_temporal_best.pt"
        if p3_path.exists():
            try:
                from ..models.pattern.temporal_model import TemporalIntensityRIModel
                model = TemporalIntensityRIModel(input_channels=3, hidden_dim=128, track_history_len=6, use_track=True)
                ckpt = torch.load(p3_path, map_location=self.device)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                model.to(self.device).eval()
                self.models["temporal"] = model
                print("[inference] loaded Phase 3 temporal model (TemporalIntensityRIModel)")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 3 temporal model: {e}")

        # Phase 4: Multimodal Fusion Track Model
        p4_path = self.checkpoint_dir / "phase4_fusion_best.pt"
        if p4_path.exists():
            try:
                from ..models.fusion.multimodal import SatelliteTrackFusionModel
                model = SatelliteTrackFusionModel(track_dim=2, weather_dim=64, sat_embed_dim=256, hidden_dim=128)
                ckpt = torch.load(p4_path, map_location=self.device)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                model.to(self.device).eval()
                self.models["fusion"] = model
                print("[inference] loaded Phase 4 fusion track model (SatelliteTrackFusionModel)")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 4 fusion model: {e}")

        # Phase 1: Track-only baseline fallback
        p1_path = self.checkpoint_dir / "phase1_track_best.pt"
        if p1_path.exists() and "fusion" not in self.models:
            try:
                from ..models.prediction.track_predictor import GRUTrackModel
                model = GRUTrackModel(input_dim=2, weather_dim=64)
                ckpt = torch.load(p1_path, map_location=self.device)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                model.to(self.device).eval()
                self.models["track"] = model
                print("[inference] loaded Phase 1 track model (GRUTrackModel)")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 1 track model: {e}")

        # Phase 7: Structural Cloud-Pattern Recognition (additive module)
        p7_path = self.checkpoint_dir / "phase7_structural_best.pt"
        if p7_path.exists():
            try:
                from ..models.pattern.structural import StructuralPatternModel, STRUCTURE_LABELS
                ckpt = torch.load(p7_path, map_location=self.device)
                model = StructuralPatternModel(
                    n_structure_classes=len(STRUCTURE_LABELS),
                    embed_dim=ckpt.get("config", {}).get("embed_dim", 128),
                    image_size=ckpt.get("config", {}).get("image_size", 128))
                model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                model.to(self.device).eval()
                self.models["structural"] = model
                self.structural_label_names = ckpt.get("label_names", STRUCTURE_LABELS)
                self.structural_thresholds = ckpt.get(
                    "results", {}).get("decision_thresholds_val_tuned",
                                       [0.5] * len(STRUCTURE_LABELS))
                self.structural_eye_stats = ckpt.get("train_eye_stats",
                                                     {"mean": 0.0, "std": 1.0})
                print("[inference] loaded Phase 7 structural pattern model "
                      "(StructuralPatternModel)")
            except Exception as e:
                print(f"[inference] warning: could not load Phase 7 structural model: {e}")

    def _prepare_image_tensor(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """Convert input image (H, W) or (C, H, W) or (H, W, C) to (1, 3, target_H, target_W)."""
        img = np.nan_to_num(image.astype(np.float32), 0.0)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        elif img.ndim == 3:
            if img.shape[2] in (1, 3):
                img = img.transpose(2, 0, 1)
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
        
        # Scale to [0, 1] if needed
        if img.max() > 2.0:
            img = img / 255.0

        # Resize if dimensions differ
        c, h, w = img.shape
        if (h, w) != target_size:
            import torch.nn.functional as F
            t = torch.from_numpy(img).unsqueeze(0)
            t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
            return t.to(self.device)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def detect(self, satellite_image: np.ndarray) -> Dict[str, Any]:
        """Module 1 / Phase 5: detect cyclone presence and locate center."""
        if "detector" in self.models:
            tensor = self._prepare_image_tensor(satellite_image, target_size=(128, 128))
            model = self.models["detector"]
            with torch.no_grad():
                out = model(tensor)
                presence_logit = out["presence_logits"]
                presence_prob = torch.sigmoid(presence_logit).item()
                heatmap = torch.sigmoid(out["heatmap"]).cpu().numpy()[0, 0]
            
            if "center" in out and out["center"] is not None:
                center_t = out["center"].detach().cpu().numpy()[0]
                cx, cy = float(center_t[0]), float(center_t[1])
                cx = max(0.05, min(0.95, cx))
                cy = max(0.05, min(0.95, cy))
                peak_conf = float(np.max(heatmap))
            else:
                from ..models.detection.detector import extract_heatmap_center
                cx, cy, peak_conf = extract_heatmap_center(heatmap)

            center_lat, center_lon = self.geo.pixel_to_latlon(cx * 512, cy * 512)
            has_tc = presence_prob >= 0.5

            half_box = 0.15
            bbox = [max(0.0, cx - half_box), max(0.0, cy - half_box),
                    min(1.0, cx + half_box), min(1.0, cy + half_box)]

            return {
                "cyclone_detected": bool(has_tc),
                "confidence": float(presence_prob),
                "peak_confidence": float(peak_conf),
                "center": {"lat": float(center_lat), "lon": float(center_lon)},
                "center_normalized": [float(cx), float(cy)],
                "bbox_normalized": bbox,
            }

        # Fallback heuristic
        has_tc, center_lat, center_lon, bbox = self._heuristic_detection(satellite_image)
        return {
            "cyclone_detected": has_tc > 0.5,
            "confidence": float(has_tc),
            "center": {"lat": float(center_lat), "lon": float(center_lon)},
            "bbox_normalized": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
        }

    def _heuristic_detection(self, image: np.ndarray) -> Tuple:
        """Fallback detection based on coldest-pixel centroid (IR proxy)."""
        if image.ndim == 3:
            gray = image.mean(axis=0)
        else:
            gray = image
        y, x = np.unravel_index(np.argmin(gray), gray.shape)
        lat, lon = self.geo.pixel_to_latlon(x / gray.shape[1] * 512,
                                            y / gray.shape[0] * 512)
        centered = (abs(x - gray.shape[1] / 2) < gray.shape[1] / 4 and
                    abs(y - gray.shape[0] / 2) < gray.shape[0] / 4)
        return (0.7 if centered else 0.3, lat, lon, np.array([0.3, 0.3, 0.7, 0.7]))

    def classify(self, satellite: np.ndarray, wind_kt: Optional[float] = None) -> Dict[str, Any]:
        """Module 2 / Phase 2: IMD operational stage classification + feature embedding."""
        stage_classes = self.cfg.classification.stage_classes if (self.cfg and hasattr(self.cfg, "classification")) else [
            "Depression", "Deep Depression", "Cyclonic Storm", "Severe Cyclonic Storm",
            "Very Severe Cyclonic Storm", "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm", "Dissipated"
        ]

        if "classifier" in self.models:
            import torch.nn.functional as F
            tensor = self._prepare_image_tensor(satellite, target_size=(128, 128))
            model = self.models["classifier"]
            with torch.no_grad():
                if hasattr(model, "proj"):
                    stage_logits, emb = model(tensor, return_embedding=True)
                else:
                    stage_logits = model(tensor)
                    emb = torch.zeros((1, 256), device=self.device)

            probs = F.softmax(stage_logits, dim=-1).cpu().numpy()[0]
            stage_idx = int(stage_logits.argmax(dim=1).item())
            structure = self.structural_patterns(
                satellite, wind_kt=wind_kt)["structure"]
            return {
                "stage": stage_classes[stage_idx] if stage_idx < len(stage_classes) else "Unknown",
                "stage_idx": stage_idx,
                "confidence": float(probs[stage_idx]),
                "probabilities": {stage_classes[i]: float(probs[i]) for i in range(min(len(stage_classes), len(probs)))},
                "embedding": emb,
                "structure": structure,
            }

        # Fallback from wind
        from ..utils.cyclone import get_imd_stage_class
        stage = get_imd_stage_class(wind_kt) if wind_kt is not None else "Depression"
        structure = self.structural_patterns(
            satellite, wind_kt=wind_kt, delta_wind_24h=None)["structure"]
        return {
            "stage": stage,
            "stage_idx": stage_classes.index(stage) if stage in stage_classes else 0,
            "confidence": 0.85 if wind_kt else 0.5,
            "probabilities": {},
            "embedding": torch.zeros((1, 256), device=self.device),
            "structure": structure,
        }

    def structural_patterns(self, satellite: np.ndarray,
                            center_normalized: Tuple[float, float] = (0.5, 0.5),
                            wind_kt: Optional[float] = None,
                            delta_wind_24h: Optional[float] = None,
                            shear: Optional[float] = None,
                            has_eye: Optional[bool] = None,
                            asymmetric: Optional[bool] = None) -> Dict[str, Any]:
        """Module 7 (additive): structural cloud-pattern recognition.

        Runs the Phase 7 StructuralPatternModel on a storm-centered crop and
        returns the structural multi-label vector, per-label probabilities and
        the three geometric diagnostics. Falls back to the rule-based
        heuristic when the checkpoint is unavailable.
        """
        model = self.models.get("structural")
        if model is None:
            return self._structural_heuristic(
                wind_kt=wind_kt, delta_wind_24h=delta_wind_24h,
                shear=shear, has_eye=has_eye, asymmetric=asymmetric)

        names = getattr(self, "structural_label_names", None)
        if names is None:
            from ..models.pattern.structural import STRUCTURE_LABELS
            names = STRUCTURE_LABELS
        thresholds = getattr(self, "structural_thresholds", None) or [0.5] * len(names)
        eye_stats = getattr(self, "structural_eye_stats", {"mean": 0.0, "std": 1.0})

        # Storm-centered crop (256px at 512 grid -> 128px model input)
        import torch.nn.functional as F
        tensor = self._prepare_image_tensor(satellite, target_size=(512, 512))
        frame = tensor[0].cpu().numpy()
        cx = float(center_normalized[0]) * (512 - 1)
        cy = float(center_normalized[1]) * (512 - 1)
        half = 128
        x0, y0 = int(cx - half), int(cy - half)
        crop = frame[:, max(y0, 0):min(y0 + 2 * half, 512),
                     max(x0, 0):min(x0 + 2 * half, 512)]
        pad_l = max(0, -x0); pad_r = max(0, x0 + 2 * half - 512)
        pad_t = max(0, -y0); pad_b = max(0, y0 + 2 * half - 512)
        if pad_t or pad_b or pad_l or pad_r:
            crop = np.pad(crop, ((0, 0), (pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        crop_t = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0)
        crop_t = F.interpolate(crop_t, size=(128, 128), mode="bilinear", align_corners=False)

        with torch.no_grad():
            out = model(crop_t.to(self.device))
        probs = torch.sigmoid(out["structure_logits"]).cpu().numpy()[0]
        eye_raw = float(out["eye_idx"].item() * eye_stats["std"] + eye_stats["mean"])
        asym = float(out["asym_idx"].item())
        org = float(out["org_idx"].item())

        structure = [names[i] for i in range(len(names))
                     if probs[i] >= float(thresholds[i])]
        return {
            "structure": structure,
            "structure_probs": {names[i]: float(probs[i]) for i in range(len(names))},
            "eye_index": eye_raw,
            "asymmetry_index": asym,
            "organization": org,
            "method": "neural",
        }

    def _structural_heuristic(self, wind_kt=None, delta_wind_24h=None,
                              shear=None, has_eye=None, asymmetric=None) -> Dict[str, Any]:
        """Rule-based structural fallback consistent with the weak-label rules."""
        from ..data.structural_labels import derive_structural_labels, STRUCTURE_LABELS
        meta = {
            "wind_kt": wind_kt if wind_kt is not None else 0.0,
            "pressure_hpa": 1013.0,
            "delta_wind_24h": delta_wind_24h or 0.0,
            "ri_flag": int((delta_wind_24h or 0.0) >= 25.0),
            "stage": 0,
        }
        indices = {"eye_idx": 0.0, "shear_arc": 1.0 - float(shear) / 30.0 if shear else 1.0,
                   "org_continuity": 0.5}
        thresholds = {"eye_q30": -0.05, "eye_q60": 0.0, "org_q50": 0.5}
        labels = derive_structural_labels(meta, indices, thresholds)
        structure = [STRUCTURE_LABELS[i] for i, v in enumerate(labels) if v]
        prob = 0.85 if (wind_kt or delta_wind_24h) else 0.5
        return {
            "structure": structure,
            "structure_probs": {name: (prob if name in structure else 0.3)
                                for name in STRUCTURE_LABELS},
            "eye_index": 0.0,
            "asymmetry_index": 1.0 - (float(shear) / 30.0 if shear else 0.0),
            "organization": 0.5,
            "method": "heuristic",
        }

    def pattern(self, satellite_sequence: Optional[np.ndarray] = None,
                track_history: Optional[np.ndarray] = None,
                current_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Module 3 / Phase 3: temporal intensity change (24h) & Rapid Intensification (RI)."""
        if "temporal" in self.models:
            # Prepare sequence (1, T, 3, 128, 128)
            if satellite_sequence is not None:
                seq_arr = satellite_sequence
                if seq_arr.ndim == 4:  # (T, C, H, W) or (T, H, W, C)
                    frames = [self._prepare_image_tensor(seq_arr[t], (128, 128)) for t in range(seq_arr.shape[0])]
                    seq_t = torch.cat(frames, dim=0).unsqueeze(0)  # (1, T, 3, 128, 128)
                else:
                    seq_t = self._prepare_image_tensor(current_frame if current_frame is not None else np.zeros((128, 128)), (128, 128)).unsqueeze(1).repeat(1, 4, 1, 1, 1)
            elif current_frame is not None:
                frame_t = self._prepare_image_tensor(current_frame, (128, 128))
                seq_t = frame_t.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            else:
                seq_t = torch.zeros((1, 4, 3, 128, 128), device=self.device)

            track_t = None
            if track_history is not None:
                track_t = torch.from_numpy(track_history[:, :2]).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.models["temporal"](seq_t, track_t)
                delta_wind = out["delta_wind_24h"].item()
                ri_prob = torch.sigmoid(out["ri_logits"]).item()
                emb = out["embedding"]

            trend = "Intensifying" if delta_wind > 5.0 else ("Weakening" if delta_wind < -5.0 else "Steady")
            return {
                "delta_wind_24h_kt": round(float(delta_wind), 2),
                "ri_probability": round(float(ri_prob), 3),
                "rapid_intensification": bool(ri_prob >= 0.5),
                "intensity_trend": trend,
                "temporal_embedding": emb,
                "patterns": ["Rapid Intensification Alert"] if ri_prob >= 0.5 else [f"Trend: {trend}"],
            }

        return {
            "delta_wind_24h_kt": 0.0,
            "ri_probability": 0.05,
            "rapid_intensification": False,
            "intensity_trend": "Steady",
            "patterns": ["Steady"],
        }

    def predict_track(self, track_history: np.ndarray,
                      environment: Optional[np.ndarray] = None,
                      sat_embedding: Optional[torch.Tensor] = None,
                      current_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Module 4 / Phase 4: multimodal fusion forecast (+6h/+12h/+24h) + uncertainty."""
        device = self.device
        if track_history.ndim == 2:
            trk_pts = track_history[:, :2]
        else:
            trk_pts = track_history

        curr = current_position if current_position is not None else trk_pts[-1]
        curr_t = torch.from_numpy(curr.astype(np.float32)).unsqueeze(0).to(device)

        # Convert track history to relative offsets from current position
        rel_history = trk_pts - curr
        history_t = torch.from_numpy(rel_history.astype(np.float32)).unsqueeze(0).to(device)

        # Weather features
        if environment is not None:
            weather_t = torch.from_numpy(environment.astype(np.float32)).unsqueeze(0).to(device)
        else:
            weather_t = torch.zeros((1, 64), dtype=torch.float32, device=device)

        # Satellite embedding
        if sat_embedding is not None:
            sat_emb_t = sat_embedding if isinstance(sat_embedding, torch.Tensor) else torch.from_numpy(sat_embedding).to(device)
            if sat_emb_t.ndim == 1:
                sat_emb_t = sat_emb_t.unsqueeze(0)
        else:
            sat_emb_t = torch.zeros((1, 256), dtype=torch.float32, device=device)

        result = {
            "current": {"lat": float(curr[0]), "lon": float(curr[1])},
            "forecasts": {}
        }

        has_features = (environment is not None) or (sat_embedding is not None)
        if "fusion" in self.models and has_features:
            model = self.models["fusion"]
            with torch.no_grad():
                out = model(history_t, weather_t, sat_emb_t, current_position=curr_t)
            for h in [6, 12, 24]:
                hkey = str(h)
                pos = out[hkey]["position"].cpu().numpy()[0]
                entry = {"lat": float(pos[0]), "lon": float(pos[1])}
                if "log_variance" in out[hkey]:
                    logvar = out[hkey]["log_variance"].cpu().numpy()[0]
                    # Convert variance to radial uncertainty (approximate km)
                    deg_std = np.sqrt(np.exp(np.clip(logvar, -4.0, 4.0)))
                    entry["uncertainty_radius_km"] = float(round(float(np.mean(deg_std) * 111.0), 1))
                else:
                    entry["uncertainty_radius_km"] = float({6: 25.0, 12: 45.0, 24: 85.0}.get(h, 50.0))
                result["forecasts"][f"{h}h"] = entry

        elif "track" in self.models:
            model = self.models["track"]
            with torch.no_grad():
                out = model(history_t, weather=weather_t, current=curr_t)
            for h in [6, 12, 24]:
                hkey = str(h)
                pos = out[hkey]["position"].cpu().numpy()[0]
                entry = {
                    "lat": float(pos[0]),
                    "lon": float(pos[1]),
                    "uncertainty_radius_km": float({6: 28.0, 12: 52.0, 24: 95.0}.get(h, 50.0))
                }
                result["forecasts"][f"{h}h"] = entry

        else:
            dr = self._dead_reckoning(trk_pts)
            for h in [6, 12, 24]:
                pos = dr[str(h)]["position"].cpu().numpy()[0]
                result["forecasts"][f"{h}h"] = {
                    "lat": float(pos[0]),
                    "lon": float(pos[1]),
                    "uncertainty_radius_km": float(h * 4.5),
                }

        return result

    def _dead_reckoning(self, track_history: np.ndarray) -> Dict[str, Dict]:
        """Fallback linear extrapolation of the last motion vector."""
        hist = track_history
        n = len(hist)
        horizons = self.cfg.track.forecast_horizons if (self.cfg and hasattr(self.cfg, "track")) else [6, 12, 24]
        result = {}
        if n >= 2:
            vel = (hist[-1] - hist[0]) / (n - 1)
        else:
            vel = np.zeros(2)
        for h in horizons:
            steps = h / 6.0
            pos = hist[-1] + vel * steps if n >= 2 else np.zeros(2)
            log_var = np.log(np.array([(h * 3) ** 2, (h * 3) ** 2]))
            result[str(h)] = {
                "position": torch.from_numpy(pos.astype(np.float32)).unsqueeze(0),
                "log_variance": torch.from_numpy(log_var.astype(np.float32)).unsqueeze(0),
            }
        return result

    def full_inference(self, satellite: np.ndarray,
                        satellite_sequence: Optional[np.ndarray] = None,
                        track_history: Optional[np.ndarray] = None,
                        environment: Optional[np.ndarray] = None,
                        wind_kt: Optional[float] = None,
                        current_position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run the complete chained runtime workflow (plan.md Phase 6)."""
        # 1. Quality Control
        qc = self.sat_preprocessor.quality_check(satellite)

        # 2. Cyclone Detection & Center Localization (Phase 5)
        detection = self.detect(satellite)

        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "quality_control": qc,
            "detection": detection,
            "pipeline_status": "COMPLETED",
        }

        # 3. If detected (or forced for analysis), run classification, pattern & forecast
        if detection.get("cyclone_detected", True):
            # Center coordinates
            det_lat = detection["center"]["lat"]
            det_lon = detection["center"]["lon"]
            curr_pos = current_position if current_position is not None else np.array([det_lat, det_lon])

            # Crop or use satellite image for Stage Classification (Phase 2)
            classification = self.classify(satellite, wind_kt=wind_kt)
            result["classification"] = {
                "stage": classification["stage"],
                "stage_idx": classification.get("stage_idx"),
                "confidence": classification.get("confidence", 0.0),
                "probabilities": classification.get("probabilities", {}),
                "structure": classification.get("structure", []),
            }

            # 4. Temporal Intensity Trend & RI Probability (Phase 3)
            temp = self.pattern(
                satellite_sequence=satellite_sequence,
                track_history=track_history,
                current_frame=satellite,
            )
            result["pattern"] = temp
            result["temporal_pattern"] = temp

            # 5. Multimodal Track Forecast (Phase 4)
            if track_history is None:
                # Plausible 36h historical track terminating at current position
                steps = 6
                track_history = np.array([
                    [curr_pos[0] - 0.25 * (steps - 1 - i), curr_pos[1] - 0.20 * (steps - 1 - i)]
                    for i in range(steps)
                ])

            sat_emb = classification.get("embedding")
            forecast = self.predict_track(
                track_history=track_history,
                environment=environment,
                sat_embedding=sat_emb,
                current_position=curr_pos,
            )
            result["track_forecast"] = forecast

        return result

    def provide_feedback(self, forecast: Dict, actual_lat: float, actual_lon: float,
                         horizon_h: int, cyclone_id: str = "LIVE") -> Optional[float]:
        """DPE verification when actual track arrives (plan.md Phase 1 / Phase 4 verification)."""
        fc_entry = forecast.get("forecasts", {}).get(f"{horizon_h}h")
        if fc_entry is None:
            return None
        dpe = direct_positional_error(
            fc_entry["lat"], fc_entry["lon"], actual_lat, actual_lon)
        fc_entry["dpe_km"] = float(round(dpe, 2))
        fc_entry["dpe_target_km"] = {6: 25.0, 12: 50.0, 24: 100.0}.get(horizon_h, 50.0)
        fc_entry["dpe_met"] = bool(dpe <= fc_entry["dpe_target_km"])
        return dpe


class CascadeInferenceEngine(InferenceEngine):
    """Inference engine that loads individual stage models (cascade)."""

    def detect(self, satellite_image: np.ndarray) -> Dict[str, Any]:
        if "detector" not in self.models:
            return super().detect(satellite_image)
        img = self.sat_preprocessor.preprocess_image(satellite_image, "IR")
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            output = self.models["detector"](tensor)
        if isinstance(output, dict) and "has_tc" in output:
            has_tc = output["has_tc"].item()
            bbox = output["bbox"].cpu().numpy()[0]
            center_xy = output["center"].cpu().numpy()[0]
        elif isinstance(output, list):
            # YOLO format: list of detections
            det = output[0]
            has_tc = 1.0 if len(det["boxes"]) else 0.0
            if len(det["boxes"]):
                bbox = det["boxes"][0]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                center_xy = np.array([cx / 512, cy / 512])
            else:
                center_xy = np.array([0.5, 0.5])
                bbox = np.zeros(4)
        else:
            return super().detect(satellite_image)
        lon, lat = self.geo.pixel_to_latlon(center_xy[0] * 512, center_xy[1] * 512)
        return {
            "cyclone_detected": has_tc > 0.5,
            "confidence": float(has_tc),
            "center": {"lat": float(lat), "lon": float(lon)},
            "bbox_normalized": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
        }