"""Data preprocessing: quality control, cloud masking, geo-location, normalization, resampling."""
import numpy as np
import cv2
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy import ndimage
from datetime import datetime, timedelta


class SatellitePreprocessor:
    """Preprocessing and quality control for satellite imagery."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True, cloud_mask: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.cloud_mask = cloud_mask

    def preprocess_image(self, image: np.ndarray, channel: str = "IR",
                         apply_geo_correction: bool = True) -> np.ndarray:
        """Full preprocessing pipeline for a single image."""
        img = image.copy()

        # Convert to float
        img = img.astype(np.float32)

        # Radiometric calibration (channel-specific)
        img = self._calibrate(img, channel)

        # Resize
        img = self._resize(img)

        # Cloud masking
        if self.cloud_mask:
            img = self._mask_clouds(img, channel)

        # Normalize to [0, 1]
        if self.normalize:
            img = self._normalize(img)

        return img

    def _calibrate(self, img: np.ndarray, channel: str) -> np.ndarray:
        """Apply channel-specific calibration."""
        if channel == "IR":
            # Convert brightness temperature from counts (common INSAT scaling)
            # This is a placeholder; actual calibration depends on the product
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        elif channel == "WV":
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        elif channel == "VIS":
            img = img / 255.0 if img.max() > 2.0 else img
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        return img

    def _mask_clouds(self, img: np.ndarray, channel: str) -> np.ndarray:
        """Simple cloud masking based on brightness temperature thresholds."""
        if channel == "IR":
            # Deep convection (cold, high clouds) shows low brightness temperature
            # Cloud mask: temperature below ~235K indicates clouds
            # This is a placeholder - real implementation uses calibrated brightness temp
            return img
        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img -= img.mean()
        img /= (img.std() + 1e-6)
        return img

    def quality_check(self, image: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Compute quality metrics for an image. Returns dict with scores."""
        if image.size == 0:
            return {"valid": 0.0, "coverage": 0.0, "noise": 1.0}
        img = image
        if img.ndim == 3:
            img = img.mean(axis=0) if img.shape[0] < img.shape[-1] else img.mean(axis=-1)
        valid_fraction = np.mean(~np.isnan(img))
        finite = np.nan_to_num(img.astype(np.float32), nan=0.0)
        if np.percentile(finite, 5) == np.amax(finite):
            return {"valid": float(valid_fraction), "coverage": 0.0, "noise": 0.0}
        coverage = np.mean(finite > np.percentile(finite, 5))
        # Simple noise proxy: fraction of pixels different from local mean
        blurred = cv2.GaussianBlur(finite, (5, 5), 0)
        noise = np.mean(np.abs(finite - blurred))
        return {
            "valid": float(valid_fraction),
            "coverage": float(coverage),
            "noise": float(noise),
        }

    def stack_channels(self, images: List[np.ndarray]) -> np.ndarray:
        """Stack multiple channels into a multi-channel array (C, H, W)."""
        return np.stack(images, axis=0)


class GeoProcessor:
    """Geo-location and coordinate transformations."""

    def __init__(self, extent: Tuple[float, float, float, float] = (0, 40, 60, 100),
                 grid_size: Tuple[int, int] = (512, 512)):
        """extent: (lat_min, lat_max, lon_min, lon_max)"""
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = extent
        self.grid_size = grid_size
        self._build_grid()

    def _build_grid(self):
        lat_step = (self.lat_max - self.lat_min) / self.grid_size[0]
        lon_step = (self.lon_max - self.lon_min) / self.grid_size[1]
        self.lats = np.linspace(self.lat_min + lat_step/2,
                                self.lat_max - lat_step/2, self.grid_size[0])
        self.lons = np.linspace(self.lon_min + lon_step/2,
                                self.lon_max - lon_step/2, self.grid_size[1])
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

    def pixel_to_latlon(self, x, y) -> Tuple[float, float]:
        """Convert pixel coordinates (x, y) to (lat, lon)."""
        lat = self.lat_min + (y / self.grid_size[0]) * (self.lat_max - self.lat_min)
        lon = self.lon_min + (x / self.grid_size[1]) * (self.lon_max - self.lon_min)
        return lat, lon

    def latlon_to_pixel(self, lat, lon) -> Tuple[float, float]:
        """Convert (lat, lon) to pixel coordinates (x, y)."""
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.grid_size[1]
        y = (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.grid_size[0]
        return x, y

    def extract_region(self, data: np.ndarray, lat, lon, window_km: float = 300.0):
        """Extract a square region centered at (lat, lon) with half-width window_km.

        Data assumed on lat/lon grid defined by self.lats/self.lons.
        """
        if data.ndim == 3:
            n_channels, h, w = data.shape
            res_h = h / self.grid_size[0]
            res_w = w / self.grid_size[1]
        else:
            h, w = data.shape
            res_h = h / self.grid_size[0]
            res_w = w / self.grid_size[1]

        cy, cx = self.latlon_to_pixel(lat, lon)
        cy *= res_h
        cx *= res_w

        # Convert km window to pixels
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * np.cos(np.radians(lat))
        deg_half = window_km / km_per_deg_lat
        px_half_y = deg_half * res_h / (self.lat_max - self.lat_min) * self.grid_size[0]
        px_half_x = window_km / km_per_deg_lon * res_w / (self.lon_max - self.lon_min) * self.grid_size[1]

        y0 = max(0, int(cy - px_half_y))
        y1 = min(h, int(cy + px_half_y))
        x0 = max(0, int(cx - px_half_x))
        x1 = min(w, int(cx + px_half_x))
        return data[..., y0:y1, x0:x1]


class WeatherPreprocessor:
    """Preprocessing for ERA5/weather fields."""

    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        self.target_size = target_size

    def extract_storm_environment(self, fields: Dict[str, np.ndarray], lat: float, lon: float,
                                  radius_km: float = 500.0) -> Dict[str, float]:
        """Extract storm-relative environmental features.

        Computes mean/centroid metrics of each variable within a radius around storm center.
        """
        features = {}
        lats = fields.get("lat", np.arange(0, 40, 0.25))
        lons = fields.get("lon", np.arange(50, 100, 0.25))

        # For simplicity assume 2D field layout (lat, lon)
        for var, field in fields.items():
            if var in ("lat", "lon"):
                continue
            if field.ndim == 3:
                field = field[0]  # take first level/time
            # Select region near storm
            lat_mask = np.abs(lats - lat) <= radius_km / 110.574
            lon_mask = np.abs(lons - lon) <= radius_km / (111.320 * np.cos(np.radians(lat)))
            region = field[np.ix_(lat_mask, lon_mask)]
            features[f"{var}_mean"] = float(np.nanmean(region))
            features[f"{var}_max"] = float(np.nanmax(region))
            features[f"{var}_min"] = float(np.nanmin(region))
        return features


class TimeAligner:
    """Temporal alignment of multi-source data."""

    def __init__(self, interval_hours: int = 3):
        self.interval = timedelta(hours=interval_hours)

    def align_timestamps(self, data: Dict[datetime, np.ndarray],
                         target: List[datetime]) -> Dict[datetime, np.ndarray]:
        """Align data to target timestamps (choose nearest within interval)."""
        aligned = {}
        for t in target:
            best_t = min(data.keys(), key=lambda k: abs((t - k).total_seconds()))
            if abs((t - best_t).total_seconds()) <= (self.interval.total_seconds() / 2):
                aligned[t] = data[best_t]
            else:
                aligned[t] = None
        return aligned

    def build_temporal_sequence(self, data: List[np.ndarray], sequence_length: int) -> np.ndarray:
        """Build a sliding-window temporal sequence from sorted data list."""
        if len(data) < sequence_length:
            return None
        return np.stack(data[-sequence_length:], axis=0)
