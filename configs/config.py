from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import yaml
from pathlib import Path


@dataclass
class SatelliteConfig:
    channels: List[str] = field(default_factory=lambda: ["IR", "WV", "VIS"])
    image_size: Tuple[int, int] = (512, 512)
    insat_resolution: float = 4.0  # km per pixel
    num_temporal_frames: int = 6  # T-15h to T0 in 3h steps
    temporal_interval_hours: int = 3
    quality_threshold: float = 0.7
    cloud_mask_enabled: bool = True


@dataclass
class WeatherConfig:
    era5_variables: List[str] = field(default_factory=lambda: [
        "u10", "v10", "t2m", "msl", "tcwv",
        "u200", "v200", "u850", "v850",
        "z500", "r700", "sst"
    ])
    grid_resolution: float = 0.25  # degrees
    spatial_extent: Tuple[float, float, float, float] = (0, 40, 60, 100)  # NIO bounds


@dataclass
class TrackConfig:
    history_length: int = 6  # number of past positions (every 6h)
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])
    max_track_length: int = 120  # hours


@dataclass
class DetectionConfig:
    model_type: str = "heatmap"  # heatmap (Gaussian center regression), faster_rcnn
    backbone: str = "unet"       # UNet encoder for heatmap regression
    num_classes: int = 2  # TC / No TC
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_channels: int = 3  # multi-channel input
    pretrained: bool = True


@dataclass
class ClassificationConfig:
    # Level 1: Meteorological stage
    stage_classes: List[str] = field(default_factory=lambda: [
        "no_disturbance", "low_pressure", "depression", "deep_depression",
        "cyclonic_storm", "severe_cyclonic_storm", "very_severe_cyclonic_storm",
        "extremely_severe_cyclonic_storm"
    ])
    # Level 2: Structural pattern
    structure_classes: List[str] = field(default_factory=lambda: [
        "developing", "mature", "weakening", "eye_forming", "eye_present",
        "eyewall_organized", "highly_asymmetric", "sheared",
        "rapidly_intensifying", "land_interaction", "dissipating"
    ])
    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class PatternConfig:
    temporal_classes: List[str] = field(default_factory=lambda: [
        "developing", "intensifying", "rapid_intensification", "mature",
        "weakening", "sheared", "recurving", "land_interaction", "dissipating"
    ])
    sequence_length: int = 6
    model_type: str = "convlstm"  # convlstm, transformer
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    multi_label: bool = True


@dataclass
class PredictionConfig:
    model_type: str = "gru"  # gru, transformer
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24])
    predict_intensity: bool = True
    uncertainty_estimation: bool = True
    num_mc_samples: int = 50  # MC Dropout samples


@dataclass
class FusionConfig:
    fusion_type: str = "transformer"  # transformer, concat, attention
    embedding_dim: int = 512
    num_fusion_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    # Multi-task loss weights
    lambda_detection: float = 1.0
    lambda_classification: float = 1.0
    lambda_pattern: float = 1.0
    lambda_track: float = 2.0
    lambda_intensity: float = 1.0
    lambda_uncertainty: float = 0.5
    # Staged training
    stage1_epochs: int = 30  # detection
    stage2_epochs: int = 30  # classification
    stage3_epochs: int = 30  # pattern
    stage4_epochs: int = 40  # track
    stage5_epochs: int = 20  # NWP integration
    stage6_epochs: int = 20  # full fusion
    stage7_epochs: int = 30  # joint fine-tuning


@dataclass
class DataConfig:
    root_dir: str = "./data"
    ibtracs_path: str = "./data/ibtracs_nio.csv"
    era5_path: str = "./data/era5"
    insat_path: str = "./data/insat"
    nwp_path: str = "./data/nwp"
    ocean_path: str = "./data/ocean"
    train_cyclones: List[str] = field(default_factory=list)
    val_cyclones: List[str] = field(default_factory=list)
    test_cyclones: List[str] = field(default_factory=list)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TCConfig:
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TCConfig":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for section_name, section_data in d.items():
            if hasattr(cfg, section_name) and isinstance(section_data, dict):
                section = getattr(cfg, section_name)
                for k, v in section_data.items():
                    if hasattr(section, k):
                        setattr(section, k, v)
            elif hasattr(cfg, section_name):
                setattr(cfg, section_name, section_data)
        return cfg

    def to_yaml(self, path: str):
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            if isinstance(obj, list):
                return [_to_dict(v) for v in obj]
            if isinstance(obj, tuple):
                return [_to_dict(v) for v in obj]
            return obj
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)
