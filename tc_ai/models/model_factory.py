"""Model factory: builds all models from config."""
import torch
from ..utils.common import get_device
from ..utils.cyclone import (
    get_imd_stage_class, get_imd_stage_index,
    classify_structure, get_pattern_multilabel,
)


def build_detector(cfg):
    """Build cyclone detection model (Module 1)."""
    from .detection.detector import (
        YOLODetector, FasterRCNNDetector, VisionTransformerDetector, DetectionHead,
    )
    model_type = cfg.detection.model_type
    if model_type == "yolov8":
        return YOLODetector(
            model_name=cfg.detection.backbone,
            num_classes=cfg.detection.num_classes,
            input_channels=cfg.detection.input_channels,
            conf_threshold=cfg.detection.confidence_threshold,
            pretrained=cfg.detection.pretrained,
        )
    elif model_type == "faster_rcnn":
        return FasterRCNNDetector(
            num_classes=cfg.detection.num_classes,
            pretrained=cfg.detection.pretrained,
        )
    elif model_type == "vit":
        return VisionTransformerDetector(
            img_size=cfg.satellite.image_size,
            in_channels=cfg.detection.input_channels,
            num_classes=cfg.detection.num_classes,
        )
    else:
        raise ValueError(f"Unknown detection model: {model_type}")


def build_classifier(cfg):
    """Build cyclone classifier (Module 2)."""
    from .classification.classifier import (
        CycloneClassifier, StageClassifier, StructureClassifier,
    )
    return CycloneClassifier(
        backbone=cfg.classification.backbone,
        pretrained=cfg.classification.pretrained,
        n_stage_classes=len(cfg.classification.stage_classes),
        n_structure_classes=len(cfg.classification.structure_classes),
        input_channels=cfg.detection.input_channels,
        dropout=cfg.classification.dropout,
    )


def build_pattern_model(cfg):
    """Build temporal pattern model (Module 3)."""
    from .pattern.temporal_model import TemporalPatternModel, TemporalVisionTransformer
    if cfg.pattern.model_type == "convlstm":
        return TemporalPatternModel(
            input_channels=cfg.detection.input_channels,
            hidden_dim=cfg.pattern.hidden_dim,
            num_layers=cfg.pattern.num_layers,
            n_pattern_classes=len(cfg.pattern.temporal_classes),
            sequence_length=cfg.pattern.sequence_length,
            image_size=cfg.satellite.image_size[0] // 8,
        )
    elif cfg.pattern.model_type == "transformer":
        return TemporalVisionTransformer(
            input_channels=cfg.detection.input_channels,
            seq_len=cfg.pattern.sequence_length,
            image_size=cfg.satellite.image_size[0],
            embed_dim=cfg.pattern.hidden_dim,
            depth=cfg.pattern.num_layers,
            num_heads=cfg.pattern.num_heads,
            num_classes=len(cfg.pattern.temporal_classes),
        )
    else:
        raise ValueError(f"Unknown pattern model: {cfg.pattern.model_type}")


def build_track_model(cfg, env_dim: int = 64):
    """Build track prediction model (Module 4)."""
    from .prediction.track_predictor import (
        GRUTrackModel, TransformerTrackModel, TrackPredictionModel, NWPResidualTrackModel,
    )
    horizons = cfg.prediction.forecast_horizons
    if cfg.prediction.model_type == "gru":
        return GRUTrackModel(
            hidden_dim=cfg.prediction.hidden_dim,
            num_layers=cfg.prediction.num_layers,
            horizons=horizons,
            predict_intensity=cfg.prediction.predict_intensity,
            uncertainty_estimation=cfg.prediction.uncertainty_estimation,
        )
    elif cfg.prediction.model_type == "transformer":
        return TransformerTrackModel(
            hidden_dim=cfg.prediction.hidden_dim,
            num_layers=cfg.prediction.num_layers,
            num_heads=cfg.prediction.num_heads,
            horizons=horizons,
            predict_intensity=cfg.prediction.predict_intensity,
            uncertainty_estimation=cfg.prediction.uncertainty_estimation,
        )
    else:
        raise ValueError(f"Unknown track model: {cfg.prediction.model_type}")


def build_multitask_model(cfg, env_dim: int = 64, temporal_satellite: bool = False):
    """Build the final multi-task multimodal model."""
    from .fusion.multimodal import MultiTaskModel
    return MultiTaskModel(
        satellite_channels=cfg.detection.input_channels,
        n_stage_classes=len(cfg.classification.stage_classes),
        n_structure_classes=len(cfg.classification.structure_classes),
        n_pattern_classes=len(cfg.pattern.temporal_classes),
        horizons=cfg.prediction.forecast_horizons,
        weather_dim=env_dim,
        track_history_len=cfg.track.history_length,
        embed_dim=cfg.fusion.embedding_dim,
        image_size=cfg.satellite.image_size[0] // 4,
        satellite_backbone=cfg.classification.backbone,
        predict_intensity=cfg.prediction.predict_intensity,
        uncertainty_estimation=cfg.prediction.uncertainty_estimation,
        fusion_layers=cfg.fusion.num_fusion_layers,
        fusion_heads=cfg.fusion.num_heads,
        temporal_satellite=temporal_satellite,
    )