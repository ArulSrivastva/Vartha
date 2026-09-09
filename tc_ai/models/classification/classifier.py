"""Module 2: Tropical Cyclone Classification.

Level 1 - Meteorological stage (IMD classification).
Level 2 - Structural pattern (multi-label).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class ClassificationHead(nn.Module):
    """Multi-task classification head for stage + structural pattern."""

    def __init__(self, in_features: int, n_stage_classes: int = 8,
                 n_structure_classes: int = 11, dropout: float = 0.3):
        super().__init__()
        self.stage_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_stage_classes),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_structure_classes),
        )

    def forward(self, x, return_features: bool = False):
        stage_logits = self.stage_head(x)
        structure_logits = self.structure_head(x)
        if return_features:
            return stage_logits, structure_logits, x
        return stage_logits, structure_logits


class CycloneClassifier(nn.Module):
    """Full image classifier with shared CNN backbone.

    Supports EffNet, ResNet, or ViT backbones.
    """

    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True,
                 n_stage_classes: int = 8, n_structure_classes: int = 11,
                 input_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=input_channels
        )
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            self.backbone.eval()
            feat_dim = self.backbone(dummy).shape[-1]
            self.backbone.train()

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = ClassificationHead(512, n_stage_classes, n_structure_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.fc(feat)
        return self.classifier(feat, return_features=True)


class StageClassifier(nn.Module):
    """Satellite-based stage classifier (Phase 2).
    
    Inputs: Cropped INSAT IR + WV channels (C, H, W).
    Outputs: IMD operational stage logits (n_classes) + feature embedding for Phase 4 fusion.
    """

    def __init__(self, backbone: str = "resnet18", n_classes: int = 8,
                 pretrained: bool = False, input_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        import timm
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
                in_chans=input_channels,
            )
        except Exception:
            self.backbone = timm.create_model(
                backbone, pretrained=False, num_classes=0,
                in_chans=input_channels,
            )
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, input_channels, 128, 128)
            feat_dim = self.backbone(dummy).shape[-1]
            self.backbone.train()

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embedding: bool = False):
        feat = self.backbone(x)
        emb = self.proj(feat)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class StructureClassifier(nn.Module):
    """Structural pattern (multi-label) classifier (Level 2)."""

    def __init__(self, backbone: str = "efficientnet_b4", n_classes: int = 11,
                 pretrained: bool = True, input_channels: int = 3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=input_channels
        )
        with torch.no_grad():
            self.backbone.eval()
            feat_dim = self.backbone(torch.zeros(1, input_channels, 224, 224)).shape[-1]
            self.backbone.train()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return torch.sigmoid(self.head(feat))
