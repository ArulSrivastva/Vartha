"""Module 7: Structural Cloud-Pattern Recognition (additive phase).

Closes the Phase-6 audit gap on "different tropical cyclone patterns": the
system identifies the *structural* cloud organization of a cyclone (eye
formation, organized eyewall, shear-induced asymmetry, spiral banding) from
storm-centered satellite imagery, complementing the intensity/stage patterns
already served by Module 3.

Structural multi-label vocabulary exactly mirrors
`configs/config.yaml -> classification.structure_classes`:

    developing, mature, weakening, eye_forming, eye_present,
    eyewall_organized, highly_asymmetric, sheared,
    rapidly_intensifying, land_interaction, dissipating

Input : (B, 3, H, W) storm-centered satellite crop (channels x H x W)
Output:
    - structure_logits : (B, K) multi-label logits (K = 11)
    - eye_idx          : (B, 1) inner-core vs eyewall-ring contrast proxy
    - asym_idx         : (B, 1) shear/arc-coverage asymmetry proxy in [0, 1]
    - org_idx          : (B, 1) spiral-band azimuthal continuity proxy in [0, 1]
    - embedding        : (B, E) learnt structural feature embedding
"""
import torch
import torch.nn as nn
from typing import List

STRUCTURE_LABELS: List[str] = [
    "developing",
    "mature",
    "weakening",
    "eye_forming",
    "eye_present",
    "eyewall_organized",
    "highly_asymmetric",
    "sheared",
    "rapidly_intensifying",
    "land_interaction",
    "dissipating",
]


class StructuralPatternModel(nn.Module):
    """Lightweight CNN backbone + structural multi-label and diagnostic heads.

    The three regression heads predict the same geometric proxies that are
    used to *derive* the weak training labels (see tc_ai/data/structural_labels.py).
    This gives a consistent, physically interpretable joint output and a
    self-consistency signal during training.
    """

    def __init__(self, input_channels: int = 3, n_structure_classes: int = 11,
                 embed_dim: int = 128, image_size: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.n_structure_classes = n_structure_classes
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Storm-centered feature encoder (strided conv -> global average pool)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),   # S/2
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # S/4
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),             # S/8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Multi-label structural head
        self.structure_head = nn.Linear(embed_dim, n_structure_classes)
        # Diagnostic regression heads (weak-label proxies)
        self.eye_head = nn.Linear(embed_dim, 1)
        self.asym_head = nn.Linear(embed_dim, 1)
        self.org_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)  # (B, embed_dim)
        return {
            "structure_logits": self.structure_head(emb),
            "eye_idx": self.eye_head(emb),
            "asym_idx": torch.sigmoid(self.asym_head(emb)),
            "org_idx": torch.sigmoid(self.org_head(emb)),
            "embedding": emb,
        }

    @torch.no_grad()
    def predict_structure_probs(self, x: torch.Tensor):
        """Return per-label sigmoid probabilities for a batch."""
        out = self.forward(x)
        probs = torch.sigmoid(out["structure_logits"]).cpu().numpy()
        return probs