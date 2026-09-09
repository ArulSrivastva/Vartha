"""Module 1: Tropical Cyclone Identification (detection) models.

Models:
  - YOLOv8 (fast, real-time)
  - Faster R-CNN (accuracy)
  - Vision Transformer (ViT-based detection head)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class YOLODetector(nn.Module):
    """Wrapper around YOLOv8 for cyclone detection."""

    def __init__(self, model_name: str = "yolov8m", num_classes: int = 2,
                 input_channels: int = 3, conf_threshold: float = 0.5,
                 pretrained: bool = True):
        super().__init__()
        from ultralytics import YOLO
        self.conf_threshold = conf_threshold
        self.model = YOLO(f"{model_name}.pt" if pretrained else f"{model_name}.yaml")
        if input_channels != 3:
            # Adjust first conv layer for multi-channel input
            self._adjust_input_channels(input_channels)
        self.num_classes = num_classes

    def _adjust_input_channels(self, n_channels: int):
        """Replace first conv layer to accept n_channels input."""
        # YOLOv8 uses different config structures; handle common case
        model = self.model.model
        first_conv = model[0]
        if isinstance(first_conv, nn.Conv2d):
            new_conv = nn.Conv2d(n_channels, first_conv.out_channels,
                                  first_conv.kernel_size, first_conv.stride,
                                  first_conv.padding, first_conv.bias)
            # Average the first 3 channels' weights to initialize
            with torch.no_grad():
                new_conv.weight.copy_(
                    first_conv.weight.mean(dim=1, keepdim=True).expand(-1, n_channels, -1, -1)
                    if n_channels >= 3 else first_conv.weight[:, :n_channels]
                )
                new_conv.bias.copy_(first_conv.bias)
            model[0] = new_conv

    def forward(self, images: torch.Tensor, targets: Optional[dict] = None):
        """Run YOLO. images: (N, C, H, W) 0-1 normalized."""
        results = self.model.predict(
            images, conf=self.conf_threshold, verbose=False
        )
        detections = []
        for r in results:
            detections.append({
                "boxes": r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4)),
                "scores": r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty(0),
                "labels": r.boxes.cls.cpu().numpy() if r.boxes is not None else np.empty(0),
            })
        return detections

    def train_step(self, images, labels_boxes, labels_cls, imgsz=512):
        """Training-specific forward pass."""
        loss, results = self.model.train(
            data=None,  # handled externally
            imgsz=imgsz,
            model=self.model,
            verbose=False,
        )
        # NOTE: For full YOLO training, training data must be in YOLO format.
        # This method is a simplified placeholder. See scripts/train.py for correct workflow.
        raise NotImplementedError(
            "YOLO training requires dataset in YOLO format. "
            "Use ultralytics API directly or export dataset to YOLO format."
        )


class FasterRCNNDetector(nn.Module):
    """Faster R-CNN with ResNet/FasterRCNN backbone for cyclone detection."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("Targets required during training")
            return self.model(images, targets)
        return self.model(images)


class DetectionHead(nn.Module):
    """Simple detection head producing center + bbox from feature maps."""

    def __init__(self, in_channels: int, num_classes: int = 2,
                 image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        # Center heatmap head
        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )
        # Bbox head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * num_classes, 1),
        )

    def forward(self, features: torch.Tensor):
        heatmap = self.center_head(features)
        bbox = self.bbox_head(features)
        return heatmap, bbox

    def decode(self, heatmap: torch.Tensor, bbox: torch.Tensor,
               conf_threshold: float = 0.5) -> List[dict]:
        """Decode heatmap + bbox into detections."""
        results = []
        batch_size = heatmap.shape[0]
        for b in range(batch_size):
            detections = {"boxes": [], "scores": [], "labels": []}
            for c in range(self.num_classes):
                hmap = torch.sigmoid(heatmap[b, c])
                peak_mask = (hmap > conf_threshold).nonzero()
                for y, x in peak_mask:
                    x, y = x.item(), y.item()
                    bx = bbox[b, c*4:(c+1)*4, y, x].tolist()
                    # bbox relative: (x1, y1, x2, y2)
                    x1, y1, x2, y2 = bx
                    detections["boxes"].append([x1, y1, x2, y2])
                    detections["scores"].append(hmap[y, x].item())
                    detections["labels"].append(c)
            detections["boxes"] = np.array(detections["boxes"]) if detections["boxes"] else np.empty((0, 4))
            detections["scores"] = np.array(detections["scores"]) if detections["scores"] else np.empty(0)
            detections["labels"] = np.array(detections["labels"]) if detections["labels"] else np.empty(0)
            results.append(detections)
        return results


class VisionTransformerDetector(nn.Module):
    """A Vision Transformer for cyclone detection (patch + classification/regression)."""

    def __init__(self, img_size: Tuple[int, int] = (512, 512),
                 patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # no final classifier
            in_chans=in_channels,
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.embed_dim = self.backbone.embed_dim

        # Task heads
        self.has_tc_head = nn.Linear(self.embed_dim, 1)          # TC/No TC
        self.center_head = nn.Linear(self.embed_dim, 2)          # center (x, y) normalized
        self.bbox_head = nn.Linear(self.embed_dim, 4)            # x1, y1, x2, y2 normalized

    def forward(self, images: torch.Tensor):
        feat = self.backbone.forward_features(images)  # (N, D) after pool
        has_tc = torch.sigmoid(self.has_tc_head(feat))
        center = torch.sigmoid(self.center_head(feat))
        bbox = torch.sigmoid(self.bbox_head(feat))
        return {
            "has_tc": has_tc,
            "center": center,
            "bbox": bbox,
        }


def generate_gaussian_heatmap(center: Tuple[float, float], size: Tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """Generate 2D Gaussian heatmap centered on (cx, cy) normalized coordinates [0, 1]."""
    h, w = size
    hmap = np.zeros((h, w), dtype=np.float32)
    cx = int(round(center[0] * (w - 1)))
    cy = int(round(center[1] * (h - 1)))
    radius = int(math.ceil(3.0 * sigma))

    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)

    if x1 > x0 and y1 > y0:
        y, x = np.ogrid[y0:y1, x0:x1]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        patch = np.exp(-dist_sq / (2.0 * sigma ** 2))
        hmap[y0:y1, x0:x1] = patch.astype(np.float32)
    return hmap



def extract_heatmap_center(heatmap: np.ndarray, threshold: float = 0.05) -> Tuple[float, float, float]:
    """Extract (cx, cy, peak_conf) normalized coordinates [0, 1] from a 2D heatmap."""
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(0)
    h, w = heatmap.shape
    # Mask out outer margin to prevent deconvolution boundary padding artifacts
    pad = max(2, min(h, w) // 16)
    inner = heatmap.copy()
    inner[:pad, :] = 0.0
    inner[-pad:, :] = 0.0
    inner[:, :pad] = 0.0
    inner[:, -pad:] = 0.0

    idx = np.argmax(inner)
    cy, cx = np.unravel_index(idx, (h, w))
    conf = float(inner[cy, cx])
    # If peak is too weak / flat (uninitialized or empty), default to center (0.5, 0.5)
    if conf < threshold:
        return 0.5, 0.5, float(heatmap.max())
    return float(cx / max(w - 1, 1)), float(cy / max(h - 1, 1)), conf


class SpatialSoftArgmax2d(nn.Module):
    """Differentiable spatial soft-argmax coordinate extraction."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        b, c, h, w = x.shape
        flat = x.view(b, c, -1) / max(self.temperature, 1e-4)
        weights = F.softmax(flat, dim=-1).view(b, c, h, w)

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        coord_x = torch.sum(weights * pos_x.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
        coord_y = torch.sum(weights * pos_y.unsqueeze(0).unsqueeze(0), dim=(-2, -1))
        return torch.cat([coord_x, coord_y], dim=-1)  # (B, 2) normalized [0, 1]


class HeatmapCenterDetector(nn.Module):
    """Phase 5 Center-Point Heatmap Regression Detector (plan.md Section 7).

    Takes a basin-wide satellite frame and predicts:
      1. has_tc: cyclone presence probability (binary classification)
      2. heatmap: Gaussian-peaked heatmap centered on true storm position
      3. center: Differentiable sub-pixel center coordinate prediction via spatial soft-argmax
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Presence classification head
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Heatmap decoder (upsampling back to /2 or full resolution)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_out = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

        self.soft_argmax = SpatialSoftArgmax2d(temperature=0.5)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        presence_logits = self.presence_head(e3)

        d2 = self.dec2(e3) + e2
        d1 = self.dec1(d2) + e1
        heatmap = self.heatmap_out(d1)

        center = self.soft_argmax(heatmap)

        return {
            "presence_logits": presence_logits,
            "heatmap": heatmap,
            "center": center,
        }

