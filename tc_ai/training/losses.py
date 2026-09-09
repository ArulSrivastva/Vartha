"""Loss functions for all tasks (plan section 11)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


def detection_loss(predicted, targets, lambda_box: float = 1.0):
    """Detection loss = BCE (has_tc) + L1 (center + bbox)."""
    loss = 0.0
    if "has_tc" in predicted:
        loss += F.binary_cross_entropy_with_logits(predicted["has_tc"], targets["has_tc"])
    if "center" in predicted and "center" in targets:
        loss += F.mse_loss(predicted["center"], targets["center"])
    if "bbox" in predicted and "bbox" in targets:
        loss += lambda_box * F.mse_loss(predicted["bbox"], targets["bbox"])
    return loss


def classification_loss(logits, targets):
    """Stage (cross entropy) + structure (binary CE for multi-label)."""
    loss = 0.0
    if "stage_logits" in logits and "stage" in targets:
        loss += F.cross_entropy(logits["stage_logits"], targets["stage"])
    if "structure_logits" in logits and "structure" in targets:
        loss += F.binary_cross_entropy_with_logits(logits["structure_logits"], targets["structure"])
    return loss


class OrdinalStageLoss(nn.Module):
    """Ordinal-aware loss for IMD stage classification (plan.md Phase 2).
    
    Penalizes cross-entropy + expected distance between predicted class and true stage.
    """

    def __init__(self, n_classes: int = 8, alpha: float = 0.5, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.register_buffer("class_indices", torch.arange(n_classes, dtype=torch.float32))
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        probs = F.softmax(logits, dim=-1)
        expected_class = torch.sum(probs * self.class_indices.to(logits.device), dim=-1)
        distance_loss = F.l1_loss(expected_class, targets.float())
        return ce_loss + self.alpha * distance_loss


class CoralOrdinalLoss(nn.Module):
    """CORAL (Consistent Rank Logits) ordinal loss with margin penalty.
    
    Guarantees monotonic stage transition probabilities across Depression -> ESCS.
    """

    def __init__(self, n_classes: int = 6, margin: float = 0.15):
        super().__init__()
        self.n_classes = n_classes
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = self.n_classes
        B = targets.size(0)
        binary_targets = torch.zeros(B, K - 1, device=targets.device)
        for k in range(K - 1):
            binary_targets[:, k] = (targets > k).float()

        if logits.shape[-1] == K - 1:
            coral_logits = logits
            probs_gt = torch.sigmoid(coral_logits)
        else:
            probs = F.softmax(logits, dim=-1)
            # P(y > k) = sum_{j=k+1}^{K-1} probs[:, j] = 1 - cumsum(probs)[:, k]
            cum_probs = torch.cumsum(probs, dim=-1)[:, :K - 1]
            probs_gt = (1.0 - cum_probs).clamp(1e-6, 1.0 - 1e-6)
            coral_logits = torch.logit(probs_gt)

        bce = F.binary_cross_entropy_with_logits(coral_logits, binary_targets)
        pred_ranks = probs_gt.sum(dim=-1)
        dist_loss = F.smooth_l1_loss(pred_ranks, targets.float(), beta=0.5)
        return bce + self.margin * dist_loss



def horizon_weighted_directional_track_loss(
    predicted: Dict[str, dict],
    targets: Dict[str, torch.Tensor],
    weights: Optional[Dict[int, float]] = None,
    lambda_dir: float = 0.35,
    lambda_spd: float = 0.15,
) -> torch.Tensor:
    """Horizon-weighted track loss with cosine heading and translation speed penalty.
    
    L_track = sum_{h in {6, 12, 24}} w_h * [ SmoothL1(p_hat_h, p_h) +
                                            lambda_dir * (1 - cos(theta_hat_h - theta_h)) +
                                            lambda_spd * |v_hat_h - v_h| ]
    where w_6 = 1.0, w_12 = 0.85, w_24 = 0.70.
    """
    if weights is None:
        weights = {6: 1.0, 12: 0.85, 24: 0.70}
    device = list(predicted.values())[0]["position"].device
    total_loss = torch.tensor(0.0, device=device)
    total_w = 0.0

    for h_str, pred in predicted.items():
        h = int(h_str)
        w = weights.get(h, 1.0)
        target = targets.get(h) if h in targets else targets.get(h_str)
        if target is None:
            continue
        p_pred = pred["position"]  # (B, 2)
        p_tgt = target             # (B, 2)

        l1 = F.smooth_l1_loss(p_pred, p_tgt, beta=0.1)

        pred_norm = torch.norm(p_pred, dim=-1, keepdim=True) + 1e-6
        tgt_norm = torch.norm(p_tgt, dim=-1, keepdim=True) + 1e-6
        cosine_sim = torch.sum((p_pred / pred_norm) * (p_tgt / tgt_norm), dim=-1)
        dir_loss = (1.0 - cosine_sim).mean()

        spd_loss = F.smooth_l1_loss(pred_norm.squeeze(-1), tgt_norm.squeeze(-1), beta=0.1)

        step_loss = l1 + lambda_dir * dir_loss + lambda_spd * spd_loss
        total_loss = total_loss + w * step_loss
        total_w += w

    return total_loss / max(total_w, 1e-5)


def pattern_loss(logits, targets):
    """Multi-label binary cross entropy for pattern recognition."""
    if "pattern" in logits and "pattern" in targets:
        return F.binary_cross_entropy_with_logits(logits["pattern"], targets["pattern"])
    return torch.tensor(0.0, device=logits.get("pattern", torch.tensor(0.0)).device)


def gaussian_nll_loss(mean, log_var, target):
    """Gaussian NLL loss with predicted heteroscedastic uncertainty."""
    var = torch.exp(log_var)
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var).mean()


def track_loss(predicted, targets, use_uncertainty: bool = True):
    """Track position loss with optional uncertainty-aware NLL."""
    loss = 0.0
    device = list(predicted.values())[0]["position"].device
    n_h = len(predicted)
    for h, pred in predicted.items():
        target = targets.get(int(h)) if isinstance(int(h), int) else targets.get(h)
        if target is None and str(h) in targets:
            target = targets[str(h)]
        if target is None:
            continue
        if isinstance(target, torch.Tensor) and target.dim() > 0:
            pass
        else:
            target = torch.tensor([target], device=device).float()

        if use_uncertainty and "log_variance" in pred:
            loss += gaussian_nll_loss(pred["position"], pred["log_variance"], target)
        else:
            loss += F.mse_loss(pred["position"], target)
    if n_h == 0:
        return torch.tensor(0.0, device=device)
    return loss / n_h


def kinematic_track_loss(predicted: Dict[str, dict], targets: Dict[str, torch.Tensor],
                         lambda_turn: float = 0.5, theta_max_deg: float = 60.0,
                         use_uncertainty: bool = True) -> torch.Tensor:
    """Kinematic track loss combining Smooth L1 / Gaussian NLL with curvature turn penalty.

    Penalizes non-physical trajectory turns (> theta_max_deg across 6h intervals) to enforce momentum.
    """
    device = list(predicted.values())[0]["position"].device
    theta_max_rad = math.radians(theta_max_deg)

    # 1. Base positional loss (Smooth L1 + optional Gaussian NLL)
    base_loss = 0.0
    horizons = sorted([int(h) for h in predicted.keys()])

    for h in horizons:
        pred = predicted[str(h)]
        target = targets.get(h) if h in targets else targets.get(str(h))
        if target is None:
            continue
        pos_pred = pred["position"]
        huber = F.smooth_l1_loss(pos_pred, target, beta=0.1)
        if use_uncertainty and "log_variance" in pred:
            nll = gaussian_nll_loss(pos_pred, pred["log_variance"], target)
            base_loss += 0.5 * huber + 0.5 * nll
        else:
            base_loss += huber

    base_loss = base_loss / max(len(horizons), 1)

    # 2. Kinematic turn angle / curvature penalty across predicted steps
    turn_loss = torch.tensor(0.0, device=device)
    if all(str(h) in predicted for h in [6, 12, 24]):
        p1 = predicted["6"]["position"]    # (B, 2)
        p2 = predicted["12"]["position"]   # (B, 2)
        p3 = predicted["24"]["position"]   # (B, 2)

        # Step vectors
        v1 = p1                            # step 0 -> 6h
        v2 = p2 - p1                       # step 6h -> 12h
        v3 = p3 - p2                       # step 12h -> 24h

        # Heading angles (atan2)
        theta1 = torch.atan2(v1[:, 1], v1[:, 0] + 1e-6)
        theta2 = torch.atan2(v2[:, 1], v2[:, 0] + 1e-6)
        theta3 = torch.atan2(v3[:, 1], v3[:, 0] + 1e-6)

        # Angular differences wrapped to [-pi, pi]
        d_theta_12 = torch.atan2(torch.sin(theta2 - theta1), torch.cos(theta2 - theta1))
        d_theta_23 = torch.atan2(torch.sin(theta3 - theta2), torch.cos(theta3 - theta2))

        # Penalty for turning more than theta_max
        pen_12 = F.relu(torch.abs(d_theta_12) - theta_max_rad)
        pen_23 = F.relu(torch.abs(d_theta_23) - theta_max_rad)

        turn_loss = (pen_12.mean() + pen_23.mean()) * 0.5

    return base_loss + lambda_turn * turn_loss


def intensity_loss(predicted, targets):
    """Intensity loss (wind + pressure per horizon)."""
    loss = 0.0
    count = 0
    device = None
    for h, pred in predicted.items():
        if "intensity" not in pred:
            continue
        device = pred["intensity"].device
        target = targets.get(int(h)) if isinstance(int(h), int) else targets.get(h)
        if target is None and str(h) in targets:
            target = targets[str(h)]
        if target is None:
            continue
        if "intensity" in target:
            target = target["intensity"]
        loss += F.mse_loss(pred["intensity"], target.float())
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss / count


def regression_nll_loss(pred_mean, pred_logvar, target):
    """Negative log likelihood for regression with uncertainty."""
    var = torch.exp(pred_logvar)
    return 0.5 * (torch.log(var) + (target - pred_mean) ** 2 / var).mean()


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss (plan section 11):

    L = λ1 L_detection + λ2 L_classification + λ3 L_pattern
        + λ4 L_track + λ5 L_intensity + λ6 L_uncertainty
    """

    def __init__(self, lambda_detection=1.0, lambda_classification=1.0,
                 lambda_pattern=1.0, lambda_track=2.0, lambda_intensity=1.0,
                 lambda_uncertainty=0.5, use_uncertainty_loss=True):
        super().__init__()
        self.lambda_detection = lambda_detection
        self.lambda_classification = lambda_classification
        self.lambda_pattern = lambda_pattern
        self.lambda_track = lambda_track
        self.lambda_intensity = lambda_intensity
        self.lambda_uncertainty = lambda_uncertainty
        self.use_uncertainty_loss = use_uncertainty_loss

    def forward(self, outputs, targets) -> Dict[str, torch.Tensor]:
        """outputs: dict from MultiTaskModel. targets: dict per-task."""
        losses = {}

        if "detection" in outputs:
            det_targets = {
                k: targets[k] for k in ("has_tc", "center", "bbox")
                if k in targets
            }
            if det_targets:
                losses["detection"] = detection_loss(outputs["detection"], det_targets)

        if "classification" in outputs:
            cls_targets = {
                k: targets[k] for k in ("stage", "structure")
                if k in targets
            }
            if cls_targets:
                losses["classification"] = classification_loss(
                    outputs["classification"], cls_targets)

        if "pattern" in outputs and "pattern" in targets:
            losses["pattern"] = pattern_loss({"pattern": outputs["pattern"]}, targets)

        if "prediction" in outputs:
            pred = outputs["prediction"]
            pos_targets = {k: targets[k] for k in targets
                           if k in [str(h) for h in targets] or isinstance(k, int)}
            # Only position targets
            position_targets = {}
            for h in pred:
                if h in targets:
                    position_targets[h] = targets[h]
            if position_targets:
                if self.use_uncertainty_loss and all(
                        "log_variance" in pred[h] for h in position_targets):
                    losses["track"] = track_loss(pred, position_targets,
                                                 use_uncertainty=True)
                else:
                    losses["track"] = track_loss(pred, position_targets,
                                                 use_uncertainty=False)
            if "intensity_targets" in targets and targets["intensity_targets"]:
                losses["intensity"] = intensity_loss(pred, targets["intensity_targets"])

        # Weighted sum
        total = torch.tensor(0.0, device=(list(outputs.values())[0] if isinstance(list(outputs.values())[0], torch.Tensor)
                                          else next(iter(outputs.values())).get("has_tc", torch.zeros(1)).device))
        weight_map = {
            "detection": self.lambda_detection,
            "classification": self.lambda_classification,
            "pattern": self.lambda_pattern,
            "track": self.lambda_track,
            "intensity": self.lambda_intensity,
        }
        for name, loss in losses.items():
            if name in weight_map:
                total = total + weight_map[name] * loss
            else:
                total = total + loss
        losses["total"] = total
        return losses


def centernet_focal_loss(pred_logits: torch.Tensor, target_heatmap: torch.Tensor,
                         alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """Penalty-reduced pixelwise logistic regression with focal loss (CenterNet / CornerNet).

    pred_logits: (N, C, H, W) raw unnormalized heatmap logits
    target_heatmap: (N, C, H, W) ground truth Gaussian heatmap [0, 1]
    """
    pred_prob = torch.sigmoid(pred_logits).clamp(min=eps, max=1.0 - eps)
    pos_mask = target_heatmap.ge(0.99)
    neg_mask = target_heatmap.lt(0.99)

    pos_loss = torch.log(pred_prob) * torch.pow(1.0 - pred_prob, alpha) * pos_mask.float()
    neg_loss = torch.log(1.0 - pred_prob) * torch.pow(pred_prob, alpha) * torch.pow(1.0 - target_heatmap, beta) * neg_mask.float()

    num_pos = pos_mask.float().sum()
    if num_pos > 0:
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return -neg_loss.mean()



def ri_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for severe class imbalance in Rapid Intensification binary classification."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_t * torch.pow(1.0 - p_t, gamma)
    return (focal_weight * bce).mean()