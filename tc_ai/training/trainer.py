"""Training machinery: data collator, schedulers, staged training pipeline."""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
import time
from typing import Optional, Dict, List, Callable
from pathlib import Path

from ..utils.logger import ExperimentLogger
from ..utils.common import set_seed, get_device


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    optimizer_type: str = "adamw"):
    if optimizer_type == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)


def build_scheduler(optimizer, scheduler_type: str, num_epochs: int,
                    warmup_epochs: int = 5):
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1))
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=max(num_epochs // 3, 1), gamma=0.1)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    else:
        return CosineAnnealingLR(optimizer, T_max=num_epochs)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device,
                    compile_metrics: Optional[Callable] = None,
                    grad_clip: float = 1.0, grad_scaler=None,
                    tag: str = "train"):
    """Standard single-epoch training loop. Returns dict of mean losses."""
    model.train()
    losses = {}
    n_batches = 0
    for batch in dataloader:
        batch = to_device(batch, device)
        optimizer.zero_grad()

        inputs, targets = split_inputs_targets(batch, tag)
        with torch.amp.autocast(device_type=device.type, enabled=grad_scaler is not None):
            outputs = model(**inputs)
            task_losses = loss_fn(outputs, targets)

        total = task_losses["total"]
        if grad_scaler is not None:
            grad_scaler.scale(total).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in task_losses.items():
            losses[k] = losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses.items()}


def validate(model, dataloader, loss_fn, device, tag: str = "val",
             compute_metrics: Optional[Callable] = None):
    """Validation loop. Returns dict of mean losses and metrics."""
    model.eval()
    losses = {}
    n_batches = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            inputs, targets = split_inputs_targets(batch, tag)
            outputs = model(**inputs)
            task_losses = loss_fn(outputs, targets)
            for k, v in task_losses.items():
                losses[k] = losses.get(k, 0.0) + v.item()
            all_preds.append(outputs)
            all_targets.append(targets)
            n_batches += 1

    result = {k: v / max(n_batches, 1) for k, v in losses.items()}
    if compute_metrics is not None:
        try:
            result.update(compute_metrics(all_preds, all_targets))
        except Exception as e:
            print(f"  (metrics computation skipped: {e})")
    return result


def to_device(batch, device):
    """Recursively move batch tensors to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [to_device(v, device) for v in batch]
    return batch


def split_inputs_targets(batch, tag: str = "train"):
    """Split a batch dict into model inputs and target dict.

    Expected batch keys:
      - inputs: 'satellite', 'weather', 'track_history', 'current_position',
                'satellite_sequence', 'history', 'nwp_positions', 'sequence', 'images'
      - targets: rest ('labels', 'has_tc', 'bbox', 'stage', 'structure',
                'pattern', 'future_positions', '24', '6', '12', ...)
    """
    input_keys = [
        "satellite", "weather", "track_history", "current_position",
        "satellite_sequence", "history", "nwp_positions", "sequence",
        "images", "environment", "current",
    ]
    inputs = {}
    targets = {}

    if "labels" in batch and isinstance(batch["labels"], dict):
        # TCDataset style: nested targets dict
        for k, v in batch.items():
            if k == "labels":
                targets = dict(batch["labels"])
            elif k in input_keys:
                inputs[k] = v
            else:
                targets[k] = v
    else:
        for k, v in batch.items():
            if k in input_keys:
                inputs[k] = v
            else:
                targets[k] = v
    return inputs, targets


class StagedTrainer:
    """Staged training pipeline per plan section 18.

    Stage 1: Train cyclone detector
    Stage 2: Train classifier
    Stage 3: Train pattern model
    Stage 4: Train track model
    Stage 5: Add NWP
    Stage 6: Add satellite (full fusion)
    Stage 7: Joint fine-tuning
    """

    def __init__(self, cfg, output_dir: str = "./experiments"):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)
        self.device = get_device(require_gpu=True)
        self.logger = ExperimentLogger(str(self.output_dir), "tc_ai")
        self.checkpoints = {}

    def run_stage(self, stage_name: str, model, model_key: str,
                  train_loader, val_loader, loss_fn,
                  num_epochs: int, lr: float, grad_clip: float = 1.0,
                  post_epoch: Optional[Callable] = None) -> nn.Module:
        """Run a training stage. Returns best model (by loss)."""
        print(f"\n=== Stage {stage_name} ===")
        model = model.to(self.device)
        optimizer = build_optimizer(model, lr, self.cfg.training.weight_decay)
        scheduler = build_scheduler(optimizer, self.cfg.training.scheduler, num_epochs,
                                    self.cfg.training.warmup_epochs)
        grad_scaler = torch.amp.GradScaler("cuda") if self.cfg.training.mixed_precision and "cuda" in str(self.device) else None

        best_loss = float("inf")
        best_state = None
        for epoch in range(num_epochs):
            t0 = time.time()
            train_losses = train_one_epoch(
                model, train_loader, optimizer, loss_fn, self.device,
                grad_clip=grad_clip, grad_scaler=grad_scaler, tag="train")
            val_result = validate(model, val_loader, loss_fn, self.device, tag="val",
                                  compute_metrics=post_epoch)
            val_loss = val_result.get("total", val_result.get("loss", float("inf")))

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{num_epochs} | train: {train_losses.get('total', 0):.4f} "
                  f"| val: {val_loss:.4f} | {elapsed:.1f}s")

            self.logger.log_scalar(f"{model_key}_train_loss", train_losses.get("total", 0), epoch)
            self.logger.log_scalar(f"{model_key}_val_loss", val_loss, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        self.checkpoints[model_key] = f"{self.output_dir}/{model_key}_best.pt"
        torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_loss},
                   self.checkpoints[model_key])
        print(f"  Stage {stage_name} complete. Best val loss: {best_loss:.4f}")
        return model

    def fine_tune_joint(self, model, train_loader, val_loader, loss_fn,
                        model_key: str = "multitask"):
        """Stage 7: Joint fine-tuning of the final model."""
        return self.run_stage(
            "7", model, model_key, train_loader, val_loader, loss_fn,
            self.cfg.training.stage7_epochs, self.cfg.training.learning_rate / 10,
            post_epoch=None,
        )

    def load_best(self, model_key: str) -> Optional[str]:
        return self.checkpoints.get(model_key)