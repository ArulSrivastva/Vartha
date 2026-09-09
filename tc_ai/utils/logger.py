import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Simple experiment tracking logger."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[float]] = {}
        self.metrics: Dict[str, Any] = {}
        self.best_metrics: Dict[str, float] = {}

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self.history.setdefault(name, []).append(value)
        if step is None:
            step = len(self.history[name]) - 1
        self._append_to_csv(name, step, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def save_metrics(self, metrics: Dict[str, Any], tag: str):
        self.metrics[tag] = metrics
        path = self.log_dir / f"metrics_{tag}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_model(self, model, optimizer, epoch, path: str):
        torch_path = self.log_dir / path
        torch_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        }, torch_path)
        return str(torch_path)

    def _append_to_csv(self, name: str, step: int, value: float):
        csv_path = self.log_dir / f"{name}.log"
        with open(csv_path, "a") as f:
            f.write(f"{step}\t{value}\n")
