# TC-AI Project Guidelines & Rules

## Hardware Acceleration: Mandatory GPU for Training
- **Always use GPU (`cuda`) for model training**: All training scripts (`scripts/train_*.py`, `tc_ai/training/trainer.py`) and gradient update loops must execute on GPU (`cuda`), never CPU.
- **Fail-Fast on Missing CUDA**: If `torch.cuda.is_available()` is `False`, raise an explicit `RuntimeError` prohibiting training on CPU. Never silently fall back to CPU for training.
- **Inference**: Inference pipelines prefer GPU and can safely run on available hardware.

## Methodology & Architecture
- Follow the phased roadmap in `plan.md`.
- Strictly enforce seasonal train/val/test splits (zero temporal data leakage).
- All evaluation results must maintain metric tables and JSON summaries in `experiments/`.
