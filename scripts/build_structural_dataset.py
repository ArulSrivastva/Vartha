"""Build the Phase 7 structural weak-label registry from REAL cyclone samples.

Usage:
  python scripts/build_structural_dataset.py --data-dir data/real \
      --out data/structural/structural_registry.json

Outputs a JSON registry at data/structural/structural_registry.json with:
  - vocabulary         : 11 structural classes (matches configs/config.yaml)
  - split_sizes        : train/val/test/recent per season (matches benchmarks)
  - thresholds         : geometric quantiles fit on TRAIN split only
  - samples            : per-sample weak multi-label vector + diagnostics

Pipeline-consistency check: asserts split sizes == (385, 188, 331, 162) as in
experiments/real_metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tc_ai.data.structural_labels import build_structural_registry  # noqa: E402

EXPECTED_SPLITS = {"train": 385, "val": 188, "test": 331, "recent": 162}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "real"))
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "structural" / "structural_registry.json"))
    ap.add_argument("--skip-size-check", action="store_true")
    args = ap.parse_args()

    registry = build_structural_registry(Path(args.data_dir) / "samples", Path(args.out))

    sizes = registry["split_sizes"]
    print(f"[build_structural] vocabulary: {registry['vocabulary']}")
    print(f"[build_structural] split sizes: {sizes}")
    print(f"[build_structural] thresholds (train-fit): {registry['thresholds']}")

    if not args.skip_size_check:
        mismatch = {k: (v, sizes.get(k)) for k, v in EXPECTED_SPLITS.items() if sizes.get(k) != v}
        if mismatch:
            raise SystemExit(
                f"[build_structural] SPLIT SIZE MISMATCH vs benchmarks: {mismatch}")
        print("[build_structural] split sizes OK (match experiments/real_metrics.json)")

    from collections import Counter
    dist = Counter()
    recent_dist = Counter()
    for e in registry["samples"]:
        split = e["split"]
        bucket = [split, "recent" if e.get("is_recent") else None]
        for lab, val in zip(registry["vocabulary"], e["labels"]):
            if val:
                dist[(split, lab)] += 1
                if e.get("is_recent"):
                    recent_dist[lab] += 1
    print("[build_structural] per-split label prevalence:")
    for split in ["train", "val", "test", "recent"]:
        label_str = ", ".join(
            f"{lab}={dist[(split, lab)]}" for lab in registry["vocabulary"])
        print(f"  {split}: {label_str}")
    print("[build_structural] recent (2024-2025 subset): " + ", ".join(
        f"{lab}={recent_dist[lab]}" for lab in registry["vocabulary"]))
    print(f"[build_structural] wrote {args.out}")


if __name__ == "__main__":
    main()