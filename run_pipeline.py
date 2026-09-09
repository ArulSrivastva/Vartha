"""Master Pipeline Orchestrator for TC-AI (plan.md).

Allows executing individual phases or the complete end-to-end system:
  Phase 0: Setup & Data Acquisition (IBTrACS + ERA5 + INSAT with seasonal splits & checksum manifest)
  Phase 1: MVP Track Predictor (Persistence & Climatology baselines vs. GRU sequence model)
  Phase 2: Satellite-Based Stage Classification (IMD categories & off-by-one ordinal error analysis)
  Phase 3: Temporal Pattern & Rapid Intensification Model (24h intensity change & RI >=30kt/24h)
  Phase 4: Multimodal Fusion Model & Systematic Ablation Study (Track + ERA5 + Satellite CNN embedding)
  Phase 5 (Stretch): Detection & Center Localization (Gaussian heatmap regression from basin frames)
  Phase 6 (Stretch): Chained Real-time Inference Pipeline & Operational Dashboard

Usage:
  python run_pipeline.py --phase 1
  python run_pipeline.py --phase 4
  python run_pipeline.py --phase all
  python run_pipeline.py --dashboard
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
PYTHON = sys.executable

PHASE_SCRIPTS = {
    0: {
        "name": "Phase 0: Setup & Data Acquisition",
        "script": SCRIPTS_DIR / "build_demo_data.py",
        "args": ["--num-storms", "15", "--data-dir", "./data"],
        "desc": "Generates multi-season synthetic storms (2012-2023), strict seasonal splits & SHA256 manifest",
    },
    1: {
        "name": "Phase 1: MVP Track Predictor",
        "script": SCRIPTS_DIR / "train_phase1_track.py",
        "args": ["--epochs", "5"],
        "desc": "Evaluates Persistence and Climatology baselines vs GRU track model on 6h/12h/24h DPE",
    },
    2: {
        "name": "Phase 2: Satellite-Based Stage Classification",
        "script": SCRIPTS_DIR / "train_phase2_stage.py",
        "args": ["--epochs", "5"],
        "desc": "Trains CNN stage classifier on IMD operational categories with ordinal off-by-one analysis",
    },
    3: {
        "name": "Phase 3: Temporal Intensity & Rapid Intensification",
        "script": SCRIPTS_DIR / "train_phase3_intensity_ri.py",
        "args": ["--epochs", "5"],
        "desc": "Trains temporal sequence model for 24h wind change (kt) and Rapid Intensification (RI >= 30kt/24h)",
    },
    4: {
        "name": "Phase 4: Multimodal Fusion Model & Ablation Study",
        "script": SCRIPTS_DIR / "train_phase4_fusion.py",
        "args": ["--epochs", "5"],
        "desc": "Fuses Track + ERA5 + Satellite CNN embeddings and produces systematic ablation study table",
    },
    5: {
        "name": "Phase 5 (Stretch): Detection & Center Localization",
        "script": SCRIPTS_DIR / "train_phase5_detection.py",
        "args": ["--epochs", "5"],
        "desc": "Automates cyclone detection and center localization via Gaussian heatmap regression",
    },
    6: {
        "name": "Phase 6 (Stretch): Chained Real-time Inference",
        "script": SCRIPTS_DIR / "predict.py",
        "args": [],
        "desc": "Executes chained runtime inference: Detection -> Stage Classification -> RI -> Fusion Forecast",
    },
}


def run_command(cmd, name):
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    start = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT_DIR))
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"\n[-] ERROR: {name} failed with exit code {res.returncode}")
        return False, elapsed
    print(f"\n[+] SUCCESS: {name} completed in {elapsed:.1f}s")
    return True, elapsed


def print_final_summary(results):
    print("\n" + "=" * 80)
    print("TC-AI FINAL PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Phase':<10} | {'Status':<10} | {'Time (s)':<10} | {'Description'}")
    print("-" * 80)
    for phase_idx, (ok, elapsed, desc) in results.items():
        status = "PASSED" if ok else "FAILED"
        print(f"Phase {phase_idx:<4} | {status:<10} | {elapsed:<10.1f} | {desc}")
    print("=" * 80)
    print("Key Evaluation Artifacts in 'experiments/':")
    print("  - experiments/phase1_results.json        (Track baseline vs GRU DPE table)")
    print("  - experiments/phase2_results.json        (IMD stage accuracy, F1, off-by-one error)")
    print("  - experiments/phase3_results.json        (24h wind change MAE/RMSE, RI precision/recall)")
    print("  - experiments/phase4_ablation_results.json (Core 4-model systematic ablation table)")
    print("  - experiments/ablation_study.md          (Formatted markdown ablation report)")
    print("  - experiments/phase5_results.json        (Heatmap detection presence F1 & center error)")
    print("  - experiments/inference_output.json      (Chained Phase 6 real-time forecast record)")
    print("  - experiments/real_metrics.json          (REAL IBTrACS track + intensity/RI metrics)")
    print("  - experiments/phase1_real_results.json   (REAL Phase 1 track comparison table)")
    print("  - experiments/phase3_real_results.json   (REAL Phase 3 intensity/RI metrics)")
    print("=" * 80 + "\n")


def run_realtime(refresh_data: bool = True, epochs: int = 15):
    """Run the REAL-time data path: fetch IBTrACS -> build -> evaluate -> dashboard-ready JSON.

    Returns list of (step_name, ok, elapsed).
    """
    results = []
    if refresh_data:
        cmd = [PYTHON, str(SCRIPTS_DIR / "fetch_real_data.py"),
               "--out-dir", "./data/real"]
        ok, elapsed = run_command(cmd, "Realtime: Fetch IBTrACS + build REAL dataset")
        results.append(("Fetch+Build", ok, elapsed))
        if not ok:
            return results
    cmd = [PYTHON, str(SCRIPTS_DIR / "eval_real_metrics.py"),
           "--data-dir", "./data/real", "--output-dir", "./experiments",
           "--epochs", str(epochs)]
    ok, elapsed = run_command(cmd, "Realtime: Evaluate Phase 1 + Phase 3 on REAL data")
    results.append(("Evaluate", ok, elapsed))
    return results


def main():
    parser = argparse.ArgumentParser(description="TC-AI Master Pipeline Runner")
    parser.add_argument(
        "--phase",
        default="all",
        help="Phase to execute: '0', '1', '2', '3', '4', '5', '6', 'realtime', or 'all'",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Dash interactive dashboard on localhost:8050",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run the REAL-time path: fetch IBTrACS -> build real dataset -> eval real metrics",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="With --realtime: reuse cached IBTrACS + built real dataset (skip fetch/build)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for the real-data evaluation",
    )
    args = parser.parse_args()

    if args.dashboard:
        dash_script = ROOT_DIR / "dashboard" / "app.py"
        print(f"Launching TC-AI dashboard via {dash_script}...")
        subprocess.run([PYTHON, str(dash_script)], cwd=str(ROOT_DIR))
        return

    if args.realtime or args.phase.lower() == "realtime":
        results = run_realtime(refresh_data=not args.no_refresh, epochs=args.epochs)
        print("\n" + "=" * 80)
        print("REALTIME PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        for step, ok, elapsed in results:
            print(f"  {'[+]' if ok else '[-]'} {step:<12} | {'PASSED' if ok else 'FAILED'} | {elapsed:.1f}s")
        print("  Real-data metrics JSON: experiments/real_metrics.json")
        print("=" * 80 + "\n")
        return

    if args.phase.lower() == "all":
        phases_to_run = [0, 1, 2, 3, 4, 5, 6]
    else:
        try:
            phases_to_run = [int(args.phase)]
        except ValueError:
            print(f"Invalid phase argument: {args.phase}. Choose from 0-6, 'all', or 'realtime'.")
            sys.exit(1)

    summary = {}
    for p in phases_to_run:
        if p not in PHASE_SCRIPTS:
            print(f"Unknown phase {p}")
            continue
        info = PHASE_SCRIPTS[p]
        cmd = [PYTHON, str(info["script"])] + info["args"]
        ok, elapsed = run_command(cmd, info["name"])
        summary[p] = (ok, elapsed, info["name"])
        if not ok and args.phase.lower() == "all":
            print(f"Pipeline stopped at Phase {p} due to error.")
            break

    print_final_summary(summary)


if __name__ == "__main__":
    main()
