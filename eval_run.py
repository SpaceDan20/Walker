"""Batch evaluator — runs eval_checkpoint.py for selected checkpoints, then generates summary charts."""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_ROOT = r"S:\IL-RL\Walker\logs\h1-balance"
EVAL_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_checkpoint.py"
)

parser = argparse.ArgumentParser(
    description="Generate evaluation charts for selected checkpoints in a run."
)
parser.add_argument(
    "--run", type=str, required=True, help="Run folder name, e.g. run_021."
)
parser.add_argument(
    "--episodes", type=int, default=5, help="Episodes per checkpoint (default 5)."
)
parser.add_argument(
    "--step", type=int, default=2,
    help="Evaluate every Nth checkpoint (default 2 = every other)."
)
args = parser.parse_args()

run_dir = os.path.join(LOG_ROOT, args.run)
if not os.path.isdir(run_dir):
    raise FileNotFoundError(f"Run folder not found: {run_dir}")

all_checkpoints = sorted(
    [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")],
    key=lambda f: int(re.search(r"\d+", f).group()),
)

if not all_checkpoints:
    raise RuntimeError(f"No model_*.pt files found in {run_dir}")

checkpoints = all_checkpoints[::args.step]

print(f"[INFO] Run      : {args.run}")
print(f"[INFO] Found    : {len(all_checkpoints)} checkpoints total")
print(f"[INFO] Step     : every {args.step} checkpoint(s) ({len(checkpoints)} selected)")
print(f"[INFO] Episodes : {args.episodes} per checkpoint")
print()

for i, ckpt in enumerate(checkpoints):
    path = os.path.join(run_dir, ckpt)
    print(f"[{i + 1}/{len(checkpoints)}] {ckpt}")
    subprocess.run(
        [
            sys.executable,
            EVAL_SCRIPT,
            "--checkpoint", path,
            "--episodes", str(args.episodes),
            "--parallel",
        ],
        check=True,
    )

# ── Summary charts ────────────────────────────────────────────────────────────

stats_dir   = os.path.join(run_dir, "charts", "stats")
summary_dir = os.path.join(run_dir, "charts", "summary")
os.makedirs(summary_dir, exist_ok=True)

stat_files = sorted(
    glob.glob(os.path.join(stats_dir, "model_*_stats.json")),
    key=lambda p: int(m.group()) if (m := re.search(r"\d+", os.path.basename(p))) else 0,
)

if not stat_files:
    print("[WARN] No stats JSON files found — skipping summary charts.")
else:
    records = [json.load(open(p)) for p in stat_files]
    iters   = [r["iteration"] for r in records]

    summary_cfgs = [
        dict(mean_key="drift_mean",     std_key="drift_std",
             ylabel="Mean XY Drift (m)",  title="Drift over Training",  fname="drift_summary.png"),
        dict(mean_key="pitch_abs_mean", std_key="pitch_abs_std",
             ylabel="Mean |Pitch| (°)",   title="Pitch over Training",  fname="pitch_summary.png"),
        dict(mean_key="roll_abs_mean",  std_key="roll_abs_std",
             ylabel="Mean |Roll| (°)",    title="Roll over Training",   fname="roll_summary.png"),
    ]

    for cfg in summary_cfgs:
        means = [r[cfg["mean_key"]] for r in records]
        stds  = [r[cfg["std_key"]]  for r in records]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"H1 Balance — {cfg['title']}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Checkpoint Iteration")
        ax.set_ylabel(cfg["ylabel"])
        ax.plot(iters, means, color="steelblue", linewidth=2.0, marker="o", markersize=4, label="Mean")
        if len(records) >= 2:
            means_arr = np.array(means)
            stds_arr  = np.array(stds)
            ax.fill_between(
                iters,
                means_arr - stds_arr,  # type: ignore[arg-type]
                means_arr + stds_arr,  # type: ignore[arg-type]
                color="steelblue", alpha=0.2, label="±1σ",
            )
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_facecolor("#f8f8f8")
        plt.tight_layout()
        out = os.path.join(summary_dir, cfg["fname"])
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved: {out}")

    print(f"\n[INFO] Done. Charts saved under {os.path.join(run_dir, 'charts')}")
