"""Batch phase portrait generator — runs eval.py once per checkpoint in a run folder."""

import argparse
import os
import re
import subprocess
import sys

LOG_ROOT = r"S:\IL-RL\Walker\logs\h1-balance"
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_checkpoint.py")

parser = argparse.ArgumentParser(description="Generate phase portraits for every checkpoint in a run.")
parser.add_argument("--run", type=str, required=True, help="Run folder name, e.g. run_021.")
parser.add_argument("--episodes", type=int, default=1, help="Episodes per checkpoint (default 1).")
args = parser.parse_args()

run_dir = os.path.join(LOG_ROOT, args.run)
if not os.path.isdir(run_dir):
    raise FileNotFoundError(f"Run folder not found: {run_dir}")

checkpoints = sorted(
    [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")],
    key=lambda f: int(re.search(r"\d+", f).group()),
)

if not checkpoints:
    raise RuntimeError(f"No model_*.pt files found in {run_dir}")

print(f"[INFO] Run    : {args.run}")
print(f"[INFO] Found  : {len(checkpoints)} checkpoints")
print(f"[INFO] Episodes per checkpoint: {args.episodes}")
print()

for i, ckpt in enumerate(checkpoints):
    path = os.path.join(run_dir, ckpt)
    print(f"[{i + 1}/{len(checkpoints)}] {ckpt}")
    subprocess.run(
        [sys.executable, EVAL_SCRIPT,
         "--checkpoint", path,
         "--episodes", str(args.episodes),
         "--headless"],
        check=True,
    )

print(f"\n[INFO] Done. PNGs saved to {run_dir}")
