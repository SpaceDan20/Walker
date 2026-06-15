"""Headless policy evaluation — generates phase portrait plots (sagittal + frontal)."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a trained H1Balance policy and plot phase portraits."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file."
)
parser.add_argument(
    "--episodes", type=int, default=5, help="Number of complete episodes to run."
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Directory to save plots (default: checkpoint folder).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────

import sys

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import Walker  # noqa: F401

from Walker.agents.rsl_rl_ppo_cfg import H1BalancePPORunnerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Configs ───────────────────────────────────────────────────────────────────

agent_cfg = H1BalancePPORunnerCfg()

env_cfg = load_cfg_from_registry("Isaac-Balance-H1-FocusPlay-v0", "env_cfg_entry_point")
env_cfg.sim.device = (
    args_cli.device if args_cli.device is not None else env_cfg.sim.device
)
env_cfg.seed = agent_cfg.seed
env_cfg.scene.num_envs = 1

checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

output_dir = args_cli.output if args_cli.output else os.path.dirname(checkpoint_path)
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Checkpoint : {checkpoint_path}")
print(f"[INFO] Episodes   : {args_cli.episodes}")
print(f"[INFO] Output dir : {output_dir}")

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make("Isaac-Balance-H1-FocusPlay-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

robot = env.unwrapped.scene["robot"]

# ── Runner / policy ───────────────────────────────────────────────────────────

train_cfg = agent_cfg.to_dict()
train_cfg["algorithm"].pop("optimizer", None)

runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
runner.load(checkpoint_path)

policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_module = runner.alg.policy

# ── Data collection ───────────────────────────────────────────────────────────

RAD_TO_DEG = 180.0 / np.pi

# Each episode: list of (pitch_deg, pitch_vel, roll_deg, roll_vel)
episodes: list[dict[str, np.ndarray]] = []

current: dict[str, list] = {"pitch": [], "pitch_vel": [], "roll": [], "roll_vel": []}
episodes_done = 0

obs = env.get_observations()

print(f"[INFO] Collecting data...")
while simulation_app.is_running() and episodes_done < args_cli.episodes:
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_module.reset(dones)

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    roll_vel = robot.data.root_ang_vel_b[0, 0].item()
    pitch_vel = robot.data.root_ang_vel_b[0, 1].item()

    current["pitch"].append(pitch[0].item() * RAD_TO_DEG)
    current["pitch_vel"].append(pitch_vel * RAD_TO_DEG)
    current["roll"].append(roll[0].item() * RAD_TO_DEG)
    current["roll_vel"].append(roll_vel * RAD_TO_DEG)

    if dones[0]:
        episodes.append({k: np.array(v) for k, v in current.items()})
        current = {"pitch": [], "pitch_vel": [], "roll": [], "roll_vel": []}
        episodes_done += 1
        print(
            f"[INFO] Episode {episodes_done}/{args_cli.episodes} complete ({len(episodes[-1]['pitch'])} steps)"
        )

# ── Plotting ──────────────────────────────────────────────────────────────────

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ARROW_EVERY = 30  # draw a direction arrow every N steps

cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("H1 Balance — Phase Portraits", fontsize=14, fontweight="bold")

plane_cfgs = [
    dict(
        ax=axes[0],
        xkey="pitch",
        ykey="pitch_vel",
        title="Sagittal plane (pitch)",
        xlabel="Pitch (°)",
        ylabel="Pitch rate (°/s)",
    ),
    dict(
        ax=axes[1],
        xkey="roll",
        ykey="roll_vel",
        title="Frontal plane (roll)",
        xlabel="Roll (°)",
        ylabel="Roll rate (°/s)",
    ),
]

for cfg in plane_cfgs:
    ax: plt.Axes = cfg["ax"]
    ax.set_title(cfg["title"])
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_facecolor("#f8f8f8")

    for ep_idx, ep in enumerate(episodes):
        color = cmap(ep_idx % 10)
        x = ep[cfg["xkey"]]
        y = ep[cfg["ykey"]]

        ax.plot(
            x, y, color=color, linewidth=0.9, alpha=0.85, label=f"Episode {ep_idx + 1}"
        )

        # Direction arrows
        for i in range(ARROW_EVERY, len(x), ARROW_EVERY):
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            ax.annotate(
                "",
                xy=(x[i], y[i]),
                xytext=(x[i] - dx, y[i] - dy),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

plt.tight_layout()

checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[
    0
]  # e.g. "model_499"
base_name = f"{checkpoint_stem}_pp"
candidate = os.path.join(output_dir, f"{base_name}.png")
if os.path.exists(candidate):
    n = 2
    while os.path.exists(os.path.join(output_dir, f"{base_name}{n}.png")):
        n += 1
    candidate = os.path.join(output_dir, f"{base_name}{n}.png")

plt.savefig(candidate, dpi=150, bbox_inches="tight")
print(f"[INFO] Saved: {candidate}")

env.close()
simulation_app.close()
