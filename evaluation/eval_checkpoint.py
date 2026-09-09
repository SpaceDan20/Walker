"""Headless policy evaluation — generates phase portrait plots (sagittal + frontal)."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a trained Walker policy and plot phase portraits."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file."
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task name (e.g. h1-walk) or gym ID. Defaults to inferring it from the "
    "checkpoint path.",
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
parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,
    help="Run all episodes as parallel envs simultaneously (faster).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # always run headless — no viewport needed for data collection

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────

import json
import re
import sys

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# This file lives in Walker/evaluation/, so put the folder *containing* the
# Walker package on sys.path
_WALKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_WALKER_DIR))
import Walker  # noqa: F401

from Walker.agents import rsl_rl_ppo_cfg
from Walker.tasks import resolve_task
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Task / configs ────────────────────────────────────────────────────────────

task = resolve_task(args_cli.checkpoint, args_cli.task)

agent_cfg = getattr(rsl_rl_ppo_cfg, task.runner_cfg_class)()

env_cfg = load_cfg_from_registry(task.focus_play_id, "env_cfg_entry_point")
env_cfg.sim.device = (
    args_cli.device if args_cli.device is not None else env_cfg.sim.device
)
env_cfg.seed = agent_cfg.seed
env_cfg.scene.num_envs = args_cli.episodes if args_cli.parallel else 1

checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

charts_root = args_cli.output if args_cli.output else os.path.join(os.path.dirname(checkpoint_path), "charts")
pp_dir    = os.path.join(charts_root, "pps")
drift_dir = os.path.join(charts_root, "drifts")
pitch_dir = os.path.join(charts_root, "pitch")
roll_dir  = os.path.join(charts_root, "roll")
stats_dir = os.path.join(charts_root, "stats")
for d in (pp_dir, drift_dir, pitch_dir, roll_dir, stats_dir):
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Task       : {task.name} ({task.focus_play_id})")
print(f"[INFO] Checkpoint : {checkpoint_path}")
print(f"[INFO] Episodes   : {args_cli.episodes} ({'parallel' if args_cli.parallel else 'sequential'})")
print(f"[INFO] Charts dir : {charts_root}")

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make(task.focus_play_id, cfg=env_cfg)
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

episodes: list[dict[str, np.ndarray]] = []
obs = env.get_observations()

print("[INFO] Collecting data...")

# (N, 2) — actual spawn XY (reset events randomize pose ±0.5m around env origins,
# so measure drift from where the robot really starts, matching torso_drift_l2)
spawn_xy = robot.data.root_pos_w[:, :2].clone()

if not args_cli.parallel:
    # ── Sequential: 1 env, N episodes one after another ──────────────────
    current: dict[str, list] = {"pitch": [], "pitch_vel": [], "roll": [], "roll_vel": [], "drift": []}
    episodes_done = 0

    while simulation_app.is_running() and episodes_done < args_cli.episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_module.reset(dones)

        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        roll_vel  = robot.data.root_ang_vel_b[0, 0].item()
        pitch_vel = robot.data.root_ang_vel_b[0, 1].item()
        drift_m   = torch.norm(robot.data.root_pos_w[:, :2] - spawn_xy, dim=-1)

        current["pitch"].append(pitch[0].item() * RAD_TO_DEG)
        current["pitch_vel"].append(pitch_vel * RAD_TO_DEG)
        current["roll"].append(roll[0].item() * RAD_TO_DEG)
        current["roll_vel"].append(roll_vel * RAD_TO_DEG)
        current["drift"].append(drift_m[0].item())

        if dones[0]:
            episodes.append({k: np.array(v) for k, v in current.items()})
            current = {"pitch": [], "pitch_vel": [], "roll": [], "roll_vel": [], "drift": []}
            # env auto-reset inside step() — root_pos_w already holds the new spawn
            spawn_xy = robot.data.root_pos_w[:, :2].clone()
            episodes_done += 1
            print(f"[INFO] Episode {episodes_done}/{args_cli.episodes} complete ({len(episodes[-1]['pitch'])} steps)")

else:
    # ── Parallel: N envs simultaneously, capture each env's first episode ─
    num_envs = args_cli.episodes
    buffers: dict[int, dict[str, list]] = {
        i: {"pitch": [], "pitch_vel": [], "roll": [], "roll_vel": [], "drift": []}
        for i in range(num_envs)
    }
    captured: set[int] = set()

    while simulation_app.is_running() and len(captured) < num_envs:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_module.reset(dones)

        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)   # (N,)
        roll_vel  = robot.data.root_ang_vel_b[:, 0]                     # (N,)
        pitch_vel = robot.data.root_ang_vel_b[:, 1]                     # (N,)
        drift_m   = torch.norm(robot.data.root_pos_w[:, :2] - spawn_xy, dim=-1)  # (N,)

        for i in range(num_envs):
            if i in captured:
                continue
            buffers[i]["pitch"].append(pitch[i].item() * RAD_TO_DEG)
            buffers[i]["pitch_vel"].append(pitch_vel[i].item() * RAD_TO_DEG)
            buffers[i]["roll"].append(roll[i].item() * RAD_TO_DEG)
            buffers[i]["roll_vel"].append(roll_vel[i].item() * RAD_TO_DEG)
            buffers[i]["drift"].append(drift_m[i].item())

            if dones[i]:
                episodes.append({k: np.array(v) for k, v in buffers[i].items()})
                captured.add(i)
                print(f"[INFO] Env {i + 1} done ({len(episodes[-1]['pitch'])} steps) — {len(captured)}/{num_envs} complete")

# ── Plotting ──────────────────────────────────────────────────────────────────

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ARROW_EVERY = 30  # draw a direction arrow every N steps

cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"{task.display_name} — Phase Portraits", fontsize=14, fontweight="bold")

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

        label = f"Env {ep_idx + 1}" if args_cli.parallel else f"Episode {ep_idx + 1}"
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.85, label=label)

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
candidate = os.path.join(pp_dir, f"{base_name}.png")
if os.path.exists(candidate):
    n = 2
    while os.path.exists(os.path.join(pp_dir, f"{base_name}{n}.png")):
        n += 1
    candidate = os.path.join(pp_dir, f"{base_name}{n}.png")

plt.savefig(candidate, dpi=150, bbox_inches="tight")
print(f"[INFO] Saved: {candidate}")

# ── Drift plot ────────────────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle(
    f"{task.display_name} — XY Drift from Spawn", fontsize=14, fontweight="bold"
)
ax2.set_xlabel("Step")
ax2.set_ylabel("XY Drift (m)")
ax2.set_facecolor("#f8f8f8")
ax2.grid(True, linewidth=0.4, alpha=0.5)

for ep_idx, ep in enumerate(episodes):
    label = f"Env {ep_idx + 1}" if args_cli.parallel else f"Episode {ep_idx + 1}"
    ax2.plot(ep["drift"], color=cmap(ep_idx % 10), linewidth=0.9, alpha=0.5, label=label)

if len(episodes) >= 1:
    max_len = max(len(ep["drift"]) for ep in episodes)
    padded = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        padded[i, : len(ep["drift"])] = ep["drift"]
    mean_drift = np.nanmean(padded, axis=0)
    ax2.plot(mean_drift, color="black", linewidth=2.0, label="Mean")
    if len(episodes) >= 2:
        std_drift = np.nanstd(padded, axis=0)
        ax2.fill_between(
            range(max_len),
            mean_drift - std_drift,
            mean_drift + std_drift,
            color="black",
            alpha=0.12,
            label="±1σ",
        )

ax2.legend(fontsize=8, loc="upper left")
plt.tight_layout()

drift_base = f"{checkpoint_stem}_drift"
drift_candidate = os.path.join(drift_dir, f"{drift_base}.png")
if os.path.exists(drift_candidate):
    n = 2
    while os.path.exists(os.path.join(drift_dir, f"{drift_base}{n}.png")):
        n += 1
    drift_candidate = os.path.join(drift_dir, f"{drift_base}{n}.png")

plt.savefig(drift_candidate, dpi=150, bbox_inches="tight")
print(f"[INFO] Saved: {drift_candidate}")

# ── Pitch chart ───────────────────────────────────────────────────────────────

def _timeseries_plot(title: str, ylabel: str, key: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{task.display_name} — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    for ep_idx, ep in enumerate(episodes):
        label = f"Env {ep_idx + 1}" if args_cli.parallel else f"Episode {ep_idx + 1}"
        ax.plot(ep[key], color=cmap(ep_idx % 10), linewidth=0.9, alpha=0.5, label=label)
    if episodes:
        max_len = max(len(ep[key]) for ep in episodes)
        padded = np.full((len(episodes), max_len), np.nan)
        for i, ep in enumerate(episodes):
            padded[i, : len(ep[key])] = ep[key]
        mean_vals = np.nanmean(padded, axis=0)
        ax.plot(mean_vals, color="black", linewidth=2.0, label="Mean")
        if len(episodes) >= 2:
            std_vals = np.nanstd(padded, axis=0)
            ax.fill_between(range(max_len), mean_vals - std_vals, mean_vals + std_vals,
                            color="black", alpha=0.12, label="±1σ")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig


def _save_chart(fig, directory: str, stem: str) -> None:
    candidate = os.path.join(directory, f"{stem}.png")
    if os.path.exists(candidate):
        n = 2
        while os.path.exists(os.path.join(directory, f"{stem}{n}.png")):
            n += 1
        candidate = os.path.join(directory, f"{stem}{n}.png")
    fig.savefig(candidate, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {candidate}")
    plt.close(fig)


pitch_fig = _timeseries_plot("Pitch over Episode", "Pitch (°)", "pitch")
_save_chart(pitch_fig, pitch_dir, f"{checkpoint_stem}_pitch")

# ── Roll chart ────────────────────────────────────────────────────────────────

roll_fig = _timeseries_plot("Roll over Episode", "Roll (°)", "roll")
_save_chart(roll_fig, roll_dir, f"{checkpoint_stem}_roll")

# ── Per-checkpoint stats JSON ─────────────────────────────────────────────────

def _episode_stat(key: str, transform=None):
    per_ep = []
    for ep in episodes:
        arr = ep[key] if transform is None else transform(ep[key])
        per_ep.append(float(np.mean(arr)))
    return float(np.mean(per_ep)), float(np.std(per_ep))


_iter_match = re.search(r"\d+", checkpoint_stem)
iteration = int(_iter_match.group()) if _iter_match else 0
drift_mean,     drift_std     = _episode_stat("drift")
pitch_abs_mean, pitch_abs_std = _episode_stat("pitch", np.abs)
roll_abs_mean,  roll_abs_std  = _episode_stat("roll",  np.abs)

stats = {
    "iteration":      iteration,
    "drift_mean":     drift_mean,
    "drift_std":      drift_std,
    "pitch_abs_mean": pitch_abs_mean,
    "pitch_abs_std":  pitch_abs_std,
    "roll_abs_mean":  roll_abs_mean,
    "roll_abs_std":   roll_abs_std,
}
stats_path = os.path.join(stats_dir, f"{checkpoint_stem}_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"[INFO] Saved: {stats_path}")

env.close()
simulation_app.close()
