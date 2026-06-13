"""Focus-play viewer: 1 env with a live joint-torque panel (10 Hz UI updates)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Focus-play a trained H1Balance policy.")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file."
)
parser.add_argument(
    "--real_time",
    action="store_true",
    default=False,
    help="Throttle to real-time speed (equivalent to --speed 1.0).",
)
parser.add_argument(
    "--speed",
    type=float,
    default=0.25,
    help="Playback speed multiplier (e.g. 0.25 for quarter-speed slow motion). Implies real-time throttling.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────

import os
import sys
import time

import gymnasium as gym
import omni.ui as ui
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs.ui import EmptyWindow
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import Walker  # noqa: F401

from Walker.agents.rsl_rl_ppo_cfg import H1BalancePPORunnerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Configs ───────────────────────────────────────────────────────────────────

agent_cfg = H1BalancePPORunnerCfg()

env_cfg = load_cfg_from_registry("Isaac-Balance-H1-DeepPlay-v0", "env_cfg_entry_point")
env_cfg.sim.device = (
    args_cli.device if args_cli.device is not None else env_cfg.sim.device
)
env_cfg.seed = agent_cfg.seed
env_cfg.scene.num_envs = 1

checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print(f"[INFO] Loading checkpoint: {checkpoint_path}")

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make("Isaac-Balance-H1-DeepPlay-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

num_envs = env.unwrapped.num_envs

# ── Joint torque panel ────────────────────────────────────────────────────────

robot = env.unwrapped.scene["robot"]
joint_names: list[str] = list(robot.data.joint_names)
max_name_len = max(len(n) for n in joint_names)

# Pre-resolve knee joint indices
knee_indices = [i for i, n in enumerate(joint_names) if "knee" in n]
knee_names = [joint_names[i] for i in knee_indices]

RAD_TO_DEG = 180.0 / 3.14159265

# Penalty tracking — read term names from the reward manager
reward_manager = env.unwrapped.reward_manager
penalty_names: list[str] = [n for n in reward_manager._episode_sums if n != "termination_penalty"]
max_penalty_name_len = max(len(n) for n in penalty_names)
prev_ep_sums: dict[str, float] = {name: 0.0 for name in penalty_names}

torque_window = EmptyWindow(env.unwrapped, "H1 Joint Torques")

penalty_label: ui.Label
knee_label: ui.Label
env_labels: list[ui.Label] = []
with torque_window.ui_window_elements["main_vstack"]:
    # ── Penalty tracker ───────────────────────────────────────────────────
    ui.Label("Penalties", style={"font_size": 20, "color": 0xFFAAAAAA})
    penalty_label = ui.Label("", style={"font_size": 20}, word_wrap=True)
    ui.Spacer(height=10)

    # ── Knee bend indicator ───────────────────────────────────────────────
    ui.Label("Knee Bend", style={"font_size": 20, "color": 0xFFAAAAAA})
    knee_label = ui.Label("", style={"font_size": 20}, word_wrap=True)
    ui.Spacer(height=10)

    # ── Joint torques ─────────────────────────────────────────────────────
    ui.Label("Torques", style={"font_size": 20, "color": 0xFFAAAAAA})
    lbl = ui.Label("", style={"font_size": 20}, word_wrap=True)
    env_labels.append(lbl)
    ui.Spacer(height=6)


def _format_penalties(was_reset: bool) -> str:
    lines = []
    for name in penalty_names:
        current = reward_manager._episode_sums[name][0].item()
        if was_reset:
            step_val = 0.0
            prev_ep_sums[name] = 0.0
        else:
            step_val = current - prev_ep_sums[name]
            prev_ep_sums[name] = current
        lines.append(
            f"  {name:<{max_penalty_name_len}}  {step_val:+.4f} ({current:+.3f})"
        )
    return "\n".join(lines)


def _format_knee_bend(env_idx: int) -> str:
    pos = robot.data.joint_pos[env_idx]
    lines = []
    for name, idx in zip(knee_names, knee_indices):
        deg = pos[idx].item() * RAD_TO_DEG
        lines.append(f"  {name:<{max_name_len}}  {deg:+6.1f}°")
    return "\n".join(lines)


def _format_torques(env_idx: int) -> str:
    torques = robot.data.applied_torque[env_idx]
    lines = []
    for name, t in zip(joint_names, torques.tolist()):
        lines.append(f"  {name:<{max_name_len}}  {t:+7.2f} Nm")
    return "\n".join(lines)


# ── Runner / policy ───────────────────────────────────────────────────────────

train_cfg = agent_cfg.to_dict()
train_cfg["algorithm"].pop("optimizer", None)

runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
runner.load(checkpoint_path)

policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_module = runner.alg.policy

# ── Inference loop ────────────────────────────────────────────────────────────

dt = env.unwrapped.step_dt
# UI updates at 10 Hz; policy runs at 50 Hz → update every 5 steps
UI_UPDATE_EVERY = max(1, round(0.1 / dt))

obs = env.get_observations()
step = 0
episode_reset = False

while simulation_app.is_running():
    t0 = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_module.reset(dones)

    if dones[0]:
        episode_reset = True

    step += 1
    if step % UI_UPDATE_EVERY == 0:
        penalty_label.text = _format_penalties(episode_reset)
        knee_label.text = _format_knee_bend(0)
        for i, lbl in enumerate(env_labels):
            lbl.text = _format_torques(i)
        episode_reset = False

    if args_cli.real_time or args_cli.speed != 1.0:
        sleep = (dt / args_cli.speed) - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

env.close()
simulation_app.close()
