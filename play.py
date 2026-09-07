"""Watch a trained policy from a checkpoint.

The task is taken from the checkpoint's log folder (logs/<task>/run_NNN/...),
so no task flag is needed for checkpoints written by train.py.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Walker policy.")
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
    "--real_time",
    action="store_true",
    default=False,
    help="Throttle to real-time speed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────

import os
import sys
import time

import carb.input
import gymnasium as gym
import omni.appwindow
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register the environment
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import Walker  # noqa: F401

from Walker.agents import rsl_rl_ppo_cfg
from Walker.tasks import resolve_task
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Task / configs ────────────────────────────────────────────────────────────

task = resolve_task(args_cli.checkpoint, args_cli.task)
print(f"[INFO] Task: {task.name} ({task.play_id})")

agent_cfg = getattr(rsl_rl_ppo_cfg, task.runner_cfg_class)()

env_cfg = load_cfg_from_registry(task.play_id, "env_cfg_entry_point")
env_cfg.sim.device = (
    args_cli.device if args_cli.device is not None else env_cfg.sim.device
)
env_cfg.seed = agent_cfg.seed

checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print(f"[INFO] Loading checkpoint: {checkpoint_path}")

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make(task.play_id, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ── Runner / policy ───────────────────────────────────────────────────────────

train_cfg = agent_cfg.to_dict()
train_cfg["algorithm"].pop(
    "optimizer", None
)  # not yet accepted by installed rsl-rl PPO

runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
runner.load(checkpoint_path)

policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_module = runner.alg.policy

# ── Keyboard reset (R key) ────────────────────────────────────────────────────

_force_reset = False


def _on_key(event, *args, **kwargs):
    global _force_reset
    if (
        event.type == carb.input.KeyboardEventType.KEY_PRESS
        and event.input == carb.input.KeyboardInput.R
    ):
        _force_reset = True
    return True


_input = carb.input.acquire_input_interface()
_keyboard = omni.appwindow.get_default_app_window().get_keyboard()
_key_sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)

print("[INFO] Press R to manually reset all environments.")

# ── Inference loop ────────────────────────────────────────────────────────────

dt = env.unwrapped.step_dt
obs = env.get_observations()

while simulation_app.is_running():
    if _force_reset:
        with torch.inference_mode():
            obs, _ = env.reset()
        policy_module.reset(
            torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        )
        _force_reset = False
        print("[INFO] Manual reset triggered.")

    t0 = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_module.reset(dones)

    if args_cli.real_time:
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

env.close()
simulation_app.close()
