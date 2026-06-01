"""Watch a trained H1Balance policy from a checkpoint."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained H1Balance policy.")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file."
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

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register the environment
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import Walker  # noqa: F401

from Walker.agents.rsl_rl_ppo_cfg import H1BalancePPORunnerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Configs ───────────────────────────────────────────────────────────────────

agent_cfg = H1BalancePPORunnerCfg()

env_cfg = load_cfg_from_registry("Isaac-Balance-H1-Play-v0", "env_cfg_entry_point")
env_cfg.sim.device = (
    args_cli.device if args_cli.device is not None else env_cfg.sim.device
)
env_cfg.seed = agent_cfg.seed

checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print(f"[INFO] Loading checkpoint: {checkpoint_path}")

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make("Isaac-Balance-H1-Play-v0", cfg=env_cfg)
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

# ── Inference loop ────────────────────────────────────────────────────────────

dt = env.unwrapped.step_dt
obs = env.get_observations()

while simulation_app.is_running():
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
