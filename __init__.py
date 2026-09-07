import gymnasium as gym

from . import agents
from .tasks import TASKS

# Every task registers three variants built from its gym prefix:
#   Isaac-<Task>-<Robot>-v0            training
#   Isaac-<Task>-<Robot>-Play-v0       many-env viewer
#   Isaac-<Task>-<Robot>-FocusPlay-v0  single env, diagnostics panel / evaluation
# Add a task in tasks.py — nothing here needs to change.
for _spec in TASKS.values():
    for _task_id, _cfg_class in (
        (_spec.train_id, _spec.env_cfg_class),
        (_spec.play_id, f"{_spec.env_cfg_class}_PLAY"),
        (_spec.focus_play_id, f"{_spec.env_cfg_class}_FOCUS_PLAY"),
    ):
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{_spec.env_cfg_module}:{_cfg_class}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:{_spec.runner_cfg_class}"
                ),
            },
        )
