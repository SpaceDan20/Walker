"""Task registry — the one place that knows which Walker tasks exist.

Two jobs:

1. Drives gym registration in ``__init__.py``, so each task gets its training,
   play and focus-play IDs from a single entry.
2. Lets play.py / focus_play.py / eval_checkpoint.py / eval_run.py work out
   which task a checkpoint belongs to from its path alone::

       logs/h1-balance/run_012/model_500.pt  ->  the h1-balance task
       logs/h1-walk/run_003/model_900.pt     ->  the h1-walk task

   train.py writes checkpoints to ``logs/<task name>/`` (derived from the PPO
   cfg's ``experiment_name``), so the folder name is the task name.

Adding a task: add one TaskSpec below, plus its env cfg module and PPO runner
cfg. Nothing else needs to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
"""Root of the per-task log tree: logs/<task name>/run_NNN/."""


@dataclass(frozen=True)
class TaskSpec:
    """Everything the scripts need to know about one task."""

    name: str
    """Log folder name. Must match the PPO cfg's experiment_name with '-' for '_'."""

    gym_prefix: str
    """Gym ID stem — the three variant IDs are built from it."""

    env_cfg_module: str
    """Module inside the Walker package holding the env cfg classes."""

    env_cfg_class: str
    """Base env cfg class. Play variants are this + _PLAY / _FOCUS_PLAY."""

    runner_cfg_class: str
    """RSL-RL runner cfg class inside agents.rsl_rl_ppo_cfg."""

    @property
    def train_id(self) -> str:
        return f"{self.gym_prefix}-v0"

    @property
    def play_id(self) -> str:
        """Many-env viewer variant."""
        return f"{self.gym_prefix}-Play-v0"

    @property
    def focus_play_id(self) -> str:
        """Single-env variant, also used for headless evaluation."""
        return f"{self.gym_prefix}-FocusPlay-v0"

    @property
    def log_root(self) -> str:
        return os.path.join(LOGS_ROOT, self.name)

    @property
    def display_name(self) -> str:
        """Human-readable form for chart titles: 'h1-balance' -> 'H1 Balance'."""
        return " ".join(
            word.upper() if len(word) <= 2 else word.capitalize()
            for word in self.name.split("-")
        )


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            name="h1-balance",
            gym_prefix="Isaac-Balance-H1",
            env_cfg_module="h1_balance_env_cfg",
            env_cfg_class="H1BalanceEnvCfg",
            runner_cfg_class="H1BalancePPORunnerCfg",
        ),
        TaskSpec(
            name="h1-walk",
            gym_prefix="Isaac-Walk-H1",
            env_cfg_module="h1_walk_env_cfg",
            env_cfg_class="H1WalkEnvCfg",
            runner_cfg_class="H1WalkPPORunnerCfg",
        ),
    )
}


def _known() -> str:
    return ", ".join(sorted(TASKS))


def task_from_name(name: str) -> TaskSpec:
    """Look up a task by folder name ('h1-walk') or by any of its gym IDs."""
    key = name.strip().replace("_", "-").lower()
    if key in TASKS:
        return TASKS[key]
    for spec in TASKS.values():
        ids = (spec.train_id, spec.play_id, spec.focus_play_id, spec.gym_prefix)
        if key in (i.lower() for i in ids):
            return spec
    raise ValueError(f"Unknown task: {name!r}. Known tasks: {_known()}")


def task_from_path(path: str) -> TaskSpec:
    """Infer the task from a checkpoint or run path.

    Expects the layout train.py writes: ``.../logs/<task name>/run_NNN/...``.
    Falls back to any path component that names a known task, so a checkpoint
    copied out of the log tree still resolves as long as the task name survives.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)

    # Search right-to-left so the nearest 'logs' anchor wins
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "logs" and i + 1 < len(parts):
            return task_from_name(parts[i + 1])

    for part in reversed(parts):
        if part.replace("_", "-").lower() in TASKS:
            return task_from_name(part)

    raise ValueError(
        f"Could not work out the task from path: {path}\n"
        f"Expected .../logs/<task>/run_NNN/model_N.pt. "
        f"Pass --task explicitly instead. Known tasks: {_known()}"
    )


def resolve_task(path: str, override: str | None = None) -> TaskSpec:
    """Task from an explicit --task override, else inferred from a path."""
    return task_from_name(override) if override else task_from_path(path)
