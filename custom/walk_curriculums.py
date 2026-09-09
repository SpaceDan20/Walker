"""Custom curriculum terms for H1 walking training.

Empty for now. The balance task showed that a curriculum is not free — a
badly timed promotion can destabilise an already-improving agent — so add one
here only when a fixed setting demonstrably blocks learning.

The obvious candidate for walking is a command curriculum: start with a narrow
lin_vel_x range and widen it (and later unlock lin_vel_y / ang_vel_z) once
tracking error stays low. Curriculum terms mutate manager configs in place,
e.g. env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = ...
"""

from __future__ import annotations

from collections import deque  # noqa: F401
from collections.abc import Sequence  # noqa: F401
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
