"""Custom observation terms for H1 walking training.

Empty for now. Each term is a function returning a (num_envs, term_dim)
tensor; the observation manager concatenates them in declaration order.

    def my_term(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # noqa: F401

from isaaclab.managers import SceneEntityCfg  # noqa: F401
from isaaclab.sensors import ContactSensor  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
