"""Custom reward terms for H1 walking training.

Empty for now. Add terms here only when the built-in mdp / locomotion terms
cannot express what the task needs.

Function-style term:

    def my_term(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        ...  # -> (num_envs,) tensor

Class-style term (use when the term needs per-episode state, e.g. a value
captured at reset — subclass ManagerTermBase and implement reset/__call__).
"""

from __future__ import annotations

from collections.abc import Sequence  # noqa: F401
from typing import TYPE_CHECKING

import torch  # noqa: F401

from isaaclab.assets import Articulation  # noqa: F401
from isaaclab.managers import (  # noqa: F401
    ManagerTermBase,
    RewardTermCfg,
    SceneEntityCfg,
)
from isaaclab.sensors import ContactSensor  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
