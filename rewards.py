from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torso_drift_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal drift of the robot root from its spawn position (XY plane only).

    Uses the initial root position recorded at the start of each episode as the
    reference. Only XY is penalised — Z drift is intentionally ignored so the
    reward does not conflict with height-keeping terms.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Current XY position in world frame
    current_xy = asset.data.root_pos_w[:, :2]

    # Spawn XY — recorded once per episode at reset time
    origin_xy = env.scene.env_origins[:, :2]

    return torch.sum(torch.square(current_xy - origin_xy), dim=1)
