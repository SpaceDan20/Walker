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

    current_xy = asset.data.root_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]

    return torch.sum(torch.square(current_xy - origin_xy), dim=1)


def knee_excess_bend_l2(
    env: ManagerBasedRLEnv,
    threshold_deg: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize knee bend exceeding a threshold beyond the default position using L2.

    No penalty is applied within the dead zone; outside it the penalty grows
    quadratically, giving a progressively stronger signal the further the knee
    goes past the limit.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    threshold_rad = torch.tensor(threshold_deg * (3.14159265 / 180.0), device=env.device)

    deviation = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    excess = (torch.abs(deviation) - threshold_rad).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)
