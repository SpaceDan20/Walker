from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class torso_drift_l2(ManagerTermBase):
    """Penalize horizontal drift of the robot root from its spawn position (XY plane only).

    The reference is the actual root position recorded at each episode reset —
    not the env origin — so randomized spawn offsets do not bake a penalty into
    the start of the episode. Only XY is penalised; Z drift is intentionally
    ignored so the reward does not conflict with height-keeping terms.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._spawn_xy = self._asset.data.root_pos_w[:, :2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset events (randomized spawn pose) run before reward-manager reset,
        # so root_pos_w already holds the new spawn position here.
        if env_ids is None:
            env_ids = slice(None)
        self._spawn_xy[env_ids] = self._asset.data.root_pos_w[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        current_xy = self._asset.data.root_pos_w[:, :2]
        return torch.sum(torch.square(current_xy - self._spawn_xy), dim=1)


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
