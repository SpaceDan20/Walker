"""Custom curriculum terms for H1 balance training."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class survival_rate_reward_weight(ManagerTermBase):
    """Promotes a single reward weight once survival rate crosses a threshold.

    Survival is defined as an episode reaching max_episode_length without falling.
    Watches a rolling window of completed episodes and promotes the target weight
    permanently once the fraction of survivors >= threshold.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        window = cfg.params.get("window", 500)
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._promoted = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        threshold: float = 0.8,
        window: int = 500,
    ) -> float:
        if self._promoted:
            return self._term_cfg.weight

        if len(env_ids) > 0:
            lengths = env.episode_length_buf[env_ids]
            for length in lengths.tolist():
                self._outcomes.append(length >= env.max_episode_length - 1)

        if len(self._outcomes) == window:
            rate = sum(self._outcomes) / window
            if rate >= threshold:
                self._term_cfg.weight = weight
                env.reward_manager.set_term_cfg(term_name, self._term_cfg)
                self._promoted = True
                print(
                    f"[Curriculum] '{term_name}' weight promoted to {weight:.4f} "
                    f"(survival rate {rate:.1%} over {window} episodes, "
                    f"step {env.common_step_counter})"
                )

        return self._term_cfg.weight


class survival_rate_reward_weights(ManagerTermBase):
    """Promotes multiple shaped-penalty weights simultaneously once survival rate crosses a threshold.

    All listed terms share one rolling survival window and trigger together.
    The full intended weights are read from the reward config at init, then
    immediately scaled down by `scale`. On promotion they are restored to the
    original values — so the rewards config stays as the single source of truth.

    Example: scale=0.05 starts every listed penalty at 5% of its configured weight,
    then promotes to 100% once `threshold` survival rate is sustained over `window` episodes.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terms: list[str] = cfg.params["terms"]
        scale: float = cfg.params.get("scale", 0.05)

        self._term_cfgs = {
            name: env.reward_manager.get_term_cfg(name) for name in terms
        }
        # Save full target weights from the rewards config, then apply scaled-down versions
        self._target_weights: dict[str, float] = {}
        for name, term_cfg in self._term_cfgs.items():
            self._target_weights[name] = term_cfg.weight
            term_cfg.weight = term_cfg.weight * scale
            env.reward_manager.set_term_cfg(name, term_cfg)

        window = cfg.params.get("window", 500)
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._promoted = False

        names = ", ".join(terms)
        print(f"[Curriculum] Shaped penalties initialized at {scale:.0%}: [{names}]")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        terms: list[str],
        scale: float = 0.05,  # applied once in __init__; must stay here for the framework kwargs pass-through
        threshold: float = 0.8,
        window: int = 500,
    ) -> float:
        if self._promoted:
            return 1.0

        if len(env_ids) > 0:
            lengths = env.episode_length_buf[env_ids]
            for length in lengths.tolist():
                self._outcomes.append(length >= env.max_episode_length - 1)

        if len(self._outcomes) == window:
            rate = sum(self._outcomes) / window
            if rate >= threshold:
                for name, target_weight in self._target_weights.items():
                    term_cfg = self._term_cfgs[name]
                    term_cfg.weight = target_weight
                    env.reward_manager.set_term_cfg(name, term_cfg)
                self._promoted = True
                names = ", ".join(terms)
                print(
                    f"[Curriculum] Shaped penalties promoted to full weight: [{names}] "
                    f"(survival rate {rate:.1%} over {window} episodes, "
                    f"step {env.common_step_counter})"
                )

        return float(self._promoted)
