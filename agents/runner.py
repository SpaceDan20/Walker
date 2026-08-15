"""Custom OnPolicyRunner with alive-gradient-fraction TensorBoard diagnostics."""

from __future__ import annotations

from collections import deque

import torch
from rsl_rl.runners import OnPolicyRunner


class H1BalanceOnPolicyRunner(OnPolicyRunner):
    """Extends OnPolicyRunner with per-term alive-gradient-fraction logging.

    Patches reward_manager.reset() at init to capture per-env episode sums at
    the exact moment of termination — before IsaacLab zeroes them — giving a
    buffer of complete episode returns for each reward term.

    At each log interval, computes the std fraction of each term over a rolling
    window of the last `EPISODE_WINDOW` completed episodes:
        alive_fraction[term] = std[term] / Σ std[all terms]

    The rolling window keeps the sample size constant regardless of how many
    episodes complete per rollout — without it, long-surviving policies yield
    only a handful of completions per iteration and the fractions turn to noise.

    Each term is logged via add_scalar (stays inside the run's own event file),
    and a custom scalar layout groups all lines onto one chart in TensorBoard's
    CUSTOM SCALARS tab. This avoids the sub-run clutter that add_scalars creates.

    All lines sum to 100% at every log point:
        ~0%  → dead gradient (term not differentiating good vs bad episodes)
        high → that term is dominating the current policy update
    """

    EPISODE_WINDOW = 500
    """Rolling window size: number of most-recent completed episodes per term."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._term_buffers: dict[str, deque[float]] = {}
        self._layout_registered = False
        self._patch_reward_manager()

    def _patch_reward_manager(self) -> None:
        """Replace rm.reset with a closure that snapshots episode sums before clearing."""
        try:
            rm = self.env.unwrapped.reward_manager
        except AttributeError:
            return
        if not hasattr(rm, "_episode_sums"):
            return

        buffers = self._term_buffers
        window = self.EPISODE_WINDOW
        unwrapped_env = self.env.unwrapped
        original_reset = rm.reset  # bound method — keeps rm as implicit self

        def _capturing_reset(env_ids):
            # Skip the warmup phase: init_at_random_ep_len pre-fills episode_length_buf,
            # so first episodes are partial-length and would pollute the return stats.
            past_warmup = (
                unwrapped_env.common_step_counter > unwrapped_env.max_episode_length
            )
            if past_warmup and len(env_ids) > 0:
                for term_name, ep_sums in rm._episode_sums.items():
                    completed = ep_sums[env_ids].float().cpu()
                    if term_name not in buffers:
                        buffers[term_name] = deque(maxlen=window)
                    buffers[term_name].extend(completed.tolist())
            return original_reset(env_ids)

        rm.reset = _capturing_reset

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)

        if not self._term_buffers or self.writer is None:
            return

        stds: dict[str, float] = {}
        for term_name, returns in self._term_buffers.items():
            if len(returns) < 2:
                continue
            stds[term_name] = torch.tensor(returns).std().item()

        if stds:
            # Register layout once — groups all per-term scalars onto one chart
            # in TensorBoard's CUSTOM SCALARS tab, inside the run's own event file.
            if not self._layout_registered:
                tags = [f"GradientAlive/{name}" for name in stds]
                self.writer.add_custom_scalars(
                    {"Gradient Alive": {"AliveFraction_%": ["Multiline", tags]}}
                )
                self._layout_registered = True

            total_std = sum(stds.values()) + 1e-8
            it = self.current_learning_iteration
            for name, s in stds.items():
                self.writer.add_scalar(
                    f"GradientAlive/{name}",
                    (s / total_std) * 100.0,
                    it,
                )
