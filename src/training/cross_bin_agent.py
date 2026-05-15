"""
CrossBinAgent — wraps an inner BaseAgent trained on one raise_bin set so it
can act in an environment whose action space uses a different raise_bin set.

The env emits actions through its native ``Discrete(2 + len(env_bins) + 1)``
space. An inner PPO policy trained on ``inner_bins`` outputs indices in its
own ``Discrete(2 + len(inner_bins) + 1)`` space. Passing those raw indices
to the env would mis-translate every raise: e.g. the inner agent's all-in
index would be interpreted by the env as some mid-pot raise.

This wrapper bridges the gap:
- Fold (0) and check/call (1) pass through unchanged.
- The inner agent's all-in index maps to the env's all-in index.
- Raise indices are translated via the inner agent's bin's pot-fraction,
  then mapped onto the env's bins via randomised pseudo-harmonic mapping
  (Ganzfried-Sandholm): ``P(A) = (B - x)(1 + A) / ((B - A)(1 + x))`` for
  ``A < x < B``.

Used by duel/regression tooling that needs to compare models with different
training action spaces.
"""

from typing import List, Optional
import random

import numpy as np

from src.agents.base_agent import BaseAgent


class CrossBinAgent(BaseAgent):
    def __init__(
        self,
        inner: BaseAgent,
        inner_bins: List[float],
        env_bins: List[float],
        rng: Optional[random.Random] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or f"CrossBin({inner.name})")
        self.inner = inner
        self.inner_bins = list(inner_bins)
        self.env_bins = list(env_bins)
        self.rng = rng if rng is not None else random.Random(0)
        self._inner_all_in_idx = 2 + len(self.inner_bins)
        self._env_all_in_idx = 2 + len(self.env_bins)
        self._identity = self.inner_bins == self.env_bins

    def select_action(
        self, observation: np.ndarray, valid_actions: Optional[list] = None
    ) -> int:
        a = self.inner.select_action(observation, valid_actions)
        if self._identity:
            return a
        if a <= 1:
            return a
        if a == self._inner_all_in_idx:
            return self._env_all_in_idx
        bin_idx = a - 2
        if bin_idx >= len(self.inner_bins):
            # Defensive: inner returned an out-of-range index. Treat as
            # all-in (closest semantic match).
            return self._env_all_in_idx
        x = self.inner_bins[bin_idx]
        return self._fraction_to_env_action(x)

    def _fraction_to_env_action(self, x: float) -> int:
        bins = self.env_bins
        for i, b in enumerate(bins):
            if abs(b - x) < 1e-9:
                return 2 + i
        if x <= bins[0]:
            return 2
        if x >= bins[-1]:
            return 2 + len(bins) - 1
        for i in range(len(bins) - 1):
            A, B = bins[i], bins[i + 1]
            if A < x < B:
                p_a = ((B - x) * (1 + A)) / ((B - A) * (1 + x))
                p_a = max(0.0, min(1.0, p_a))
                return 2 + (i if self.rng.random() < p_a else i + 1)
        return 2
