"""Action-mask utilities: sampling and log-probability for valid actions.

Shared by PPO training, evaluation, and benchmark scripts. Algorithm-agnostic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def random_action_from_mask(mask: NDArray) -> int:
    """Sample a random valid action from the action mask."""
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def uniform_log_prob(mask: NDArray) -> float:
    """Log probability for a uniform random choice over valid actions."""
    n = int(mask.sum())
    if n <= 0:
        return 0.0
    return float(np.log(1.0 / n))
