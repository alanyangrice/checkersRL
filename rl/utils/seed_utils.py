"""Reproducibility: seed control for random, numpy, and torch."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, fully_deterministic: bool = False) -> None:
    """Set seeds for random, numpy, and torch to enable reproducible runs.

    Call this at the start of training when --seed is provided.
    CUDA nondeterministic ops (e.g. some atomics) may still cause variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            # Reduce nondeterminism; causes significant performance drop on CNNs
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
