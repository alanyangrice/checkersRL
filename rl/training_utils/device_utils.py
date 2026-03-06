"""Device selection and torch.compile availability.

Used by both PPO and MCTS. Centralizes CUDA/MPS/CPU selection and Triton checks
for torch.compile support (required on CUDA; unavailable on Windows).
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_compile_available() -> bool:
    """Return True only when torch.compile can fully execute on CUDA.

    torch.compile with both mode='default' and mode='reduce-overhead' uses
    TorchInductor to generate optimized CUDA kernels, which requires Triton.
    Triton is Linux-only in standard PyTorch releases and is not available on
    Windows.  Attempting to compile without Triton raises TritonMissing on the
    first forward call, not at compile time, causing a silent crash deep in
    training.  This guard prevents that.
    """
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "compile"):
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False
