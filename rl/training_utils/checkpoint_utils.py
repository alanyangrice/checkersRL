"""Generic checkpoint discovery and pruning utilities.

Used by both PPO and MCTS training pipelines. Algorithm-agnostic.
"""

from __future__ import annotations

import os


def find_latest_checkpoint_path(
    directory: str,
    prefix: str = "agent_epoch_",
    suffix: str = ".pt",
) -> str | None:
    """Return full path to latest checkpoint by epoch number, or None if none exist.

    Args:
        directory: Path to the checkpoint directory.
        prefix: Filename prefix (e.g. "agent_epoch_" for PPO, "az_epoch_" for MCTS).
        suffix: Filename suffix (e.g. ".pt").

    Returns:
        Full path to the latest checkpoint file, or None if no matching files exist.
    """
    if not os.path.isdir(directory):
        return None
    files = [f for f in os.listdir(directory)
             if f.startswith(prefix) and f.endswith(suffix)]
    if not files:
        return None
    latest = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    return os.path.join(directory, latest)


def prune_checkpoints(
    model_dir: str,
    keep_last: int | None,
    prefix: str = "agent_epoch_",
    suffix: str = ".pt",
) -> None:
    """Delete old checkpoint files, retaining only the most recent keep_last.

    If keep_last is None, all checkpoints are kept.

    Args:
        model_dir: Path to the checkpoint directory.
        keep_last: Number of checkpoints to retain (most recent by epoch number).
        prefix: Filename prefix (e.g. "agent_epoch_", "az_epoch_").
        suffix: Filename suffix (e.g. ".pt").
    """
    if keep_last is None:
        return
    if not os.path.isdir(model_dir):
        return
    files = [f for f in os.listdir(model_dir)
             if f.startswith(prefix) and f.endswith(suffix)]
    files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    for old in files[:-keep_last]:
        os.remove(os.path.join(model_dir, old))
