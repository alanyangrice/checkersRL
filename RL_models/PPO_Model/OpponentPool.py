import os
import random
import torch


class OpponentPool:
    """Manages a pool of past model checkpoints for diverse self-play training.

    Supports two modes:

    Single-agent mode (original behaviour):
        pool.save(state_dict, epoch=10)           → saves pool_epoch_10.pt
        pool.sample()                             → samples from pool_epoch_*.pt

    League mode (multi-agent):
        pool.save(state_dict, epoch=10,           → saves tactical_epoch_10.pt
                  agent_name="tactical")
        pool.sample()                             → samples from ALL *.pt files
                                                     (cross-agent exposure)

    In league mode, eviction is performed per agent_name so that each agent
    type maintains its own window of max_size checkpoints.
    """

    def __init__(self, pool_dir, max_size=20):
        self.pool_dir = pool_dir
        self.max_size = max_size
        os.makedirs(pool_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_checkpoints_for(self, agent_name=None):
        """Return sorted filenames belonging to a specific agent_name.

        If agent_name is None, returns single-agent 'pool_epoch_N.pt' files.
        """
        if agent_name is None:
            files = [f for f in os.listdir(self.pool_dir)
                     if f.startswith("pool_epoch_") and f.endswith(".pt")]
            files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        else:
            prefix = f"{agent_name}_epoch_"
            files = [f for f in os.listdir(self.pool_dir)
                     if f.startswith(prefix) and f.endswith(".pt")]
            files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        return files

    def _list_all_checkpoints(self):
        """Return all .pt files in the pool directory (cross-agent sampling)."""
        files = [f for f in os.listdir(self.pool_dir) if f.endswith(".pt")]
        return files

    # Legacy alias used by single-agent code paths
    def _list_checkpoints(self):
        return self._list_checkpoints_for(agent_name=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state_dict, epoch, agent_name=None):
        """Save a model snapshot to the pool.

        Args:
            state_dict: Model state dict to save.
            epoch: Training epoch number.
            agent_name: Optional agent type identifier (e.g. "tactical").
                        If None, uses the original pool_epoch_N.pt naming.
        """
        if agent_name is None:
            path = os.path.join(self.pool_dir, f"pool_epoch_{epoch}.pt")
        else:
            path = os.path.join(self.pool_dir, f"{agent_name}_epoch_{epoch}.pt")
        torch.save(state_dict, path)

        # Evict oldest checkpoints for this agent_name only
        checkpoints = self._list_checkpoints_for(agent_name)
        while len(checkpoints) > self.max_size:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.pool_dir, oldest))

    def sample(self):
        """Return the full path to a random checkpoint from the pool.

        In league mode (multiple agent types present) samples uniformly from
        ALL checkpoints — providing cross-style exposure.  In single-agent
        mode, samples from pool_epoch_*.pt files only.

        Returns None if the pool is empty.
        """
        # Prefer cross-agent sampling when agent-namespaced files exist
        all_files = self._list_all_checkpoints()
        if not all_files:
            return None
        chosen = random.choice(all_files)
        return os.path.join(self.pool_dir, chosen)

    def should_use_opponent(self, prob=0.3):
        """With the given probability, decide to use a pool opponent."""
        if not self._list_all_checkpoints():
            return False
        return random.random() < prob

    @property
    def size(self):
        """Total number of checkpoints across all agent types."""
        return len(self._list_all_checkpoints())
