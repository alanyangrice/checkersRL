import os
import random
import torch


class OpponentPool:
    """Manages a pool of past model checkpoints for diverse self-play training.

    Instead of always playing against itself (pure self-play), the current agent
    sometimes plays against a past version of itself sampled from this pool. This
    prevents catastrophic forgetting and policy cycling.

    Usage:
        pool = OpponentPool(pool_dir, max_size=20)
        pool.save(agent.policy.state_dict(), epoch=10)

        if pool.should_use_opponent(prob=0.3):
            opponent_path = pool.sample()
            # load opponent_path into a second agent
    """

    def __init__(self, pool_dir, max_size=20):
        self.pool_dir = pool_dir
        self.max_size = max_size
        os.makedirs(pool_dir, exist_ok=True)

    def _list_checkpoints(self):
        """Return sorted list of checkpoint filenames in the pool."""
        files = [f for f in os.listdir(self.pool_dir)
                 if f.startswith("pool_epoch_") and f.endswith(".pt")]
        files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        return files

    def save(self, state_dict, epoch):
        """Save a model snapshot to the pool. Evicts oldest if pool is full."""
        path = os.path.join(self.pool_dir, f"pool_epoch_{epoch}.pt")
        torch.save(state_dict, path)

        # Evict oldest checkpoints if pool exceeds max size
        checkpoints = self._list_checkpoints()
        while len(checkpoints) > self.max_size:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.pool_dir, oldest))

    def sample(self):
        """Return the full path to a random checkpoint from the pool, or None if empty."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        chosen = random.choice(checkpoints)
        return os.path.join(self.pool_dir, chosen)

    def should_use_opponent(self, prob=0.3):
        """With the given probability, decide to use a pool opponent (if pool is non-empty)."""
        if not self._list_checkpoints():
            return False
        return random.random() < prob

    @property
    def size(self):
        return len(self._list_checkpoints())
