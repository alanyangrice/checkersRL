from rl.configs.ppo_config import PPOConfig

default_config = PPOConfig()
import json
import csv
import os
import random

import numpy as np
import torch



class OpponentPool:
    """Manages a pool of past model checkpoints for diverse self-play training.

    Supports two modes:

    Single-agent mode (original behaviour):
        pool.save(state_dict, epoch=10)           → saves pool_epoch_10.pt
        pool.sample()                             → weighted sample from pool_epoch_*.pt

    League mode (multi-agent):
        pool.save(state_dict, epoch=10,           → saves tactical_epoch_10.pt
                  agent_name="tactical")
        pool.sample(agent_name="tactical")        → weighted sample from ALL *.pt files
                                                     with weights based on win-rate stats
                                                     tracked per agent_name

    Sampling uses softmax-weighted selection biased toward opponents with lower
    historical win rates (PFSP-style: sample opponents you struggle against more often).
    Stats are persisted per agent in win_rates_{agent_name}.csv files.

    In league mode, eviction from the active sampling window is performed per agent_name
    so that each agent type maintains its own window of max_size checkpoints.
    (Older checkpoints are kept on disk but no longer sampled).
    """

    def __init__(self, pool_dir, max_size=20, config=None):
        config = config or default_config
        self.config = config
        self.pool_dir = pool_dir
        self.max_size = max_size
        os.makedirs(pool_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers — checkpoint listing
    # ------------------------------------------------------------------

    def _list_checkpoints_for(self, agent_name=None):
        """Return sorted active filenames belonging to a specific agent_name."""
        if agent_name is None:
            files = [f for f in os.listdir(self.pool_dir)
                     if f.startswith("pool_epoch_") and f.endswith(".pt")]
        else:
            prefix = f"{agent_name}_epoch_"
            files = [f for f in os.listdir(self.pool_dir)
                     if f.startswith(prefix) and f.endswith(".pt")]
        files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        return files[-self.max_size:] if self.max_size > 0 else files

    def _list_all_checkpoints(self):
        """Return active .pt files in the pool directory (cross-agent sampling)."""
        all_agents = [None]
        if hasattr(self.config, "ACTIVE_AGENTS") and self.config.ACTIVE_AGENTS:
            all_agents.extend(self.config.ACTIVE_AGENTS)
            
        active = []
        for a in all_agents:
            active.extend(self._list_checkpoints_for(a))
        return active

    # ------------------------------------------------------------------
    # Internal helpers — win-rate stats persistence
    # ------------------------------------------------------------------

    def _stats_path(self, agent_name=None):
        key = agent_name if agent_name is not None else "default"
        return os.path.join(self.pool_dir, f"win_rates_{key}.csv")

    def _load_stats(self, agent_name=None):
        path = self._stats_path(agent_name)
        json_path = os.path.join(self.pool_dir, f"win_rates_{agent_name if agent_name else 'default'}.json")
        
        stats = {}
        # Fallback to json if csv doesn't exist yet
        if os.path.exists(json_path) and not os.path.exists(path):
            try:
                with open(json_path, "r") as f:
                    stats = json.load(f)
                return stats
            except (json.JSONDecodeError, IOError):
                pass

        if not os.path.exists(path):
            return {}
            
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) == 3:
                        fname, win_rate, n = row
                        stats[fname] = {"win_rate": float(win_rate), "n": int(n)}
        except Exception:
            pass
        return stats

    def _save_stats(self, stats, agent_name=None):
        path = self._stats_path(agent_name)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["opponent_checkpoint", "win_rate", "games_played"])
            for fname in sorted(stats.keys()):
                entry = stats[fname]
                writer.writerow([fname, entry.get("win_rate", 0.5), entry.get("n", 0)])

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

        # Cleanup stats for checkpoints that are no longer in the active window.
        # We do NOT evict from disk, but we stop tracking them in the win_rates stats.
        active_checkpoints = set(self._list_all_checkpoints())
        
        all_agents = [None]
        if hasattr(self.config, "ACTIVE_AGENTS") and self.config.ACTIVE_AGENTS:
            all_agents.extend(self.config.ACTIVE_AGENTS)
        
        for a_name in all_agents:
            stats = self._load_stats(a_name)
            changed = False
            for old_f in list(stats.keys()):
                if old_f not in active_checkpoints:
                    del stats[old_f]
                    changed = True
            if changed:
                self._save_stats(stats, a_name)

    def batch_update_stats(self, all_results, epoch=None, agent_name=None):
        """Update win-rate stats from a completed epoch's game results.

        Takes the full all_results list, filters for games that used a pool
        opponent (opponent_path is not None), and performs a single JSON/CSV
        load + batch of increments + single save.  Call once per epoch,
        NOT once per game, to keep I/O to a single round-trip.

        Args:
            all_results: list of result dicts from run_games_on_gpu.
            epoch: the current training epoch (used for historical logging).
            agent_name: The agent whose stats to update (e.g. "tactical").
        """
        pool_results = [
            r for r in all_results
            if r.get("opponent_path") is not None and r.get("agent_win") is not None
        ]
        if not pool_results:
            return

        stats = self._load_stats(agent_name)

        for r in pool_results:
            fname = os.path.basename(r["opponent_path"])
            entry = stats.setdefault(fname, {"win_rate": 0.5, "n": 0})
            
            # Transition old stats format to EMA
            if "win_rate" not in entry:
                if "w" in entry and entry.get("n", 0) > 0:
                    entry["win_rate"] = entry["w"] / entry["n"]
                else:
                    entry["win_rate"] = 0.5
                for key in ["w", "l", "t"]:
                    if key in entry:
                        del entry[key]

            # Exponential Moving Average (alpha = 0.05)
            alpha = 0.05
            is_win = 1.0 if r["agent_win"] else 0.0
            
            if entry["n"] == 0:
                entry["win_rate"] = is_win
            else:
                entry["win_rate"] = (alpha * is_win) + ((1 - alpha) * entry["win_rate"])
                
            entry["n"] += 1

        self._save_stats(stats, agent_name)

        # Log history if epoch is provided
        if epoch is not None:
            history_path = os.path.join(self.pool_dir, f"win_rates_history_{agent_name if agent_name else 'default'}.csv")
            write_header = not os.path.exists(history_path)
            
            with open(history_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "opponent_checkpoint", "win_rate", "games_played"])
                
                for fname in sorted(stats.keys()):
                    entry = stats[fname]
                    writer.writerow([epoch, fname, entry.get("win_rate", 0.5), entry.get("n", 0)])

    def sample(self, agent_name=None):
        """Return the full path to a checkpoint, weighted by difficulty.

        Checkpoints the agent historically struggles against (lower win rate)
        receive higher sampling weight — analogous to AlphaStar's PFSP.

        Weight formula:  weight_i = (1 - win_rate_i) / temperature
        Probabilities:   softmax(weights)

        New or unseen checkpoints use OPPONENT_PRIOR_WIN_RATE (0.5) until
        OPPONENT_MIN_GAMES games have been played against them.

        Returns None if the pool is empty.
        """
        all_files = self._list_all_checkpoints()
        if not all_files:
            return None

        stats = self._load_stats(agent_name)
        weights = []
        for fname in all_files:
            entry = stats.get(fname, {})
            n = entry.get("n", 0)
            
            if "win_rate" not in entry:
                if "w" in entry and n > 0:
                    win_rate = entry["w"] / n
                else:
                    win_rate = self.config.OPPONENT_PRIOR_WIN_RATE
            else:
                win_rate = entry["win_rate"]

            if n < self.config.OPPONENT_MIN_GAMES:
                win_rate = self.config.OPPONENT_PRIOR_WIN_RATE
                
            weights.append((1.0 - win_rate) / self.config.OPPONENT_SAMPLING_TEMPERATURE)

        # Numerically stable softmax
        w = np.array(weights, dtype=np.float64)
        w = np.exp(w - w.max())
        w /= w.sum()

        chosen = np.random.choice(all_files, p=w)
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
