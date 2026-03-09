import math
import os
from multiprocessing import cpu_count
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Centralized configuration for PPO training and agent hyperparameters.

    All tunable settings live here so they can be adjusted in one place
    without hunting through train_parallel.py, train_gpu_parallel.py, or agent.py.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Agent / PPO hyperparameters
    # ─────────────────────────────────────────────────────────────────────

    LEARNING_RATE: float = 1e-4
    GAMMA: float = 0.99                # discount factor (raised from 0.95 — higher gamma makes terminal
                                  # tie/loss penalties less discounted, reducing the incentive to
                                  # force early ties to escape future losses)
    EPS_CLIP: float = 0.2                # PPO clipping range
    K_EPOCHS: float = 4                  # PPO update epochs per batch
    GAE_LAMBDA: float = 0.95             # GAE smoothing parameter
    AUGMENT: bool = True                # DrAC-style data augmentation
    AUGMENT_NOISE: float = 0.05          # std of Gaussian noise for augmentation
    MINI_BATCH_SIZE: int = 2048        # mini-batch size for PPO updates
    LR_SCHEDULER_T_MAX: int = 250      # CosineAnnealingLR period
    LR_SCHEDULER_ETA_MIN: float = 1e-6   # CosineAnnealingLR minimum LR


    # ─────────────────────────────────────────────────────────────────────
    # Training loop settings
    # ─────────────────────────────────────────────────────────────────────

    NUM_EPOCHS: int = 1000
    NUM_GAMES: int = 5000              # games per epoch
    POOL_OPPONENT_PROB_CURRICULUM: float = 0.25  # pool prob during curriculum phase
    POOL_OPPONENT_PROB_FULL: float = 0.50        # pool prob after curriculum (single-agent mode)
    POOL_EPSILON: float = 0.15           # exploration rate for pool opponents
    POOL_SAVE_INTERVAL: int = 10       # save to opponent pool every N epochs
    POOL_MAX_SIZE: int = 20            # max checkpoints per agent in the opponent pool
    BENCHMARK_INTERVAL: int = 10       # run benchmark every N epochs
    BENCHMARK_GAMES: int = 500         # games per benchmark evaluation


    # ─────────────────────────────────────────────────────────────────────
    # Multi-Agent League Play
    # ─────────────────────────────────────────────────────────────────────

    # Reward-diverse agent profiles. Each entry fully specifies the reward
    # function and PPO hyperparameters for one agent type. Add new agent
    # types here without any code changes elsewhere.
    #
    # Keys:
    #   capture_bonus        reward for capturing a piece (per hop)
    #   capture_penalty      retroactive penalty applied to the victim (per piece)
    #   king_bonus           one-time reward for king promotion
    #   tie_base             base penalty for a draw
    #   shaping_scale        multiplier on all shaped (non-terminal) rewards
    #   time_penalty_scale   multiplier on the per-move time penalty
    #   gamma                discount factor (controls planning horizon)
    #   entropy_bonus        entropy regularisation coefficient in PPO loss
    from dataclasses import field
    LEAGUE_AGENTS: dict = field(default_factory=lambda: {
        "tactical": {
            # Balanced style — captures + king promotion + tie aversion.
            # Represents the current well-rounded baseline.
            "capture_bonus":      10.0,
            "capture_penalty":    -5.0,
            "king_bonus":         15.0,
            "tie_base":          -90,
            "shaping_scale":       0.5,
            "time_penalty_scale":  1.0,
            "gamma":               0.99,
            "entropy_bonus":       0.01,
        },
        "terminal": {
            # Win/loss only — no intermediate shaping whatsoever.
            # Forces the agent to develop long-horizon positional reasoning
            # rather than myopic material counting.
            "capture_bonus":      0.0,
            "capture_penalty":    0.0,
            "king_bonus":         0.0,
            "tie_base":          -90,    # accepts draws more readily; winning is the only signal
            "shaping_scale":      0.0,
            "time_penalty_scale": 0.0,
            "gamma":              0.995,  # needs very long horizon since terminal signal is all there is
            "entropy_bonus":      0.02,   # extra exploration — must discover positional wins
        },
        "aggressive": {
            # Amplified tactical rewards + extreme tie aversion.
            # Piece-hungry, forces exchanges, demolishes passive draw-seekers.
            # Reward magnitudes reduced vs run 6 (20/30 → 15/20) to cut gradient
            # variance; entropy raised to 0.02 to match terminal and prevent the
            # policy-sharpening collapse observed at run 6 epochs 73-75.
            "capture_bonus":      15.0,
            "capture_penalty":    -8.0,
            "king_bonus":         20.0,
            "tie_base":           -90,
            "shaping_scale":       0.8,
            "time_penalty_scale":  2.0,  # urgency: finish faster
            "gamma":               0.99,
            "entropy_bonus":       0.02,  # was 0.01 — slows policy sharpening
        },
    })

    # Which agent types are active in the current league run.
    # Remove an entry to disable that agent; add a new LEAGUE_AGENTS key to enable it.
    ACTIVE_AGENTS: list = field(default_factory=lambda: ["tactical", "terminal", "aggressive"])

    # Opponent pool probability for league mode — ramped after curriculum.
    # During curriculum (epoch < 20), position diversity is already high so
    # cross-style opponents add less; 10% keeps noise low.  After epoch 20,
    # 30% gives meaningful cross-style exposure without crowding out self-play.
    LEAGUE_POOL_OPPONENT_PROB_EARLY: float = 0.20
    LEAGUE_POOL_OPPONENT_PROB_FULL: float = 0.30


    # ─────────────────────────────────────────────────────────────────────
    # Curriculum learning
    # ─────────────────────────────────────────────────────────────────────

    CURRICULUM_ENABLED: bool = True

    # Phase 1: mid-game positions (4-9 pieces per side)
    # NOTE: 2-5 piece endgame phases were avoided — very small positions are
    # often theoretical draws and train passive play rather than curing it.
    CURRICULUM_PHASE1_END_EPOCH: int = 20  # switch to full 12v12 after this epoch
    CURRICULUM_PHASE1_PIECES_MIN: int = 4
    CURRICULUM_PHASE1_PIECES_MAX: int = 9


    # ─────────────────────────────────────────────────────────────────────
    # Exploration schedule
    # ─────────────────────────────────────────────────────────────────────

    EPSILON_START: float = 1.0           # initial random-action probability
    EPSILON_END: float = 0.08            # minimum random-action probability
    EPSILON_DECAY_EPOCHS: int = 100    # epochs over which epsilon decays


    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────────────────────────────────

    CHECKPOINT_KEEP_LAST: int | None = None   # None = keep all checkpoints; set to an int to prune old ones


    # ─────────────────────────────────────────────────────────────────────
    # Prioritized opponent sampling
    # ─────────────────────────────────────────────────────────────────────

    OPPONENT_SAMPLING_TEMPERATURE: float = 0.5  # lower = exploit harder opponents more aggressively
    OPPONENT_PRIOR_WIN_RATE: float = 0.5  # assumed win rate for unseen/new checkpoints
    OPPONENT_MIN_GAMES: int = 5    # games required before stats influence sampling weight


    # ─────────────────────────────────────────────────────────────────────
    # Parallelism
    # ─────────────────────────────────────────────────────────────────────

    # CPU-only parallel (train_parallel.py)
    CPU_WORKER_FRACTION: float = 0.5     # fraction of cpu_count for CPU-only mode

    # GPU-accelerated parallel (train_gpu_parallel.py)
    GPU_WORKER_FRACTION: float = 0.5    # fraction of cpu_count for GPU mode


    def get_num_workers_cpu(self):
        """Worker count for self.CPU-only parallel training."""
        return max(2, int(cpu_count() * self.CPU_WORKER_FRACTION))


    def get_num_workers_gpu(self):
        """Worker count for self.GPU-accelerated parallel training."""
        return max(2, int(cpu_count() * self.GPU_WORKER_FRACTION))


    def get_epsilon(self, epoch):
        """Compute exploration epsilon for the given epoch."""
        # For resumed runs, check if epoch exceeds decay epochs
        if epoch >= self.EPSILON_DECAY_EPOCHS:
            return self.EPSILON_END
        return max(self.EPSILON_END, self.EPSILON_START - epoch / self.EPSILON_DECAY_EPOCHS)


    def get_league_pool_prob(self, league_epoch):
        """Return pool opponent probability for the current league epoch.

        Lower during curriculum (epoch < self.CURRICULUM_PHASE1_END_EPOCH) and higher
        once all three agent types have had a chance to add checkpoints.
        """
        if self.CURRICULUM_ENABLED and league_epoch < self.CURRICULUM_PHASE1_END_EPOCH:
            return self.LEAGUE_POOL_OPPONENT_PROB_EARLY
        return self.LEAGUE_POOL_OPPONENT_PROB_FULL


    def get_pool_opponent_prob(self, epoch):
        """Return pool opponent probability for the current epoch.

        Lower during curriculum (position diversity already helps) and higher
        once full 12v12 games start (needed to fight passive co-evolution).
        """
        if self.CURRICULUM_ENABLED and epoch < self.CURRICULUM_PHASE1_END_EPOCH:
            return self.POOL_OPPONENT_PROB_CURRICULUM
        return self.POOL_OPPONENT_PROB_FULL


    def get_curriculum_options(self, epoch):
        """Generate guaranteed-asymmetric board options for the current curriculum phase.

        The weak side draws its piece count first, then the strong side draws from
        [weak+1, max], guaranteeing a strict material advantage on every game.
        Which color is the strong side is re-rolled 50/50 each game.
        """
        if not self.CURRICULUM_ENABLED:
            return None
        if epoch < self.CURRICULUM_PHASE1_END_EPOCH:
            import random as _random
            return {"num_blue": _random.randint(self.CURRICULUM_PHASE1_PIECES_MIN,
                                                self.CURRICULUM_PHASE1_PIECES_MAX),
                    "num_red":  _random.randint(self.CURRICULUM_PHASE1_PIECES_MIN,
                                                self.CURRICULUM_PHASE1_PIECES_MAX)}
        return None  # full 12v12