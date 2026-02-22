"""Centralized configuration for PPO training and agent hyperparameters.

All tunable settings live here so they can be adjusted in one place
without hunting through train_parallel.py, train_gpu_parallel.py, or Agent.py.
"""

from multiprocessing import cpu_count


# ─────────────────────────────────────────────────────────────────────
# Agent / PPO hyperparameters
# ─────────────────────────────────────────────────────────────────────

LEARNING_RATE = 1e-4
GAMMA = 0.99                # discount factor (raised from 0.95 — higher gamma makes terminal
                              # tie/loss penalties less discounted, reducing the incentive to
                              # force early ties to escape future losses)
EPS_CLIP = 0.2                # PPO clipping range
K_EPOCHS = 4                  # PPO update epochs per batch
GAE_LAMBDA = 0.95             # GAE smoothing parameter
AUGMENT = True                # DrAC-style data augmentation
AUGMENT_NOISE = 0.05          # std of Gaussian noise for augmentation
MINI_BATCH_SIZE = 2048        # mini-batch size for PPO updates
LR_SCHEDULER_T_MAX = 500      # CosineAnnealingLR period
LR_SCHEDULER_ETA_MIN = 1e-6   # CosineAnnealingLR minimum LR


# ─────────────────────────────────────────────────────────────────────
# Training loop settings
# ─────────────────────────────────────────────────────────────────────

NUM_EPOCHS = 1000
NUM_GAMES = 5000              # games per epoch
POOL_OPPONENT_PROB_CURRICULUM = 0.25  # pool prob during curriculum phase (diversity already comes from position randomness)
POOL_OPPONENT_PROB_FULL = 0.50        # pool prob after curriculum ends (stronger anti-passive-play pressure)
POOL_EPSILON = 0.15           # exploration rate for pool opponents
POOL_SAVE_INTERVAL = 10       # save to opponent pool every N epochs
POOL_MAX_SIZE = 20            # max checkpoints in the opponent pool
BENCHMARK_INTERVAL = 10       # run benchmark every N epochs
BENCHMARK_GAMES = 500         # games per benchmark evaluation


# ─────────────────────────────────────────────────────────────────────
# Curriculum learning
# ─────────────────────────────────────────────────────────────────────

CURRICULUM_ENABLED = True

# Phase 1: mid-game positions (4-9 pieces per side)
# NOTE: 2-5 piece endgame phases were avoided — very small positions are
# often theoretical draws and train passive play rather than curing it.
CURRICULUM_PHASE1_END_EPOCH = 80  # switch to full 12v12 after this epoch
CURRICULUM_PHASE1_PIECES_MIN = 4
CURRICULUM_PHASE1_PIECES_MAX = 9


# ─────────────────────────────────────────────────────────────────────
# Exploration schedule
# ─────────────────────────────────────────────────────────────────────

EPSILON_START = 1.0           # initial random-action probability
EPSILON_END = 0.08            # minimum random-action probability
EPSILON_DECAY_EPOCHS = 100    # epochs over which epsilon decays


# ─────────────────────────────────────────────────────────────────────
# Parallelism
# ─────────────────────────────────────────────────────────────────────

# CPU-only parallel (train_parallel.py)
CPU_WORKER_FRACTION = 0.5     # fraction of cpu_count for CPU-only mode

# GPU-accelerated parallel (train_gpu_parallel.py)
GPU_WORKER_FRACTION = 0.75    # fraction of cpu_count for GPU mode


def get_num_workers_cpu():
    """Worker count for CPU-only parallel training."""
    return max(2, int(cpu_count() * CPU_WORKER_FRACTION))


def get_num_workers_gpu():
    """Worker count for GPU-accelerated parallel training."""
    return max(2, int(cpu_count() * GPU_WORKER_FRACTION))


def get_epsilon(epoch):
    """Compute exploration epsilon for the given epoch."""
    return max(EPSILON_END, EPSILON_START - epoch / EPSILON_DECAY_EPOCHS)


def get_pool_opponent_prob(epoch):
    """Return pool opponent probability for the current epoch.

    Lower during curriculum (position diversity already helps) and higher
    once full 12v12 games start (needed to fight passive co-evolution).
    """
    if CURRICULUM_ENABLED and epoch < CURRICULUM_PHASE1_END_EPOCH:
        return POOL_OPPONENT_PROB_CURRICULUM
    return POOL_OPPONENT_PROB_FULL


def get_curriculum_options(epoch):
    """Return env.reset() options for the current curriculum phase.

    Phase 1 (epochs 0 to CURRICULUM_PHASE1_END_EPOCH - 1):
        Random boards with 4-9 pieces per side. Forces the agent to learn
        decisive mid-game play before encountering full 12v12 games.
        NOTE: 2-5 piece (endgame) positions are intentionally skipped —
        they are often theoretical draws and reinforce passive play.
    Phase 2 (CURRICULUM_PHASE1_END_EPOCH+):
        Standard 12v12 starting position.
    """
    if not CURRICULUM_ENABLED:
        return None
    if epoch < CURRICULUM_PHASE1_END_EPOCH:
        import random as _random
        return {"num_pieces": _random.randint(CURRICULUM_PHASE1_PIECES_MIN,
                                              CURRICULUM_PHASE1_PIECES_MAX)}
    return None  # full 12v12

