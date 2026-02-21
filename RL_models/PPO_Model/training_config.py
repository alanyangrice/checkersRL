"""Centralized configuration for PPO training and agent hyperparameters.

All tunable settings live here so they can be adjusted in one place
without hunting through train_parallel.py, train_gpu_parallel.py, or Agent.py.
"""

from multiprocessing import cpu_count


# ─────────────────────────────────────────────────────────────────────
# Agent / PPO hyperparameters
# ─────────────────────────────────────────────────────────────────────

LEARNING_RATE = 1e-4
GAMMA = 0.95                  # discount factor
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
POOL_OPPONENT_PROB = 0.15     # probability of using a pool opponent
POOL_EPSILON = 0.15           # exploration rate for pool opponents
POOL_SAVE_INTERVAL = 10       # save to opponent pool every N epochs
POOL_MAX_SIZE = 20            # max checkpoints in the opponent pool
BENCHMARK_INTERVAL = 10       # run benchmark every N epochs
BENCHMARK_GAMES = 500         # games per benchmark evaluation


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

