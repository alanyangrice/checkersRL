"""Hyperparameter configuration for AlphaZero (MCTS) training."""


# ---------------------------------------------------------------------------
# Self-play exploration (Dirichlet noise added to root priors)
# ---------------------------------------------------------------------------
DIRICHLET_ALPHA = 1.0
DIRICHLET_EPSILON = 0.25

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
# Weight >1 prioritises the value head, which was the bottleneck in the
# first run (value MSE stalled at 0.31).  2.0 roughly triples the value
# gradient relative to a policy loss of ~1.0.
VALUE_LOSS_WEIGHT = 2.0

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
BACKBONE_CHANNELS = 256
NUM_RES_BLOCKS = 5
POLICY_HEAD_CHANNELS = 2
VALUE_HEAD_CHANNELS = 1

# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------
NUM_SIMULATIONS = 400              # Full-board self-play
NUM_SIMULATIONS_CURRICULUM_P1 = 75 # Phase 1 endgames (2–5 pcs, tiny trees)
NUM_SIMULATIONS_CURRICULUM_P2 = 200 # Phase 2 mid-game (4–9 pcs)
C_PUCT = 1.5

# ---------------------------------------------------------------------------
# Network & optimiser
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# ---------------------------------------------------------------------------
# LR scheduler (CosineAnnealingLR)
# ---------------------------------------------------------------------------
LR_T_MAX = 500
LR_ETA_MIN = 1e-6

# ---------------------------------------------------------------------------
# Replay buffer & training
# ---------------------------------------------------------------------------
BUFFER_SIZE = 500_000
BATCH_SIZE = 256

# Adaptive training steps: instead of a fixed step count, scale the number
# of gradient updates proportional to how many new positions were collected
# this epoch.  This prevents massive overfitting in curriculum phase 1 where
# only ~1–2K positions are generated per epoch (previously hammered by
# 400 steps × 256 = 102K samples).
#
# Formula: steps = clip(REPLAY_RATIO * new_positions / BATCH_SIZE,
#                       TRAIN_STEPS_MIN, TRAIN_STEPS_MAX)
REPLAY_RATIO = 25           # Each new position is seen ~25 times on average
TRAIN_STEPS_MIN = 50        # Floor: always do at least this many updates
TRAIN_STEPS_MAX = 400       # Ceiling: cap compute for large data epochs


def get_train_steps(new_positions):
    """Compute adaptive training steps based on fresh data volume."""
    raw = int(REPLAY_RATIO * new_positions / BATCH_SIZE)
    return max(TRAIN_STEPS_MIN, min(raw, TRAIN_STEPS_MAX))


# Temperature: controls stochasticity of the *played* move (not the training
# target, which is always the raw visit-count distribution).
#
# Phase-dependent schedule:
#   Curriculum endgames are short; greedy play (low T) is fine and produces
#   decisive outcomes.  Full-board games need more state diversity (higher T).
TEMPERATURE_EARLY = 1.0     # First N moves: full exploration
TEMPERATURE_THRESHOLD_CURRICULUM = 10  # Shorter games → switch earlier
TEMPERATURE_THRESHOLD_FULL = 30        # Full board → more exploration moves
TEMPERATURE_LATE_CURRICULUM = 0.15     # Near-greedy for short endgames
TEMPERATURE_LATE_FULL = 0.3            # Moderate for full-board state diversity

MAX_GAME_MOVES = 150

# ---------------------------------------------------------------------------
# Curriculum learning phases
# ---------------------------------------------------------------------------
CURRICULUM_PHASE1_END = 15
CURRICULUM_PHASE2_END = 50
CURRICULUM_PHASE1_PIECES = (3, 6)
CURRICULUM_PHASE2_PIECES = (4, 9)


def get_temperature_config(epoch):
    """Return (threshold, late_temperature) for the given epoch."""
    if epoch < CURRICULUM_PHASE1_END:
        return TEMPERATURE_THRESHOLD_CURRICULUM, TEMPERATURE_LATE_CURRICULUM
    elif epoch < CURRICULUM_PHASE2_END:
        return TEMPERATURE_THRESHOLD_CURRICULUM, TEMPERATURE_LATE_CURRICULUM
    else:
        return TEMPERATURE_THRESHOLD_FULL, TEMPERATURE_LATE_FULL


def get_num_simulations(epoch):
    """Return the number of MCTS simulations for the given epoch."""
    if epoch < CURRICULUM_PHASE1_END:
        return NUM_SIMULATIONS_CURRICULUM_P1
    elif epoch < CURRICULUM_PHASE2_END:
        return NUM_SIMULATIONS_CURRICULUM_P2
    else:
        return NUM_SIMULATIONS


# ---------------------------------------------------------------------------
# Move-cap adjudication
# ---------------------------------------------------------------------------
# When a game hits MAX_GAME_MOVES, instead of always declaring "Tie" (which
# produces outcome=0 and zero value-head gradient), adjudicate based on
# material.  The side with more material wins; equal material → Tie.
# This is zero-sum compatible (winner=+1, loser=-1) and discourages passive
# play that runs out the clock with a material advantage.
#
# Material weighting: kings count as 1.5 pieces.
MOVE_CAP_ADJUDICATE = True  # False → always Tie at move cap (old behavior)
KING_MATERIAL_VALUE = 1.5

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
GAMES_PER_EPOCH = 100
NUM_EPOCHS = 500
SAVE_INTERVAL = 1

# ---------------------------------------------------------------------------
# Parallel training
# ---------------------------------------------------------------------------
NUM_WORKERS = 50


def get_num_workers_parallel():
    """Return the number of CPU worker processes for parallel self-play."""
    import os
    if NUM_WORKERS is not None:
        return NUM_WORKERS
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 24))

# ---------------------------------------------------------------------------
# Evaluation & gating
# ---------------------------------------------------------------------------
EVAL_INTERVAL = 5
EVAL_GAMES_RANDOM = 50
EVAL_GAMES_GATE = 50
EVAL_SIMULATIONS = 100
GATE_THRESHOLD = 0.55
GATE_ENABLED = True
