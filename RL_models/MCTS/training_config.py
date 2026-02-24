"""Hyperparameter configuration for AlphaZero (MCTS) training."""


# ---------------------------------------------------------------------------
# Self-play exploration (Dirichlet noise added to root priors)
# ---------------------------------------------------------------------------
# α controls the concentration of the noise distribution.
# Smaller α → noisier/more uniform; larger α → concentrated near network priors.
# Rule of thumb: α ≈ 10 / avg_branching_factor.
# Checkers has ~8 legal moves on average, so α ≈ 1.0; 0.3 is the Chess value
# and is conservative enough to start with.
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25    # Fraction of noise mixed into root priors

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
# Scales the value head loss relative to policy loss.
# policy_loss (cross-entropy) typically sits in ~[1, 3] nats.
# value_loss  (MSE vs ±1 targets) can be up to 4.0 early in training.
# A weight < 1.0 prevents the value head from dominating gradients early on.
VALUE_LOSS_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
BACKBONE_CHANNELS = 256     # Convolutional channels throughout the residual tower
NUM_RES_BLOCKS = 5          # Number of residual blocks in the shared backbone
POLICY_HEAD_CHANNELS = 2    # 1×1 conv output channels before the policy linear layer
VALUE_HEAD_CHANNELS = 1     # 1×1 conv output channels before the value linear layers

# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------
NUM_SIMULATIONS = 100       # Simulations per move during self-play
C_PUCT = 1.5                # Exploration constant in the PUCT formula

# ---------------------------------------------------------------------------
# Network & optimiser
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0        # Max gradient norm for clipping

# ---------------------------------------------------------------------------
# LR scheduler (CosineAnnealingLR)
# ---------------------------------------------------------------------------
LR_T_MAX = 500              # Number of epochs before LR hits eta_min
LR_ETA_MIN = 1e-6           # Minimum learning rate

# ---------------------------------------------------------------------------
# Replay buffer & training
# ---------------------------------------------------------------------------
BUFFER_SIZE = 50_000        # Maximum number of (state, policy, outcome) tuples
BATCH_SIZE = 256            # Mini-batch size per gradient step
TRAIN_STEPS_PER_EPOCH = 100 # Gradient updates performed after each epoch

# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
GAMES_PER_EPOCH = 100       # Self-play games generated before each update
TEMPERATURE_THRESHOLD = 15  # Moves before switching from explore (T=1) to exploit (T=0.1)
TEMPERATURE_EARLY = 1.0     # Temperature for moves 0..TEMPERATURE_THRESHOLD-1
TEMPERATURE_LATE = 0.1      # Temperature for moves >= TEMPERATURE_THRESHOLD

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
NUM_EPOCHS = 500
SAVE_INTERVAL = 5           # Save a checkpoint every N epochs

# ---------------------------------------------------------------------------
# Parallel training  (train_parallel.py)
# ---------------------------------------------------------------------------
# None = auto-detect (os.cpu_count() - 2, capped at 16)
NUM_WORKERS = None


def get_num_workers_parallel():
    """Return the number of CPU worker processes for parallel self-play."""
    import os
    if NUM_WORKERS is not None:
        return NUM_WORKERS
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 16))

# ---------------------------------------------------------------------------
# Curriculum learning phases  (used in get_curriculum_options)
# ---------------------------------------------------------------------------
CURRICULUM_PHASE1_END = 30          # Epochs 0–29: small endgame positions
CURRICULUM_PHASE2_END = 80          # Epochs 30–79: medium positions
CURRICULUM_PHASE1_PIECES = (2, 5)   # Random piece count range for phase 1
CURRICULUM_PHASE2_PIECES = (4, 9)   # Random piece count range for phase 2
# Epochs >= CURRICULUM_PHASE2_END: full board (options=None)
