"""Hyperparameter configuration for AlphaZero (MCTS) training."""


# ---------------------------------------------------------------------------
# Self-play exploration (Dirichlet noise added to root priors)
# ---------------------------------------------------------------------------
# α controls the concentration of the noise distribution.
# Rule of thumb from the AlphaZero paper: α ≈ 10 / avg_branching_factor.
#   Chess (~35 moves)  → α = 0.3
#   Shogi (~70 moves)  → α = 0.15
#   Go    (~250 moves) → α = 0.03
#   Checkers (~8 moves)→ α = 10/8 ≈ 1.25
#
# With α = 0.3 (Chess value) applied to 7–8 checkers moves the noise
# distribution is highly concentrated — most mass falls on 1–2 moves and
# the other 5–6 are barely explored during self-play.  At α ≈ 1.0–1.25
# the noise is near-uniform, guaranteeing every legal move is tried.
DIRICHLET_ALPHA = 1.0
DIRICHLET_EPSILON = 0.25    # Fraction of noise mixed into root priors

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
# The original AlphaZero paper weights policy and value losses equally (1.0).
# A lower value (e.g. 0.5) under-trains the value head — since MCTS Q-values
# are averages of backed-up network values, a poorly calibrated value head
# means poor search guidance, which then produces poor policy targets.
VALUE_LOSS_WEIGHT = 1.0

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
NUM_SIMULATIONS = 400       # Simulations per move during self-play
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
BUFFER_SIZE = 500_000       # Maximum number of (state, policy, outcome) tuples
# At 100 games/epoch × ~50 moves/game ≈ 5,000 new positions per epoch.
# 50 K fills in ~10 epochs, causing the network to only see the last
# 10 epochs of data — catastrophic forgetting for a 500-epoch run.
# 500 K retains ~100 epochs of diversity (AlphaZero uses 500 K).
BATCH_SIZE = 256            # Mini-batch size per gradient step
# 200 gradient steps × 256 batch = 51,200 training samples per epoch.
# With ~8,000 new positions added per epoch and a growing 500K buffer,
# 100 steps covers only ~5% of the buffer at capacity; 200 steps doubles
# utilisation with only a ~5–8% increase in total epoch time.
TRAIN_STEPS_PER_EPOCH = 200 # Gradient updates performed after each epoch

# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
GAMES_PER_EPOCH = 100       # Self-play games generated before each update
# Temperature controls how stochastically the final move is sampled from MCTS
# visit counts.  T=1 samples proportionally (exploration); T→0 approaches argmax
# (exploitation).  Keeping T=0.5 throughout the game (rather than near-argmax
# 0.1) prevents both sides from collapsing onto the same deterministic lines and
# drawing every game.  Note: temperature only affects MOVE SELECTION — the policy
# training target is always the raw visit-count distribution regardless of T.
TEMPERATURE_THRESHOLD = 30  # Moves before switching from T_EARLY to T_LATE
TEMPERATURE_EARLY = 1.0     # Temperature for moves 0..TEMPERATURE_THRESHOLD-1
TEMPERATURE_LATE = 0.5      # Temperature for moves >= TEMPERATURE_THRESHOLD
# Maximum full turns per game before declaring a draw.  Prevents runaway
# passive games from wasting compute; 150 full turns ≈ 300 half-moves which
# is well above any realistic checkers game.
MAX_GAME_MOVES = 150

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
NUM_EPOCHS = 500
SAVE_INTERVAL = 5           # Save a checkpoint every N epochs

# ---------------------------------------------------------------------------
# Parallel training  (train_parallel.py)
# ---------------------------------------------------------------------------
# None = auto-detect (os.cpu_count() - 2, capped at 16)
NUM_WORKERS = 24


def get_num_workers_parallel():
    """Return the number of CPU worker processes for parallel self-play."""
    import os
    if NUM_WORKERS is not None:
        return NUM_WORKERS
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 24))

# ---------------------------------------------------------------------------
# Curriculum learning phases  (used in get_curriculum_options)
# ---------------------------------------------------------------------------
CURRICULUM_PHASE1_END = 0          # Epochs 0–29: small endgame positions
CURRICULUM_PHASE2_END = 0          # Epochs 30–79: medium positions
CURRICULUM_PHASE1_PIECES = (2, 5)   # Random piece count range for phase 1
CURRICULUM_PHASE2_PIECES = (4, 9)   # Random piece count range for phase 2
# Epochs >= CURRICULUM_PHASE2_END: full board (options=None)
