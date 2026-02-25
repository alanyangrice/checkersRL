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
# The AlphaZero paper weights policy and value losses equally (1.0), but their
# training starts with a much stronger initial value signal.  In the previous
# run, policy loss was ~1.0 while value loss was ~0.31 — the value head
# received 3x less gradient.  Combined with 31% of samples having outcome=0
# (ties) that produce zero value gradient, the value head stagnated.
#
# Weight 2.0 roughly triples the effective value gradient, which should break
# the "predict zero → get ties → stay at zero" feedback loop.  Once the value
# head improves, MCTS Q-values become informative, which sharpens policy
# targets — a virtuous cycle.
VALUE_LOSS_WEIGHT = 2.0

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
NUM_SIMULATIONS = 400       # Simulations per move during self-play (full board)
NUM_SIMULATIONS_CURRICULUM = 150  # Simulations during curriculum phases (smaller boards)
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

# Tie outcome labeling:
# In the previous run, 31% of games ended in ties.  All those positions were
# labeled outcome=0.0, which produces zero gradient for the value head and
# reinforces predicting near-zero.  A small negative label teaches the value
# head that ties are undesirable — both sides learn to steer toward decisive
# play.  -0.15 is mild enough not to distort the value scale but strong
# enough to provide gradient signal from the ~30% of data that was wasted.
TIE_OUTCOME_VALUE = -0.15

BUFFER_SIZE = 500_000       # Maximum number of (state, policy, outcome) tuples
# At 100 games/epoch × ~50 moves/game ≈ 5,000 new positions per epoch.
# 50 K fills in ~10 epochs, causing the network to only see the last
# 10 epochs of data — catastrophic forgetting for a 500-epoch run.
# 500 K retains ~100 epochs of diversity (AlphaZero uses 500 K).
BATCH_SIZE = 256            # Mini-batch size per gradient step
# 400 gradient steps × 256 batch = 102,400 training samples per epoch.
# Previous run used 200 steps — each position seen ~0.1 times per epoch at
# buffer capacity.  Doubling to 400 improves utilisation significantly while
# adding only ~5–10s per epoch (training is <5% of total epoch wall-clock).
# During curriculum phase 1, games are shorter (~10 moves × 100 games ≈ 1K
# new positions), so the network can iterate more over a smaller, cleaner
# dataset — exactly what the value head needs to bootstrap.
TRAIN_STEPS_PER_EPOCH = 400 # Gradient updates performed after each epoch

# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
GAMES_PER_EPOCH = 100       # Self-play games generated before each update
# Temperature controls how stochastically the final move is sampled from MCTS
# visit counts.  T=1 samples proportionally (exploration); T→0 approaches argmax
# (exploitation).  The policy TRAINING TARGET is always the raw visit-count
# distribution regardless of T, so low T doesn't reduce target diversity —
# it only makes the played games more decisive, reducing the tie rate and
# producing stronger win/loss labels for the value head.
#
# Previous run used T_LATE=0.5 and saw a 31% tie rate with flat value learning.
# T_LATE=0.1 is near-greedy: a move with 2x visits has ~(2^10)x selection
# probability, almost always picking the most-visited move.  Dirichlet noise
# at the root still guarantees every legal move is explored *within MCTS*,
# so the training targets remain diverse even with greedy play.
TEMPERATURE_THRESHOLD = 15  # Moves before switching from T_EARLY to T_LATE
TEMPERATURE_EARLY = 1.0     # Temperature for moves 0..TEMPERATURE_THRESHOLD-1
TEMPERATURE_LATE = 0.1      # Temperature for moves >= TEMPERATURE_THRESHOLD
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
# Phase 1: Simple endgames (2–5 pieces/side).  Games are short (5–15 moves),
#   nearly always decisive, and teach the value head to distinguish won from
#   lost positions.  This breaks the "predict zero → get ties → stay at zero"
#   feedback loop that plagued the full-board-from-scratch run.
# Phase 2: Mid-game complexity (4–9 pieces/side).  Teaches capturing chains,
#   king play, and multi-piece tactics before tackling the full opening.
# Phase 3 (epoch >= PHASE2_END): Full 12v12 standard board.
CURRICULUM_PHASE1_END = 40         # Epochs 0–39: small endgame positions
CURRICULUM_PHASE2_END = 100        # Epochs 40–99: medium positions
CURRICULUM_PHASE1_PIECES = (2, 5)   # Random piece count range for phase 1
CURRICULUM_PHASE2_PIECES = (4, 9)   # Random piece count range for phase 2
# Epochs >= CURRICULUM_PHASE2_END: full board (options=None)
