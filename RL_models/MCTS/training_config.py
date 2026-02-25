"""Hyperparameter configuration for AlphaZero (MCTS) training.

All values are calibrated for 8×8 English draughts (checkers):
  - ~7–8 legal moves per position (branching factor)
  - 40–80 full turns per decisive game
  - Mandatory captures enforce tactical lines
  - Multi-jump captures keep the same player active for several tree levels
"""


# ---------------------------------------------------------------------------
# Self-play exploration (Dirichlet noise added to root priors)
# ---------------------------------------------------------------------------
# Rule of thumb from the AlphaZero paper: α ≈ 10 / avg_branching_factor.
#   Chess  (~35 moves) → α = 0.3    Go (~250 moves) → α = 0.03
#   Checkers (~7-8 moves) → α = 10/7.5 ≈ 1.25 – 1.33
# α = 1.2 keeps noise near-uniform across the ~7-8 legal moves, ensuring
# every move is explored while not swamping strong priors entirely.
DIRICHLET_ALPHA    = 1.2
# Standard ε = 0.25 from AlphaZero.  With α = 1.2 (near-uniform noise) a
# higher ε would make the root prior close to random; 0.25 is a good balance.
DIRICHLET_EPSILON  = 0.25

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
# Weight = 2.0 during bootstrapping to fix the value head faster.
# Checkers has clear win/loss signals from short curriculum games,
# so the extra value gradient will be useful. Can reduce to 1.5 after
# the network starts showing value separation (winner avg > ±0.4).
VALUE_LOSS_WEIGHT  = 2.0

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
# 5 residual blocks × 256 channels.
# Original AlphaZero used 20 blocks for chess (much larger state space).
# Checkers: ~5×10^20 positions but only 32 playable squares → 5 blocks is
# sufficient capacity for the function complexity.  256 channels matches
# the original head design.
BACKBONE_CHANNELS  = 256
NUM_RES_BLOCKS     = 5
POLICY_HEAD_CHANNELS = 2   # 1×1 conv → 2 channels before policy linear
VALUE_HEAD_CHANNELS  = 1   # 1×1 conv → 1 channel before value linear

# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------
# Simulations per move: more sims → sharper visit distributions → better
# training targets, but higher compute per game.
#
# Phase 1 (3–6 pieces/side): game trees are tiny. 75 sims is effectively
#   near-exhaustive for most 3-piece positions and more than sufficient for
#   6-piece positions.  Going higher wastes compute on already-solved trees.
#
# Phase 2 (4–9 pieces): moderate complexity.  200 sims gives meaningful
#   search depth (3–5 ply) across the broader mid-game tree.
#
# Full board (12v12): 300 sims balances quality with throughput.
#   AlphaZero used 800 sims for chess (much larger tree), but checkers'
#   smaller branching means 300 already reaches ~4–5 ply with selective
#   deepening. 400 can be used if wall-clock is not a bottleneck.
NUM_SIMULATIONS               = 400
NUM_SIMULATIONS_CURRICULUM_P1 = 75
NUM_SIMULATIONS_CURRICULUM_P2 = 200

# PUCT exploration constant.
# U(s,a) = c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
# With uniform priors P ≈ 1/7.5 ≈ 0.13 and c_puct = 1.5, the exploration
# bonus at the first visit is ≈ 0.13 × 1.5 = 0.20, which meaningfully
# competes with Q values in [-1, 1] throughout training.  Too low
# (< 1.0) and search collapses onto the top prior early; too high (> 3.0)
# and random-looking play dominates early training.
C_PUCT             = 1.5

# ---------------------------------------------------------------------------
# Network & optimiser
# ---------------------------------------------------------------------------
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
GRAD_CLIP_NORM     = 1.0

# ---------------------------------------------------------------------------
# LR scheduler (CosineAnnealingLR)
# ---------------------------------------------------------------------------
LR_T_MAX           = 500   # Anneal over full training run
LR_ETA_MIN         = 1e-6

# ---------------------------------------------------------------------------
# Replay buffer & training
# ---------------------------------------------------------------------------
# 500K positions at ~70 positions/game × 100 games/epoch = 7K positions/epoch
# → retains ~71 epochs of full-board data.  During curriculum phase 1
# (~1–2K positions/epoch) this retains 250–500 epochs — enough diversity.
BUFFER_SIZE        = 500_000
BATCH_SIZE         = 256

# Adaptive training steps: scale gradient updates with new data volume.
#   steps = clip(REPLAY_RATIO × new_positions / BATCH_SIZE,
#                TRAIN_STEPS_MIN, TRAIN_STEPS_MAX)
#
# REPLAY_RATIO = 20: each new position is trained on ~20 times on average.
#   Typical AlphaZero replays each position 4–8 times; 20 is higher but
#   justified for curriculum where positions are short and decisive (high
#   signal-to-noise).  Reduce if you see overfitting signs (eval score
#   drops while training loss keeps falling).
#
# Example:  Phase 1, 100 games × ~15 moves = 1 500 new positions
#           → steps = clip(20×1500/256, 50, 300) = clip(117, 50, 300) = 117
#           Phase full, 100 games × ~70 moves = 7 000 new positions
#           → steps = clip(20×7000/256, 50, 300) = clip(547, 50, 300) = 300
REPLAY_RATIO       = 20
TRAIN_STEPS_MIN    = 50
TRAIN_STEPS_MAX    = 300


def get_train_steps(new_positions):
    """Compute adaptive gradient steps from fresh data volume."""
    raw = int(REPLAY_RATIO * new_positions / BATCH_SIZE)
    return max(TRAIN_STEPS_MIN, min(raw, TRAIN_STEPS_MAX))


# ---------------------------------------------------------------------------
# Temperature schedule (affects move selection, NOT training targets)
# ---------------------------------------------------------------------------
# The policy training target is always the raw visit-count distribution
# regardless of temperature.  Temperature only governs which board positions
# appear in self-play, i.e. state diversity.
#
# Strategy:
#   T=1 (proportional sampling) for opening moves → maximise state diversity.
#   T=LATE (near-greedy) after the threshold → favour decisive outcomes and
#       stabilise the late-game training signal.
#
# Curriculum phases have short games (10–30 moves), so:
#   • Use a lower threshold (switch to near-greedy after move 8) to keep
#     most of the game decisive.
#   • T_LATE = 0.2 is near-greedy but retains a tiny bit of variation so
#     not every game following the same endgame line.
#
# Full board (50–80 move games):
#   • Threshold = 20 keeps ~25–40% of moves fully exploratory, matching
#     the AlphaZero chess proportion (30 / ~80 moves ≈ 38%).
#   • T_LATE = 0.4: two moves with 2x visit difference have 2^2.5 ≈ 5.7x
#     selection ratio.  Meaningfully focused but not deterministic —
#     prevents both sides from repeating the same opening every game.
TEMPERATURE_EARLY                 = 1.0
TEMPERATURE_THRESHOLD_CURRICULUM  = 8
TEMPERATURE_THRESHOLD_FULL        = 20
TEMPERATURE_LATE_CURRICULUM       = 0.2
TEMPERATURE_LATE_FULL             = 0.4

# ---------------------------------------------------------------------------
# Move caps (phase-dependent)
# ---------------------------------------------------------------------------
# Cap prevents runaway passive games; adjudication produces a winner.
#
# Curriculum phase 1 (3–6 pieces): decisive games expected in 8–20 moves.
#   Cap at 40 — generous enough not to truncate genuine long endgame fights,
#   tight enough to kill truly passive positions quickly.
#
# Curriculum phase 2 (4–9 pieces): 15–40 moves expected.
#   Cap at 80.
#
# Full board (12v12): 40–80 moves; cap at 150 (standard for tournament play).
MAX_GAME_MOVES_CURRICULUM_P1 = 40
MAX_GAME_MOVES_CURRICULUM_P2 = 80
MAX_GAME_MOVES_FULL          = 150


def get_max_game_moves(epoch):
    """Return the move cap for the given training epoch."""
    if epoch < CURRICULUM_PHASE1_END:
        return MAX_GAME_MOVES_CURRICULUM_P1
    elif epoch < CURRICULUM_PHASE2_END:
        return MAX_GAME_MOVES_CURRICULUM_P2
    return MAX_GAME_MOVES_FULL


# ---------------------------------------------------------------------------
# Curriculum learning phases
# ---------------------------------------------------------------------------
CURRICULUM_PHASE1_END    = 15   # Epochs 0–14:  3–6 pieces/side
CURRICULUM_PHASE2_END    = 50   # Epochs 15–49: 4–9 pieces/side
CURRICULUM_PHASE1_PIECES = (3, 6)
CURRICULUM_PHASE2_PIECES = (4, 9)


def get_temperature_config(epoch):
    """Return (threshold, late_temperature) for the given epoch."""
    if epoch < CURRICULUM_PHASE2_END:   # covers both curriculum phases
        return TEMPERATURE_THRESHOLD_CURRICULUM, TEMPERATURE_LATE_CURRICULUM
    return TEMPERATURE_THRESHOLD_FULL, TEMPERATURE_LATE_FULL


def get_num_simulations(epoch):
    """Return MCTS simulations per move for the given epoch."""
    if epoch < CURRICULUM_PHASE1_END:
        return NUM_SIMULATIONS_CURRICULUM_P1
    elif epoch < CURRICULUM_PHASE2_END:
        return NUM_SIMULATIONS_CURRICULUM_P2
    return NUM_SIMULATIONS


# ---------------------------------------------------------------------------
# Move-cap adjudication
# ---------------------------------------------------------------------------
# Zero-sum: winner = side with more material; equal → Tie.
# Kings count as 1.5 regular pieces (standard checkers valuation).
MOVE_CAP_ADJUDICATE  = True
KING_MATERIAL_VALUE  = 1.5

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
GAMES_PER_EPOCH = 100
NUM_EPOCHS      = 500
SAVE_INTERVAL   = 1     # Save every 1 epochs (500 checkpoints total)

# ---------------------------------------------------------------------------
# Parallel training
# ---------------------------------------------------------------------------
NUM_WORKERS = 24


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
EVAL_INTERVAL      = 5     # Run evaluation every N epochs
EVAL_GAMES_RANDOM  = 50    # Games vs random (25 as BLUE, 25 as RED)
EVAL_GAMES_GATE    = 50    # Gating games (25 as BLUE, 25 as RED)
EVAL_SIMULATIONS   = 100   # MCTS sims per move during eval
GATE_THRESHOLD     = 0.55  # score = (wins + 0.5*ties) / games
GATE_ENABLED       = True
