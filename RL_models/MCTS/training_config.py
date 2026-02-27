"""Hyperparameter configuration for AlphaZero (MCTS) training.

All values are calibrated for 8×8 English draughts (checkers):
  - ~7–8 legal moves per position (branching factor)
  - 40–80 full turns per decisive game
  - Mandatory captures enforce tactical lines
  - Multi-jump captures keep the same player active for several tree levels
"""

import math


# ---------------------------------------------------------------------------
# Self-play exploration (Dirichlet noise added to root priors)
# ---------------------------------------------------------------------------
# Rule of thumb from the AlphaZero paper: α ≈ 10 / avg_branching_factor.
#   Chess  (~35 moves) → α = 0.3    Go (~250 moves) → α = 0.03
#   Checkers (~7-8 moves) → α = 10/7.5 ≈ 1.25 – 1.33
# α = 1.2 keeps noise near-uniform across the ~7-8 legal moves, ensuring
# every move is explored while not swamping strong priors entirely.
DIRICHLET_ALPHA    = 1.2
# ε = 0.35: increased from 0.25 to counteract policy collapse onto a single
# opening move.  With α = 1.2 and ~7 legal moves, each move gets ~14% noise
# so ε = 0.35 adds meaningful exploration without drowning the learned prior.
DIRICHLET_EPSILON  = 0.35

# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------
# Weight = 3.0: value component is only ~7% of total loss at 2.0, which
# is too weak to calibrate the value head on harder Phase 2/3 positions.
# At 3.0 it becomes ~11% — enough to drive calibration without harming policy.
VALUE_LOSS_WEIGHT  = 3.0

# Weight applied to the MCTS-Q auxiliary value loss (see Fix 1).
# The primary value target is the final game outcome (±1/0).
# For non-natural terminations (cap, repetition), the MCTS Q-value replaces
# the outcome entirely (MCTS_VALUE_WEIGHT is not used as a blend weight here).
MCTS_VALUE_WEIGHT  = 1.0   # reserved for future blending; currently a flag

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
POLICY_HEAD_CHANNELS = 2    # 1×1 conv → 2 channels before policy linear
# 32 channels (vs previous 1): the 1-channel bottleneck compressed 256×8×8
# backbone features down to 64 scalars before the hidden layer — far too
# little capacity for complex positional evaluation.  32 channels gives 2048
# features, matching AlphaZero's original value head design intent.
# NOTE: changing this requires training from scratch (checkpoint incompatible).
VALUE_HEAD_CHANNELS  = 32  # 1×1 conv → 32 channels (2048 features) before value FC

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
# Doubled at every phase so MCTS can see deeper into game trees:
#   Phase 1 (3–6 pieces, ~35 move games): 150 sims ≈ 6-ply effective depth
#   Phase 2 (4–9 pieces, ~55 move games): 400 sims ≈ 7-8-ply
#   Phase 3 (12v12, 80+ move games):      800 sims ≈ 9-10-ply
# Epoch time roughly doubles vs previous values at each phase.
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
LEARNING_RATE      = 1e-3   # initial value; overridden per-epoch by get_lr()
WEIGHT_DECAY       = 1e-4
GRAD_CLIP_NORM     = 1.0

# ---------------------------------------------------------------------------
# Per-phase LR schedule (cosine with warm restart at each curriculum boundary)
# ---------------------------------------------------------------------------
# Each curriculum phase gets its own full cosine cycle so the model can
# rapidly re-adapt when it encounters harder positions.
#
#  Phase 1 (epochs 0–14):   1e-3  → 1e-5  over 15 epochs
#  Phase 2 (epochs 15–49):  1e-3  → 1e-5  over 35 epochs  ← warm restart
#  Phase 3 (epochs 50–499): 5e-4  → 1e-6  over 450 epochs ← warm restart
#                            (lower peak: model is mature, less re-exploration)
LR_PHASE1_MAX = 1e-3
LR_PHASE1_MIN = 1e-5
LR_PHASE2_MAX = 1e-3
LR_PHASE2_MIN = 1e-5
LR_PHASE3_MAX = 5e-4
LR_PHASE3_MIN = 1e-6


def get_lr(epoch):
    """Return the learning rate for the given epoch (0-indexed).

    Uses a cosine schedule that warm-restarts at each curriculum phase
    boundary, giving each phase its own full decay cycle.
    """
    if epoch < CURRICULUM_PHASE1_END:
        t = epoch / CURRICULUM_PHASE1_END
        return LR_PHASE1_MIN + 0.5 * (LR_PHASE1_MAX - LR_PHASE1_MIN) * (
            1 + math.cos(math.pi * t)
        )
    elif epoch < CURRICULUM_PHASE2_END:
        phase_len = CURRICULUM_PHASE2_END - CURRICULUM_PHASE1_END
        t = (epoch - CURRICULUM_PHASE1_END) / phase_len
        return LR_PHASE2_MIN + 0.5 * (LR_PHASE2_MAX - LR_PHASE2_MIN) * (
            1 + math.cos(math.pi * t)
        )
    else:
        phase_len = NUM_EPOCHS - CURRICULUM_PHASE2_END
        t = (epoch - CURRICULUM_PHASE2_END) / phase_len
        return LR_PHASE3_MIN + 0.5 * (LR_PHASE3_MAX - LR_PHASE3_MIN) * (
            1 + math.cos(math.pi * t)
        )

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
# Curriculum phases have short games (~60–80 moves with 3–6 pieces):
#   • Threshold = 15 keeps the first ~20% of a game fully exploratory.
#     The original threshold of 8 was too low — it made 88%+ of moves
#     greedy, generating homogeneous game trajectories.
#   • T_LATE = 0.5 (was 0.2): with T=0.2 the exponent is 5, collapsing even
#     mild visit preferences to near-delta-function actions and preventing
#     diverse endgame exploration.  T=0.5 (exponent 2) meaningfully focuses
#     the distribution on the best move while still sampling occasionally from
#     second-best lines, giving the replay buffer richer state coverage.
#
#   NOTE: Temperature here only governs ACTION SELECTION (which states appear
#   in the replay buffer).  The policy training TARGET is always the raw
#   N/ΣN visit distribution, independent of temperature — see train loop.
#
# Full board (50–80 move games):
#   • Threshold = 20 keeps ~25–40% of moves fully exploratory, matching
#     the AlphaZero chess proportion (30 / ~80 moves ≈ 38%).
#   • T_LATE = 0.4: two moves with 2x visit difference have 2^2.5 ≈ 5.7x
#     selection ratio.  Meaningfully focused but not deterministic —
#     prevents both sides from repeating the same opening every game.
TEMPERATURE_EARLY                 = 1.0
TEMPERATURE_THRESHOLD_CURRICULUM  = 15
TEMPERATURE_THRESHOLD_FULL        = 20
TEMPERATURE_LATE_CURRICULUM       = 0.5
TEMPERATURE_LATE_FULL             = 0.4

# ---------------------------------------------------------------------------
# Move caps (all phases)
# ---------------------------------------------------------------------------
# The 40-move no-progress rule (NO_PROGRESS_DRAW_MOVES) now terminates all
# passive/stalling games.  The cap's only remaining purpose is as a hard
# safety net for games still making *active* progress (captures or promotions
# continuing) that nonetheless take a very long time to resolve.
#
# All phases use 250 because:
#   • Endgame conversion is harder with fewer pieces — a 3v1 king endgame
#     can require 15–20 moves of maneuvering between each capture, pushing
#     a decisive game well past 80 or 160 turns.
#   • The old lower phase caps were sized to prevent oscillation, which the
#     no-progress rule now handles.  They were prematurely terminating active
#     decisive games.
#   • Epoch training time is protected by the no-progress rule; stall games
#     end within 40 moves instead of running to cap.
MAX_GAME_MOVES_CURRICULUM_P1 = 250
MAX_GAME_MOVES_CURRICULUM_P2 = 250
MAX_GAME_MOVES_FULL          = 250


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
CURRICULUM_PHASE1_END    = 0   # Epochs 0–14:  3–6 pieces/side
CURRICULUM_PHASE2_END    = 15   # Epochs 15–49: 4–9 pieces/side
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
# When False (recommended): cap → Tie for BOTH MCTS simulations and training
# labels.  This is correct because the cap is a training infrastructure
# artifact — real checkers is won by capturing all pieces or blocking the
# opponent, not by having more material when time expires.  Declaring cap
# games as draws forces the network to learn actual winning play (conversion)
# rather than a stalling meta-strategy that doesn't exist in real checkers.
#
# When True: cap → material adjudication (more pieces = win).  This sounds
# informative but creates a destructive incentive in MCTS: a player ahead in
# material will search for ways to stall rather than capture, corrupting both
# the policy and value heads.
#
# Kings count as KING_MATERIAL_VALUE pieces (only used when True).
MOVE_CAP_ADJUDICATE  = False
KING_MATERIAL_VALUE  = 1.5

# ---------------------------------------------------------------------------
# No-progress draw rule
# ---------------------------------------------------------------------------
# Mirrors the WCDF (World Checkers Draughts Federation) rule: if neither
# player captures a piece within N consecutive full turns, the game is
# declared a draw.  Applied in both the outer self-play loop and inside
# NumpyCheckersEnv MCTS simulations so the two stay consistent.
#
# Standard English draughts uses 40 moves.  This also terminates the
# king-vs-king oscillation games that caused ~45 % of phase-2 games to
# reach the move cap, without requiring any curriculum change.
NO_PROGRESS_DRAW_MOVES = 40

# ---------------------------------------------------------------------------
# Contempt factor
# ---------------------------------------------------------------------------
# AlphaZero treats draws as value 0, which means a drawn position is as good
# as an unknown one.  Setting CONTEMPT_VALUE to a small negative number makes
# the agent prefer any winning attempt over a certain draw.
#
# CONTEMPT_VALUE: base contempt applied when material is approximately equal
#   at the time of the draw (repetition, no-progress, or move-cap).
#   -0.3 (was -0.1): ratio of draw signal to win signal drops from 10:1 to
#   3.3:1.  With 57% draw games flooding the buffer the old -0.1 created
#   near-zero signal for the majority of training positions.  At -0.3 draws
#   are clearly undesirable but not catastrophic — the agent learns to press
#   advantages without becoming reckless about losing.
#
# CONTEMPT_MATERIAL_SCALE: extra contempt added for the side that held a
#   material advantage at the draw.  Scales linearly with material share
#   above 50%.  The trailing side always receives only the base contempt.
#   Clamped at -0.95 in get_contempt_value() so targets stay within [-1, 1]
#   (required for MSE against tanh value head — see that function's docstring).
#
#   Examples at CONTEMPT_VALUE=-0.3, CONTEMPT_MATERIAL_SCALE=1.2:
#     6v6 draw  (share=0.50) → -0.30   (fair draw)
#     8v4 draw  (share=0.67) → -0.54   (clear failure to convert)
#     9v3 draw  (share=0.75) → -0.66   (strong advantage wasted)
#     11v1 draw (share=0.92) → -0.85   (dominant position thrown away)
#   The weaker side always receives only the base contempt (-0.30).
#
# Contempt is applied only in training labels (replay buffer), NOT in the
# MCTS terminal backup (which must stay zero-sum).  See _outcome_value().
CONTEMPT_VALUE          = -0.3
CONTEMPT_MATERIAL_SCALE = 1.2


def get_contempt_value(my_material, opp_material):
    """Variable contempt for the player at *my_material* in a drawn game.

    Returns a value in [-|CONTEMPT_VALUE|, -0.95].
    Only the side that was AHEAD in material is penalised more; the trailing
    side always receives the base contempt.

    Hard-clamped to -0.95 so value targets never exceed the tanh output
    range [-1, 1].  Pushing targets below -1.0 would saturate the value
    head (gradient → 0 at tanh → -1), destabilising training.

    Examples with CONTEMPT_VALUE=-0.3, CONTEMPT_MATERIAL_SCALE=1.2:
      6v6 draw  (share=0.50) → -0.30  (fair draw)
      7v5 draw  (share=0.58) → -0.40  (slight advantage wasted)
      8v4 draw  (share=0.67) → -0.54  (clear failure to convert)
      9v3 draw  (share=0.75) → -0.66  (strong advantage wasted)
      11v1 draw (share=0.92) → -0.85  (dominant position thrown away)
    """
    total = my_material + opp_material
    if total <= 0:
        return CONTEMPT_VALUE
    my_share = my_material / total          # 0.5 for equal, → 1.0 as I dominate
    extra = CONTEMPT_MATERIAL_SCALE * max(0.0, my_share - 0.5)
    raw = -(abs(CONTEMPT_VALUE) + extra)
    return max(raw, -0.95)                  # clamp: value targets must stay in [-1, 1]

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
EVAL_GAMES_GATE    = 50    # Gating games (25 as BLUE, 25 as RED)
EVAL_SIMULATIONS   = 100   # MCTS sims per move during eval
GATE_THRESHOLD     = 0.55  # score = (wins + 0.5*ties) / games
GATE_ENABLED       = True
