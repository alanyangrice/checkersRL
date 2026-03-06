import math
import os
from dataclasses import dataclass

@dataclass
class MCTSConfig:
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
    DIRICHLET_ALPHA: float = 1.2
    DIRICHLET_EPSILON: float = 0.35

    # ---------------------------------------------------------------------------
    # Loss weighting
    # ---------------------------------------------------------------------------
    VALUE_LOSS_WEIGHT: float = 3.0
    MCTS_VALUE_WEIGHT: float = 1.0

    # ---------------------------------------------------------------------------
    # MCTS search
    # ---------------------------------------------------------------------------
    NUM_SIMULATIONS: int = 400
    NUM_SIMULATIONS_CURRICULUM_P1: int = 75
    NUM_SIMULATIONS_CURRICULUM_P2: int = 200
    C_PUCT: float = 1.5
    C_PUCT_WDL: float = 1.5

    # ---------------------------------------------------------------------------
    # Network & optimiser
    # ---------------------------------------------------------------------------
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0

    LR_PHASE1_MAX: float = 1e-3
    LR_PHASE1_MIN: float = 1e-5
    LR_PHASE2_MAX: float = 1e-3
    LR_PHASE2_MIN: float = 1e-5
    LR_PHASE3_MAX: float = 5e-4
    LR_PHASE3_MIN: float = 1e-6

    # ---------------------------------------------------------------------------
    # Replay buffer & training
    # ---------------------------------------------------------------------------
    BUFFER_SIZE: int = 500_000
    BATCH_SIZE: int = 256
    REPLAY_RATIO: int = 35
    TRAIN_STEPS_MIN: int = 50
    TRAIN_STEPS_MAX: int = 500

    # ---------------------------------------------------------------------------
    # Temperature schedule
    # ---------------------------------------------------------------------------
    TEMPERATURE_EARLY: float = 1.0
    TEMPERATURE_THRESHOLD_CURRICULUM: int = 15
    TEMPERATURE_THRESHOLD_FULL: int = 20
    TEMPERATURE_LATE_CURRICULUM: float = 0.5
    TEMPERATURE_LATE_FULL: float = 0.4

    # ---------------------------------------------------------------------------
    # Move caps
    # ---------------------------------------------------------------------------
    MAX_GAME_MOVES_CURRICULUM_P1: int = 250
    MAX_GAME_MOVES_CURRICULUM_P2: int = 250
    MAX_GAME_MOVES_FULL: int = 250

    # ---------------------------------------------------------------------------
    # Curriculum learning phases
    # ---------------------------------------------------------------------------
    CURRICULUM_PHASE1_END: int = 15
    CURRICULUM_PHASE2_END: int = 65
    CURRICULUM_PHASE1_WEAK_MIN: int = 1
    CURRICULUM_PHASE1_WEAK_MAX: int = 5
    CURRICULUM_PHASE1_STRONG_MAX: int = 6
    CURRICULUM_PHASE2_WEAK_MIN: int = 5
    CURRICULUM_PHASE2_WEAK_MAX: int = 9
    CURRICULUM_PHASE2_STRONG_MAX: int = 10

    # ---------------------------------------------------------------------------
    # Move-cap adjudication
    # ---------------------------------------------------------------------------
    MOVE_CAP_ADJUDICATE: bool = False
    KING_MATERIAL_VALUE: float = 1.5

    # ---------------------------------------------------------------------------
    # No-progress draw rule
    # ---------------------------------------------------------------------------
    NO_PROGRESS_DRAW_MOVES: int = 80

    # ---------------------------------------------------------------------------
    # Contempt factor
    # ---------------------------------------------------------------------------
    CONTEMPT_VALUE: float = -0.3
    CONTEMPT_MATERIAL_SCALE: float = 1.2

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    GAMES_PER_EPOCH: int = 100
    NUM_EPOCHS: int = 500
    SAVE_INTERVAL: int = 1

    # ---------------------------------------------------------------------------
    # Parallel training
    # ---------------------------------------------------------------------------
    NUM_WORKERS: int = 24

    # ---------------------------------------------------------------------------
    # Evaluation & gating
    # ---------------------------------------------------------------------------
    EVAL_INTERVAL: int = 5
    EVAL_GAMES_GATE: int = 50
    EVAL_SIMULATIONS: int = 400
    GATE_THRESHOLD: float = 0.55
    GATE_ENABLED: bool = True

    # ---------------------------------------------------------------------------
    # Soft-Z value blending
    # ---------------------------------------------------------------------------
    SOFT_Z_ALPHA: float = 1.0

    # ---------------------------------------------------------------------------
    # Regret buffer
    # ---------------------------------------------------------------------------
    REGRET_SAMPLE_PROB: float = 0.20
    REGRET_BUFFER_SIZE: int = 5000
    HIGH_REGRET_THRESHOLD: float = 0.5


    def get_lr(self, epoch):
        if epoch < self.CURRICULUM_PHASE1_END:
            t = epoch / self.CURRICULUM_PHASE1_END
            return self.LR_PHASE1_MIN + 0.5 * (self.LR_PHASE1_MAX - self.LR_PHASE1_MIN) * (
                1 + math.cos(math.pi * t)
            )
        elif epoch < self.CURRICULUM_PHASE2_END:
            phase_len = self.CURRICULUM_PHASE2_END - self.CURRICULUM_PHASE1_END
            t = (epoch - self.CURRICULUM_PHASE1_END) / phase_len
            return self.LR_PHASE2_MIN + 0.5 * (self.LR_PHASE2_MAX - self.LR_PHASE2_MIN) * (
                1 + math.cos(math.pi * t)
            )
        else:
            phase_len = self.NUM_EPOCHS - self.CURRICULUM_PHASE2_END
            t = (epoch - self.CURRICULUM_PHASE2_END) / phase_len
            return self.LR_PHASE3_MIN + 0.5 * (self.LR_PHASE3_MAX - self.LR_PHASE3_MIN) * (
                1 + math.cos(math.pi * t)
            )


    def get_train_steps(self, new_positions):
        raw = int(self.REPLAY_RATIO * new_positions / self.BATCH_SIZE)
        return max(self.TRAIN_STEPS_MIN, min(raw, self.TRAIN_STEPS_MAX))


    def get_max_game_moves(self, epoch):
        if epoch < self.CURRICULUM_PHASE1_END:
            return self.MAX_GAME_MOVES_CURRICULUM_P1
        elif epoch < self.CURRICULUM_PHASE2_END:
            return self.MAX_GAME_MOVES_CURRICULUM_P2
        return self.MAX_GAME_MOVES_FULL


    def get_temperature_config(self, epoch):
        if epoch < self.CURRICULUM_PHASE2_END:
            return self.TEMPERATURE_THRESHOLD_CURRICULUM, self.TEMPERATURE_LATE_CURRICULUM
        return self.TEMPERATURE_THRESHOLD_FULL, self.TEMPERATURE_LATE_FULL


    def get_num_simulations(self, epoch):
        if epoch < self.CURRICULUM_PHASE1_END:
            return self.NUM_SIMULATIONS_CURRICULUM_P1
        elif epoch < self.CURRICULUM_PHASE2_END:
            return self.NUM_SIMULATIONS_CURRICULUM_P2
        return self.NUM_SIMULATIONS


    def get_contempt_value(self, my_material, opp_material):
        total = my_material + opp_material
        if total <= 0:
            return self.CONTEMPT_VALUE
        my_share = my_material / total
        extra = self.CONTEMPT_MATERIAL_SCALE * max(0.0, my_share - 0.5)
        raw = -(abs(self.CONTEMPT_VALUE) + extra)
        return max(raw, -0.95)


    def get_num_workers_parallel(self):
        import os
        if self.NUM_WORKERS is not None:
            return self.NUM_WORKERS
        cpu = os.cpu_count() or 4
        return max(1, min(cpu - 2, 24))