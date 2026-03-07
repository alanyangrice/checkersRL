"""AlphaZero-style training loop for checkers.

This module provides an alternative to the PPO training pipeline. Instead of
using PPO with reward shaping, it uses MCTS self-play to generate training data
and trains the network to match MCTS visit-count distributions (policy) and
game outcomes (value).

Usage:
    python -m RL_models.MCTS.alphazero_trainer
"""

import os
import csv
import copy
import random
import zipfile
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RL_models.checkers_env import CheckersEnv
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.MCTS.AlphaZeroNetwork import AlphaZeroNetwork
from RL_models.MCTS.WDLAlphaZeroNetwork import WDLAlphaZeroNetwork
from RL_models.MCTS import training_config as cfg
from RL_models.PPO_Model.Agent import get_device
from checkers_game.constants import BLUE, RED, NUM_ACTIONS


def _outcome_to_wdl(v):
    """Convert a scalar outcome in [-1, 1] to a [P(win), P(draw), P(loss)] array."""
    w = float(max(0.0, v))
    l = float(max(0.0, -v))
    d = max(0.0, 1.0 - w - l)
    return np.array([w, d, l], dtype=np.float32)


class AlphaZeroTrainer:
    """AlphaZero training: MCTS self-play + supervised network updates.

    Training loop:
        1. Play games using MCTS (generates (state, mcts_policy, outcome) tuples).
        2. Store data in a replay buffer.
        3. Sample mini-batches and train the network:
           - Policy loss: cross-entropy between network output and MCTS visit distribution.
           - Value loss: MSE between network value and actual game outcome.
        4. Repeat.
    """

    def __init__(self, batch_size=cfg.BATCH_SIZE, buffer_size=cfg.BUFFER_SIZE,
                 device=None, use_wdl=False):
        self.device = device or get_device()
        self.batch_size = batch_size
        self.use_wdl = use_wdl
        # Temperature config is set per-game by self_play_game() via
        # cfg.get_temperature_config(epoch); these are just defaults.
        self.temperature_threshold = cfg.TEMPERATURE_THRESHOLD_FULL
        self._temp_late = cfg.TEMPERATURE_LATE_FULL

        # Network
        input_shape  = (4, 8, 8)
        NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork
        self.network = NetworkClass(input_shape, NUM_ACTIONS).to(self.device)
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
        )
        # Per-phase cosine LR schedule with warm restarts at curriculum boundaries.
        # cfg.get_lr() returns an absolute LR; LambdaLR needs a multiplier relative
        # to the initial base_lr, so we normalise by LEARNING_RATE.
        # PyTorch calls step() once during construction, so after __init__ the
        # optimizer LR is already set correctly for epoch 0.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: cfg.get_lr(epoch) / cfg.LEARNING_RATE,
        )

        # MCTS — sim count is overridden per-game by cfg.get_num_simulations(epoch)
        self.mcts = MCTSSearch(
            self.network,
            num_simulations=cfg.NUM_SIMULATIONS,
            c_puct=cfg.C_PUCT,
            dirichlet_alpha=cfg.DIRICHLET_ALPHA,
            dirichlet_epsilon=cfg.DIRICHLET_EPSILON,
            device=self.device,
        )

        # Replay buffer: stores (state, mcts_policy, outcome)
        self.replay_buffer = deque(maxlen=buffer_size)

    def self_play_game(self, curriculum_options=None, epoch=0,
                       start_board=None, start_turn=None):
        """Play one game using MCTS, collecting training data.

        Args:
            curriculum_options: Passed to env.reset() for curriculum boards.
            epoch:              Current training epoch.
            start_board:        Optional (4,8,8) absolute board state from
                                env.get_absolute_board_state() for diverse starts.
            start_turn:         Color (BLUE/RED) to move first; required when
                                start_board is provided.

        Returns:
            game_data:        list of (state, mcts_policy, player_color, mcts_q) tuples.
            abs_board_states: parallel list of absolute board states (for regret buffer).
            winner:           the game winner (BLUE, RED, or "Tie").
            num_moves:        number of full turns in the game.
            game_stats:       dict with per-game diagnostic signals.
        """
        env = CheckersEnv()
        if start_board is not None and start_turn is not None:
            env.load_absolute_board_state(start_board, start_turn)
        else:
            env.reset(options=curriculum_options)

        game_data        = []   # (state, mcts_policy, player_at_this_step, mcts_q)
        abs_board_states = []   # absolute board state at each step
        value_log        = []   # (player_color, root_value) per move
        entropy_log      = []   # float per move
        move_count       = 0
        done             = False
        info             = {}

        self.network.eval()
        self.mcts._root = None

        while not done:
            if move_count >= cfg.get_max_game_moves(epoch):
                from RL_models.MCTS.train_gpu_parallel import _adjudicate_move_cap
                info = {"winner": _adjudicate_move_cap(env)}
                break

            action_mask = env.get_action_mask()

            if action_mask.sum() == 0:
                _, _, done, _, info = env.step(0)
                break

            temperature = (
                cfg.TEMPERATURE_EARLY
                if move_count < self.temperature_threshold
                else self._temp_late
            )

            # Record the state and current player BEFORE the action
            state      = env.get_board_state()
            abs_state  = env.get_absolute_board_state()   # for regret buffer
            current_player = env.game.turn

            # Run MCTS with Dirichlet noise (self-play exploration)
            action, mcts_policy, root_value = self.mcts.select_action(
                env, temperature=temperature, add_noise=True,
                no_progress_count=env.game._no_progress_count,
            )
            mcts_q_value = self.mcts._root.q_value if self.mcts._root is not None else root_value

            # Per-step diagnostics
            eps = 1e-10
            entropy = float(-np.sum(mcts_policy * np.log(mcts_policy + eps)))
            value_log.append((current_player, root_value))
            entropy_log.append(entropy)

            # Store training data (outcome will be filled in after the game)
            game_data.append((state, mcts_policy, current_player, mcts_q_value))
            abs_board_states.append(abs_state)

            # Apply the action
            _, _, done, _, info = env.step(action)

            # Advance the cached tree root so the next search reuses statistics
            self.mcts.update_root(action)

            # Count full turns (not intermediate capture hops)
            if info.get("turn_complete", True):
                move_count += 1

        winner = info.get("winner", "Tie")

        # Compute per-game summary stats.
        # For ties neither side "won", so both averages are set to 0.
        is_decisive = winner not in ("Tie", "None")
        winner_vals = [v for p, v in value_log if is_decisive and p == winner]
        loser_vals  = [v for p, v in value_log if is_decisive and p != winner]
        game_stats = {
            "avg_root_val_winner": float(np.mean(winner_vals)) if winner_vals else 0.0,
            "avg_root_val_loser":  float(np.mean(loser_vals))  if loser_vals  else 0.0,
            "avg_policy_entropy":  float(np.mean(entropy_log)) if entropy_log  else 0.0,
            "move_sequence":       ", ".join(env.game.moves),
        }

        return game_data, abs_board_states, winner, move_count, game_stats

    def add_game_to_buffer(self, game_data, winner, regret_buffer=None):
        """Label game data with outcomes and add to the replay buffer.

        In scalar mode: stores scalar outcome (+1 win, -1 loss, 0 tie).
        In WDL mode:    stores soft-Z blended WDL [P(win), P(draw), P(loss)].

        Args:
            game_data:     list of (state, mcts_policy, player_color, mcts_q) tuples.
            winner:        game winner (BLUE, RED, or "Tie").
            regret_buffer: optional deque to populate with high-regret abs states.
                           Must be paired with abs_board_states available externally.
        """
        for state, mcts_policy, player_color, mcts_qval in game_data:
            if winner == "Tie" or winner == "None":
                scalar_outcome = cfg.CONTEMPT_VALUE
            elif winner == player_color:
                scalar_outcome = 1.0
            else:
                scalar_outcome = -1.0

            if self.use_wdl:
                game_wdl   = _outcome_to_wdl(scalar_outcome)
                mcts_wdl   = _outcome_to_wdl(float(mcts_qval))
                target     = (cfg.SOFT_Z_ALPHA * game_wdl
                              + (1.0 - cfg.SOFT_Z_ALPHA) * mcts_wdl)
            else:
                target = scalar_outcome

            self.replay_buffer.append((state, mcts_policy, target))

    def train_network(self, new_positions=None):
        """Sample from replay buffer and update the network.

        Args:
            new_positions: number of positions added this epoch (for adaptive
                           step count).  If None, uses TRAIN_STEPS_MAX.

        Returns:
            avg_policy_loss, avg_value_loss, avg_total_loss, avg_grad_norm,
            train_steps
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0, 0

        train_steps = (cfg.get_train_steps(new_positions)
                       if new_positions is not None
                       else cfg.TRAIN_STEPS_MAX)

        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        total_grad_norm = 0.0

        buffer_snapshot = list(self.replay_buffer)

        for _ in range(train_steps):
            batch = random.sample(buffer_snapshot, self.batch_size)
            states, target_policies, target_values = zip(*batch)

            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            target_policies_t = torch.FloatTensor(np.array(target_policies)).to(self.device)

            if self.use_wdl:
                tv = torch.FloatTensor(np.array(target_values)).to(self.device)  # (B,3)
                logits, wdl_pred, _ = self.network.forward_wdl(states_t)
                log_probs = torch.log_softmax(logits, dim=1)
                policy_loss = -(target_policies_t * log_probs).sum(dim=1).mean()
                value_loss  = -(tv * torch.log(wdl_pred + 1e-8)).sum(dim=1).mean()
                # Color-swap value augmentation (batch-time, value head only)
                st_aug = torch.flip(states_t, dims=[2])
                st_aug = torch.cat([st_aug[:, 2:4], st_aug[:, 0:2]], dim=1)
                tv_aug = torch.stack([tv[:, 2], tv[:, 1], tv[:, 0]], dim=1)
                _, wdl_aug, _ = self.network.forward_wdl(st_aug)
                vl_aug = -(tv_aug * torch.log(wdl_aug + 1e-8)).sum(dim=1).mean()
                value_loss = 0.5 * (value_loss + vl_aug)
            else:
                tv = torch.FloatTensor(np.array(target_values)).unsqueeze(1).to(self.device)
                logits, values = self.network(states_t)
                log_probs = torch.log_softmax(logits, dim=1)
                policy_loss = -(target_policies_t * log_probs).sum(dim=1).mean()
                value_loss  = nn.MSELoss()(values, tv)

            loss = policy_loss + cfg.VALUE_LOSS_WEIGHT * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            pre_clip_norm = nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=cfg.GRAD_CLIP_NORM
            )
            total_grad_norm += pre_clip_norm.item()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()

        n = train_steps
        return (
            total_policy_loss / n,
            total_value_loss / n,
            total_loss / n,
            total_grad_norm / n,
            train_steps,
        )

    def buffer_policy_entropy(self, sample_size=2048):
        """Return the mean Shannon entropy of MCTS policy targets in the buffer.

        Interpretation:
          High entropy (~log(NUM_LEGAL_MOVES)) → the search is not
          discriminating between moves; 100 simulations may be too few to
          produce sharp targets and the network has little to learn from them.

          Low entropy → the search is confidently directing most visits to a
          small number of moves; the policy targets are informative.

        As training progresses the entropy should decrease — the network's
        priors get better, MCTS focuses its budget more, and the targets
        become sharper.  If entropy stays high throughout training that is a
        signal to increase NUM_SIMULATIONS.
        """
        if len(self.replay_buffer) < 1:
            return 0.0
        n = min(sample_size, len(self.replay_buffer))
        sample = random.sample(self.replay_buffer, n)
        policies = np.array([s[1] for s in sample])          # (n, NUM_ACTIONS)
        # Shannon entropy H = -Σ p log p  (skip zeros to avoid log(0))
        eps = 1e-10
        entropy = -(policies * np.log(policies + eps)).sum(axis=1)
        return float(entropy.mean())

    def save_checkpoint(self, path, epoch, stats=None):
        """Save model checkpoint."""
        data = {
            "epoch": epoch,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if stats:
            data.update(stats)
        torch.save(data, path)

    def load_checkpoint(self, path):
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.mcts.network = self.network
        loaded_epoch = checkpoint.get("epoch", 0)
        # Reconstruct the scheduler at the correct epoch.  last_epoch=(loaded_epoch-1)
        # causes LambdaLR.__init__'s internal step() to advance it to loaded_epoch,
        # so the LR is correct for the first training batch after resuming.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda e: cfg.get_lr(e) / cfg.LEARNING_RATE,
            last_epoch=loaded_epoch - 1,
        )
        return loaded_epoch


def get_curriculum_options(epoch):
    """Generate guaranteed-asymmetric board options for the current curriculum phase.

    The weak side draws its piece count first, then the strong side draws from
    [weak+1, strong_max], guaranteeing a strict material advantage on every game.
    Which color is the strong side is re-rolled 50/50 each game so both BLUE and
    RED learn to play from both material situations equally.

    Phase 1: weak ∈ [1, 5],  strong ∈ [weak+1, 6]  — endgame positions
    Phase 2: weak ∈ [5, 8],  strong ∈ [weak+1, 9]  — mid-game positions (≥5 pieces/side)
    """
    if epoch < cfg.CURRICULUM_PHASE1_END:
        weak   = random.randint(1, cfg.CURRICULUM_PHASE1_WEAK_MAX)
        strong = random.randint(weak + 1, cfg.CURRICULUM_PHASE1_STRONG_MAX)
    elif epoch < cfg.CURRICULUM_PHASE2_END:
        weak   = random.randint(cfg.CURRICULUM_PHASE2_WEAK_MIN, cfg.CURRICULUM_PHASE2_WEAK_MAX)
        strong = random.randint(weak + 1, cfg.CURRICULUM_PHASE2_STRONG_MAX)
    else:
        return None

    if random.random() < 0.5:
        return {"num_blue": strong, "num_red": weak}
    return {"num_blue": weak, "num_red": strong}


def main(use_wdl=False):
    """Main AlphaZero training loop (sequential, single-process).

    Args:
        use_wdl: If True, use WDLAlphaZeroNetwork with soft-Z blending and
                 diverse starting positions (--network-type wdl).
    """
    num_epochs      = cfg.NUM_EPOCHS
    games_per_epoch = cfg.GAMES_PER_EPOCH
    batch_size      = cfg.BATCH_SIZE
    buffer_size     = cfg.BUFFER_SIZE
    save_interval   = cfg.SAVE_INTERVAL
    network_type_str = "wdl" if use_wdl else "scalar"

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    model_dir    = os.path.join(base_dir, "alphazero_checkpoints")
    detailed_dir = os.path.join(base_dir, "az_detailed_games")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "alphazero_training_progress.csv")

    trainer = AlphaZeroTrainer(
        batch_size=batch_size,
        buffer_size=buffer_size,
        use_wdl=use_wdl,
    )

    # Regret buffer for diverse starting positions
    regret_buffer = deque(maxlen=cfg.REGRET_BUFFER_SIZE)

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("az_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(model_dir, latest)
        ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        ckpt_network_type = ckpt.get("network_type", "scalar")
        if ckpt_network_type != network_type_str:
            raise RuntimeError(
                f"Checkpoint network_type='{ckpt_network_type}' does not match "
                f"--network-type '{network_type_str}'."
            )
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        print(f"Resumed from epoch {start_epoch}")
    else:
        # Write CSV header
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "games", "blue_wins", "red_wins", "ties",
                "avg_moves", "policy_loss", "value_loss", "total_loss",
                "avg_grad_norm", "policy_entropy_nats",
                "avg_root_val_winner", "avg_root_val_loser",
                "buffer_size", "time",
            ])

    detailed_headers = [
        "game_id", "epoch", "game_num", "winner", "num_moves",
        "avg_root_val_winner", "avg_root_val_loser",
        "avg_policy_entropy_nats", "move_sequence",
    ]

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = datetime.now()
        blue_wins, red_wins, ties, total_moves = 0, 0, 0, 0
        total_root_val_winner = 0.0
        total_root_val_loser  = 0.0
        total_entropy         = 0.0

        epoch_sims = cfg.get_num_simulations(epoch)
        print(f"\nEpoch {epoch + 1}/{num_epochs} -- Self-play ({games_per_epoch} games, "
              f"{epoch_sims} sims/move)...")

        # Open per-epoch detailed game CSV (zipped after the epoch)
        detailed_csv_path = os.path.join(
            detailed_dir, f"az_games_epoch_{epoch + 1}.csv"
        )
        with open(detailed_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(detailed_headers)

        # --- Self-play phase ---
        new_positions   = 0
        regret_snapshot = list(regret_buffer)   # snapshot for consistent sampling
        for game_idx in range(games_per_epoch):
            curriculum_opts = get_curriculum_options(epoch)
            trainer.mcts.num_simulations = cfg.get_num_simulations(epoch)
            trainer.mcts.c_puct = cfg.C_PUCT_WDL if use_wdl else cfg.C_PUCT
            temp_threshold, temp_late = cfg.get_temperature_config(epoch)
            trainer.temperature_threshold = temp_threshold
            trainer._temp_late = temp_late

            # Inject regret-buffer start state for diverse starting positions
            start_board = start_turn = None
            if regret_snapshot and random.random() < cfg.REGRET_SAMPLE_PROB:
                start_board, start_turn = random.choice(regret_snapshot)

            game_data, abs_board_states, winner, num_moves, game_stats = (
                trainer.self_play_game(
                    curriculum_opts, epoch=epoch,
                    start_board=start_board, start_turn=start_turn,
                )
            )
            trainer.add_game_to_buffer(game_data, winner)
            new_positions += len(game_data)

            # Populate regret buffer
            if abs_board_states:
                buf_items = list(trainer.replay_buffer)
                recent = buf_items[-len(game_data):]   # the positions just added
                for i, (state, mcts_policy, target) in enumerate(recent):
                    if i >= len(game_data):
                        break
                    _, _, player_color, mcts_qval = game_data[i]
                    if winner in ("Tie", "None"):
                        scalar_outcome = cfg.CONTEMPT_VALUE
                    elif winner == player_color:
                        scalar_outcome = 1.0
                    else:
                        scalar_outcome = -1.0
                    if abs(float(mcts_qval) - scalar_outcome) > cfg.HIGH_REGRET_THRESHOLD:
                        regret_buffer.append((abs_board_states[i], player_color))

            total_moves += num_moves
            if winner == BLUE:
                blue_wins += 1
            elif winner == RED:
                red_wins += 1
            else:
                ties += 1

            total_root_val_winner += game_stats["avg_root_val_winner"]
            total_root_val_loser  += game_stats["avg_root_val_loser"]
            total_entropy         += game_stats["avg_policy_entropy"]

            # Append one row per game to the detailed CSV
            with open(detailed_csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    epoch * games_per_epoch + game_idx + 1,
                    epoch + 1,
                    game_idx + 1,
                    winner,
                    num_moves,
                    round(game_stats["avg_root_val_winner"], 4),
                    round(game_stats["avg_root_val_loser"],  4),
                    round(game_stats["avg_policy_entropy"],  4),
                    game_stats["move_sequence"],
                ])

            if (game_idx + 1) % 10 == 0:
                print(f"  Game {game_idx + 1}/{games_per_epoch} done "
                      f"(buffer: {len(trainer.replay_buffer)})")

        # Compress and remove the per-epoch CSV to keep disk usage manageable
        epoch_zip_path = os.path.join(
            detailed_dir, f"az_games_epoch_{epoch + 1}.zip"
        )
        with zipfile.ZipFile(epoch_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(detailed_csv_path, arcname=os.path.basename(detailed_csv_path))
        os.remove(detailed_csv_path)

        avg_moves         = total_moves           / games_per_epoch if games_per_epoch else 0
        avg_root_val_winner = total_root_val_winner / games_per_epoch if games_per_epoch else 0.0
        avg_root_val_loser  = total_root_val_loser  / games_per_epoch if games_per_epoch else 0.0
        epoch_avg_entropy   = total_entropy         / games_per_epoch if games_per_epoch else 0.0

        # --- Training phase ---
        print(f"  Training on {len(trainer.replay_buffer)} positions...")
        p_loss, v_loss, t_loss, avg_grad_norm, train_steps = trainer.train_network(
            new_positions=new_positions
        )

        # Buffer-level policy entropy (sampled from the full replay buffer,
        # complementing the per-game entropy from self-play above)
        buffer_entropy = trainer.buffer_policy_entropy()

        elapsed = (datetime.now() - epoch_start).total_seconds()

        print(f"  Epoch {epoch + 1} complete in {elapsed:.0f}s")
        print(f"  Blue: {blue_wins}, Red: {red_wins}, Ties: {ties}, "
              f"Avg Moves: {avg_moves:.1f}")
        print(f"  Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f}, "
              f"Total: {t_loss:.4f}")
        print(f"  Grad Norm (pre-clip): {avg_grad_norm:.4f}  "
              f"[clip={cfg.GRAD_CLIP_NORM}  "
              f"{'BINDING' if avg_grad_norm > cfg.GRAD_CLIP_NORM * 0.9 else 'not binding'}]")
        print(f"  Buffer Policy Entropy: {buffer_entropy:.4f} nats  "
              f"(max uniform ~{np.log(8):.2f} for 8-move branching)")
        print(f"  Value calibration — winner avg: {avg_root_val_winner:+.3f}  "
              f"loser avg: {avg_root_val_loser:+.3f}  "
              f"(ideal: +1.0 / -1.0)")
        current_lr = trainer.scheduler.get_last_lr()[0]
        print(f"  LR: {current_lr:.2e}, "
              f"Buffer: {len(trainer.replay_buffer)}")

        # Log to CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, games_per_epoch, blue_wins, red_wins, ties,
                avg_moves, p_loss, v_loss, t_loss,
                avg_grad_norm, buffer_entropy,
                round(avg_root_val_winner, 4), round(avg_root_val_loser, 4),
                len(trainer.replay_buffer),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(model_dir, f"az_epoch_{epoch + 1}.pt")
            trainer.save_checkpoint(path, epoch + 1, {
                "blue_wins":    blue_wins,
                "red_wins":     red_wins,
                "ties":         ties,
                "network_type": network_type_str,   # for resume validation
            })
            print(f"  Checkpoint saved: {path}")

        # Advance the LR schedule for the next epoch
        trainer.scheduler.step()

    print("\nAlphaZero training complete.")


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(
        description="Sequential AlphaZero training for checkers"
    )
    _parser.add_argument(
        "--network-type", choices=["scalar", "wdl"], default="scalar",
        dest="network_type",
        help=(
            "scalar (default): AlphaZeroNetwork with tanh value head and MSE loss. "
            "wdl: WDLAlphaZeroNetwork with softmax WDL head, cross-entropy loss, "
            "and soft-Z blending (new run required)."
        ),
    )
    _args = _parser.parse_args()
    main(use_wdl=(_args.network_type == "wdl"))
