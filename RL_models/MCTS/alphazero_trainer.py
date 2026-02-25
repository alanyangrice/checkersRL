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
from RL_models.MCTS import training_config as cfg
from RL_models.PPO_Model.Agent import get_device
from checkers_game.constants import BLUE, RED, NUM_ACTIONS


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

    def __init__(
        self,
        num_simulations=cfg.NUM_SIMULATIONS,
        c_puct=cfg.C_PUCT,
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        buffer_size=cfg.BUFFER_SIZE,
        batch_size=cfg.BATCH_SIZE,
        train_steps_per_epoch=cfg.TRAIN_STEPS_PER_EPOCH,
        temperature_threshold=cfg.TEMPERATURE_THRESHOLD,
        device=None,
    ):
        self.device = device or get_device()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.train_steps_per_epoch = train_steps_per_epoch
        self.temperature_threshold = temperature_threshold

        # Network
        input_shape = (4, 8, 8)
        self.network = AlphaZeroNetwork(input_shape, NUM_ACTIONS).to(self.device)
        # AdamW applies weight decay decoupled from the gradient update,
        # which is correct regularisation for adaptive-moment optimisers.
        # torch.optim.Adam with weight_decay adds λ·θ to the gradient *before*
        # the adaptive scaling, making the effective decay vary per-parameter.
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.LR_T_MAX, eta_min=cfg.LR_ETA_MIN
        )

        # MCTS
        self.mcts = MCTSSearch(
            self.network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=cfg.DIRICHLET_ALPHA,
            dirichlet_epsilon=cfg.DIRICHLET_EPSILON,
            device=self.device,
        )

        # Replay buffer: stores (state, mcts_policy, outcome)
        self.replay_buffer = deque(maxlen=buffer_size)

    def self_play_game(self, curriculum_options=None):
        """Play one game using MCTS, collecting training data.

        Returns:
            game_data:  list of (state, mcts_policy, player_color) tuples.
            winner:     the game winner (BLUE, RED, or "Tie").
            num_moves:  number of full turns in the game.
            game_stats: dict with per-game diagnostic signals:
                avg_root_val_winner  — mean root value for moves played by the
                                       winning side (should approach +1 as
                                       training improves).
                avg_root_val_loser   — mean root value for moves played by the
                                       losing side (should approach -1).
                avg_policy_entropy   — mean Shannon entropy of MCTS visit
                                       distributions (nats).  Should decrease
                                       as the network's priors sharpen.
                move_sequence        — human-readable move history string.
        """
        env = CheckersEnv()
        env.reset(options=curriculum_options)

        game_data   = []   # (state, mcts_policy, player_at_this_step)
        value_log   = []   # (player_color, root_value) per move
        entropy_log = []   # float per move
        move_count  = 0
        done        = False
        info        = {}

        self.network.eval()
        self.mcts._root = None  # start each game with a fresh search tree
        self.mcts.num_simulations = (
            cfg.NUM_SIMULATIONS_CURRICULUM
            if curriculum_options is not None
            else cfg.NUM_SIMULATIONS
        )

        while not done:
            if move_count >= cfg.MAX_GAME_MOVES:
                info = {"winner": "Tie"}
                break

            action_mask = env.get_action_mask()

            if action_mask.sum() == 0:
                # No legal moves -- game over
                _, _, done, _, info = env.step(0)
                break

            temperature = (
                cfg.TEMPERATURE_EARLY
                if move_count < self.temperature_threshold
                else cfg.TEMPERATURE_LATE
            )

            # Record the state and current player BEFORE the action
            state = env.get_board_state()
            current_player = env.game.turn

            # Run MCTS with Dirichlet noise (self-play exploration)
            action, mcts_policy, root_value = self.mcts.select_action(
                env, temperature=temperature, add_noise=True
            )

            # Per-step diagnostics
            eps = 1e-10
            entropy = float(-np.sum(mcts_policy * np.log(mcts_policy + eps)))
            value_log.append((current_player, root_value))
            entropy_log.append(entropy)

            # Store training data (outcome will be filled in after the game)
            game_data.append((state, mcts_policy, current_player))

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

        return game_data, winner, move_count, game_stats

    def add_game_to_buffer(self, game_data, winner):
        """Label game data with outcomes and add to the replay buffer.

        Each position is labeled with:
            +1   if the player at that position won
            -1   if the player at that position lost
            cfg.TIE_OUTCOME_VALUE  if the game was a tie (slightly negative
                 to give the value head gradient signal from drawn games)
        """
        for state, mcts_policy, player_color in game_data:
            if winner == "Tie" or winner == "None":
                outcome = cfg.TIE_OUTCOME_VALUE
            elif winner == player_color:
                outcome = 1.0
            else:
                outcome = -1.0

            self.replay_buffer.append((state, mcts_policy, outcome))

    def train_network(self):
        """Sample from replay buffer and update the network.

        Returns:
            avg_policy_loss, avg_value_loss, avg_total_loss, avg_grad_norm
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0

        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        total_grad_norm = 0.0

        # Snapshot deque → list once so random.sample uses O(1) index access
        # instead of O(n) deque pointer walks for each of the batch items.
        buffer_snapshot = list(self.replay_buffer)

        for _ in range(self.train_steps_per_epoch):
            # Sample a mini-batch
            batch = random.sample(buffer_snapshot, self.batch_size)
            states, target_policies, target_values = zip(*batch)

            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            target_policies_t = torch.FloatTensor(np.array(target_policies)).to(self.device)
            target_values_t = torch.FloatTensor(np.array(target_values)).unsqueeze(1).to(self.device)

            # Forward pass
            logits, values = self.network(states_t)

            # Policy loss: cross-entropy with MCTS visit distribution
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(target_policies_t * log_probs).sum(dim=1).mean()

            # Value loss: MSE between predicted and actual outcome
            value_loss = nn.MSELoss()(values, target_values_t)

            # Total loss
            loss = policy_loss + cfg.VALUE_LOSS_WEIGHT * value_loss

            self.optimizer.zero_grad()
            loss.backward()

            # Measure pre-clip gradient norm to check whether the clip is binding.
            # If avg_grad_norm << GRAD_CLIP_NORM the clip is a no-op; if it is
            # close to or above it, clipping is actively changing the update.
            pre_clip_norm = nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=cfg.GRAD_CLIP_NORM
            )
            total_grad_norm += pre_clip_norm.item()

            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()

        n = self.train_steps_per_epoch
        return (
            total_policy_loss / n,
            total_value_loss / n,
            total_loss / n,
            total_grad_norm / n,
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
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if stats:
            data.update(stats)
        torch.save(data, path)

    def load_checkpoint(self, path):
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.mcts.network = self.network
        return checkpoint.get("epoch", 0)


def get_curriculum_options(epoch):
    """Curriculum phases for AlphaZero training."""
    if epoch < cfg.CURRICULUM_PHASE1_END:
        lo, hi = cfg.CURRICULUM_PHASE1_PIECES
        return {"num_pieces": random.randint(lo, hi)}
    elif epoch < cfg.CURRICULUM_PHASE2_END:
        lo, hi = cfg.CURRICULUM_PHASE2_PIECES
        return {"num_pieces": random.randint(lo, hi)}
    else:
        return None


def main():
    """Main AlphaZero training loop."""
    num_epochs = cfg.NUM_EPOCHS
    games_per_epoch = cfg.GAMES_PER_EPOCH
    num_simulations = cfg.NUM_SIMULATIONS
    batch_size = cfg.BATCH_SIZE
    train_steps = cfg.TRAIN_STEPS_PER_EPOCH
    buffer_size = cfg.BUFFER_SIZE
    save_interval = cfg.SAVE_INTERVAL

    # Directories
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    model_dir    = os.path.join(base_dir, "alphazero_checkpoints")
    detailed_dir = os.path.join(base_dir, "az_detailed_games")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "alphazero_training_progress.csv")

    # Initialize trainer
    trainer = AlphaZeroTrainer(
        num_simulations=num_simulations,
        batch_size=batch_size,
        train_steps_per_epoch=train_steps,
        buffer_size=buffer_size,
    )

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("az_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(model_dir, latest)
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

        print(f"\nEpoch {epoch + 1}/{num_epochs} -- Self-play ({games_per_epoch} games, "
              f"{num_simulations} sims/move)...")

        # Open per-epoch detailed game CSV (zipped after the epoch)
        detailed_csv_path = os.path.join(
            detailed_dir, f"az_games_epoch_{epoch + 1}.csv"
        )
        with open(detailed_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(detailed_headers)

        # --- Self-play phase ---
        for game_idx in range(games_per_epoch):
            curriculum_opts = get_curriculum_options(epoch)
            game_data, winner, num_moves, game_stats = trainer.self_play_game(
                curriculum_opts
            )
            trainer.add_game_to_buffer(game_data, winner)

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
        p_loss, v_loss, t_loss, avg_grad_norm = trainer.train_network()
        trainer.scheduler.step()

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
        print(f"  LR: {trainer.scheduler.get_last_lr()[0]:.2e}, "
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
                "blue_wins": blue_wins,
                "red_wins": red_wins,
                "ties": ties,
            })
            print(f"  Checkpoint saved: {path}")

    print("\nAlphaZero training complete.")


if __name__ == "__main__":
    main()
