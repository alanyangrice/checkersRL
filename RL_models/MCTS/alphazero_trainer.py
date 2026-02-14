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
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RL_models.checkers_env import CheckersEnv
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork
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
        num_simulations=100,
        c_puct=1.5,
        lr=1e-3,
        weight_decay=1e-4,
        buffer_size=50000,
        batch_size=256,
        train_steps_per_epoch=100,
        temperature_threshold=15,
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
        self.network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-6
        )

        # MCTS
        self.mcts = MCTSSearch(
            self.network, num_simulations=num_simulations,
            c_puct=c_puct, device=self.device
        )

        # Replay buffer: stores (state, mcts_policy, outcome)
        self.replay_buffer = deque(maxlen=buffer_size)

    def self_play_game(self, curriculum_options=None):
        """Play one game using MCTS, collecting training data.

        Returns:
            game_data: list of (state, mcts_policy, player_color) tuples.
            winner: the game winner (BLUE, RED, or "Tie").
            num_moves: number of full turns in the game.
        """
        env = CheckersEnv()
        env.reset(options=curriculum_options)

        game_data = []  # (state, mcts_policy, player_at_this_step)
        move_count = 0
        done = False

        self.network.eval()

        while not done:
            action_mask = env.get_action_mask()

            if action_mask.sum() == 0:
                # No legal moves -- game over
                _, _, done, _, info = env.step(0)
                break

            # Choose temperature based on move count
            # Early game: temperature=1 (explore), later: temperature=0 (exploit)
            temperature = 1.0 if move_count < self.temperature_threshold else 0.1

            # Record the state and current player BEFORE the action
            state = env.get_board_state()
            current_player = copy.deepcopy(env.game.turn)

            # Run MCTS to get action and visit-count distribution
            action, mcts_policy = self.mcts.select_action(env, temperature=temperature)

            # Store training data (outcome will be filled in after the game)
            game_data.append((state, mcts_policy, current_player))

            # Apply the action
            _, _, done, _, info = env.step(action)

            # Count full turns (not intermediate capture hops)
            if info.get("turn_complete", True):
                move_count += 1

        # Determine the winner
        winner = info.get("winner", "Tie")

        return game_data, winner, move_count

    def add_game_to_buffer(self, game_data, winner):
        """Label game data with outcomes and add to the replay buffer.

        Each position is labeled with:
            +1 if the player at that position won
            -1 if the player at that position lost
             0 if the game was a tie
        """
        for state, mcts_policy, player_color in game_data:
            if winner == "Tie" or winner == "None":
                outcome = 0.0
            elif winner == player_color:
                outcome = 1.0
            else:
                outcome = -1.0

            self.replay_buffer.append((state, mcts_policy, outcome))

    def train_network(self):
        """Sample from replay buffer and update the network.

        Returns:
            avg_policy_loss, avg_value_loss, avg_total_loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for _ in range(self.train_steps_per_epoch):
            # Sample a mini-batch
            batch = random.sample(self.replay_buffer, self.batch_size)
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
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()

        n = self.train_steps_per_epoch
        return total_policy_loss / n, total_value_loss / n, total_loss / n

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
        # Update MCTS reference to the loaded network
        self.mcts.network = self.network
        return checkpoint.get("epoch", 0)


def get_curriculum_options(epoch):
    """Curriculum phases for AlphaZero training."""
    if epoch < 30:
        return {"num_pieces": random.randint(2, 5)}
    elif epoch < 80:
        return {"num_pieces": random.randint(4, 9)}
    else:
        return None


def main():
    """Main AlphaZero training loop."""
    # Hyperparameters
    num_epochs = 500
    games_per_epoch = 100
    num_simulations = 100
    batch_size = 256
    train_steps = 100
    buffer_size = 50000
    save_interval = 5

    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "alphazero_checkpoints")
    os.makedirs(model_dir, exist_ok=True)

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
                "buffer_size", "time"
            ])

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = datetime.now()
        blue_wins, red_wins, ties, total_moves = 0, 0, 0, 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} -- Self-play ({games_per_epoch} games, "
              f"{num_simulations} sims/move)...")

        # --- Self-play phase ---
        for game_idx in range(games_per_epoch):
            curriculum_opts = get_curriculum_options(epoch)
            game_data, winner, num_moves = trainer.self_play_game(curriculum_opts)
            trainer.add_game_to_buffer(game_data, winner)

            total_moves += num_moves
            if winner == BLUE:
                blue_wins += 1
            elif winner == RED:
                red_wins += 1
            else:
                ties += 1

            if (game_idx + 1) % 10 == 0:
                print(f"  Game {game_idx + 1}/{games_per_epoch} done "
                      f"(buffer: {len(trainer.replay_buffer)})")

        avg_moves = total_moves / games_per_epoch if games_per_epoch > 0 else 0

        # --- Training phase ---
        print(f"  Training on {len(trainer.replay_buffer)} positions...")
        p_loss, v_loss, t_loss = trainer.train_network()
        trainer.scheduler.step()

        elapsed = (datetime.now() - epoch_start).total_seconds()

        print(f"  Epoch {epoch + 1} complete in {elapsed:.0f}s")
        print(f"  Blue: {blue_wins}, Red: {red_wins}, Ties: {ties}, "
              f"Avg Moves: {avg_moves:.1f}")
        print(f"  Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f}, "
              f"Total: {t_loss:.4f}")
        print(f"  LR: {trainer.scheduler.get_last_lr()[0]:.2e}, "
              f"Buffer: {len(trainer.replay_buffer)}")

        # Log to CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, games_per_epoch, blue_wins, red_wins, ties,
                avg_moves, p_loss, v_loss, t_loss,
                len(trainer.replay_buffer),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
