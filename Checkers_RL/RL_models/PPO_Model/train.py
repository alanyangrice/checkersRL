import sys
sys.path.append(r"C:\Users\Alan Yang\Downloads\checkersRL\Checkers_RL")

from RL_models.checkers_env import CheckersEnv
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.PPO_Model.Memory import Memory
from checkers_game.constants import BLUE, RED
from util import get_action_index

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import random
import csv
import os

# Initialize environment and agents
env = CheckersEnv()
input_shape = (4, 8, 8)  # 4 channels, 8x8 board
n_actions = env.action_space.n

# Create Agents 1 and 2 to play checkers
agent = PPOAgent(input_shape, n_actions)

# Directory for saving models
model_dir = "C:/Users/Alan Yang/Downloads/checkersRL/Checkers_RL/RL_models/PPO_Model/PPO_saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Specify the path for the CSV file
csv_file_path = "C:/Users/Alan Yang/Downloads/checkersRL/Checkers_RL/RL_models/PPO_Model/training_progress.csv"

# Define CSV headers
headers = [
    "epoch", "average_epoch_reward", "average_episode_length", "win_rate_blue", "win_rate_red", 
    "tie_rate", "max_move_reward"
]

# Create the CSV file and write headers (only if it doesn't exist already)
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# Directory for saving game moves
detailed_csv_folder_path = "C:/Users/Alan Yang/Downloads/checkersRL/Checkers_RL/RL_models/PPO_Model/training_progress_detailed"
if not os.path.exists(detailed_csv_folder_path):
    os.makedirs(detailed_csv_folder_path)

# Define headers for the detailed CSV
detailed_headers = ["game_number", "epoch", "episode", "blue_win", "red_win", "reward", "time", "moves", "log_probs"]

# Training parameters
num_epochs = 1000  # Number of epochs for training
num_episodes = 5000  # Number of episodes per epoch
batch_size = 100  # Number of episodes per batch
save_interval = 1  # Save model every epoch

# Main training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    total_rewards, total_steps, blue_wins, red_wins, ties, max_move_reward = 0, 0, 0, 0, 0, 0
    blue_memory, red_memory = Memory(), Memory()  # Memory for the agent playing as BLUE and RED

    # Path for the detailed CSV file for this epoch
    detailed_csv_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv")

    # Create the CSV file for the epoch and write headers
    with open(detailed_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(detailed_headers)

    for episode in range(num_episodes):
        print(f"Episode: {episode + 1}")
        state = env.reset()  # Reset environment for each episode
        done = False  # Flag to check if game is over
        
        # Randomly choose which side starts first
        current_side = random.choice([BLUE, RED])
                
        episode_reward, episode_steps, max_episode_move_reward = 0, 0, 0
        first_move = True  # Track if it's the first moves of the game
        first_move_count = 0
        log_prob_list = []

        while not done:
            legal_moves = env.game.get_all_possible_moves()  # Get legal moves for the current player

            # Diversify the first move
            if first_move and epoch < 50:
                action = random.choice(range(len(legal_moves)))
                log_prob = np.log(1 / len(legal_moves)) if len(legal_moves) != 0 else 1
                first_move_count += 1

                if first_move_count > 10:
                    first_move = False
            else:
                if len(legal_moves) != 0:
                    epsilon = max(0.03, 1 - epoch / 50)
                    if random.random() < epsilon:
                        action = random.choice(range(len(legal_moves)))
                        log_prob = np.log(1 / len(legal_moves))
                    else:
                        action, log_prob, _ = agent.select_action(state, len(legal_moves))
                else:
                    action = 0
                    log_prob = 1

            # Make sure move is within range of legal_moves
            if action >= len(legal_moves) and len(legal_moves) != 0:
                action = random.choice(range(len(legal_moves)))
                log_prob = np.log(1 / len(legal_moves))

            # Step the environment
            next_state, reward, done, info = env.step(action, legal_moves)

            # Track rewards
            episode_reward += reward
            total_rewards += reward
            episode_steps += 1

            if reward > max_episode_move_reward and not done:  # Make sure max reward isn't from winning
                max_episode_move_reward = reward

                if max_episode_move_reward > max_move_reward:
                    max_move_reward = max_episode_move_reward

            # Record in memory based on the current side
            action_index = get_action_index(action)
            if current_side == BLUE:
                blue_memory.add(state, action_index, reward, log_prob, done)
            else:
                red_memory.add(state, action_index, reward, log_prob, done)
            log_prob_list.append(log_prob)

            # Track win/loss for the current side
            if done:
                winner = info["winner"]
                if winner == BLUE:
                    blue_wins += 1
                elif winner == RED:
                    red_wins += 1
                else:
                    ties += 1

            # Update state for next step
            state = next_state

            # Switch sides
            env.game.switch_turn()
        
        # Write the episodes data to the epoch's detailed CSV file
        with open(detailed_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch * num_episodes + episode + 1,
                epoch + 1,
                episode + 1,
                1 if winner == BLUE else 0,
                1 if winner == RED else 0,
                episode_reward,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(env.game.moves),
                ", ".join(log_prob_list)
            ])

        # Store episode data
        total_steps += episode_steps
        
        # Perform batch update
        if (episode + 1) % batch_size == 0:
            print(f"Updating agent with batch of {batch_size} episodes")
            agent.update(blue_memory)
            agent.update(red_memory)
            blue_memory.clear()

    # Calculate averages and other metrics after the epoch
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes
    blue_win_rate = blue_wins / num_episodes
    red_win_rate = red_wins / num_episodes
    tie_rate = ties / num_episodes

    # Write data for this epoch as a new row in the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1, avg_reward, avg_steps, blue_win_rate, red_win_rate, tie_rate, max_move_reward
        ])

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Win Rate Agent1: {blue_win_rate:.2%}, Win Rate Agent2: {red_win_rate:.2%}, Tie Rate: {tie_rate:.2%}, Max Move Reward: {max_move_reward}")

    # Save model after every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'win_rate_blue': blue_win_rate,
            'win_rate_red': red_win_rate,
        }, os.path.join(model_dir, f"agent_epoch_{epoch + 1}.pt"))

print("Training complete.")
env.close()