import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

from RL_models.checkers_env import CheckersEnv
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.PPO_Model.Memory import Memory
from checkers_game.constants import BLUE, RED
from util import get_action_index

import torch
import pandas as pd
import numpy as np
import random
import os

# Initialize environment and agents
env = CheckersEnv()
input_shape = (4, 8, 8)  # 4 channels, 8x8 board
n_actions = env.action_space.n

# Create Agents 1 and 2 to play checkers
agent1 = PPOAgent(input_shape, n_actions)
agent2 = PPOAgent(input_shape, n_actions)

# Track the starting agent (agent1 as BLUE initially)
agent1_side = BLUE
agent2_side = RED

# Data storage for monitoring training progress
training_data = {
    "epoch": [],
    "episode_reward": [],
    "win_rate_agent1": [],
    "win_rate_agent2": [],
    "tie_rate": [],
    "average_episode_length": []
}

# Directory for saving models
model_dir = "c:/Users/Alan Yang/Desktop/checkersRL/Checkers_RL/RL_models/PPO_Model/PPO_saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Directory for saving random games
game_info_dir = "c:/Users/Alan Yang/Desktop/checkersRL/Checkers_RL/RL_models/PPO_Model/saved_games"
if not os.path.exists(game_info_dir):
    os.makedirs(game_info_dir)

# Training parameters
num_epochs = 1000  # Number of epochs for training
num_episodes = 10000  # Number of episodes per epoch
save_interval = 1  # Save model every epoch (can adjust)

# Main training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    total_rewards, total_steps, agent1_wins, agent2_wins, tie = 0, 0, 0, 0, 0

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        state = env.reset()  # Reset environment for each episode
        done = False  # Flag to check if game is over
        memory1, memory2 = Memory(), Memory()  # Memory object for agent1 and agent2

        # Track game info if sampling condition is met
        game_info = [] if random.random() < 0.0005 else None  # Log moves for ~0.01% of games
        
        # Randomly select which agent goes first
        if random.choice([True, False]):
            current_agent, opponent_agent = agent1, agent2
            current_memory, opponent_memory = memory1, memory2
            agent1_side, agent2_side = BLUE, RED  # Agent1 is BLUE, Agent2 is RED
        else:
            current_agent, opponent_agent = agent2, agent1
            current_memory, opponent_memory = memory2, memory1
            agent1_side, agent2_side = RED, BLUE  # Agent2 is BLUE, Agent1 is RED
                
        episode_reward, episode_steps = 0, 0

        while not done:
            legal_moves = env.game.get_all_possible_moves()  # Get legal moves for the current player

            # Get action and log probabilities from the current agent
            action, log_prob, _ = current_agent.select_action(state, len(legal_moves))
            
            # Step the environment
            next_state, reward, done, info = env.step(action)

            #env.render()

            # Track rewards
            episode_reward += reward
            total_rewards += reward
            episode_steps += 1

            # Record in memory based on current agent's perspective
            action_index = get_action_index(action)
            current_memory.add(state, action_index, reward, log_prob)
            
            # Save selected games stats
            if game_info is not None:
                agent_label = "Agent 1" if current_agent == agent1 else "Agent 2"
                game_info.append({
                    "epoch": epoch,
                    "episode": episode,
                    "turn": env.game.turn,
                    "agent": agent_label,
                    "move": info["legal_moves"][action] if info["move_success"] else None,
                    "reward": reward,
                    "success": info["move_success"],
                    "winner": info["winner"]
                })

            # Track win/loss for agents
            if done:
                if info["winner"] == BLUE:
                    agent1_wins += 1 if agent1_side == BLUE else 0
                    agent2_wins += 1 if agent2_side == BLUE else 0
                elif info["winner"] == RED:
                    agent1_wins += 1 if agent1_side == RED else 0
                    agent2_wins += 1 if agent2_side == RED else 0
                else:
                    tie += 1

            # Alternate agents
            current_agent, opponent_agent = opponent_agent, current_agent
            current_memory, opponent_memory = opponent_memory, current_memory

            # Update state for next step
            state = next_state

            # Switch sides
            env.game.switch_turn()
        
        # Save game information if sampled
        if game_info:
            game_info_df = pd.DataFrame(game_info)
            game_info_df.to_csv(f"{game_info_dir}/game_info_epoch_{epoch}_episode_{episode}.csv", index=False)
        
        # Store episode data
        total_steps += episode_steps
        
        # After episode ends, update each agent using its memory
        agent1.update(memory1)
        agent2.update(memory2)
        
        # Swap sides after each episode
        agent1_side, agent2_side = agent2_side, agent1_side

    # Calculate epoch statistics
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes
    win_rate1 = agent1_wins / num_episodes
    win_rate2 = agent2_wins / num_episodes
    tie_rate = tie / num_episodes

    # Append data for this epoch
    training_data["epoch"].append(epoch)
    training_data["episode_reward"].append(avg_reward)
    training_data["win_rate_agent1"].append(win_rate1)
    training_data["win_rate_agent2"].append(win_rate2)
    training_data["tie_rate"].append(tie_rate)
    training_data["average_episode_length"].append(avg_steps)

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Reward: {avg_reward:.2f}, Win Rate Agent1: {win_rate1:.2%}, Win Rate Agent2: {win_rate2:.2%}, Tie Rate: {tie_rate:.2%}, Avg Steps: {avg_steps:.2f}")

    # Save model after every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(agent1.policy.state_dict(), os.path.join(model_dir, f"agent1_epoch_{epoch + 1}.pt"))
        torch.save(agent2.policy.state_dict(), os.path.join(model_dir, f"agent2_epoch_{epoch + 1}.pt"))

# Save training data as a CSV for later analysis
training_df = pd.DataFrame(training_data)
training_df.to_csv("training_progress.csv", index=False)

print("Training complete.")
env.close()