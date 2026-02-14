import os
import csv
import random
import zipfile
from datetime import datetime

import torch
import numpy as np

from RL_models.checkers_env import CheckersEnv
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.PPO_Model.Memory import Memory
from checkers_game.constants import BLUE, RED


def zip_csv_file(csv_file_path, zip_file_path):
    """Compress and remove the CSV file."""
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, arcname=os.path.basename(csv_file_path))
    os.remove(csv_file_path)


def main():
    # Initialize environment and agent
    env = CheckersEnv()
    input_shape = (4, 8, 8)
    n_actions = env.action_space.n
    resume_training = True

    agent = PPOAgent(input_shape, n_actions)

    # Directories (relative to project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_saved_models")
    os.makedirs(model_dir, exist_ok=True)

    csv_file_path = os.path.join(base_dir, "training_progress.csv")

    headers = [
        "epoch", "average_epoch_reward", "average_episode_length", "win_rate_blue",
        "win_rate_red", "tie_rate", "max_move_reward"
    ]

    detailed_csv_folder_path = os.path.join(base_dir, "training_progress_detailed")
    os.makedirs(detailed_csv_folder_path, exist_ok=True)

    detailed_headers = [
        "game_number", "epoch", "episode", "blue_win", "red_win", "total_reward",
        "blue_reward", "red_reward", "time", "moves", "log_probs", "reward_list"
    ]

    # Training parameters
    num_epochs = 1000
    num_episodes = 2500
    batch_size = 100
    save_interval = 1
    start_epoch = 0

    if resume_training:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("agent_epoch_") and f.endswith(".pt")]
        if checkpoints:
            latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            checkpoint_path = os.path.join(model_dir, latest)
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        else:
            print("No checkpoint found. Starting from scratch.")
            resume_training = False

    if not resume_training:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch: {epoch + 1}")
        total_rewards, total_steps, blue_wins, red_wins, ties, max_move_reward = 0, 0, 0, 0, 0, 0
        blue_memory, red_memory = Memory(), Memory()

        detailed_csv_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv")

        with open(detailed_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(detailed_headers)

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False

            episode_reward, episode_steps, max_episode_move_reward = 0, 0, 0
            first_move = True
            first_move_count = 0
            log_prob_list = []
            reward_list = []
            blue_reward = []
            red_reward = []

            while not done:
                legal_moves = env.legal_moves

                if len(legal_moves) == 0:
                    action = 49
                    log_prob = 0
                    done = True
                else:
                    # Diversify the first move
                    if first_move and epoch < 50:
                        action = random.choice(range(len(legal_moves)))
                        log_prob = np.log(1 / len(legal_moves)) if len(legal_moves) != 0 else 1
                        first_move_count += 1
                        if first_move_count > 10:
                            first_move = False
                    else:
                        epsilon = max(0.03, 1 - epoch / 50)
                        if random.random() < epsilon:
                            action = random.choice(range(len(legal_moves)))
                            log_prob = np.log(1 / len(legal_moves))
                        else:
                            action, log_prob, _ = agent.select_action(state, len(legal_moves))

                # Make sure action is in range
                if action >= len(legal_moves) and len(legal_moves) != 0:
                    action = random.choice(range(len(legal_moves)))
                    log_prob = np.log(1 / len(legal_moves))

                if isinstance(log_prob, torch.Tensor):
                    log_prob = log_prob.item()

                # Step the environment
                next_state, reward, done, _, info = env.step(action)

                episode_reward += reward
                total_rewards += reward
                episode_steps += 1

                if reward > max_episode_move_reward and not done:
                    max_episode_move_reward = reward
                    if max_episode_move_reward > max_move_reward:
                        max_move_reward = max_episode_move_reward

                # Record in memory based on current side
                if env.game.turn == BLUE:
                    blue_memory.add(state, action, reward, log_prob, done)
                    blue_reward.append(reward)
                    if done:
                        red_memory.update_last_done()
                else:
                    red_memory.add(state, action, reward, log_prob, done)
                    red_reward.append(reward)
                    if done:
                        blue_memory.update_last_done()

                log_prob_list.append(log_prob)
                reward_list.append(reward)

                if done:
                    winner = info["winner"]
                    if winner == BLUE:
                        blue_wins += 1
                    elif winner == RED:
                        red_wins += 1
                    else:
                        ties += 1

                state = next_state
                env.game.switch_turn()

                # Refresh legal moves after switching turns
                if not done:
                    env._legal_moves = env.game.get_all_possible_moves()

            # Write detailed data
            with open(detailed_csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch * num_episodes + episode + 1,
                    epoch + 1,
                    episode + 1,
                    1 if winner == BLUE else 0,
                    1 if winner == RED else 0,
                    episode_reward,
                    sum(blue_reward),
                    sum(red_reward),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ", ".join(env.game.moves),
                    ", ".join([str(prob) for prob in log_prob_list]),
                    ", ".join([str(rew) for rew in reward_list])
                ])

            total_steps += episode_steps

            # Batch update
            if (episode + 1) % batch_size == 0:
                print(f"  Updating agent with batch of {batch_size} episodes")
                agent.update(blue_memory)
                agent.update(red_memory)
                blue_memory.clear()
                red_memory.clear()

        # Epoch statistics
        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes
        blue_win_rate = blue_wins / num_episodes
        red_win_rate = red_wins / num_episodes
        tie_rate = ties / num_episodes

        # Step the learning rate scheduler
        agent.step_scheduler()

        # Zip detailed CSV
        epoch_zip_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.zip")
        zip_csv_file(detailed_csv_file_path, epoch_zip_file_path)

        # Write epoch summary
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, avg_reward, avg_steps, blue_win_rate, red_win_rate, tie_rate, max_move_reward
            ])

        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Reward: {avg_reward:.2f}, "
              f"Avg Steps: {avg_steps:.2f}, Blue Win: {blue_win_rate:.2%}, "
              f"Red Win: {red_win_rate:.2%}, Tie: {tie_rate:.2%}, "
              f"Max Move Reward: {max_move_reward}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
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


if __name__ == "__main__":
    main()
