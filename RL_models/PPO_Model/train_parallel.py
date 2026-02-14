import os
import csv
import random
import zipfile
from datetime import datetime
from multiprocessing import Pool, cpu_count

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


def write_detailed_csv(file_path, results, batch_start, epoch, num_games):
    """Write game details to a CSV file."""
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for game_id, result in enumerate(results, start=batch_start + 1):
            writer.writerow([
                game_id + epoch * num_games,
                result["blue_win"],
                result["red_win"],
                sum(result["rewards"].values()),
                result["rewards"]["blue"],
                result["rewards"]["red"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(result["moves"]),
                ", ".join(map(str, result["log_probs"])),
                ", ".join(map(str, result["rewards_list"]))
            ])


def play_game(n_actions, game_id, epoch, temp_model_path):
    """Simulate a single game of Checkers with training logic."""
    env = CheckersEnv()
    # Workers always run on CPU for multiprocessing
    agent = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
    agent.policy.load_state_dict(torch.load(temp_model_path, map_location='cpu', weights_only=False))

    blue_memory, red_memory = Memory(), Memory()
    state, _ = env.reset()
    done = False

    episode_reward, episode_steps, max_episode_move_reward = 0, 0, 0
    first_move, first_move_count = True, 0
    log_prob_list, reward_list, blue_reward, red_reward = [], [], [], []
    blue_win, red_win = 0, 0

    while not done:
        legal_moves = env.legal_moves

        if len(legal_moves) == 0:
            action = 49
            log_prob = 0
            done = True
        else:
            if first_move:
                action = random.choice(range(len(legal_moves)))
                log_prob = np.log(1 / len(legal_moves)) if len(legal_moves) != 0 else 1
                first_move_count += 1
                if first_move_count > 1:
                    first_move = False
            else:
                epsilon = max(0.08, 1 - epoch / 100)
                if random.random() < epsilon:
                    action = random.choice(range(len(legal_moves)))
                    log_prob = np.log(1 / len(legal_moves))
                else:
                    action, log_prob, _ = agent.select_action(state, len(legal_moves))

        if action >= len(legal_moves) and len(legal_moves) != 0:
            action = random.choice(range(len(legal_moves)))
            log_prob = np.log(1 / len(legal_moves))

        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.item()

        next_state, reward, done, _, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        if reward > max_episode_move_reward and not done:
            max_episode_move_reward = reward

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
                blue_win = 1
            elif winner == RED:
                red_win = 1

        state = next_state
        env.game.switch_turn()

        if not done:
            env._legal_moves = env.game.get_all_possible_moves()

    return {
        "blue_memory": blue_memory,
        "red_memory": red_memory,
        "rewards": {"blue": sum(blue_reward), "red": sum(red_reward)},
        "blue_win": blue_win,
        "red_win": red_win,
        "moves": env.game.moves,
        "log_probs": log_prob_list,
        "rewards_list": reward_list,
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "max_episode_move_reward": max_episode_move_reward,
    }


def train_parallel(num_epochs=1000, num_games=2500, batch_size=250, n_actions=50):
    """Parallelized training loop for the Checkers PPO agent."""
    input_shape = (4, 8, 8)
    agent = PPOAgent(input_shape, n_actions)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_saved_models_parallel")
    os.makedirs(model_dir, exist_ok=True)

    start_epoch = 0

    # Find and load latest checkpoint
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("agent_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(model_dir, latest)
        checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch: {start_epoch}")

    # CSV directories
    detailed_csv_folder_path = os.path.join(base_dir, "training_progress_detailed_parallel")
    os.makedirs(detailed_csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(base_dir, "training_progress_parallel.csv")

    if start_epoch == 0:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch", "average_epoch_reward", "average_blue_reward", "average_red_reward",
                "average_episode_length", "win_rate_blue", "win_rate_red", "tie_rate", "max_move_reward"
            ])

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch + 1}")

        # Save temporary model for workers
        temp_model_path = os.path.join(model_dir, f'temp_agent_model_epoch_{epoch + 1}.pt')
        torch.save(agent.policy.state_dict(), temp_model_path)

        num_processes = max(1, int(cpu_count() * 0.25))
        print(f"Using {num_processes} processes for parallel simulation.")

        total_rewards = {"blue": 0, "red": 0}
        blue_wins, red_wins, ties, max_move_reward, total_steps = 0, 0, 0, 0, 0

        detailed_csv_file_path = os.path.join(
            detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv"
        )

        with open(detailed_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "game_number", "blue_win", "red_win", "total_reward", "blue_reward",
                "red_reward", "time", "moves", "log_probs", "reward_list"
            ])

        for batch_start in range(0, num_games, batch_size):
            batch_end = min(batch_start + batch_size, num_games)

            with Pool(num_processes) as pool:
                results = pool.starmap(
                    play_game,
                    [(n_actions, game_id, epoch, temp_model_path) for game_id in range(batch_start, batch_end)]
                )
            write_detailed_csv(detailed_csv_file_path, results, batch_start, epoch, num_games)

            for result in results:
                total_rewards["blue"] += result["rewards"]["blue"]
                total_rewards["red"] += result["rewards"]["red"]
                total_steps += len(result["moves"])
                blue_wins += result["blue_win"]
                red_wins += result["red_win"]
                ties += 1 - (result["blue_win"] or result["red_win"])
                max_move_reward = max(max_move_reward, result["max_episode_move_reward"])

            # Batch update
            blue_memory, red_memory = Memory(), Memory()
            for result in results:
                blue_memory.extend(result["blue_memory"])
                red_memory.extend(result["red_memory"])
            agent.update(blue_memory)
            agent.update(red_memory)

        # Step the learning rate scheduler
        agent.step_scheduler()

        # Zip detailed CSV
        epoch_zip_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.zip")
        zip_csv_file(detailed_csv_file_path, epoch_zip_file_path)

        # Clean up temp model
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        # Log epoch statistics
        avg_reward = (total_rewards["blue"] + total_rewards["red"]) / num_games
        avg_blue_reward = total_rewards["blue"] / num_games
        avg_red_reward = total_rewards["red"] / num_games
        avg_steps = total_steps / num_games
        blue_win_rate = blue_wins / num_games
        red_win_rate = red_wins / num_games
        tie_rate = ties / num_games

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, avg_reward, avg_blue_reward, avg_red_reward,
                avg_steps, blue_win_rate, red_win_rate, tie_rate, max_move_reward
            ])

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
        }, os.path.join(model_dir, f"agent_epoch_{epoch + 1}.pt"))

        print(f"Epoch {epoch + 1} complete. Avg Reward: {avg_reward:.2f}, "
              f"Blue Win: {blue_win_rate:.2%}, Red Win: {red_win_rate:.2%}, "
              f"Tie: {tie_rate:.2%}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")


if __name__ == "__main__":
    train_parallel(num_epochs=1000, num_games=5000)
