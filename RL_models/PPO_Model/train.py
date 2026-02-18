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
from RL_models.PPO_Model.OpponentPool import OpponentPool
from checkers_game.constants import BLUE, RED, NUM_ACTIONS


def zip_csv_file(csv_file_path, zip_file_path):
    """Compress and remove the CSV file."""
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, arcname=os.path.basename(csv_file_path))
    os.remove(csv_file_path)


def get_curriculum_options(epoch):
    """Return env.reset() options for the current curriculum phase.

    Phase 1 (epochs 0-49):   Endgame practice, 2-5 pieces per side.
    Phase 2 (epochs 50-149): Mid-game, 4-9 pieces per side.
    Phase 3 (epochs 150+):   Full game, 12 pieces per side (standard).
    """
    if epoch < 50:
        return {"num_pieces": random.randint(2, 5)}
    elif epoch < 150:
        return {"num_pieces": random.randint(4, 9)}
    else:
        return None  # standard 12v12


def random_action_from_mask(mask):
    """Sample a random valid action from the action mask."""
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def uniform_log_prob(mask):
    """Log probability for a uniform random choice over valid actions."""
    n = int(mask.sum())
    if n <= 0:
        return 0.0
    return float(np.log(1.0 / n))


def select_action_for_agent(agent, state, action_mask, epoch, first_move, first_move_count):
    """Select an action using exploration/exploitation strategy."""
    if first_move and epoch < 50:
        action = random_action_from_mask(action_mask)
        log_prob = uniform_log_prob(action_mask)
        first_move_count += 1
        if first_move_count > 10:
            first_move = False
    else:
        epsilon = max(0.03, 1 - epoch / 50)
        if random.random() < epsilon:
            action = random_action_from_mask(action_mask)
            log_prob = uniform_log_prob(action_mask)
        else:
            action, log_prob, _ = agent.select_action(state, action_mask)

    if isinstance(log_prob, torch.Tensor):
        log_prob = log_prob.item()

    return action, log_prob, first_move, first_move_count


def select_action_for_opponent(opponent, state, action_mask):
    """Select an action for a pool opponent (no exploration, no gradient)."""
    with torch.no_grad():
        action, log_prob, _ = opponent.select_action(state, action_mask)
    if isinstance(log_prob, torch.Tensor):
        log_prob = log_prob.item()
    return action, log_prob


def main():
    # Initialize environment and agent
    env = CheckersEnv()
    input_shape = (4, 8, 8)
    n_actions = NUM_ACTIONS
    resume_training = True

    agent = PPOAgent(input_shape, n_actions)

    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_saved_models")
    os.makedirs(model_dir, exist_ok=True)

    # Opponent pool
    pool_dir = os.path.join(base_dir, "opponent_pool")
    pool = OpponentPool(pool_dir, max_size=20)
    pool_save_interval = 10  # Save to pool every N epochs
    pool_opponent_prob = 0.3  # 30% chance of using a pool opponent

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
        print(f"Epoch: {epoch + 1} (pool size: {pool.size})")
        total_rewards, total_steps, blue_wins, red_wins, ties, max_move_reward = 0, 0, 0, 0, 0, 0
        blue_memory, red_memory = Memory(), Memory()

        detailed_csv_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv")

        with open(detailed_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(detailed_headers)

        for episode in range(num_episodes):
            curriculum_opts = get_curriculum_options(epoch)
            state, _ = env.reset(options=curriculum_opts)
            done = False

            # Decide whether to use an opponent from the pool for this episode
            use_opponent = pool.should_use_opponent(prob=pool_opponent_prob)
            opponent = None
            opponent_color = None

            if use_opponent:
                opp_path = pool.sample()
                if opp_path:
                    opponent = PPOAgent(input_shape, n_actions, device=agent.device)
                    opp_state_dict = torch.load(opp_path, map_location=agent.device, weights_only=False)
                    opponent.policy.load_state_dict(opp_state_dict)
                    opponent.policy.eval()
                    # Randomly assign opponent to Blue or Red
                    opponent_color = BLUE if random.random() < 0.5 else RED

            episode_reward, episode_steps, max_episode_move_reward = 0, 0, 0
            first_move = True
            first_move_count = 0
            log_prob_list = []
            reward_colors = []
            blue_offset = len(blue_memory.rewards)
            red_offset = len(red_memory.rewards)
            winner = "None"

            while not done:
                action_mask = env.get_action_mask()

                if action_mask.sum() == 0:
                    next_state, reward, done, _, info = env.step(0)
                    episode_reward += reward
                    total_rewards += reward
                    episode_steps += 1
                    log_prob_list.append(0.0)

                    blue_adj = info.get("blue_reward_adjustment", 0.0)
                    red_adj = info.get("red_reward_adjustment", 0.0)
                    if blue_adj != 0.0 and blue_memory.rewards:
                        blue_memory.rewards[-1] += blue_adj
                    if red_adj != 0.0 and red_memory.rewards:
                        red_memory.rewards[-1] += red_adj

                    if done:
                        winner = info["winner"]
                        if winner == BLUE:
                            blue_wins += 1
                        elif winner == RED:
                            red_wins += 1
                        else:
                            ties += 1

                    state = next_state
                    continue

                # Determine who is acting: current agent or pool opponent
                is_opponent_turn = (opponent is not None and env.game.turn == opponent_color)

                if is_opponent_turn:
                    action, log_prob = select_action_for_opponent(opponent, state, action_mask)
                else:
                    action, log_prob, first_move, first_move_count = select_action_for_agent(
                        agent, state, action_mask, epoch, first_move, first_move_count
                    )

                # Step the environment
                next_state, reward, done, _, info = env.step(action)

                episode_reward += reward
                total_rewards += reward
                episode_steps += 1

                if reward > max_episode_move_reward and not done:
                    max_episode_move_reward = reward
                    if max_episode_move_reward > max_move_reward:
                        max_move_reward = max_episode_move_reward

                # Memory routing: only store current agent's experiences
                turn_complete = info.get("turn_complete", True)
                if turn_complete:
                    acting_color = RED if env.game.turn == BLUE else BLUE
                else:
                    acting_color = env.game.turn

                is_opponent_acting = (opponent is not None and acting_color == opponent_color)

                if not is_opponent_acting:
                    if acting_color == BLUE:
                        blue_memory.add(state, action, reward, log_prob, done)
                        reward_colors.append("blue")
                        if done:
                            red_memory.update_last_done()
                    else:
                        red_memory.add(state, action, reward, log_prob, done)
                        reward_colors.append("red")
                        if done:
                            blue_memory.update_last_done()
                else:
                    # Opponent's turn -- still update done flags if game ended
                    if done:
                        if acting_color == BLUE:
                            red_memory.update_last_done()
                        else:
                            blue_memory.update_last_done()

                # Apply per-color reward adjustments computed by the env
                blue_adj = info.get("blue_reward_adjustment", 0.0)
                red_adj = info.get("red_reward_adjustment", 0.0)
                if blue_adj != 0.0 and blue_memory.rewards:
                    blue_memory.rewards[-1] += blue_adj
                if red_adj != 0.0 and red_memory.rewards:
                    red_memory.rewards[-1] += red_adj

                log_prob_list.append(log_prob)

                if done:
                    winner = info["winner"]
                    if winner == BLUE:
                        blue_wins += 1
                    elif winner == RED:
                        red_wins += 1
                    else:
                        ties += 1

                state = next_state

            blue_memory.update_last_done()
            red_memory.update_last_done()

            reward_list = []
            bi, ri = blue_offset, red_offset
            for color in reward_colors:
                if color == "blue":
                    reward_list.append((color, blue_memory.rewards[bi]))
                    bi += 1
                else:
                    reward_list.append((color, red_memory.rewards[ri]))
                    ri += 1

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
                    sum(blue_memory.rewards[blue_offset:]),
                    sum(red_memory.rewards[red_offset:]),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ", ".join(env.game.moves),
                    ", ".join([str(p) for p in log_prob_list]),
                    ", ".join(f"{c}:{r}" for c, r in reward_list)
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

        # Save to opponent pool periodically
        if (epoch + 1) % pool_save_interval == 0:
            pool.save(agent.policy.state_dict(), epoch + 1)
            print(f"  Saved to opponent pool (size: {pool.size})")

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
