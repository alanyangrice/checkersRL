import os
import csv
import time
import random
import zipfile
from datetime import datetime
from multiprocessing import Pool

import torch
import numpy as np

from RL_models.checkers_env import CheckersEnv
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.PPO_Model.Memory import Memory
from RL_models.PPO_Model.OpponentPool import OpponentPool
from RL_models.PPO_Model import training_config as cfg
from checkers_game.constants import BLUE, RED, NUM_ACTIONS


# ── Worker-level globals (initialised once per Pool worker) ──────────
_worker_agent = None          # reused PPOAgent for the current model
_worker_opponents = {}        # path → PPOAgent cache for pool opponents


def _init_worker(n_actions, temp_model_path):
    """Pool initializer: load the model ONCE per worker process."""
    global _worker_agent
    _worker_agent = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
    _worker_agent.policy.load_state_dict(
        torch.load(temp_model_path, map_location="cpu", weights_only=False)
    )
    _worker_agent.policy.eval()


def _get_opponent(n_actions, opp_path):
    """Return a cached opponent agent, loading from disk only on first use."""
    global _worker_opponents
    if opp_path not in _worker_opponents:
        opp = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
        checkpoint = torch.load(opp_path, map_location="cpu", weights_only=False)
        # Support both full checkpoints and raw state dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        opp.policy.load_state_dict(state_dict)
        opp.policy.eval()
        _worker_opponents[opp_path] = opp
    return _worker_opponents[opp_path]


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
                result["opponent"],
                ", ".join(result["moves"]),
                ", ".join(map(str, result["log_probs"])),
                ", ".join(f"{c}:{r}" for c, r in result["rewards_list"])
            ])


def random_action_from_mask(mask):
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def uniform_log_prob(mask):
    n = int(mask.sum())
    if n <= 0:
        return 0.0
    return float(np.log(1.0 / n))


get_curriculum_options = cfg.get_curriculum_options


def play_benchmark_game(n_actions, opponent_type, opponent_model_path=None):
    """Play a single benchmark game (no memory/training, just win/loss/tie).

    Args:
        n_actions: Size of the action space.
        opponent_type: "random" or "model".
        opponent_model_path: Path to opponent model (only for opponent_type="model").

    Returns:
        dict with keys: agent_color, agent_win, opponent_win, tie, steps
    """
    env = CheckersEnv()
    agent = _worker_agent

    opponent = None
    if opponent_type == "model" and opponent_model_path is not None:
        opponent = _get_opponent(n_actions, opponent_model_path)

    # Agent plays a random color each game
    agent_color = BLUE if random.random() < 0.5 else RED
    opponent_color = RED if agent_color == BLUE else BLUE

    state, _ = env.reset()
    done = False
    steps = 0

    while not done:
        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            steps += 1
            if done:
                break
            state = env.get_board_state()
            continue

        current_turn = env.game.turn
        if current_turn == agent_color:
            # Agent plays greedily (no exploration) for benchmark
            with torch.no_grad():
                action, _, _ = agent.select_action(state, action_mask)
        elif opponent_type == "random":
            action = random_action_from_mask(action_mask)
        else:
            # Model opponent plays greedily
            with torch.no_grad():
                action, _, _ = opponent.select_action(state, action_mask)

        next_state, _, done, _, info = env.step(action)
        steps += 1
        state = next_state

    winner = info.get("winner", "Tie")
    agent_win = 1 if winner == agent_color else 0
    opponent_win = 1 if (winner != agent_color and winner != "Tie" and winner != "None") else 0
    tie = 1 if winner == "Tie" else 0

    return {
        "agent_color": "BLUE" if agent_color == BLUE else "RED",
        "agent_win": agent_win,
        "opponent_win": opponent_win,
        "tie": tie,
        "steps": steps,
    }


def play_game(n_actions, game_id, epoch, opponent_model_path=None):
    """Simulate a single game of Checkers with training logic.

    Uses the worker-level cached agent (loaded once per Pool worker via
    _init_worker) instead of creating a new one for every game.

    If opponent_model_path is provided, a past opponent plays one side (randomly
    chosen). Only the current agent's experiences are recorded in memory.
    """
    env = CheckersEnv()
    agent = _worker_agent          # reuse the pre-loaded model

    # Optionally load a pool opponent (cached per worker)
    opponent = None
    opponent_color = None
    opponent_label = "self"
    if opponent_model_path is not None:
        opponent = _get_opponent(n_actions, opponent_model_path)
        opponent_color = BLUE if random.random() < 0.5 else RED
        # Extract a readable label like "pool_epoch_80"
        opponent_label = os.path.splitext(os.path.basename(opponent_model_path))[0]

    blue_memory, red_memory = Memory(), Memory()
    curriculum_opts = get_curriculum_options(epoch)
    state, _ = env.reset(options=curriculum_opts)
    done = False

    episode_reward, episode_steps, max_episode_move_reward = 0, 0, 0
    first_move, first_move_count = True, 0
    log_prob_list = []
    reward_colors = []
    blue_win, red_win = 0, 0

    while not done:
        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            next_state, reward, done, _, info = env.step(0)
            episode_reward += reward
            episode_steps += 1
            log_prob_list.append(0.0)

            blue_adj = info.get("blue_reward_adjustment", 0.0)
            red_adj = info.get("red_reward_adjustment", 0.0)
            if blue_adj != 0.0 and len(blue_memory) > 0:
                blue_memory.rewards[-1] += blue_adj
            if red_adj != 0.0 and len(red_memory) > 0:
                red_memory.rewards[-1] += red_adj

            if done:
                winner = info["winner"]
                if winner == BLUE:
                    blue_win = 1
                elif winner == RED:
                    red_win = 1

            state = next_state
            continue

        # Determine who acts
        is_opponent_turn = (opponent is not None and env.game.turn == opponent_color)

        if is_opponent_turn:
            # Pool opponents get a fixed exploration rate to prevent
            # deterministic loops that trigger 3-fold repetition ties.
            if random.random() < cfg.POOL_EPSILON:
                action = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                with torch.no_grad():
                    action, log_prob, _ = opponent.select_action(state, action_mask)
                if isinstance(log_prob, torch.Tensor):
                    log_prob = log_prob.item()
        else:
            epsilon = cfg.get_epsilon(epoch)

            # No forced random first moves — full 12v12 from the start,
            # epsilon provides sufficient opening diversity.
            use_random_first = False

            if use_random_first:
                action = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
                first_move_count += 1
                if first_move_count > 1:
                    first_move = False
            elif random.random() < epsilon:
                action = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                action, log_prob, _ = agent.select_action(state, action_mask)

            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()

        next_state, reward, done, _, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        if reward > max_episode_move_reward and not done:
            max_episode_move_reward = reward

        # Memory routing: only store current agent's experiences
        turn_complete = info.get("turn_complete", True)
        if turn_complete:
            acting_color = RED if env.game.turn == BLUE else BLUE
        else:
            acting_color = env.game.turn

        is_opponent_acting = (opponent is not None and acting_color == opponent_color)

        if not is_opponent_acting:
            if acting_color == BLUE:
                blue_memory.add(state, action, reward, log_prob, done, action_mask)
                reward_colors.append("blue")
                if done:
                    red_memory.update_last_done()
            else:
                red_memory.add(state, action, reward, log_prob, done, action_mask)
                reward_colors.append("red")
                if done:
                    blue_memory.update_last_done()
        else:
            if done:
                if acting_color == BLUE:
                    red_memory.update_last_done()
                else:
                    blue_memory.update_last_done()

        # Apply per-color reward adjustments computed by the env
        blue_adj = info.get("blue_reward_adjustment", 0.0)
        red_adj = info.get("red_reward_adjustment", 0.0)
        if blue_adj != 0.0 and len(blue_memory) > 0:
            blue_memory.rewards[-1] += blue_adj
        if red_adj != 0.0 and len(red_memory) > 0:
            red_memory.rewards[-1] += red_adj

        log_prob_list.append(log_prob)

        if done:
            winner = info["winner"]
            if winner == BLUE:
                blue_win = 1
            elif winner == RED:
                red_win = 1

        state = next_state

    blue_memory.update_last_done()
    red_memory.update_last_done()

    reward_list = []
    bi, ri = 0, 0
    for color in reward_colors:
        if color == "blue":
            reward_list.append((color, blue_memory.rewards[bi]))
            bi += 1
        else:
            reward_list.append((color, red_memory.rewards[ri]))
            ri += 1

    return {
        "blue_memory": blue_memory,
        "red_memory": red_memory,
        "rewards": {
            "blue": sum(blue_memory.rewards) if blue_memory.rewards else 0,
            "red": sum(red_memory.rewards) if red_memory.rewards else 0,
        },
        "blue_win": blue_win,
        "red_win": red_win,
        "moves": env.game.moves,
        "log_probs": log_prob_list,
        "rewards_list": reward_list,
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "max_episode_move_reward": max_episode_move_reward,
        "opponent": opponent_label,
    }


def run_benchmark(agent, n_actions, num_processes, reference_model_path, num_games=200):
    """Evaluate the agent against random and reference opponents.

    Returns:
        dict with win/tie rates vs each opponent type.
    """
    temp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "PPO_saved_models_parallel", "_benchmark_temp.pt"
    )
    torch.save(agent.policy.state_dict(), temp_path)

    results = {}
    try:
        with Pool(num_processes, initializer=_init_worker,
                  initargs=(n_actions, temp_path)) as p:
            # --- vs Random ---
            random_args = [(n_actions, "random", None)] * num_games
            random_results = p.starmap(play_benchmark_game, random_args)

            wins = sum(r["agent_win"] for r in random_results)
            losses = sum(r["opponent_win"] for r in random_results)
            ties = sum(r["tie"] for r in random_results)
            avg_steps = sum(r["steps"] for r in random_results) / num_games
            results["vs_random"] = {
                "win_rate": wins / num_games,
                "loss_rate": losses / num_games,
                "tie_rate": ties / num_games,
                "avg_steps": avg_steps,
            }

            # --- vs Reference model ---
            if reference_model_path and os.path.exists(reference_model_path):
                ref_args = [(n_actions, "model", reference_model_path)] * num_games
                ref_results = p.starmap(play_benchmark_game, ref_args)

                wins = sum(r["agent_win"] for r in ref_results)
                losses = sum(r["opponent_win"] for r in ref_results)
                ties = sum(r["tie"] for r in ref_results)
                avg_steps = sum(r["steps"] for r in ref_results) / num_games
                results["vs_reference"] = {
                    "win_rate": wins / num_games,
                    "loss_rate": losses / num_games,
                    "tie_rate": ties / num_games,
                    "avg_steps": avg_steps,
                }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return results


def train_parallel(num_epochs=cfg.NUM_EPOCHS, num_games=cfg.NUM_GAMES,
                   batch_size=cfg.NUM_GAMES, n_actions=NUM_ACTIONS):
    """Parallelized training loop for the Checkers PPO agent (CPU-only)."""
    input_shape = (4, 8, 8)
    agent = PPOAgent(
        input_shape, n_actions,
        lr=cfg.LEARNING_RATE, gamma=cfg.GAMMA, eps_clip=cfg.EPS_CLIP,
        K_epochs=cfg.K_EPOCHS, gae_lambda=cfg.GAE_LAMBDA,
        augment=cfg.AUGMENT, augment_noise=cfg.AUGMENT_NOISE,
        mini_batch_size=cfg.MINI_BATCH_SIZE,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_saved_models_parallel")
    os.makedirs(model_dir, exist_ok=True)

    # Opponent pool
    pool_dir = os.path.join(base_dir, "opponent_pool_parallel")
    pool = OpponentPool(pool_dir, max_size=cfg.POOL_MAX_SIZE)

    start_epoch = 0

    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("agent_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(model_dir, latest)
        checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch: {start_epoch}")

    detailed_csv_folder_path = os.path.join(base_dir, "training_progress_detailed_parallel")
    os.makedirs(detailed_csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(base_dir, "training_progress_parallel.csv")
    benchmark_csv_path = os.path.join(base_dir, "benchmark_parallel.csv")

    # Save the starting model as a permanent reference for benchmarking
    reference_model_path = os.path.join(model_dir, "reference_model.pt")
    if not os.path.exists(reference_model_path):
        torch.save(agent.policy.state_dict(), reference_model_path)
        print(f"Saved reference model for benchmarking: {reference_model_path}")

    if start_epoch == 0:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch", "average_epoch_reward", "average_blue_reward", "average_red_reward",
                "average_episode_length", "win_rate_blue", "win_rate_red", "tie_rate", "max_move_reward",
                "epoch_time_s"
            ])

    # Always create benchmark CSV with header if it doesn't exist yet
    if not os.path.exists(benchmark_csv_path):
        with open(benchmark_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "vs_random_win", "vs_random_loss", "vs_random_tie", "vs_random_avg_steps",
                "vs_reference_win", "vs_reference_loss", "vs_reference_tie", "vs_reference_avg_steps",
            ])

    num_processes = cfg.get_num_workers_cpu()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        print(f"Starting epoch {epoch + 1} (pool size: {pool.size})")

        temp_model_path = os.path.join(model_dir, f'temp_agent_model_epoch_{epoch + 1}.pt')
        torch.save(agent.policy.state_dict(), temp_model_path)

        print(f"Using {num_processes} processes for parallel simulation.")

        total_rewards = {"blue": 0, "red": 0}
        blue_wins, red_wins, ties, max_move_reward, total_steps = 0, 0, 0, 0, 0
        pool_games = 0

        detailed_csv_file_path = os.path.join(
            detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv"
        )

        with open(detailed_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "game_number", "blue_win", "red_win", "total_reward", "blue_reward",
                "red_reward", "time", "opponent", "moves", "log_probs", "reward_list"
            ])

        # Create pool ONCE per epoch with initializer that loads the model
        # once per worker process (instead of once per game).
        with Pool(num_processes, initializer=_init_worker,
                  initargs=(n_actions, temp_model_path)) as p:

            # Submit all games and collect results, printing progress every 500
            all_results = []
            log_interval = 500

            for batch_start in range(0, num_games, batch_size):
                batch_end = min(batch_start + batch_size, num_games)

                # For each game in the batch, decide whether to use a pool opponent
                game_args = []
                for game_id in range(batch_start, batch_end):
                    opp_path = None
                    if pool.should_use_opponent(prob=cfg.get_pool_opponent_prob(epoch)):
                        opp_path = pool.sample()
                    game_args.append((n_actions, game_id, epoch, opp_path))

                # Process in chunks for progress logging
                for chunk_start in range(0, len(game_args), log_interval):
                    chunk_args = game_args[chunk_start : chunk_start + log_interval]
                    chunk_results = p.starmap(play_game, chunk_args)
                    all_results.extend(chunk_results)
                    games_done = batch_start + chunk_start + len(chunk_args)
                    print(f"  Games {games_done}/{num_games} finished")

                write_detailed_csv(detailed_csv_file_path, all_results[batch_start:], batch_start, epoch, num_games)

            for result in all_results:
                total_rewards["blue"] += result["rewards"]["blue"]
                total_rewards["red"] += result["rewards"]["red"]
                total_steps += result["episode_steps"]
                blue_wins += result["blue_win"]
                red_wins += result["red_win"]
                ties += 1 - (result["blue_win"] or result["red_win"])
                max_move_reward = max(max_move_reward, result["max_episode_move_reward"])
                if result["opponent"] != "self":
                    pool_games += 1

            # Combine blue + red into ONE memory to avoid doubling GPU work
            combined_memory = Memory()
            for result in all_results:
                combined_memory.extend(result["blue_memory"])
                combined_memory.extend(result["red_memory"])
            agent.update(combined_memory)

        # Step the learning rate scheduler
        agent.step_scheduler()

        # Save to opponent pool periodically
        if (epoch + 1) % cfg.POOL_SAVE_INTERVAL == 0:
            pool.save(agent.policy.state_dict(), epoch + 1)
            print(f"  Saved to opponent pool (size: {pool.size})")

        # Zip detailed CSV
        epoch_zip_file_path = os.path.join(detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.zip")
        zip_csv_file(detailed_csv_file_path, epoch_zip_file_path)

        # Clean up temp model
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        # Log epoch statistics
        epoch_duration = time.perf_counter() - epoch_start
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
                avg_steps, blue_win_rate, red_win_rate, tie_rate, max_move_reward,
                round(epoch_duration, 2)
            ])

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
        }, os.path.join(model_dir, f"agent_epoch_{epoch + 1}.pt"))

        print(f"Epoch {epoch + 1} complete in {epoch_duration:.1f}s. "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Blue Win: {blue_win_rate:.2%}, Red Win: {red_win_rate:.2%}, "
              f"Tie: {tie_rate:.2%}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")

        # --- Benchmark evaluation every N epochs ---
        if (epoch + 1) % cfg.BENCHMARK_INTERVAL == 0:
            print(f"  Running benchmark ({cfg.BENCHMARK_GAMES} games each vs random & reference)...")
            bench = run_benchmark(
                agent, n_actions, num_processes,
                reference_model_path, num_games=cfg.BENCHMARK_GAMES,
            )
            vr = bench["vs_random"]
            print(f"  vs Random:    Win {vr['win_rate']:.1%}  Loss {vr['loss_rate']:.1%}  "
                  f"Tie {vr['tie_rate']:.1%}  AvgSteps {vr['avg_steps']:.0f}")

            vref = bench.get("vs_reference", {})
            if vref:
                print(f"  vs Reference: Win {vref['win_rate']:.1%}  Loss {vref['loss_rate']:.1%}  "
                      f"Tie {vref['tie_rate']:.1%}  AvgSteps {vref['avg_steps']:.0f}")

            with open(benchmark_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    vr["win_rate"], vr["loss_rate"], vr["tie_rate"], vr["avg_steps"],
                    vref.get("win_rate", ""), vref.get("loss_rate", ""),
                    vref.get("tie_rate", ""), vref.get("avg_steps", ""),
                ])

            # Auto-advance reference model to current epoch so the next
            # benchmark always measures improvement over the last tested epoch.
            torch.save(agent.policy.state_dict(), reference_model_path)
            print(f"  Reference model advanced to epoch {epoch + 1}")


if __name__ == "__main__":
    train_parallel()
