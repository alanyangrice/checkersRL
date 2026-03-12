"""Shared PPO training utilities: CSV, checkpoint, agent factory, stats aggregation."""

import os
import csv
import zipfile
from datetime import datetime

import torch
import numpy as np

from rl.configs.ppo_config import PPOConfig
from rl.algorithms.ppo.agent import PPOAgent
from rl.algorithms.ppo.torch_helpers import get_policy_state_dict, load_policy_state_dict
from rl.algorithms.ppo.memory import Memory

default_config = PPOConfig()


def zip_csv_file(csv_file_path, zip_file_path):
    """Compress the CSV file and remove the original."""
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


def find_latest_checkpoint_path(model_dir, prefix="agent_epoch_"):
    """Find the path to the most recent checkpoint file."""
    if not os.path.exists(model_dir):
        return None
    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    return os.path.join(model_dir, files[-1])


def prune_checkpoints(model_dir, keep_last, prefix="agent_epoch_"):
    """Delete old agent_epoch_N.pt files, retaining only the most recent keep_last."""
    if keep_last is None or keep_last <= 0:
        return
    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
    files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    
    if len(files) > keep_last:
        for f in files[:-keep_last]:
            os.remove(os.path.join(model_dir, f))


def build_agent(agent_type, n_actions, device, config=None):
    config = config or default_config
    """Create a PPOAgent using hyperparameters for agent_type (or defaults if None)."""
    if agent_type is not None and agent_type in config.LEAGUE_AGENTS:
        ac = config.LEAGUE_AGENTS[agent_type]
        gamma = ac.get("gamma", config.GAMMA)
        entropy_bonus = ac.get("entropy_bonus", 0.01)
    else:
        gamma = config.GAMMA
        entropy_bonus = 0.01

    return PPOAgent(
        (4, 8, 8), n_actions, device=device,
        lr=config.LEARNING_RATE, gamma=gamma, eps_clip=config.EPS_CLIP,
        K_epochs=config.K_EPOCHS, gae_lambda=config.GAE_LAMBDA,
        augment=config.AUGMENT, augment_noise=config.AUGMENT_NOISE,
        mini_batch_size=config.MINI_BATCH_SIZE,
        entropy_bonus=entropy_bonus,
    )


def aggregate_epoch_stats(all_results, num_games):
    """Aggregate epoch statistics from game results. Returns dict with all derived stats."""
    total_rewards = {"blue": 0, "red": 0}
    blue_wins = red_wins = ties = total_steps = pool_games = 0
    max_move_reward = float('-inf')

    for r in all_results:
        total_rewards["blue"] += r["rewards"]["blue"]
        total_rewards["red"] += r["rewards"]["red"]
        total_steps += r["episode_steps"]
        blue_wins += r["blue_win"]
        red_wins += r["red_win"]
        ties += 1 - (r["blue_win"] or r["red_win"])
        max_move_reward = max(max_move_reward, r["max_episode_move_reward"])
        if r["opponent"] != "self":
            pool_games += 1

    return {
        "total_rewards": total_rewards,
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "ties": ties,
        "total_steps": total_steps,
        "pool_games": pool_games,
        "max_move_reward": max_move_reward,
        "avg_reward": (total_rewards["blue"] + total_rewards["red"]) / num_games,
        "avg_blue_reward": total_rewards["blue"] / num_games,
        "avg_red_reward": total_rewards["red"] / num_games,
        "avg_steps": total_steps / num_games,
        "blue_win_rate": blue_wins / num_games,
        "red_win_rate": red_wins / num_games,
        "tie_rate": ties / num_games,
    }


def resume_ppo_checkpoint(model_dir, agent, device):
    """Load latest checkpoint into agent. Returns start_epoch (0 if none found)."""
    path = find_latest_checkpoint_path(model_dir)
    if path is None:
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    load_policy_state_dict(agent.policy, ckpt["model_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        agent.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "rng_state_torch" in ckpt:
        torch.random.set_rng_state(ckpt["rng_state_torch"])
    if "rng_state_numpy" in ckpt:
        np.random.set_state(ckpt["rng_state_numpy"])
    if "rng_state_python" in ckpt:
        import random
        random.setstate(ckpt["rng_state_python"])
    start_epoch = ckpt["epoch"]
    print(f"Resuming training from epoch: {start_epoch}")
    return start_epoch


def ensure_reference_model(agent, path):
    """Save agent's policy as reference model if path does not exist."""
    if not os.path.exists(path):
        torch.save(get_policy_state_dict(agent.policy), path)
        print(f"Saved reference model for benchmarking: {path}")


def init_training_csv_headers(csv_path, bench_path, start_epoch):
    """Write CSV headers when starting fresh (start_epoch == 0)."""
    if start_epoch != 0:
        return
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "average_epoch_reward", "average_blue_reward",
            "average_red_reward", "average_episode_length",
            "win_rate_blue", "win_rate_red", "tie_rate", "max_move_reward",
            "epoch_time_s",
        ])
    if not os.path.exists(bench_path):
        with open(bench_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "vs_random_win", "vs_random_loss", "vs_random_tie",
                "vs_random_avg_steps",
                "vs_reference_win", "vs_reference_loss", "vs_reference_tie",
                "vs_reference_avg_steps",
            ])


def build_epoch_train_tasks(num_games, epoch, pool, agent_type, reward_config,
                            pool_prob_fn):
    """Build list of train task dicts for one epoch.

    pool_prob_fn(epoch) returns the probability of using a pool opponent;
    use config.get_pool_opponent_prob for single-agent, config.get_league_pool_prob
    for league training.
    """
    tasks = []
    for _ in range(num_games):
        opp_path = None
        if pool.should_use_opponent(prob=pool_prob_fn(epoch)):
            opp_path = pool.sample(agent_name=agent_type)
        tasks.append({
            "mode": "train",
            "epoch": epoch,
            "opponent_model_path": opp_path,
            "reward_config": reward_config,
        })
    return tasks


def run_ppo_epoch_update(agent, all_results):
    """Combine memories from all_results, run agent.update(), step scheduler."""
    combined = Memory()
    for r in all_results:
        combined.extend(r["blue_memory"])
        combined.extend(r["red_memory"])
    agent.update(combined)
    agent.step_scheduler()


def write_epoch_csv_row(csv_path, epoch, stats, epoch_duration):
    """Append one row to the training progress CSV."""
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, stats["avg_reward"], stats["avg_blue_reward"],
            stats["avg_red_reward"], stats["avg_steps"],
            stats["blue_win_rate"], stats["red_win_rate"], stats["tie_rate"],
            stats["max_move_reward"], round(epoch_duration, 2),
        ])


def run_and_log_benchmark(ctx, agent, device, n_actions, num_workers,
                          reference_model_path, benchmark_csv_path,
                          run_benchmark_fn, num_games, epoch, config=None):
    """Run benchmark, print results, write CSV row, advance reference model."""
    print(f"  Running benchmark ({num_games} games each vs random & reference)...")
    bench = run_benchmark_fn(
        ctx, device, n_actions, num_workers,
        reference_model_path, num_games=num_games, config=config
    )
    vr = bench["vs_random"]
    print(f"  vs Random:    Win {vr['win_rate']:.1%}  Loss {vr['loss_rate']:.1%}  "
          f"Tie {vr['tie_rate']:.1%}  AvgSteps {vr['avg_steps']:.0f}")
    vref = bench.get("vs_reference", {})
    if vref:
        print(f"  vs Reference: Win {vref['win_rate']:.1%}  "
              f"Loss {vref['loss_rate']:.1%}  Tie {vref['tie_rate']:.1%}  "
              f"AvgSteps {vref['avg_steps']:.0f}")
    with open(benchmark_csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            vr["win_rate"], vr["loss_rate"], vr["tie_rate"], vr["avg_steps"],
            vref.get("win_rate", ""), vref.get("loss_rate", ""),
            vref.get("tie_rate", ""), vref.get("avg_steps", ""),
        ])
    torch.save(get_policy_state_dict(agent.policy), reference_model_path)
    print(f"  Reference model advanced to epoch {epoch + 1}")
