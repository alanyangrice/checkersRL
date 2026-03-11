"""GPU-accelerated parallel training for the Checkers PPO agent.

Architecture:
    - Main process hosts a GPU Inference Server thread that batches
      forward-pass requests from many CPU workers.
    - CPU worker processes run game simulation (CheckersEnv) and send
      (state, action_mask) to the server when the current agent needs
      to act.  Opponent inference stays CPU-local in each worker.
    - After all games finish, the main process collects memories and
      runs the standard PPO update on GPU (unchanged from train_parallel).

Run from repo root:
    python -m rl.algorithms.ppo.trainer
"""

import os
import csv
import time
import argparse
import random

import torch
import numpy as np

from rl.configs.ppo_config import PPOConfig
from rl.envs.checkers_env import CheckersEnv
from rl.training_utils.gpu_inference_server import PPOInferenceServer, SHUTDOWN
from rl.training_utils.worker_pool import BaseWorkerContext
from rl.algorithms.ppo.agent import PPOAgent
from rl.algorithms.ppo.opponent_pool import OpponentPool
from rl.algorithms.ppo import utils as ppo_utils
from rl.algorithms.ppo.parallel_infra import WorkerContext
from rl.utils.seed_utils import set_seed
from checkers_game.constants import NUM_ACTIONS
from rl.eval.ppo_evaluate import run_benchmark
from rl.training_utils.device_utils import get_device

default_config = PPOConfig()


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

def train_gpu_parallel(num_epochs=None, num_games=None,
                       n_actions=NUM_ACTIONS, num_workers=None,
                       agent_type=None, reward_config=None, seed=None, config=None):
    config = config or default_config
    if num_epochs is None: num_epochs = config.NUM_EPOCHS
    if num_games is None: num_games = config.NUM_GAMES
    """GPU-accelerated parallelized training loop for the Checkers PPO agent.

    Args:
        num_workers:   Number of CPU worker processes.
        agent_type:    League agent name (e.g. "tactical"). If provided,
                       reward_config and PPO hyperparams (gamma, entropy_bonus)
                       are looked up from config.LEAGUE_AGENTS[agent_type].
        reward_config: Explicit reward config dict (overrides agent_type lookup).
        seed: Random seed for reproducibility (optional).
    """
    if seed is not None:
        pass # Seed is set at entry point now
    input_shape = (4, 8, 8)
    device = get_device()

    # Resolve reward config (for game tasks)
    if agent_type is not None and agent_type in config.LEAGUE_AGENTS:
        league_cfg = config.LEAGUE_AGENTS[agent_type]
        if reward_config is None:
            reward_config = league_cfg
    # else: reward_config stays as passed (or None)

    agent = ppo_utils.build_agent(agent_type, n_actions, device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "training_results", "ppo")
    model_dir = os.path.join(results_dir, "ppo_saved_models_parallel")
    os.makedirs(model_dir, exist_ok=True)

    pool_dir = os.path.join(results_dir, "opponent_pool_parallel")
    pool = OpponentPool(pool_dir, max_size=config.POOL_MAX_SIZE)

    detailed_csv_folder_path = os.path.join(results_dir, "training_progress_detailed_parallel")
    os.makedirs(detailed_csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(results_dir, "training_progress_parallel.csv")
    benchmark_csv_path = os.path.join(results_dir, "benchmark_parallel.csv")
    reference_model_path = os.path.join(model_dir, "reference_model.pt")

    start_epoch = ppo_utils.resume_ppo_checkpoint(model_dir, agent, device)
    ppo_utils.ensure_reference_model(agent, reference_model_path)
    ppo_utils.init_training_csv_headers(csv_file_path, benchmark_csv_path, start_epoch)

    if num_workers is None:
        num_workers = config.get_num_workers_gpu()

    with WorkerContext(agent.get_policy_state_dict(), device, n_actions, num_workers, config=config, seed=seed) as ctx:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            print(f"\nStarting epoch {epoch + 1} (pool size: {pool.size})")
            print(f"Using {num_workers} CPU workers + GPU inference server.")

            game_tasks = ppo_utils.build_epoch_train_tasks(
                num_games, epoch, pool, agent_type, reward_config,
                pool_prob_fn=config.get_pool_opponent_prob,
            )

            # Detailed CSV
            detailed_csv_file_path = os.path.join(
                detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.csv"
            )
            with open(detailed_csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "game_number", "blue_win", "red_win", "total_reward",
                    "blue_reward", "red_reward", "time", "opponent", "moves",
                    "log_probs", "reward_list"
                ])

            # ── Run all games with GPU inference server ───────────────
            ctx.update_model(agent.get_policy_state_dict())
            all_results = ctx.run_tasks(
                game_tasks, label=f"Epoch {epoch+1}"
            )

            ppo_utils.write_detailed_csv(
                detailed_csv_file_path, all_results, 0, epoch, num_games
            )

            stats = ppo_utils.aggregate_epoch_stats(all_results, num_games)
            pool.batch_update_stats(all_results, epoch=epoch + 1, agent_name=agent_type)
            ppo_utils.run_ppo_epoch_update(agent, all_results)

            # Save to opponent pool periodically (namespaced by agent_type if set)
            if (epoch + 1) % config.POOL_SAVE_INTERVAL == 0:
                pool.save(agent.get_policy_state_dict(), epoch + 1, agent_name=agent_type)
                print(f"  Saved to opponent pool (size: {pool.size})")

            epoch_zip_file_path = os.path.join(
                detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.zip"
            )
            ppo_utils.zip_csv_file(detailed_csv_file_path, epoch_zip_file_path)
            epoch_duration = time.perf_counter() - epoch_start

            ppo_utils.write_epoch_csv_row(csv_file_path, epoch + 1, stats, epoch_duration)

            # NaN guard: skip checkpoint save if policy has NaN (matches train_league)
            if any(torch.isnan(p).any() for p in agent.policy.parameters()):
                print(f"  CRITICAL: NaN in policy weights after update — skipping checkpoint save.")
            else:
                # Save checkpoint (plain state dict — no _orig_mod. prefix)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': agent.get_policy_state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'scheduler_state_dict': agent.scheduler.state_dict(),
                    'rng_state_torch': torch.random.get_rng_state(),
                    'rng_state_numpy': np.random.get_state(),
                    'rng_state_python': random.getstate(),
                }, os.path.join(model_dir, f"agent_epoch_{epoch + 1}.pt"))
                ppo_utils.prune_checkpoints(model_dir, config.CHECKPOINT_KEEP_LAST)

                print(f"Epoch {epoch + 1} complete in {epoch_duration:.1f}s. "
                      f"Avg Reward: {stats['avg_reward']:.2f}, "
                      f"Blue Win: {stats['blue_win_rate']:.2%}, Red Win: {stats['red_win_rate']:.2%}, "
                      f"Tie: {stats['tie_rate']:.2%}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")

            from rl.eval.ppo_evaluate import run_benchmark
            if (epoch + 1) % config.BENCHMARK_INTERVAL == 0:
                ctx.update_model(agent.get_policy_state_dict())
                ppo_utils.run_and_log_benchmark(
                        ctx, agent, device, n_actions, num_workers,
                        reference_model_path, benchmark_csv_path,
                        run_benchmark, config.BENCHMARK_GAMES, epoch, config=config
                    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fully-deterministic", action="store_true", help="Force CuDNN determinism (slower)")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed, fully_deterministic=args.fully_deterministic)
    train_gpu_parallel(seed=args.seed)

