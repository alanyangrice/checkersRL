"""Multi-agent league self-play training for the Checkers PPO agent.

Architecture:
    - N agent types defined in config.LEAGUE_AGENTS (training_config.py).
    - One interleaved training loop: each league epoch trains every agent
      type for one epoch in sequence, then repeats.
    - All agents share a single opponent pool (opponent_pool_league/).
      Pool checkpoints are namespaced by agent type so cross-style sampling
      is automatic: a tactical agent might face an aggressive opponent.
    - Each agent gets its own:
        checkpoint directory:  ppo_saved_models_{agent_type}/
        training CSV:          training_progress_{agent_type}.csv
        benchmark CSV:         benchmark_{agent_type}.csv

Motivation:
    Pure self-play converges to a mutual draw equilibrium (co-evolution).
    Reward-diverse agents develop structurally different play styles that
    break each other's passive equilibria.  See:
    - AlphaStar league play (Vinyals et al., Nature 2019)
    - OpenAI Five PBT reward diversity (Berner et al., arXiv:1912.06680)
    - PSRO game-theoretic foundation (Lanctot et al., NeurIPS 2017)

Run from repo root:
    python -m rl.algorithms.ppo.trainer_league
"""

import copy
import os
import csv
import time
import random
import numpy as np
import zipfile
from datetime import datetime
from itertools import combinations
from multiprocessing import cpu_count

import torch

from rl.configs.ppo_config import PPOConfig
from rl.algorithms.ppo.agent import PPOAgent
from rl.algorithms.ppo.opponent_pool import OpponentPool
from rl.algorithms.ppo.parallel_infra import WorkerContext
from rl.algorithms.ppo import utils as ppo_utils
from rl.utils.seed_utils import set_seed
from checkers_game.constants import NUM_ACTIONS
from rl.eval.ppo_evaluate import run_benchmark
from rl.training_utils.device_utils import get_device

default_config = PPOConfig()


# ─────────────────────────────────────────────────────────────────────
# Per-agent state (checkpoint, CSVs, PPOAgent instance)
# ─────────────────────────────────────────────────────────────────────

def load_checkpoint(agent, model_dir, device):
    """Load the latest checkpoint in model_dir into agent. Returns start_epoch."""
    ckpt_path = ppo_utils.find_latest_checkpoint_path(model_dir)
    if ckpt_path is None:
        return 0
    cp = torch.load(ckpt_path, map_location=device, weights_only=False)
    ppo_utils.load_policy_state_dict(agent.policy, cp["model_state_dict"])
    agent.optimizer.load_state_dict(cp["optimizer_state_dict"])
    if "scheduler_state_dict" in cp:
        agent.scheduler.load_state_dict(cp["scheduler_state_dict"])
    print(f"    Resumed {os.path.basename(model_dir)} from epoch {cp['epoch']}")
    return cp["epoch"]


# ─────────────────────────────────────────────────────────────────────
# Main league training loop
# ─────────────────────────────────────────────────────────────────────

def run_league_benchmark(agents, device, n_actions, num_workers,
                          model_dirs, ref_paths, num_games, config=None, seed=None):
    """Consolidated benchmark for all agents.

    Per-agent:
      - vs random          (absolute strength)
      - vs own reference   (improvement since last benchmark)
    Per unique pair (A, B):
      - A vs B             (cross-style matchup)
      - B vs A             (derived by flipping A vs B win/loss — no extra games)

    Total games: N_agents × 2 × num_games  (vs random + vs self-ref)
               + N_pairs  × num_games       (pairwise, one direction each)

    Returns a dict: {match_key → {win_rate, loss_rate, tie_rate, avg_steps}}
    e.g. "tactical_vs_random", "tactical_vs_self", "tactical_vs_terminal"
    """
    config = config or default_config
    results = {}
    agent_list = config.ACTIVE_AGENTS

    # Each agent vs random + vs its own reference (one GPU server session each)
    for agent_type in agent_list:
        has_ref = os.path.exists(ref_paths[agent_type])
        with WorkerContext(agents[agent_type].get_policy_state_dict(), device, n_actions, num_workers, config=config, seed=seed) as ctx:
            ctx.update_model(agents[agent_type].get_policy_state_dict())
            bench = run_benchmark(
                ctx, device, n_actions, num_workers,
                reference_model_path=ref_paths[agent_type] if has_ref else None,
                num_games=num_games,
                include_random=True,
                config=config
            )
        results[f"{agent_type}_vs_random"] = bench["vs_random"]
        if "vs_reference" in bench:
            results[f"{agent_type}_vs_self"] = bench["vs_reference"]

    # Each unique pair (A, B) — one direction; derive reverse for free
    for type_a, type_b in combinations(agent_list, 2):
        ckpt_b = ppo_utils.find_latest_checkpoint_path(model_dirs[type_b])
        if not ckpt_b:
            continue
        with WorkerContext(agents[type_a].get_policy_state_dict(), device, n_actions, num_workers, config=config, seed=seed) as ctx:
            ctx.update_model(agents[type_a].get_policy_state_dict())
            bench_ab = run_benchmark(
                ctx, device, n_actions, num_workers,
                reference_model_path=None,
                num_games=num_games,
                extra_opponents={type_b: ckpt_b},
                include_random=False,
                config=config
            )
        a_vs_b = bench_ab.get(f"vs_{type_b}", {})
        results[f"{type_a}_vs_{type_b}"] = a_vs_b
        # B_vs_A: win ↔ loss, tie unchanged — no extra games needed
        results[f"{type_b}_vs_{type_a}"] = {
            "win_rate":  a_vs_b.get("loss_rate", ""),
            "loss_rate": a_vs_b.get("win_rate",  ""),
            "tie_rate":  a_vs_b.get("tie_rate",  ""),
            "avg_steps": a_vs_b.get("avg_steps", ""),
        }

    return results


def train_league(num_league_epochs=None,
                 num_games=None,
                 n_actions=NUM_ACTIONS,
                 num_workers=None, seed=None, config=None):
    config = config or default_config
    if num_league_epochs is None: num_league_epochs = config.NUM_EPOCHS
    if num_games is None: num_games = config.NUM_GAMES
    """Interleaved multi-agent league training.

    Each 'league epoch' trains every ACTIVE_AGENT for one epoch in sequence.
    All agents share a unified opponent pool for cross-style exposure.
    """
    if seed is not None:
        pass # Seed is set at entry point now
    if num_workers is None:
        num_workers = config.get_num_workers_gpu()

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Active agents: {config.ACTIVE_AGENTS}")
    print(f"Workers: {num_workers}  |  Games per agent per epoch: {num_games}")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "training_results", "ppo")

    # ── Shared opponent pool ────────────────────────────────────────
    league_pool_dir = os.path.join(results_dir, "opponent_pool_league")
    pool = OpponentPool(league_pool_dir, max_size=config.POOL_MAX_SIZE)

    # ── Per-agent state ─────────────────────────────────────────────
    agents      = {}   # agent_type → PPOAgent
    model_dirs  = {}   # agent_type → checkpoint dir path
    start_epochs = {}  # agent_type → epoch to resume from
    csv_paths   = {}   # agent_type → training_progress CSV
    ref_paths   = {}   # agent_type → reference model path
    detail_dirs = {}   # agent_type → detailed CSV folder

    for agent_type in config.ACTIVE_AGENTS:
        agents[agent_type]     = ppo_utils.build_agent(agent_type, n_actions, device)
        model_dir              = os.path.join(results_dir, f"ppo_saved_models_{agent_type}")
        model_dirs[agent_type] = model_dir
        os.makedirs(model_dir, exist_ok=True)

        start_epochs[agent_type] = load_checkpoint(
            agents[agent_type], model_dir, device
        )

        csv_paths[agent_type] = os.path.join(results_dir, f"training_progress_{agent_type}.csv")
        ref_paths[agent_type] = os.path.join(model_dir, "reference_model.pt")
        detail_dirs[agent_type] = os.path.join(results_dir, f"training_progress_detailed_{agent_type}")
        os.makedirs(detail_dirs[agent_type], exist_ok=True)

        # Write CSV headers if starting fresh
        if start_epochs[agent_type] == 0:
            with open(csv_paths[agent_type], "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch", "average_epoch_reward", "average_blue_reward",
                    "average_red_reward", "average_episode_length",
                    "win_rate_blue", "win_rate_red", "tie_rate",
                    "max_move_reward", "epoch_time_s",
                ])

        # Save reference model if new run
        if not os.path.exists(ref_paths[agent_type]):
            torch.save(agents[agent_type].get_policy_state_dict(),
                       ref_paths[agent_type])
            print(f"  [{agent_type}] Saved reference model (epoch 0)")

    # ── Consolidated league benchmark CSV ──────────────────────────
    bench_league_path = os.path.join(results_dir, "benchmark_league.csv")
    if not os.path.exists(bench_league_path):
        # Header: epoch + per-agent (vs random + vs self) + pairwise (both dirs)
        header = ["epoch"]
        for at in config.ACTIVE_AGENTS:
            for opp in ["random", "self"]:
                header += [f"{at}_vs_{opp}_win", f"{at}_vs_{opp}_loss",
                           f"{at}_vs_{opp}_tie", f"{at}_vs_{opp}_avg_steps"]
        for type_a, type_b in combinations(config.ACTIVE_AGENTS, 2):
            for ta, tb in [(type_a, type_b), (type_b, type_a)]:
                header += [f"{ta}_vs_{tb}_win", f"{ta}_vs_{tb}_loss",
                           f"{ta}_vs_{tb}_tie", f"{ta}_vs_{tb}_avg_steps"]
        with open(bench_league_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Determine the minimum start_epoch across all agents so that no agent
    # skips epochs — all agents advance together from the earliest checkpoint.
    global_start_epoch = min(start_epochs.values())

    def build_tasks(agent_type, league_epoch):
        """Build the task list for one agent-epoch."""
        reward_cfg = config.LEAGUE_AGENTS[agent_type]
        return ppo_utils.build_epoch_train_tasks(
            num_games, league_epoch, pool, agent_type, reward_cfg,
            pool_prob_fn=config.get_league_pool_prob,
        )

    # ── League training loop (WorkerContext, sequential) ──────────────────
    # Workers are kept alive across all agent-epochs within the entire run,
    # eliminating per-agent spawn overhead (~1-2s/worker on Windows).
    #
    # Collection and GPU update are strictly sequential: all CPU workers
    # finish collecting games before the PPO update starts, so CPU and GPU
    # are never both under full load at the same time.

    first_agent = config.ACTIVE_AGENTS[0]
    initial_sd  = agents[first_agent].get_policy_state_dict()

    with WorkerContext(initial_sd, device, n_actions, num_workers, config=config, seed=seed) as ctx:

        for league_epoch in range(global_start_epoch, num_league_epochs):
            print(f"\n{'='*60}")
            print(f"  League epoch {league_epoch + 1}  |  pool size: {pool.size}")
            print(f"{'='*60}")

            for agent_idx, agent_type in enumerate(config.ACTIVE_AGENTS):
                agent      = agents[agent_type]
                reward_cfg = config.LEAGUE_AGENTS[agent_type]

                print(f"\n  [{agent_type}] epoch {league_epoch + 1}  "
                      f"(gamma={reward_cfg['gamma']}, "
                      f"tie_base={reward_cfg['tie_base']})")

                epoch_start = time.perf_counter()

                # Collect games with up-to-date weights, then wait for all workers
                # to finish before starting the GPU PPO update.
                ctx.update_model(agent.get_policy_state_dict())
                all_results = ctx.run_tasks(
                    build_tasks(agent_type, league_epoch),
                    f"[{agent_type}] ep{league_epoch + 1}",
                )

                # Detailed CSV for this agent-epoch
                detail_csv = os.path.join(
                    detail_dirs[agent_type],
                    f"detailed_games_epoch_{league_epoch + 1}.csv"
                )
                with open(detail_csv, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "game_number", "blue_win", "red_win", "total_reward",
                        "blue_reward", "red_reward", "time", "opponent",
                        "moves", "log_probs", "reward_list"
                    ])

                ppo_utils.write_detailed_csv(detail_csv, all_results, 0,
                                   league_epoch, num_games)

                # Zip detailed CSV immediately after writing
                zip_path = detail_csv.replace(".csv", ".zip")
                ppo_utils.zip_csv_file(detail_csv, zip_path)

                stats = ppo_utils.aggregate_epoch_stats(all_results, num_games)
                pool.batch_update_stats(all_results, epoch=league_epoch + 1, agent_name=agent_type)
                ppo_utils.run_ppo_epoch_update(agent, all_results)

                # Guard: skip saving if weights went NaN during update
                if any(torch.isnan(p).any()
                       for p in agent.policy.parameters()):
                    print(f"    [{agent_type}] CRITICAL: NaN in policy weights after "
                          f"update at epoch {league_epoch + 1} — skipping checkpoint save. "
                          f"Consider rolling back to previous epoch.")
                    continue

                # Pool save
                if (league_epoch + 1) % config.POOL_SAVE_INTERVAL == 0:
                    pool.save(agent.get_policy_state_dict(),
                              league_epoch + 1,
                              agent_name=agent_type)
                    print(f"    [{agent_type}] saved to league pool "
                          f"(total pool size: {pool.size})")

                epoch_duration = time.perf_counter() - epoch_start
                ppo_utils.write_epoch_csv_row(
                    csv_paths[agent_type], league_epoch + 1, stats, epoch_duration,
                )

                # Save checkpoint
                torch.save({
                    "epoch": league_epoch + 1,
                    "model_state_dict": agent.get_policy_state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "scheduler_state_dict": agent.scheduler.state_dict(),
                    "rng_state_torch": torch.random.get_rng_state(),
                    "rng_state_numpy": np.random.get_state(),
                    "rng_state_python": random.getstate(),
                }, os.path.join(model_dirs[agent_type],
                                f"agent_epoch_{league_epoch + 1}.pt"))

                ppo_utils.prune_checkpoints(model_dirs[agent_type], config.CHECKPOINT_KEEP_LAST)

                print(f"    [{agent_type}] ep{league_epoch + 1} done in "
                      f"{epoch_duration:.0f}s. "
                      f"Reward: {stats['avg_reward']:.1f}  "
                      f"Blue: {stats['blue_win_rate']:.1%}  Red: {stats['red_win_rate']:.1%}  "
                      f"Tie: {stats['tie_rate']:.1%}  "
                      f"LR: {agent.scheduler.get_last_lr()[0]:.2e}")



            # ── Consolidated benchmark (once per league epoch, after all agents) ──
            if (league_epoch + 1) % config.BENCHMARK_INTERVAL == 0:
                print(f"\n  League benchmark (epoch {league_epoch + 1}) — "
                      f"{config.BENCHMARK_GAMES} games each...")
                bench = run_league_benchmark(
                    agents, device, n_actions, num_workers,
                    model_dirs, ref_paths, num_games=config.BENCHMARK_GAMES,
                    seed=seed
                )

                # Print per-agent rows
                for at in config.ACTIVE_AGENTS:
                    vr = bench.get(f"{at}_vs_random", {})
                    vs = bench.get(f"{at}_vs_self",   {})
                    if vr:
                        print(f"  {at:<12} vs random: "
                              f"Win {vr['win_rate']:.1%}  "
                              f"Loss {vr['loss_rate']:.1%}  "
                              f"Tie {vr['tie_rate']:.1%}  "
                              f"AvgSteps {vr['avg_steps']:.0f}")
                    if vs:
                        print(f"  {at:<12} vs self:   "
                              f"Win {vs['win_rate']:.1%}  "
                              f"Loss {vs['loss_rate']:.1%}  "
                              f"Tie {vs['tie_rate']:.1%}  "
                              f"AvgSteps {vs['avg_steps']:.0f}")

                # Print cross-agent rows
                for type_a, type_b in combinations(config.ACTIVE_AGENTS, 2):
                    v = bench.get(f"{type_a}_vs_{type_b}", {})
                    if v:
                        print(f"  {type_a} vs {type_b}: "
                              f"Win {v['win_rate']:.1%}  "
                              f"Loss {v['loss_rate']:.1%}  "
                              f"Tie {v['tie_rate']:.1%}  "
                              f"AvgSteps {v['avg_steps']:.0f}")

                # Write CSV row — order matches header exactly
                row = [league_epoch + 1]
                for at in config.ACTIVE_AGENTS:
                    for opp in ["random", "self"]:
                        v = bench.get(f"{at}_vs_{opp}", {})
                        row += [v.get("win_rate", ""), v.get("loss_rate", ""),
                                v.get("tie_rate", ""), v.get("avg_steps", "")]
                for type_a, type_b in combinations(config.ACTIVE_AGENTS, 2):
                    for ta, tb in [(type_a, type_b), (type_b, type_a)]:
                        v = bench.get(f"{ta}_vs_{tb}", {})
                        row += [v.get("win_rate", ""), v.get("loss_rate", ""),
                                v.get("tie_rate", ""), v.get("avg_steps", "")]
                with open(bench_league_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                # Auto-advance each agent's self-reference to current epoch
                for at in config.ACTIVE_AGENTS:
                    torch.save(agents[at].get_policy_state_dict(), ref_paths[at])
                print(f"  All reference models advanced to epoch {league_epoch + 1}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fully-deterministic", action="store_true", help="Force CuDNN determinism (slower)")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed, fully_deterministic=args.fully_deterministic)
    train_league(seed=args.seed)

