"""Multi-agent league self-play training for the Checkers PPO agent.

Architecture:
    - N agent types defined in cfg.LEAGUE_AGENTS (training_config.py).
    - One interleaved training loop: each league epoch trains every agent
      type for one epoch in sequence, then repeats.
    - All agents share a single opponent pool (opponent_pool_league/).
      Pool checkpoints are namespaced by agent type so cross-style sampling
      is automatic: a tactical agent might face an aggressive opponent.
    - Each agent gets its own:
        checkpoint directory:  PPO_saved_models_{agent_type}/
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
    python -m RL_models.PPO_Model.train_league
"""

import copy
import os
import csv
import time
import random
import zipfile
from datetime import datetime
from multiprocessing import cpu_count

import torch

from RL_models.PPO_Model.Agent import (PPOAgent, get_device,
                                        get_policy_state_dict,
                                        load_policy_state_dict)
from RL_models.PPO_Model.Memory import Memory
from RL_models.PPO_Model.OpponentPool import OpponentPool
from RL_models.PPO_Model import training_config as cfg
from itertools import combinations

from RL_models.PPO_Model.train_gpu_parallel import (
    WorkerContext,
    _prune_checkpoints,
    run_benchmark,
    write_detailed_csv,
    zip_csv_file,
)
from checkers_game.constants import NUM_ACTIONS


# ─────────────────────────────────────────────────────────────────────
# Per-agent state (checkpoint, CSVs, PPOAgent instance)
# ─────────────────────────────────────────────────────────────────────

def _build_agent(agent_type, n_actions, device):
    """Create a PPOAgent using the hyperparameters for agent_type."""
    ac = cfg.LEAGUE_AGENTS[agent_type]
    return PPOAgent(
        (4, 8, 8), n_actions, device=device,
        lr=cfg.LEARNING_RATE,
        gamma=ac.get("gamma", cfg.GAMMA),
        eps_clip=cfg.EPS_CLIP,
        K_epochs=cfg.K_EPOCHS,
        gae_lambda=cfg.GAE_LAMBDA,
        augment=cfg.AUGMENT,
        augment_noise=cfg.AUGMENT_NOISE,
        mini_batch_size=cfg.MINI_BATCH_SIZE,
        entropy_bonus=ac.get("entropy_bonus", 0.01),
    )


def _latest_checkpoint_path(model_dir):
    """Return the path to the latest agent_epoch_N.pt file, or None."""
    files = [f for f in os.listdir(model_dir)
             if f.startswith("agent_epoch_") and f.endswith(".pt")]
    if not files:
        return None
    latest = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    return os.path.join(model_dir, latest)


def _load_checkpoint(agent, model_dir, device):
    """Load the latest checkpoint in model_dir into agent. Returns start_epoch."""
    checkpoints = [f for f in os.listdir(model_dir)
                   if f.startswith("agent_epoch_") and f.endswith(".pt")]
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    cp = torch.load(os.path.join(model_dir, latest), map_location=device,
                    weights_only=False)
    load_policy_state_dict(agent.policy, cp["model_state_dict"])
    agent.optimizer.load_state_dict(cp["optimizer_state_dict"])
    if "scheduler_state_dict" in cp:
        agent.scheduler.load_state_dict(cp["scheduler_state_dict"])
    print(f"    Resumed {os.path.basename(model_dir)} from epoch {cp['epoch']}")
    return cp["epoch"]


# ─────────────────────────────────────────────────────────────────────
# Main league training loop
# ─────────────────────────────────────────────────────────────────────

def _run_league_benchmark(agents, device, n_actions, num_workers,
                          model_dirs, ref_paths, num_games):
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
    results = {}
    agent_list = cfg.ACTIVE_AGENTS

    # Each agent vs random + vs its own reference (one GPU server session each)
    for agent_type in agent_list:
        has_ref = os.path.exists(ref_paths[agent_type])
        bench = run_benchmark(
            agents[agent_type].policy, device, n_actions, num_workers,
            reference_model_path=ref_paths[agent_type] if has_ref else None,
            num_games=num_games,
            include_random=True,
        )
        results[f"{agent_type}_vs_random"] = bench["vs_random"]
        if "vs_reference" in bench:
            results[f"{agent_type}_vs_self"] = bench["vs_reference"]

    # Each unique pair (A, B) — one direction; derive reverse for free
    for type_a, type_b in combinations(agent_list, 2):
        ckpt_b = _latest_checkpoint_path(model_dirs[type_b])
        if not ckpt_b:
            continue
        bench_ab = run_benchmark(
            agents[type_a].policy, device, n_actions, num_workers,
            reference_model_path=None,
            num_games=num_games,
            extra_opponents={type_b: ckpt_b},
            include_random=False,
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


def train_league(num_league_epochs=cfg.NUM_EPOCHS,
                 num_games=cfg.NUM_GAMES,
                 n_actions=NUM_ACTIONS,
                 num_workers=None):
    """Interleaved multi-agent league training.

    Each 'league epoch' trains every ACTIVE_AGENT for one epoch in sequence.
    All agents share a unified opponent pool for cross-style exposure.
    """
    if num_workers is None:
        num_workers = cfg.get_num_workers_gpu()

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Active agents: {cfg.ACTIVE_AGENTS}")
    print(f"Workers: {num_workers}  |  Games per agent per epoch: {num_games}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Shared opponent pool ────────────────────────────────────────
    league_pool_dir = os.path.join(base_dir, "opponent_pool_league")
    pool = OpponentPool(league_pool_dir, max_size=cfg.POOL_MAX_SIZE)

    # ── Per-agent state ─────────────────────────────────────────────
    agents      = {}   # agent_type → PPOAgent
    model_dirs  = {}   # agent_type → checkpoint dir path
    start_epochs = {}  # agent_type → epoch to resume from
    csv_paths   = {}   # agent_type → training_progress CSV
    ref_paths   = {}   # agent_type → reference model path
    detail_dirs = {}   # agent_type → detailed CSV folder

    for agent_type in cfg.ACTIVE_AGENTS:
        agents[agent_type]     = _build_agent(agent_type, n_actions, device)
        model_dir              = os.path.join(base_dir, f"PPO_saved_models_{agent_type}")
        model_dirs[agent_type] = model_dir
        os.makedirs(model_dir, exist_ok=True)

        start_epochs[agent_type] = _load_checkpoint(
            agents[agent_type], model_dir, device
        )

        csv_paths[agent_type] = os.path.join(base_dir, f"training_progress_{agent_type}.csv")
        ref_paths[agent_type] = os.path.join(model_dir, "reference_model.pt")
        detail_dirs[agent_type] = os.path.join(base_dir, f"training_progress_detailed_{agent_type}")
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
    bench_league_path = os.path.join(base_dir, "benchmark_league.csv")
    if not os.path.exists(bench_league_path):
        # Header: epoch + per-agent (vs random + vs self) + pairwise (both dirs)
        header = ["epoch"]
        for at in cfg.ACTIVE_AGENTS:
            for opp in ["random", "self"]:
                header += [f"{at}_vs_{opp}_win", f"{at}_vs_{opp}_loss",
                           f"{at}_vs_{opp}_tie", f"{at}_vs_{opp}_avg_steps"]
        for type_a, type_b in combinations(cfg.ACTIVE_AGENTS, 2):
            for ta, tb in [(type_a, type_b), (type_b, type_a)]:
                header += [f"{ta}_vs_{tb}_win", f"{ta}_vs_{tb}_loss",
                           f"{ta}_vs_{tb}_tie", f"{ta}_vs_{tb}_avg_steps"]
        with open(bench_league_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Determine the minimum start_epoch across all agents so that no agent
    # skips epochs — all agents advance together from the earliest checkpoint.
    global_start_epoch = min(start_epochs.values())

    def _build_tasks(agent_type, league_epoch):
        """Build the task list for one agent-epoch."""
        reward_cfg = cfg.LEAGUE_AGENTS[agent_type]
        tasks = []
        for _ in range(num_games):
            opp_path = None
            if pool.should_use_opponent(prob=cfg.get_league_pool_prob(league_epoch)):
                opp_path = pool.sample(agent_name=agent_type)
            tasks.append({
                "mode": "train",
                "epoch": league_epoch,
                "opponent_model_path": opp_path,
                "reward_config": reward_cfg,
            })
        return tasks

    # ── League training loop (WorkerContext, sequential) ──────────────────
    # Workers are kept alive across all agent-epochs within the entire run,
    # eliminating per-agent spawn overhead (~1-2s/worker on Windows).
    #
    # Collection and GPU update are strictly sequential: all CPU workers
    # finish collecting games before the PPO update starts, so CPU and GPU
    # are never both under full load at the same time.

    first_agent = cfg.ACTIVE_AGENTS[0]
    initial_sd  = agents[first_agent].get_policy_state_dict()

    with WorkerContext(initial_sd, device, n_actions, num_workers) as ctx:

        for league_epoch in range(global_start_epoch, num_league_epochs):
            print(f"\n{'='*60}")
            print(f"  League epoch {league_epoch + 1}  |  pool size: {pool.size}")
            print(f"{'='*60}")

            for agent_idx, agent_type in enumerate(cfg.ACTIVE_AGENTS):
                agent      = agents[agent_type]
                reward_cfg = cfg.LEAGUE_AGENTS[agent_type]

                print(f"\n  [{agent_type}] epoch {league_epoch + 1}  "
                      f"(gamma={reward_cfg['gamma']}, "
                      f"tie_base={reward_cfg['tie_base']})")

                epoch_start = time.perf_counter()

                # Collect games with up-to-date weights, then wait for all workers
                # to finish before starting the GPU PPO update.
                ctx.update_model(agent.get_policy_state_dict())
                all_results = ctx.run_tasks(
                    _build_tasks(agent_type, league_epoch),
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

                write_detailed_csv(detail_csv, all_results, 0,
                                   league_epoch, num_games)

                # Zip detailed CSV immediately after writing
                zip_path = detail_csv.replace(".csv", ".zip")
                zip_csv_file(detail_csv, zip_path)

                # Aggregate statistics
                total_rewards = {"blue": 0, "red": 0}
                blue_wins = red_wins = ties = total_steps = pool_games = 0
                max_move_reward = float('-inf')

                for r in all_results:
                    total_rewards["blue"] += r["rewards"]["blue"]
                    total_rewards["red"]  += r["rewards"]["red"]
                    total_steps       += r["episode_steps"]
                    blue_wins         += r["blue_win"]
                    red_wins          += r["red_win"]
                    ties              += 1 - (r["blue_win"] or r["red_win"])
                    max_move_reward    = max(max_move_reward,
                                            r["max_episode_move_reward"])
                    if r["opponent"] != "self":
                        pool_games += 1

                # Update prioritized opponent sampling stats (single JSON round-trip)
                pool.batch_update_stats(all_results, agent_name=agent_type)

                # PPO update (GPU) — workers are idle while this runs
                combined_memory = Memory()
                for r in all_results:
                    combined_memory.extend(r["blue_memory"])
                    combined_memory.extend(r["red_memory"])
                agent.update(combined_memory)
                agent.step_scheduler()

                # Guard: skip saving if weights went NaN during update
                if any(torch.isnan(p).any()
                       for p in agent.policy.parameters()):
                    print(f"    [{agent_type}] CRITICAL: NaN in policy weights after "
                          f"update at epoch {league_epoch + 1} — skipping checkpoint save. "
                          f"Consider rolling back to previous epoch.")
                    continue

                # Pool save
                if (league_epoch + 1) % cfg.POOL_SAVE_INTERVAL == 0:
                    pool.save(agent.get_policy_state_dict(),
                              league_epoch + 1,
                              agent_name=agent_type)
                    print(f"    [{agent_type}] saved to league pool "
                          f"(total pool size: {pool.size})")

                # Log epoch statistics
                epoch_duration = time.perf_counter() - epoch_start
                avg_reward      = (total_rewards["blue"] + total_rewards["red"]) / num_games
                avg_blue_reward = total_rewards["blue"] / num_games
                avg_red_reward  = total_rewards["red"]  / num_games
                avg_steps       = total_steps / num_games
                blue_win_rate   = blue_wins / num_games
                red_win_rate    = red_wins  / num_games
                tie_rate        = ties      / num_games

                with open(csv_paths[agent_type], "a", newline="") as f:
                    csv.writer(f).writerow([
                        league_epoch + 1, avg_reward, avg_blue_reward,
                        avg_red_reward, avg_steps, blue_win_rate,
                        red_win_rate, tie_rate, max_move_reward,
                        round(epoch_duration, 2),
                    ])

                # Save checkpoint
                torch.save({
                    "epoch": league_epoch + 1,
                    "model_state_dict": agent.get_policy_state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "scheduler_state_dict": agent.scheduler.state_dict(),
                }, os.path.join(model_dirs[agent_type],
                                f"agent_epoch_{league_epoch + 1}.pt"))

                # Prune old checkpoints — keep only the most recent N
                _prune_checkpoints(model_dirs[agent_type], cfg.CHECKPOINT_KEEP_LAST)

                print(f"    [{agent_type}] ep{league_epoch + 1} done in "
                      f"{epoch_duration:.0f}s. "
                      f"Reward: {avg_reward:.1f}  "
                      f"Blue: {blue_win_rate:.1%}  Red: {red_win_rate:.1%}  "
                      f"Tie: {tie_rate:.1%}  "
                      f"LR: {agent.scheduler.get_last_lr()[0]:.2e}")



            # ── Consolidated benchmark (once per league epoch, after all agents) ──
            if (league_epoch + 1) % cfg.BENCHMARK_INTERVAL == 0:
                print(f"\n  League benchmark (epoch {league_epoch + 1}) — "
                      f"{cfg.BENCHMARK_GAMES} games each...")
                bench = _run_league_benchmark(
                    agents, device, n_actions, num_workers,
                    model_dirs, ref_paths, num_games=cfg.BENCHMARK_GAMES,
                )

                # Print per-agent rows
                for at in cfg.ACTIVE_AGENTS:
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
                for type_a, type_b in combinations(cfg.ACTIVE_AGENTS, 2):
                    v = bench.get(f"{type_a}_vs_{type_b}", {})
                    if v:
                        print(f"  {type_a} vs {type_b}: "
                              f"Win {v['win_rate']:.1%}  "
                              f"Loss {v['loss_rate']:.1%}  "
                              f"Tie {v['tie_rate']:.1%}  "
                              f"AvgSteps {v['avg_steps']:.0f}")

                # Write CSV row — order matches header exactly
                row = [league_epoch + 1]
                for at in cfg.ACTIVE_AGENTS:
                    for opp in ["random", "self"]:
                        v = bench.get(f"{at}_vs_{opp}", {})
                        row += [v.get("win_rate", ""), v.get("loss_rate", ""),
                                v.get("tie_rate", ""), v.get("avg_steps", "")]
                for type_a, type_b in combinations(cfg.ACTIVE_AGENTS, 2):
                    for ta, tb in [(type_a, type_b), (type_b, type_a)]:
                        v = bench.get(f"{ta}_vs_{tb}", {})
                        row += [v.get("win_rate", ""), v.get("loss_rate", ""),
                                v.get("tie_rate", ""), v.get("avg_steps", "")]
                with open(bench_league_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                # Auto-advance each agent's self-reference to current epoch
                for at in cfg.ACTIVE_AGENTS:
                    torch.save(agents[at].get_policy_state_dict(), ref_paths[at])
                print(f"  All reference models advanced to epoch {league_epoch + 1}")


if __name__ == "__main__":
    train_league()

