"""PPO evaluation and benchmarking utilities.

Provides:
  - play_benchmark_game: single game vs random or model opponent (CPU, for Pool workers)
  - run_benchmark_cpu: evaluate agent vs random + reference using multiprocessing.Pool

For GPU-accelerated benchmarking (WorkerContext), see train_gpu_parallel.run_benchmark.
"""

import os
import random
from multiprocessing import Pool

import torch

from checkers_game.constants import BLUE, RED

from rl.envs import CheckersEnv
from rl.configs.ppo_config import PPOConfig
default_config = PPOConfig()
from rl.algorithms.ppo.agent import PPOAgent
from rl.algorithms.ppo.torch_helpers import get_policy_state_dict, load_policy_state_dict
from rl.utils.action_utils import random_action_from_mask


# Worker-level global for benchmark Pool (set by _init_benchmark_worker)
_benchmark_agent = None
_benchmark_opponents = {}


def init_benchmark_worker(n_actions, temp_model_path, seed=None):
    """Pool initializer: load the agent model once per worker process."""
    if seed is not None:
        from rl.utils.seed_utils import set_seed
        set_seed(seed + os.getpid())
    global _benchmark_agent
    _benchmark_agent = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
    ckpt = torch.load(temp_model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    load_policy_state_dict(_benchmark_agent.policy, sd)
    _benchmark_agent.policy.eval()


def get_benchmark_opponent(n_actions, opp_path):
    """Return cached opponent agent for benchmark, loading from disk on first use."""
    global _benchmark_opponents
    if opp_path not in _benchmark_opponents:
        opp = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
        checkpoint = torch.load(opp_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        load_policy_state_dict(opp.policy, state_dict)
        opp.policy.eval()
        _benchmark_opponents[opp_path] = opp
    return _benchmark_opponents[opp_path]


def play_benchmark_game(n_actions, opponent_type, opponent_model_path=None, agent_color=None):
    """Play a single benchmark game (no memory/training, just win/loss/tie).

    Uses the worker-level _benchmark_agent set by _init_benchmark_worker.
    Must be called from a Pool worker that was initialized with that function.

    Args:
        n_actions: Size of the action space.
        opponent_type: "random" or "model".
        opponent_model_path: Path to opponent model (only for opponent_type="model").
        agent_color: BLUE or RED. If None, chosen randomly.

    Returns:
        dict with keys: agent_color, agent_win, opponent_win, tie, steps
    """
    agent = _benchmark_agent

    opponent = None
    if opponent_type == "model" and opponent_model_path is not None:
        opponent = get_benchmark_opponent(n_actions, opponent_model_path)

    if agent_color is None:
        agent_color = BLUE if random.random() < 0.5 else RED
    opponent_color = RED if agent_color == BLUE else BLUE

    env = CheckersEnv()
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
            with torch.no_grad():
                # Agent always plays deterministically in evaluation
                action, _, _ = agent.select_action(state, action_mask, deterministic=False)
        elif opponent_type == "random":
            action = random_action_from_mask(action_mask)
        else:
            with torch.no_grad():
                # Opponent always plays deterministically in evaluation
                action, _, _ = opponent.select_action(state, action_mask, deterministic=False)

        next_state, _, done, _, info = env.step(action)
        steps += 1
        state = next_state

    winner = info.get("winner", "Tie")
    agent_win = 1 if winner == agent_color else 0
    opponent_win = 1 if winner == opponent_color else 0
    
    # A tie is everything else (actual "Tie", or "None" if max moves hit, or just neither won)
    tie = 1 if (agent_win == 0 and opponent_win == 0) else 0

    return {
        "agent_color": "BLUE" if agent_color == BLUE else "RED",
        "agent_win": agent_win,
        "opponent_win": opponent_win,
        "tie": tie,
        "steps": steps,
    }


def run_benchmark_cpu(agent, n_actions, num_processes, reference_model_path, num_games=200, seed=None):
    """Evaluate the agent against random and reference opponents (CPU, multiprocessing).

    Saves the agent to a temp file, spawns a Pool of workers that load it,
    then runs games in parallel.

    Returns:
        dict with keys: vs_random, vs_reference (if reference exists)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_path = os.path.join(base_dir, "training_results", "ppo", "ppo_saved_models_parallel", f"_benchmark_temp_{os.getpid()}.pt")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    torch.save({"model_state_dict": get_policy_state_dict(agent.policy)}, temp_path)

    results = {}
    try:
        with Pool(num_processes, initializer=init_benchmark_worker,
                  initargs=(n_actions, temp_path, seed)) as p:
            random_args = [(n_actions, "random", None, BLUE if i < num_games // 2 else RED) for i in range(num_games)]
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

            if reference_model_path and os.path.exists(reference_model_path):
                ref_args = [(n_actions, "model", reference_model_path, BLUE if i < num_games // 2 else RED) for i in range(num_games)]
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


# ─────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────

def run_benchmark(ctx, device, n_actions, num_workers,
                  reference_model_path, num_games=200,
                  extra_opponents=None, include_random=True, config=None):
    config = config or default_config
    """Evaluate the agent against multiple opponents using the GPU server.

    Args:
        ctx: WorkerContext instance with the model already loaded.
        reference_model_path: Path to the self-reference model (own past weights).
        extra_opponents: Optional dict of {label: model_path} for additional
                         opponents to benchmark against (e.g. other league agent
                         types). Each gets num_games games. Missing paths are
                         skipped gracefully.

    Returns:
        dict with keys "vs_random", "vs_reference", and one key per
        extra_opponents entry (only if the path exists).
        Each value: {"win_rate", "loss_rate", "tie_rate", "avg_steps"}.
    """
    tasks = []

    # vs Random (optional — skip for pairwise cross-agent benchmarks)
    if include_random:
        for i in range(num_games):
            tasks.append({
                "mode": "benchmark",
                "opponent_type": "random",
                "opponent_model_path": None,
                "agent_color": BLUE if i < num_games // 2 else RED,
            })

    # vs Reference model
    has_ref = reference_model_path and os.path.exists(reference_model_path)
    if has_ref:
        for i in range(num_games):
            tasks.append({
                "mode": "benchmark",
                "opponent_type": "model",
                "opponent_model_path": reference_model_path,
                "agent_color": BLUE if i < num_games // 2 else RED,
            })

    # vs Extra opponents (e.g. other league agent types)
    valid_extras = {}
    if extra_opponents:
        for label, path in extra_opponents.items():
            if path and os.path.exists(path):
                valid_extras[label] = path
                for i in range(num_games):
                    tasks.append({
                        "mode": "benchmark",
                        "opponent_type": "model",
                        "opponent_model_path": path,
                        "agent_color": BLUE if i < num_games // 2 else RED,
                    })

    ctx.set_deterministic(False)
    all_results = ctx.run_tasks(tasks, label="benchmark")

    def tally(slice_results):
        return {
            "win_rate":  sum(r["agent_win"]    for r in slice_results) / num_games,
            "loss_rate": sum(r["opponent_win"]  for r in slice_results) / num_games,
            "tie_rate":  sum(r["tie"]           for r in slice_results) / num_games,
            "avg_steps": sum(r["steps"]         for r in slice_results) / num_games,
        }

    results = {}
    offset = 0

    if include_random:
        results["vs_random"] = tally(all_results[offset : offset + num_games])
        offset += num_games

    if has_ref:
        results["vs_reference"] = tally(all_results[offset : offset + num_games])
        offset += num_games

    for label in valid_extras:
        results[f"vs_{label}"] = tally(all_results[offset : offset + num_games])
        offset += num_games

    return results
