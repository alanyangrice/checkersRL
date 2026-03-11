"""PPO parallel training infrastructure: workers, worker context, orchestration."""

import os
import random
import threading
import time
from multiprocessing import Process, Queue, Event
from multiprocessing import shared_memory as _shm_module

import torch
import numpy as np

from rl.configs.ppo_config import PPOConfig

default_config = PPOConfig()
from rl.envs import CheckersEnv
from rl.training_utils.gpu_inference_server import PPOInferenceServer, SHUTDOWN
from rl.training_utils.worker_pool import BaseWorkerContext

from rl.algorithms.ppo.agent import (PPOAgent, get_device,
                                        get_policy_state_dict,
                                        load_policy_state_dict)
from rl.algorithms.ppo.torch_helpers import torch_compile_available
from rl.algorithms.ppo.memory import Memory
from rl.networks import PPOPolicyNetwork
from rl.utils.action_utils import random_action_from_mask, uniform_log_prob
from checkers_game.constants import BLUE, RED, NUM_ACTIONS

# ─────────────────────────────────────────────────────────────────────
# Worker process — plays games, uses GPU server for agent inference
# ─────────────────────────────────────────────────────────────────────

def load_opponent(n_actions, opp_path, cache):
    """Return a cached CPU opponent agent."""
    if opp_path not in cache:
        opp = PPOAgent((4, 8, 8), n_actions, device=torch.device("cpu"))
        checkpoint = torch.load(opp_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        load_policy_state_dict(opp.policy, state_dict)
        opp.policy.eval()
        cache[opp_path] = opp
    return cache[opp_path]


def play_game(worker_id, request_queue, response_queue,
               n_actions, epoch, opponent_model_path, opp_cache,
               reward_config=None,
               state_buf=None, mask_buf=None, config=None):
    config = config or default_config
    """Play one game using the GPU server for agent inference."""
    env = CheckersEnv(reward_config=reward_config)

    # Optionally load a pool opponent (CPU-local, cached)
    opponent = None
    opponent_color = None
    opponent_label = "self"
    if opponent_model_path is not None:
        opponent = load_opponent(n_actions, opponent_model_path, opp_cache)
        opponent_color = BLUE if random.random() < 0.5 else RED
        opponent_label = os.path.splitext(os.path.basename(opponent_model_path))[0]

    blue_memory, red_memory = Memory(), Memory()
    curriculum_opts = config.get_curriculum_options(epoch)
    state, _ = env.reset(options=curriculum_opts)
    done = False

    episode_reward, episode_steps, max_episode_move_reward = 0, 0, float('-inf')
    log_prob_list = []
    reward_colors = []
    blue_win, red_win = 0, 0

    while not done:
        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            next_state, reward, done, _, info = env.step(0)
            episode_reward += reward
            episode_steps += 1

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
        value = 0.0  # default; overwritten by server response for agent turns

        if is_opponent_turn:
            if random.random() < config.POOL_EPSILON:
                action = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                with torch.no_grad():
                    action, log_prob, _ = opponent.select_action(state, action_mask)
                if isinstance(log_prob, torch.Tensor):
                    log_prob = log_prob.item()
        else:
            # Current agent — use GPU server or random exploration
            epsilon = config.get_epsilon(epoch)

            # ── GPU inference via server ──────────────────────
            # Always query the server to get the accurate value estimate
            # (prevents value clipping corruption during PPO updates)
            if state_buf is not None:
                state_buf[worker_id][:] = state
                mask_buf[worker_id][:]  = action_mask
                request_queue.put(worker_id)
            else:
                request_queue.put((worker_id, state, action_mask))
            
            srv_action, srv_log_prob, _, value = response_queue.get()

            if random.random() < epsilon:
                action   = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                action   = srv_action
                log_prob = srv_log_prob

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
                blue_memory.add(state, action, reward, log_prob, done, action_mask, value=value)
                reward_colors.append("blue")
            else:
                red_memory.add(state, action, reward, log_prob, done, action_mask, value=value)
                reward_colors.append("red")

        # Apply per-color reward adjustments computed by the env
        blue_adj = info.get("blue_reward_adjustment", 0.0)
        red_adj = info.get("red_reward_adjustment", 0.0)
        if blue_adj != 0.0 and len(blue_memory) > 0:
            blue_memory.rewards[-1] += blue_adj
        if red_adj != 0.0 and len(red_memory) > 0:
            red_memory.rewards[-1] += red_adj

        # Only log agent log_probs (opponent steps excluded) so log_probs
        # aligns with rewards_list in the detailed CSV
        if not is_opponent_acting:
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

    # Compute agent_win for prioritized opponent sampling stats.
    # Only meaningful when there is a pool opponent; self-play has no "agent side".
    agent_win = None
    if opponent is not None:
        agent_color = RED if opponent_color == BLUE else BLUE
        agent_win   = blue_win if agent_color == BLUE else red_win

    return {
        "blue_memory": blue_memory,
        "red_memory": red_memory,
        "rewards": {
            "blue": float(blue_memory.rewards.sum()) if len(blue_memory) > 0 else 0.0,
            "red":  float(red_memory.rewards.sum())  if len(red_memory)  > 0 else 0.0,
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
        "opponent_path": opponent_model_path,  # None for self-play
        "agent_win": agent_win,                # None for self-play
    }


def play_benchmark_game(worker_id, request_queue, response_queue,
                         n_actions, opponent_type, opponent_model_path, opp_cache,
                         reward_config=None,
                         state_buf=None, mask_buf=None, agent_color=None):
    """Play a single benchmark game using GPU server for agent inference."""
    env = CheckersEnv(reward_config=None)  # benchmarks only track outcomes, not rewards

    opponent = None
    if opponent_type == "model" and opponent_model_path is not None:
        opponent = load_opponent(n_actions, opponent_model_path, opp_cache)

    if agent_color is None:
        agent_color = BLUE if random.random() < 0.5 else RED
    opponent_color = RED if agent_color == BLUE else BLUE

    # In benchmark games, we only want to track if the *agent* won or lost
    state, _ = env.reset()
    done = False
    steps = 0
    info = {}

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
            # Agent plays greedily via GPU server
            if state_buf is not None:
                state_buf[worker_id][:] = state
                mask_buf[worker_id][:]  = action_mask
                request_queue.put(worker_id)
            else:
                request_queue.put((worker_id, state, action_mask))
            action, _, _, _ = response_queue.get()
        elif opponent_type == "random":
            action = random_action_from_mask(action_mask)
        else:
            with torch.no_grad():
                action, _, _ = opponent.select_action(state, action_mask, deterministic=False)

        next_state, _, done, _, info = env.step(action)
        steps += 1
        state = next_state

    winner = info.get("winner", "Tie")
    # For agent_win/opponent_win, check against exactly who won
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


# Task-queue sentinels
# InferenceServer's request-queue sentinel is SHUTDOWN
from rl.training_utils.parallel_utils import WORKER_EXIT, WORKER_BATCH_DONE


def worker_fn(worker_id, request_queue, response_queue, results_queue,
              n_actions, task_queue, progress_queue,
              shm_state_name=None, shm_mask_name=None, num_workers=None, config=None, seed=None):
    if seed is not None:
        from rl.utils.seed_utils import set_seed
        set_seed(seed + worker_id)

    config = config or default_config
    """Persistent worker process entry point.

    Workers pull tasks from task_queue in a loop.  Three sentinel values:
      _WORKER_EXIT       (None)         → terminate the process
      _WORKER_BATCH_DONE ("BATCH_DONE") → echo back and wait for next batch
      (task_index, task_dict)           → play one game and return result

    When shm_state_name/shm_mask_name are provided, the worker attaches to the
    shared-memory buffers and writes state/mask directly without pickling.

    task_dict keys:
        mode:             "train" or "benchmark"
        epoch:            (train only)
        opponent_model_path: path or None
        opponent_type:    (benchmark only) "random" or "model"
        reward_config:    optional dict of reward parameters for the env
    """
    # Attach to shared-memory buffers if provided
    from rl.training_utils.parallel_utils import attach_shm_buffers
    _shm_state, _shm_mask, _state_buf, _mask_buf = attach_shm_buffers(
        shm_state_name, shm_mask_name, num_workers, obs_shape=(4, 8, 8), num_actions=n_actions
    )

    opp_cache = {}  # per-worker opponent model cache
    games_played = 0

    while True:
        item = task_queue.get()

        if item is WORKER_EXIT:
            if _shm_state is not None:
                _shm_state.close()
                _shm_mask.close()
            break

        if item == WORKER_BATCH_DONE:
            # Echo back so WorkerContext.run_tasks() knows this worker is idle
            results_queue.put(WORKER_BATCH_DONE)
            continue

        task_index, task = item
        reward_config = task.get("reward_config", None)

        if task["mode"] == "train":
            result = play_game(
                worker_id, request_queue, response_queue,
                n_actions, task["epoch"], task["opponent_model_path"], opp_cache,
                reward_config=reward_config,
                state_buf=_state_buf, mask_buf=_mask_buf, config=config
            )
        else:  # benchmark
            result = play_benchmark_game(
                worker_id, request_queue, response_queue,
                n_actions, task["opponent_type"],
                task["opponent_model_path"], opp_cache,
                reward_config=reward_config,
                state_buf=_state_buf, mask_buf=_mask_buf,
                agent_color=task.get("agent_color", None)
            )

        results_queue.put((task_index, result))
        games_played += 1

        if games_played % 100 == 0:
            progress_queue.put((worker_id, games_played))


# ─────────────────────────────────────────────────────────────────────
# WorkerContext — long-lived worker pool with hot-swappable model weights
# ─────────────────────────────────────────────────────────────────────

class WorkerContext(BaseWorkerContext):
    """Long-lived worker pool + inference server for use across multiple batches.

    In league training, run_games_on_gpu is called once per agent per epoch,
    each time spawning and destroying all worker processes.  On Windows (spawn
    start method) this costs ~1-2 seconds per worker per call.  WorkerContext
    keeps workers alive across calls and hot-swaps model weights via
    InferenceServer.update_weights() between batches.

    The server maintains its own PPOPolicyNetwork copy — entirely separate from
    agent.policy — so agent.update()'s backward pass cannot race with the
    server's load_state_dict() call.

    Usage (context manager):
        with WorkerContext(initial_state_dict, device, n_actions, num_workers) as ctx:
            ctx.update_model(agent.policy.state_dict())
            results = ctx.run_tasks(game_tasks, label="ep1")
    """

    def __init__(self, initial_state_dict, device, n_actions, num_workers, config=None, seed=None):
        super().__init__(device, num_workers, obs_shape=(4, 8, 8), num_actions=n_actions)
        config = config or default_config
        self.config = config

        self._server_policy = PPOPolicyNetwork((4, 8, 8), n_actions).to(device)
        load_policy_state_dict(self._server_policy, initial_state_dict)
        self._server_policy.eval()
        if torch_compile_available():
            self._server_policy = torch.compile(self._server_policy, mode="default")

        self._server = PPOInferenceServer(
            self._server_policy, device,
            self._request_queue, self._response_queues,
            self._stop_event, max_batch=num_workers * 4,
            state_buf=self._state_buf, mask_buf=self._mask_buf,
        )
        self._server.start()

        for wid in range(num_workers):
            p = Process(
                target=worker_fn,
                args=(wid, self._request_queue, self._response_queues[wid],
                      self._results_queue, n_actions,
                      self._task_queue, self._progress_queue,
                      self._shm_state.name, self._shm_mask.name, num_workers, config, seed),
            )
            p.start()
            self._workers.append(p)

    def update_model(self, state_dict):
        """Hot-swap model weights into the inference server's own policy copy."""
        self._server.update_weights(state_dict)

    def set_deterministic(self, deterministic: bool):
        self._server.deterministic = deterministic

    def run_tasks(self, game_tasks, label="games"):
        from rl.training_utils.parallel_utils import WORKER_BATCH_DONE
        return self._distribute_tasks_and_collect(game_tasks, label, log_interval=500, batch_done_sentinel=WORKER_BATCH_DONE)
