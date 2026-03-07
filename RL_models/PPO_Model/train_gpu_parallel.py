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
    python -m RL_models.PPO_Model.train_gpu_parallel
"""

import os
import csv
import random
import zipfile
import threading
import time
from datetime import datetime
from multiprocessing import Process, Queue, Event
from multiprocessing import shared_memory as _shm_module
from queue import Empty as QueueEmpty

import torch
import numpy as np
from torch.distributions import Categorical

from RL_models.checkers_env import CheckersEnv
from RL_models.PPO_Model.Agent import (PPOAgent, get_device,
                                        get_policy_state_dict,
                                        load_policy_state_dict,
                                        _torch_compile_available)
from RL_models.PPO_Model.Memory import Memory
from RL_models.PPO_Model.OpponentPool import OpponentPool
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork
from RL_models.PPO_Model import training_config as cfg
from checkers_game.constants import BLUE, RED, NUM_ACTIONS


# ─────────────────────────────────────────────────────────────────────
# Sentinel values
# ─────────────────────────────────────────────────────────────────────
_SHUTDOWN = "SHUTDOWN"


# ─────────────────────────────────────────────────────────────────────
# GPU Inference Server (runs as a thread in the main process)
# ─────────────────────────────────────────────────────────────────────

class InferenceServer(threading.Thread):
    """Batches inference requests from workers and runs them on GPU.

    Standard protocol (request items are (worker_id, state_np, mask_np)):
        request_queue items:  (worker_id, state_np, mask_np)  OR just worker_id (shm mode)
        response_queues[wid]: (action_int, log_prob_float, entropy_float, value_float)

    Shared-memory protocol (when state_buf/mask_buf are provided):
        Workers write state/mask into pre-allocated numpy arrays backed by
        shared memory, then put only the integer worker_id on the request queue.
        The server reads directly from the shared arrays — zero pickle overhead.
    """

    def __init__(self, model, device, request_queue, response_queues,
                 stop_event, max_batch=64, max_wait_ms=3.0,
                 state_buf=None, mask_buf=None):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.stop_event = stop_event
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        # Shared-memory buffers (None = classic pickle-based protocol)
        self._state_buf = state_buf
        self._mask_buf  = mask_buf
        # Pre-allocate the mask constant once
        self._neg_inf = torch.tensor(-1e10, device=device)

    def run(self):
        self.model.eval()
        req_q = self.request_queue

        while not self.stop_event.is_set():
            batch = []

            # Block until at least one request arrives (or timeout)
            try:
                item = req_q.get(timeout=0.01)
                if item == _SHUTDOWN:
                    break
                batch.append(item)
            except QueueEmpty:
                continue

            # Drain additional pending requests up to max_batch.
            # Use the live queue depth to avoid waiting the full timeout for a
            # batch that will never fill (e.g. near the end of an epoch with few
            # remaining stragglers).
            effective_max = max(8, min(req_q.qsize() + len(batch) + 1,
                                       self.max_batch))
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < effective_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = req_q.get_nowait()
                    if item == _SHUTDOWN:
                        # Put it back so the outer loop sees it
                        req_q.put(_SHUTDOWN)
                        break
                    batch.append(item)
                except QueueEmpty:
                    # Brief sleep then try once more before committing
                    if remaining > 0.0005:
                        time.sleep(0.0003)
                    else:
                        break

            if not batch:
                continue

            # ── Batched forward pass ──────────────────────────────
            if self._state_buf is not None:
                # Shared-memory path: batch items are plain worker_ids (ints).
                # Slicing the shared numpy array is zero-copy.
                worker_ids = list(batch)
                states_np  = self._state_buf[worker_ids].copy()
                masks_np   = self._mask_buf[worker_ids].copy()
            else:
                # Classic path: batch items are (worker_id, state_np, mask_np).
                worker_ids = [b[0] for b in batch]
                states_np  = np.stack([b[1] for b in batch])
                masks_np   = np.stack([b[2] for b in batch])

            states_t = torch.from_numpy(states_np).to(self.device)
            masks_t = torch.from_numpy(masks_np).to(self.device)

            with torch.no_grad():
                logits, values_t = self.model(states_t)

            if torch.isnan(logits).any():
                print(f"[InferenceServer] NaN in logits — sending random actions for this batch")
                for i, wid in enumerate(worker_ids):
                    mask = masks_np[i]
                    valid = np.where(mask > 0)[0]
                    action = int(np.random.choice(valid)) if len(valid) > 0 else 0
                    n = max(int(mask.sum()), 1)
                    log_prob = float(np.log(1.0 / n))
                    self.response_queues[wid].put((action, log_prob, 0.0, 0.0))
                continue

            masked_logits = logits + torch.where(
                masks_t > 0,
                torch.zeros_like(logits),
                torch.full_like(logits, -1e10),
            )
            probs = Categorical(logits=masked_logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)
            entropies = probs.entropy()

            # Move results to CPU once
            actions_cpu    = actions.cpu().numpy()
            log_probs_cpu  = log_probs.cpu().numpy()
            entropies_cpu  = entropies.cpu().numpy()
            values_cpu     = values_t.squeeze(-1).cpu().numpy()

            # ── Dispatch results to per-worker queues ─────────────
            for i, wid in enumerate(worker_ids):
                self.response_queues[wid].put((
                    int(actions_cpu[i]),
                    float(log_probs_cpu[i]),
                    float(entropies_cpu[i]),
                    float(values_cpu[i]),       # old value estimate for clipping
                ))

    def update_weights(self, state_dict):
        """Hot-reload model weights (called from main thread between epochs)."""
        load_policy_state_dict(self.model, state_dict)
        self.model.eval()


# ─────────────────────────────────────────────────────────────────────
# Helper functions (shared with workers — must be picklable / top-level)
# ─────────────────────────────────────────────────────────────────────

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


def zip_csv_file(csv_file_path, zip_file_path):
    """Compress the CSV file and remove the original."""
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, arcname=os.path.basename(csv_file_path))
    os.remove(csv_file_path)


def _prune_checkpoints(model_dir, keep_last):
    """Delete old agent_epoch_N.pt files, retaining only the most recent keep_last.

    If keep_last is None, all checkpoints are kept.
    """
    if keep_last is None:
        return
    files = [f for f in os.listdir(model_dir)
             if f.startswith("agent_epoch_") and f.endswith(".pt")]
    files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    for old in files[:-keep_last]:
        os.remove(os.path.join(model_dir, old))


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


# ─────────────────────────────────────────────────────────────────────
# Worker process — plays games, uses GPU server for agent inference
# ─────────────────────────────────────────────────────────────────────

def _load_opponent(n_actions, opp_path, cache):
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


def _play_game(worker_id, request_queue, response_queue,
               n_actions, epoch, opponent_model_path, opp_cache,
               reward_config=None,
               state_buf=None, mask_buf=None):
    """Play one game using the GPU server for agent inference."""
    env = CheckersEnv(reward_config=reward_config)

    # Optionally load a pool opponent (CPU-local, cached)
    opponent = None
    opponent_color = None
    opponent_label = "self"
    if opponent_model_path is not None:
        opponent = _load_opponent(n_actions, opponent_model_path, opp_cache)
        opponent_color = BLUE if random.random() < 0.5 else RED
        opponent_label = os.path.splitext(os.path.basename(opponent_model_path))[0]

    blue_memory, red_memory = Memory(), Memory()
    curriculum_opts = get_curriculum_options(epoch)
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
            if random.random() < cfg.POOL_EPSILON:
                action = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                with torch.no_grad():
                    action, log_prob, _ = opponent.select_action(state, action_mask)
                if isinstance(log_prob, torch.Tensor):
                    log_prob = log_prob.item()
        else:
            # Current agent — use GPU server or random exploration
            epsilon = cfg.get_epsilon(epoch)

            if random.random() < epsilon:
                action   = random_action_from_mask(action_mask)
                log_prob = uniform_log_prob(action_mask)
            else:
                # ── GPU inference via server ──────────────────────
                if state_buf is not None:
                    # Shared-memory path: write directly, send only worker_id
                    state_buf[worker_id][:] = state
                    mask_buf[worker_id][:]  = action_mask
                    request_queue.put(worker_id)
                else:
                    request_queue.put((worker_id, state, action_mask))
                action, log_prob, _, value = response_queue.get()

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


def _play_benchmark_game(worker_id, request_queue, response_queue,
                         n_actions, opponent_type, opponent_model_path, opp_cache,
                         reward_config=None,
                         state_buf=None, mask_buf=None):
    """Play a single benchmark game using GPU server for agent inference."""
    env = CheckersEnv(reward_config=None)  # benchmarks only track outcomes, not rewards

    opponent = None
    if opponent_type == "model" and opponent_model_path is not None:
        opponent = _load_opponent(n_actions, opponent_model_path, opp_cache)

    agent_color = BLUE if random.random() < 0.5 else RED
    opponent_color = RED if agent_color == BLUE else BLUE

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


# Task-queue sentinels — distinct names to avoid collision with the
# InferenceServer's request-queue sentinel (_SHUTDOWN = "SHUTDOWN" at line 42).
_WORKER_EXIT       = None        # terminate the worker process
_WORKER_BATCH_DONE = "BATCH_DONE"  # current batch finished; worker waits for next

# Legacy alias kept so existing _run_games_on_gpu (single-batch path) still works.
_TASK_DONE = _WORKER_EXIT


def worker_fn(worker_id, request_queue, response_queue, results_queue,
              n_actions, task_queue, progress_queue,
              shm_state_name=None, shm_mask_name=None, num_workers=None):
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
    _shm_state = _shm_mask = None
    _state_buf  = _mask_buf  = None
    if shm_state_name is not None and shm_mask_name is not None:
        _shm_state = _shm_module.SharedMemory(name=shm_state_name)
        _shm_mask  = _shm_module.SharedMemory(name=shm_mask_name)
        _state_buf = np.ndarray((num_workers, 4, 8, 8),
                                dtype=np.float32, buffer=_shm_state.buf)
        _mask_buf  = np.ndarray((num_workers, NUM_ACTIONS),
                                dtype=np.float32, buffer=_shm_mask.buf)

    opp_cache = {}  # per-worker opponent model cache
    games_played = 0

    while True:
        item = task_queue.get()

        if item is _WORKER_EXIT:
            if _shm_state is not None:
                _shm_state.close()
                _shm_mask.close()
            break

        if item == _WORKER_BATCH_DONE:
            # Echo back so WorkerContext.run_tasks() knows this worker is idle
            results_queue.put(_WORKER_BATCH_DONE)
            continue

        task_index, task = item
        reward_config = task.get("reward_config", None)

        if task["mode"] == "train":
            result = _play_game(
                worker_id, request_queue, response_queue,
                n_actions, task["epoch"], task["opponent_model_path"], opp_cache,
                reward_config=reward_config,
                state_buf=_state_buf, mask_buf=_mask_buf,
            )
        else:  # benchmark
            result = _play_benchmark_game(
                worker_id, request_queue, response_queue,
                n_actions, task["opponent_type"],
                task["opponent_model_path"], opp_cache,
                reward_config=reward_config,
                state_buf=_state_buf, mask_buf=_mask_buf,
            )

        results_queue.put((task_index, result))
        games_played += 1

        if games_played % 100 == 0:
            progress_queue.put((worker_id, games_played))


# ─────────────────────────────────────────────────────────────────────
# WorkerContext — long-lived worker pool with hot-swappable model weights
# ─────────────────────────────────────────────────────────────────────

class WorkerContext:
    """Long-lived worker pool + inference server for use across multiple batches.

    In league training, _run_games_on_gpu is called once per agent per epoch,
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

    def __init__(self, initial_state_dict, device, n_actions, num_workers):
        self.device      = device
        self.n_actions   = n_actions
        self.num_workers = num_workers

        # Own copy of the policy — never shared with agent.policy
        input_shape = (4, 8, 8)
        self._server_policy = PPOPolicyNetwork(input_shape, n_actions).to(device)
        load_policy_state_dict(self._server_policy, initial_state_dict)
        self._server_policy.eval()
        if _torch_compile_available():
            # mode="default" keeps the model on the default CUDA stream.
            # Same Triton requirement as the training model — guarded identically.
            self._server_policy = torch.compile(self._server_policy, mode="default")

        # Shared queues
        self._request_queue  = Queue()
        self._response_queues = [Queue() for _ in range(num_workers)]
        self._results_queue  = Queue()
        self._progress_queue = Queue()
        self._task_queue     = Queue()
        self._stop_event     = threading.Event()

        # Shared-memory buffers: workers write state/mask here; server reads
        # directly — zero pickle overhead for the most frequent IPC message.
        _state_bytes = num_workers * 4 * 8 * 8 * 4   # float32
        _mask_bytes  = num_workers * NUM_ACTIONS * 4  # float32
        self._shm_state = _shm_module.SharedMemory(create=True, size=_state_bytes)
        self._shm_mask  = _shm_module.SharedMemory(create=True, size=_mask_bytes)
        self._state_buf = np.ndarray((num_workers, 4, 8, 8),
                                     dtype=np.float32, buffer=self._shm_state.buf)
        self._mask_buf  = np.ndarray((num_workers, NUM_ACTIONS),
                                     dtype=np.float32, buffer=self._shm_mask.buf)

        # Start inference server (receives shared-memory arrays for zero-copy reads)
        self._server = InferenceServer(
            self._server_policy, device,
            self._request_queue, self._response_queues,
            self._stop_event, max_batch=num_workers * 4,
            state_buf=self._state_buf, mask_buf=self._mask_buf,
        )
        self._server.start()

        # Start persistent worker processes (receive shm names to attach)
        self._workers = []
        for wid in range(num_workers):
            p = Process(
                target=worker_fn,
                args=(wid, self._request_queue, self._response_queues[wid],
                      self._results_queue, n_actions,
                      self._task_queue, self._progress_queue,
                      self._shm_state.name, self._shm_mask.name, num_workers),
            )
            p.start()
            self._workers.append(p)

    def update_model(self, state_dict):
        """Hot-swap model weights into the inference server's own policy copy."""
        self._server.update_weights(state_dict)

    def run_tasks(self, game_tasks, label="games"):
        """Play all game_tasks using the persistent worker pool and return results.

        Fills the task queue, sends one _WORKER_BATCH_DONE per worker as a
        flush signal, then collects results until all num_workers echoes arrive.
        """
        total_tasks = len(game_tasks)

        # Enqueue all tasks
        for i, task in enumerate(game_tasks):
            self._task_queue.put((i, task))

        # Flush signal: one per worker so run_tasks knows when each worker
        # has finished all its tasks for this batch
        for _ in range(self.num_workers):
            self._task_queue.put(_WORKER_BATCH_DONE)

        result_map   = {}
        reported     = 0
        log_interval = 500
        idle_workers = 0

        while idle_workers < self.num_workers or len(result_map) < total_tasks:
            try:
                item = self._results_queue.get(timeout=1.0)
                if item == _WORKER_BATCH_DONE:
                    idle_workers += 1
                    continue
                task_index, result = item
                result_map[task_index] = result

                games_done = len(result_map)
                if games_done - reported >= log_interval or games_done == total_tasks:
                    alive = sum(1 for p in self._workers if p.is_alive())
                    print(f"  {label}: {games_done}/{total_tasks} finished "
                          f"({alive} workers active)")
                    reported = games_done
            except Exception:
                if (not any(p.is_alive() for p in self._workers)
                        and len(result_map) < total_tasks):
                    raise RuntimeError(
                        f"[{label}] All workers died after "
                        f"{len(result_map)}/{total_tasks} games."
                    )

        return [result_map[i] for i in range(total_tasks)]

    def shutdown(self):
        """Terminate all workers, stop the inference server, and free shared memory."""
        for _ in self._workers:
            self._task_queue.put(_WORKER_EXIT)
        for p in self._workers:
            p.join(timeout=10)

        self._stop_event.set()
        self._request_queue.put(_SHUTDOWN)
        self._server.join(timeout=5)

        # Release shared memory.  On Windows, unlink() is a no-op (OS reclaims
        # on last handle close); on Linux it removes the /dev/shm entry.
        self._shm_state.close()
        self._shm_state.unlink()
        self._shm_mask.close()
        self._shm_mask.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# ─────────────────────────────────────────────────────────────────────
# Orchestration helpers
# ─────────────────────────────────────────────────────────────────────

def _run_games_on_gpu(model, device, n_actions, game_tasks, num_workers,
                      label="games"):
    """Spin up workers + inference server, play all game_tasks, return results.

    Uses dynamic task scheduling: a shared task queue that workers pull from
    as they finish games.  Faster workers automatically get more work, keeping
    all CPUs busy until the very last game completes (no straggler tail).

    Args:
        model: PPOPolicyNetwork already on `device` (GPU).
        game_tasks: list of task dicts (see worker_fn).
        num_workers: number of CPU worker processes.
        label: for progress printing.

    Returns:
        list of result dicts in the original task order.
    """
    request_queue = Queue()
    response_queues = [Queue() for _ in range(num_workers)]
    results_queue = Queue()
    progress_queue = Queue()
    task_queue = Queue()
    stop_event = threading.Event()

    # Start inference server thread
    server = InferenceServer(
        model, device, request_queue, response_queues, stop_event,
        max_batch=num_workers * 4,
    )
    server.start()

    # Fill the shared task queue (workers pull dynamically)
    for i, task in enumerate(game_tasks):
        task_queue.put((i, task))
    # Add sentinel for each worker so they know when to stop
    for _ in range(num_workers):
        task_queue.put(_TASK_DONE)

    # Start worker processes
    workers = []
    for wid in range(num_workers):
        p = Process(
            target=worker_fn,
            args=(wid, request_queue, response_queues[wid], results_queue,
                  n_actions, task_queue, progress_queue),
        )
        p.start()
        workers.append(p)

    # Collect results as they arrive
    total_tasks = len(game_tasks)
    result_map = {}
    reported = 0
    log_interval = 500

    while len(result_map) < total_tasks:
        try:
            task_index, result = results_queue.get(timeout=1.0)
            result_map[task_index] = result

            games_done = len(result_map)
            if games_done - reported >= log_interval or games_done == total_tasks:
                alive = sum(1 for p in workers if p.is_alive())
                print(f"  {label}: {games_done}/{total_tasks} finished "
                      f"({alive} workers active)")
                reported = games_done
        except Exception:
            # Check if all workers have died before all results arrived
            if not any(p.is_alive() for p in workers) and len(result_map) < total_tasks:
                raise RuntimeError(
                    f"[{label}] All workers died after {len(result_map)}/{total_tasks} games. "
                    f"Check worker processes for exceptions."
                )

    # Stop server
    stop_event.set()
    request_queue.put(_SHUTDOWN)
    server.join(timeout=5)

    # Wait for workers
    for p in workers:
        p.join(timeout=5)

    # Reassemble results in original task order
    return [result_map[i] for i in range(total_tasks)]


# ─────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────

def run_benchmark(model, device, n_actions, num_workers,
                  reference_model_path, num_games=200,
                  extra_opponents=None, include_random=True):
    """Evaluate the agent against multiple opponents using the GPU server.

    Args:
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
        for _ in range(num_games):
            tasks.append({
                "mode": "benchmark",
                "opponent_type": "random",
                "opponent_model_path": None,
            })

    # vs Reference model
    has_ref = reference_model_path and os.path.exists(reference_model_path)
    if has_ref:
        for _ in range(num_games):
            tasks.append({
                "mode": "benchmark",
                "opponent_type": "model",
                "opponent_model_path": reference_model_path,
            })

    # vs Extra opponents (e.g. other league agent types)
    valid_extras = {}
    if extra_opponents:
        for label, path in extra_opponents.items():
            if path and os.path.exists(path):
                valid_extras[label] = path
                for _ in range(num_games):
                    tasks.append({
                        "mode": "benchmark",
                        "opponent_type": "model",
                        "opponent_model_path": path,
                    })

    all_results = _run_games_on_gpu(
        model, device, n_actions, tasks, num_workers, label="benchmark"
    )

    def _tally(slice_results):
        return {
            "win_rate":  sum(r["agent_win"]    for r in slice_results) / num_games,
            "loss_rate": sum(r["opponent_win"]  for r in slice_results) / num_games,
            "tie_rate":  sum(r["tie"]           for r in slice_results) / num_games,
            "avg_steps": sum(r["steps"]         for r in slice_results) / num_games,
        }

    results = {}
    offset = 0

    if include_random:
        results["vs_random"] = _tally(all_results[offset : offset + num_games])
        offset += num_games

    if has_ref:
        results["vs_reference"] = _tally(all_results[offset : offset + num_games])
        offset += num_games

    for label in valid_extras:
        results[f"vs_{label}"] = _tally(all_results[offset : offset + num_games])
        offset += num_games

    return results


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

def train_gpu_parallel(num_epochs=cfg.NUM_EPOCHS, num_games=cfg.NUM_GAMES,
                       n_actions=NUM_ACTIONS, num_workers=None,
                       agent_type=None, reward_config=None):
    """GPU-accelerated parallelized training loop for the Checkers PPO agent.

    Args:
        num_workers:   Number of CPU worker processes.
        agent_type:    League agent name (e.g. "tactical"). If provided,
                       reward_config and PPO hyperparams (gamma, entropy_bonus)
                       are looked up from cfg.LEAGUE_AGENTS[agent_type].
        reward_config: Explicit reward config dict (overrides agent_type lookup).
    """
    input_shape = (4, 8, 8)
    device = get_device()

    # Resolve reward config and per-agent PPO hyperparams
    if agent_type is not None and agent_type in cfg.LEAGUE_AGENTS:
        league_cfg = cfg.LEAGUE_AGENTS[agent_type]
        if reward_config is None:
            reward_config = league_cfg
        gamma         = league_cfg.get("gamma", cfg.GAMMA)
        entropy_bonus = league_cfg.get("entropy_bonus", 0.01)
    else:
        gamma         = cfg.GAMMA
        entropy_bonus = 0.01

    agent = PPOAgent(
        input_shape, n_actions, device=device,
        lr=cfg.LEARNING_RATE, gamma=gamma, eps_clip=cfg.EPS_CLIP,
        K_epochs=cfg.K_EPOCHS, gae_lambda=cfg.GAE_LAMBDA,
        augment=cfg.AUGMENT, augment_noise=cfg.AUGMENT_NOISE,
        mini_batch_size=cfg.MINI_BATCH_SIZE,
        entropy_bonus=entropy_bonus,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_saved_models_parallel")
    os.makedirs(model_dir, exist_ok=True)

    # Opponent pool (shared with train_parallel)
    pool_dir = os.path.join(base_dir, "opponent_pool_parallel")
    pool = OpponentPool(pool_dir, max_size=cfg.POOL_MAX_SIZE)

    start_epoch = 0

    checkpoints = [f for f in os.listdir(model_dir)
                   if f.startswith("agent_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(model_dir, latest)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_policy_state_dict(agent.policy, checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch: {start_epoch}")

    detailed_csv_folder_path = os.path.join(base_dir, "training_progress_detailed_parallel")
    os.makedirs(detailed_csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(base_dir, "training_progress_parallel.csv")
    benchmark_csv_path = os.path.join(base_dir, "benchmark_parallel.csv")

    # Save the starting model as a permanent reference for benchmarking
    reference_model_path = os.path.join(model_dir, "reference_model.pt")
    if not os.path.exists(reference_model_path):
        torch.save(agent.get_policy_state_dict(), reference_model_path)
        print(f"Saved reference model for benchmarking: {reference_model_path}")

    if start_epoch == 0:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch", "average_epoch_reward", "average_blue_reward",
                "average_red_reward", "average_episode_length",
                "win_rate_blue", "win_rate_red", "tie_rate", "max_move_reward",
                "epoch_time_s"
            ])

    if not os.path.exists(benchmark_csv_path):
        with open(benchmark_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "vs_random_win", "vs_random_loss", "vs_random_tie",
                "vs_random_avg_steps",
                "vs_reference_win", "vs_reference_loss", "vs_reference_tie",
                "vs_reference_avg_steps",
            ])

    if num_workers is None:
        num_workers = cfg.get_num_workers_gpu()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        print(f"\nStarting epoch {epoch + 1} (pool size: {pool.size})")
        print(f"Using {num_workers} CPU workers + GPU inference server.")

        # Build task list for this epoch
        game_tasks = []
        for game_id in range(num_games):
            opp_path = None
            if pool.should_use_opponent(prob=cfg.get_pool_opponent_prob(epoch)):
                opp_path = pool.sample(agent_name=agent_type)
            game_tasks.append({
                "mode": "train",
                "epoch": epoch,
                "opponent_model_path": opp_path,
                "reward_config": reward_config,  # None = default env rewards
            })

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
        all_results = _run_games_on_gpu(
            agent.policy, device, n_actions, game_tasks, num_workers,
            label=f"Epoch {epoch+1}",
        )

        write_detailed_csv(
            detailed_csv_file_path, all_results, 0, epoch, num_games
        )

        # ── Aggregate statistics ──────────────────────────────────
        total_rewards = {"blue": 0, "red": 0}
        blue_wins, red_wins, ties, max_move_reward, total_steps = 0, 0, 0, float('-inf'), 0
        pool_games = 0

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

        # ── Update prioritized opponent sampling stats (single JSON round-trip) ──
        pool.batch_update_stats(all_results, agent_name=agent_type)

        # ── PPO update (GPU) ──────────────────────────────────────
        combined_memory = Memory()
        for result in all_results:
            combined_memory.extend(result["blue_memory"])
            combined_memory.extend(result["red_memory"])
        agent.update(combined_memory)

        # Step the learning rate scheduler
        agent.step_scheduler()

        # Save to opponent pool periodically (namespaced by agent_type if set)
        if (epoch + 1) % cfg.POOL_SAVE_INTERVAL == 0:
            pool.save(agent.get_policy_state_dict(), epoch + 1, agent_name=agent_type)
            print(f"  Saved to opponent pool (size: {pool.size})")

        # Zip detailed CSV
        epoch_zip_file_path = os.path.join(
            detailed_csv_folder_path, f"detailed_games_epoch_{epoch + 1}.zip"
        )
        zip_csv_file(detailed_csv_file_path, epoch_zip_file_path)

        # Log epoch statistics
        avg_reward = (total_rewards["blue"] + total_rewards["red"]) / num_games
        avg_blue_reward = total_rewards["blue"] / num_games
        avg_red_reward = total_rewards["red"] / num_games
        avg_steps = total_steps / num_games
        blue_win_rate = blue_wins / num_games
        red_win_rate = red_wins / num_games
        tie_rate = ties / num_games
        epoch_duration = time.perf_counter() - epoch_start

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, avg_reward, avg_blue_reward, avg_red_reward,
                avg_steps, blue_win_rate, red_win_rate, tie_rate,
                max_move_reward, round(epoch_duration, 2)
            ])

        # Save checkpoint (plain state dict — no _orig_mod. prefix)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': agent.get_policy_state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'scheduler_state_dict': agent.scheduler.state_dict(),
        }, os.path.join(model_dir, f"agent_epoch_{epoch + 1}.pt"))

        # Prune old checkpoints — keep only the most recent N
        _prune_checkpoints(model_dir, cfg.CHECKPOINT_KEEP_LAST)

        print(f"Epoch {epoch + 1} complete in {epoch_duration:.1f}s. "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Blue Win: {blue_win_rate:.2%}, Red Win: {red_win_rate:.2%}, "
              f"Tie: {tie_rate:.2%}, LR: {agent.scheduler.get_last_lr()[0]:.2e}")

        # ── Benchmark evaluation every N epochs ───────────────────
        if (epoch + 1) % cfg.BENCHMARK_INTERVAL == 0:
            print(f"  Running benchmark ({cfg.BENCHMARK_GAMES} games each "
                  f"vs random & reference)...")
            bench = run_benchmark(
                agent.policy, device, n_actions, num_workers,
                reference_model_path, num_games=cfg.BENCHMARK_GAMES,
            )
            vr = bench["vs_random"]
            print(f"  vs Random:    Win {vr['win_rate']:.1%}  "
                  f"Loss {vr['loss_rate']:.1%}  "
                  f"Tie {vr['tie_rate']:.1%}  "
                  f"AvgSteps {vr['avg_steps']:.0f}")

            vref = bench.get("vs_reference", {})
            if vref:
                print(f"  vs Reference: Win {vref['win_rate']:.1%}  "
                      f"Loss {vref['loss_rate']:.1%}  "
                      f"Tie {vref['tie_rate']:.1%}  "
                      f"AvgSteps {vref['avg_steps']:.0f}")

            with open(benchmark_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    vr["win_rate"], vr["loss_rate"], vr["tie_rate"],
                    vr["avg_steps"],
                    vref.get("win_rate", ""), vref.get("loss_rate", ""),
                    vref.get("tie_rate", ""), vref.get("avg_steps", ""),
                ])

            # Auto-advance reference model to current epoch so the next
            # benchmark always measures improvement over the last tested epoch.
            torch.save(agent.get_policy_state_dict(), reference_model_path)
            print(f"  Reference model advanced to epoch {epoch + 1}")


if __name__ == "__main__":
    train_gpu_parallel()

