"""GPU-accelerated parallel AlphaZero training for checkers.

Architecture
------------
                        ┌─────────────────────────────────────┐
                        │          Main Process (GPU)         │
                        │                                     │
                        │  AlphaZeroInferenceServer (thread)  │
  Worker 0 ──req──►     │    batches leaf-eval requests       │
  Worker 1 ──req──►     │    runs one forward pass for all    │
    ...       ...       │    sends (logits, value) back       │
  Worker N ──req──►     │                                     │
       ▲                │  Training loop                      │
       │                │    samples replay buffer            │
       └───resp───────  │    gradient updates on GPU          │
                        └─────────────────────────────────────┘

Key difference from PPO parallel:
  PPO workers make 1 inference call per env step.
  MCTS workers make NUM_SIMULATIONS calls per move (100 by default).
  With N=16 busy workers, the server sees batches of up to 16 requests
  simultaneously → 16x better GPU utilisation vs the sequential baseline.

Workers use NumpyCheckersEnv for fast_clone() inside simulations.
The outer game loop still uses CheckersEnv (for from_env() conversion).

Shared-memory protocol: workers write (4,8,8) state and (NUM_ACTIONS,) mask
into pre-allocated numpy arrays backed by shared memory, then put only
their integer worker_id onto the request queue.  Zero pickle overhead for
the most frequent IPC message (~600,000 per epoch).

Usage
-----
    python -m RL_models.MCTS.train_parallel
    python -m RL_models.MCTS.train_parallel --workers 12
"""

import os
import csv
import random
import threading
import time
import argparse
import zipfile
from collections import deque
from multiprocessing import Process, Queue, Event
from multiprocessing import shared_memory as _shm_module
from queue import Empty as QueueEmpty

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from RL_models.checkers_env import CheckersEnv
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.MCTS.AlphaZeroNetwork import AlphaZeroNetwork
from RL_models.MCTS import training_config as cfg
from RL_models.PPO_Model.Agent import get_device
from checkers_game.constants import BLUE, RED, NUM_ACTIONS

# NumpyCheckersEnv is not used directly here but is imported so it is
# available in worker processes when MCTSSearch.search() calls
# NumpyCheckersEnv.from_env(env) for fast simulation cloning.
from RL_models.numpy_checkers_env import NumpyCheckersEnv  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Sentinels
# ─────────────────────────────────────────────────────────────────────────────

_SHUTDOWN         = "SHUTDOWN"
_WORKER_EXIT      = None          # tell worker process to terminate
_WORKER_BATCH_DONE = "BATCH_DONE"  # worker finished all tasks in current batch


# ─────────────────────────────────────────────────────────────────────────────
# GPU Inference Server  (runs as a daemon thread in the main process)
# ─────────────────────────────────────────────────────────────────────────────

class AlphaZeroInferenceServer(threading.Thread):
    """Batches MCTS leaf-evaluation requests from CPU workers onto the GPU.

    Request protocol (items on request_queue):
      Shared-memory mode:  plain int worker_id.
                           State and mask are read from pre-allocated
                           shared numpy arrays — zero pickle overhead.
      Classic mode:        (worker_id, state_np, mask_np) tuple.

    Response: each worker's response_queue receives (logits_np, value_float)
    where logits_np is shape (NUM_ACTIONS,) float32 and value is a Python float.
    """

    def __init__(self, model, device, request_queue, response_queues,
                 stop_event, max_batch=64, max_wait_ms=2.0,
                 state_buf=None, mask_buf=None):
        super().__init__(daemon=True)
        self.model          = model
        self.device         = device
        self.request_queue  = request_queue
        self.response_queues = response_queues
        self.stop_event     = stop_event
        self.max_batch      = max_batch
        self.max_wait_s     = max_wait_ms / 1000.0
        self._state_buf     = state_buf
        self._mask_buf      = mask_buf

    def run(self):
        self.model.eval()
        req_q = self.request_queue

        while not self.stop_event.is_set():
            batch = []

            # Block until at least one request arrives
            try:
                item = req_q.get(timeout=0.01)
                if item == _SHUTDOWN:
                    break
                batch.append(item)
            except QueueEmpty:
                continue

            # Drain additional pending requests up to max_batch.
            # Use live queue depth so we don't wait 2ms when requests are
            # already available (as happens when all workers are busy).
            effective_max = max(4, min(req_q.qsize() + len(batch) + 1,
                                       self.max_batch))
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < effective_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = req_q.get_nowait()
                    if item == _SHUTDOWN:
                        req_q.put(_SHUTDOWN)
                        break
                    batch.append(item)
                except QueueEmpty:
                    if remaining > 0.0005:
                        time.sleep(0.0003)
                    else:
                        break

            if not batch:
                continue

            # Build batched tensors
            if self._state_buf is not None:
                # Shared-memory path: batch items are plain worker_ids
                worker_ids = list(batch)
                states_np  = self._state_buf[worker_ids].copy()
            else:
                # Classic path: batch items are (worker_id, state_np, mask_np)
                worker_ids = [b[0] for b in batch]
                states_np  = np.stack([b[1] for b in batch])

            states_t = torch.from_numpy(states_np).to(self.device)

            with torch.no_grad():
                logits_t, values_t = self.model(states_t)

            logits_np = logits_t.cpu().numpy()              # (B, NUM_ACTIONS)
            values_np = values_t.squeeze(-1).cpu().numpy()  # (B,)

            for i, wid in enumerate(worker_ids):
                self.response_queues[wid].put(
                    (logits_np[i], float(values_np[i]))
                )

    def update_weights(self, state_dict):
        """Hot-reload model weights between epochs (called from main thread)."""
        self.model.load_state_dict(state_dict)
        self.model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Remote evaluator  (created inside each worker process)
# ─────────────────────────────────────────────────────────────────────────────

class RemoteEvaluator:
    """Drop-in for MCTSSearch.evaluator — routes inference through the GPU server.

    Called once per MCTS simulation for each leaf evaluation.  Uses the
    shared-memory protocol when buffers are available to eliminate pickling.
    """

    def __init__(self, worker_id, request_queue, response_queue,
                 state_buf=None, mask_buf=None):
        self.worker_id      = worker_id
        self.request_queue  = request_queue
        self.response_queue = response_queue
        self._state_buf     = state_buf
        self._mask_buf      = mask_buf

    def __call__(self, state, action_mask):
        """Send (state, action_mask) to GPU server, return (logits_np, value)."""
        if self._state_buf is not None:
            self._state_buf[self.worker_id][:] = state
            # mask is not needed by the server; it's used locally for masking
            self.request_queue.put(self.worker_id)
        else:
            self.request_queue.put((self.worker_id, state, action_mask))
        return self.response_queue.get()


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side: play one complete self-play game
# ─────────────────────────────────────────────────────────────────────────────

def _get_curriculum_options(epoch):
    if epoch < cfg.CURRICULUM_PHASE1_END:
        lo, hi = cfg.CURRICULUM_PHASE1_PIECES
        return {"num_pieces": random.randint(lo, hi)}
    elif epoch < cfg.CURRICULUM_PHASE2_END:
        lo, hi = cfg.CURRICULUM_PHASE2_PIECES
        return {"num_pieces": random.randint(lo, hi)}
    return None


def _play_self_play_game(worker_id, request_queue, response_queue,
                          epoch, state_buf=None, mask_buf=None):
    """Play one complete AlphaZero self-play game using the GPU inference server.

    Each MCTS leaf evaluation is routed to the GPU server via IPC.
    The outer game loop uses CheckersEnv; MCTS internally uses NumpyCheckersEnv
    for fast_clone() across the 100 simulations per move.

    Returns:
        dict with keys:
            game_data  — list of (state_np, mcts_policy_np, player_color)
            winner     — BLUE / RED / "Tie"
            num_moves  — number of completed turns
            game_stats — dict of per-game diagnostic signals:
                           avg_root_val_winner, avg_root_val_loser,
                           avg_policy_entropy, move_sequence
    """
    evaluator = RemoteEvaluator(
        worker_id, request_queue, response_queue, state_buf, mask_buf
    )

    curriculum_opts = _get_curriculum_options(epoch)
    num_sims = (cfg.NUM_SIMULATIONS_CURRICULUM
                if curriculum_opts is not None
                else cfg.NUM_SIMULATIONS)

    mcts = MCTSSearch(
        evaluator=evaluator,
        num_simulations=num_sims,
        c_puct=cfg.C_PUCT,
        dirichlet_alpha=cfg.DIRICHLET_ALPHA,
        dirichlet_epsilon=cfg.DIRICHLET_EPSILON,
    )
    env = CheckersEnv()
    env.reset(options=curriculum_opts)

    game_data   = []
    value_log   = []   # (player_color, root_value) per move
    entropy_log = []   # float per move
    move_count  = 0
    done        = False
    info        = {}
    mcts._root  = None

    while not done:
        # Hard cap: declare draw if the game exceeds MAX_GAME_MOVES full turns.
        # Prevents runaway passive games from wasting compute and polluting the
        # buffer with hundreds of near-identical draw positions.
        if move_count >= cfg.MAX_GAME_MOVES:
            info = {"winner": "Tie"}
            break

        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            break

        temperature = (
            cfg.TEMPERATURE_EARLY
            if move_count < cfg.TEMPERATURE_THRESHOLD
            else cfg.TEMPERATURE_LATE
        )

        state          = env.get_board_state()
        current_player = env.game.turn

        action, mcts_policy, root_value = mcts.select_action(
            env, temperature=temperature, add_noise=True
        )

        # Per-step diagnostics
        eps = 1e-10
        entropy = float(-np.sum(mcts_policy * np.log(mcts_policy + eps)))
        value_log.append((current_player, root_value))
        entropy_log.append(entropy)

        game_data.append((state, mcts_policy, current_player))

        _, _, done, _, info = env.step(action)
        mcts.update_root(action)

        if info.get("turn_complete", True):
            move_count += 1

    winner = info.get("winner", "Tie")

    is_decisive = winner not in ("Tie", "None")
    winner_vals = [v for p, v in value_log if is_decisive and p == winner]
    loser_vals  = [v for p, v in value_log if is_decisive and p != winner]
    game_stats = {
        "avg_root_val_winner": float(np.mean(winner_vals)) if winner_vals else 0.0,
        "avg_root_val_loser":  float(np.mean(loser_vals))  if loser_vals  else 0.0,
        "avg_policy_entropy":  float(np.mean(entropy_log)) if entropy_log  else 0.0,
        "move_sequence":       ", ".join(env.game.moves),
    }

    return {
        "game_data":  game_data,
        "winner":     winner,
        "num_moves":  move_count,
        "game_stats": game_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistent worker process entry point
# ─────────────────────────────────────────────────────────────────────────────

def worker_fn(worker_id, request_queue, response_queue, results_queue,
              task_queue, progress_queue,
              shm_state_name=None, shm_mask_name=None, num_workers=None):
    """Persistent worker process.

    Pulls tasks from task_queue in a loop.

    Sentinels:
      _WORKER_EXIT       (None)         → terminate the process
      _WORKER_BATCH_DONE ("BATCH_DONE") → echo back (signals batch complete)
      (task_index, task_dict)           → play one game and return result

    task_dict keys:
      "epoch": int — used for curriculum options
    """
    # Attach to shared-memory buffers when provided
    _shm_state = _shm_mask = None
    _state_buf  = _mask_buf  = None
    if shm_state_name is not None:
        _shm_state = _shm_module.SharedMemory(name=shm_state_name)
        _shm_mask  = _shm_module.SharedMemory(name=shm_mask_name)
        _state_buf = np.ndarray(
            (num_workers, 4, 8, 8), dtype=np.float32, buffer=_shm_state.buf
        )
        # mask buffer not used by server (masking done locally) — still allocated
        # so RemoteEvaluator can write to it without branching
        _mask_buf  = np.ndarray(
            (num_workers, NUM_ACTIONS), dtype=np.float32, buffer=_shm_mask.buf
        )

    games_played = 0

    while True:
        item = task_queue.get()

        if item is _WORKER_EXIT:
            if _shm_state is not None:
                _shm_state.close()
                _shm_mask.close()
            break

        if item == _WORKER_BATCH_DONE:
            results_queue.put(_WORKER_BATCH_DONE)
            continue

        task_index, task = item

        result = _play_self_play_game(
            worker_id, request_queue, response_queue,
            task["epoch"],
            state_buf=_state_buf,
            mask_buf=_mask_buf,
        )
        results_queue.put((task_index, result))

        games_played += 1
        if games_played % 50 == 0:
            progress_queue.put((worker_id, games_played))


# ─────────────────────────────────────────────────────────────────────────────
# WorkerContext — long-lived worker pool with hot-swappable model weights
# ─────────────────────────────────────────────────────────────────────────────

class WorkerContext:
    """Manages the persistent worker pool and GPU inference server.

    Workers are spawned once and reused across all epochs.  Model weights are
    hot-swapped between epochs via update_model() without restarting anything.

    Usage (context manager):
        with WorkerContext(model.state_dict(), device, num_workers) as ctx:
            for epoch in range(NUM_EPOCHS):
                ctx.update_model(model.state_dict())
                results = ctx.run_epoch(game_tasks, label=f"Epoch {epoch}")
    """

    def __init__(self, initial_state_dict, device, num_workers):
        self.device      = device
        self.num_workers = num_workers

        # Inference server has its own model copy — never shared with the
        # training model to avoid backward() racing with load_state_dict().
        input_shape = (4, 8, 8)
        self._server_model = AlphaZeroNetwork(input_shape, NUM_ACTIONS).to(device)
        self._server_model.load_state_dict(initial_state_dict)
        self._server_model.eval()

        # IPC channels
        self._request_queue   = Queue()
        self._response_queues = [Queue() for _ in range(num_workers)]
        self._results_queue   = Queue()
        self._progress_queue  = Queue()
        self._task_queue      = Queue()
        self._stop_event      = threading.Event()

        # Shared-memory buffers: workers write state into these; server reads.
        # Eliminates pickling of the (4,8,8) state tensor on every eval call.
        _state_bytes = num_workers * 4 * 8 * 8 * 4   # float32
        _mask_bytes  = num_workers * NUM_ACTIONS * 4  # float32 (written but not read by server)
        self._shm_state = _shm_module.SharedMemory(create=True, size=_state_bytes)
        self._shm_mask  = _shm_module.SharedMemory(create=True, size=_mask_bytes)
        self._state_buf = np.ndarray(
            (num_workers, 4, 8, 8), dtype=np.float32, buffer=self._shm_state.buf
        )
        self._mask_buf  = np.ndarray(
            (num_workers, NUM_ACTIONS), dtype=np.float32, buffer=self._shm_mask.buf
        )

        # Start inference server thread
        self._server = AlphaZeroInferenceServer(
            self._server_model, device,
            self._request_queue, self._response_queues,
            self._stop_event,
            max_batch=num_workers * 4,   # allow deep batching when workers are busy
            state_buf=self._state_buf,
            mask_buf=self._mask_buf,
        )
        self._server.start()

        # Start persistent worker processes
        self._workers = []
        for wid in range(num_workers):
            p = Process(
                target=worker_fn,
                args=(
                    wid,
                    self._request_queue,
                    self._response_queues[wid],
                    self._results_queue,
                    self._task_queue,
                    self._progress_queue,
                    self._shm_state.name,
                    self._shm_mask.name,
                    num_workers,
                ),
            )
            p.start()
            self._workers.append(p)

    def update_model(self, state_dict):
        """Push new model weights into the inference server (between epochs)."""
        self._server.update_weights(state_dict)

    def run_epoch(self, game_tasks, label="games"):
        """Distribute tasks to workers and collect results in original order.

        Fills the task queue, sends one _WORKER_BATCH_DONE sentinel per worker
        as a flush signal, then collects results until all workers are idle.

        Args:
            game_tasks: list of task dicts (each needs at least {"epoch": int}).
            label:      prefix string for progress prints.

        Returns:
            list of result dicts in the same order as game_tasks.
        """
        total = len(game_tasks)

        for i, task in enumerate(game_tasks):
            self._task_queue.put((i, task))

        for _ in range(self.num_workers):
            self._task_queue.put(_WORKER_BATCH_DONE)

        result_map   = {}
        reported     = 0
        log_interval = max(10, total // 10)
        idle_workers = 0

        while idle_workers < self.num_workers or len(result_map) < total:
            try:
                item = self._results_queue.get(timeout=1.0)
                if item == _WORKER_BATCH_DONE:
                    idle_workers += 1
                    continue
                task_idx, result = item
                result_map[task_idx] = result

                done = len(result_map)
                if done - reported >= log_interval or done == total:
                    alive = sum(1 for p in self._workers if p.is_alive())
                    print(f"  {label}: {done}/{total} games "
                          f"({alive} workers active)")
                    reported = done
            except Exception:
                if (not any(p.is_alive() for p in self._workers)
                        and len(result_map) < total):
                    raise RuntimeError(
                        f"[{label}] All workers died after "
                        f"{len(result_map)}/{total} games."
                    )

        return [result_map[i] for i in range(total)]

    def shutdown(self):
        """Terminate all workers and the inference server, release shared memory."""
        for _ in self._workers:
            self._task_queue.put(_WORKER_EXIT)
        for p in self._workers:
            p.join(timeout=15)

        self._stop_event.set()
        self._request_queue.put(_SHUTDOWN)
        self._server.join(timeout=5)

        self._shm_state.close()
        self._shm_state.unlink()
        self._shm_mask.close()
        self._shm_mask.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_alphazero_parallel(num_workers=None):
    """GPU-accelerated parallel AlphaZero training loop.

    Self-play runs across `num_workers` CPU processes.  Each MCTS leaf
    evaluation is dispatched to the GPU inference server, which batches
    requests from all workers for maximum GPU utilisation.  After all games
    finish the main process runs the supervised gradient updates on GPU.

    Args:
        num_workers: number of CPU worker processes.
                     None → auto-detect from cfg.get_num_workers_parallel().
    """
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if num_workers is None:
        num_workers = cfg.get_num_workers_parallel()
    print(f"Workers: {num_workers}")

    # ── Training network (on GPU — used only for gradient updates) ────────
    input_shape = (4, 8, 8)
    network = AlphaZeroNetwork(input_shape, NUM_ACTIONS).to(device)
    optimizer = optim.AdamW(
        network.parameters(),
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.LR_T_MAX, eta_min=cfg.LR_ETA_MIN
    )

    replay_buffer = deque(maxlen=cfg.BUFFER_SIZE)

    # ── Directories / CSV ─────────────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    model_dir    = os.path.join(base_dir, "alphazero_checkpoints")
    detailed_dir = os.path.join(base_dir, "az_detailed_games_parallel")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "alphazero_training_progress_parallel.csv")

    # ── Resume from latest checkpoint ────────────────────────────────────
    start_epoch = 0
    checkpoints = [f for f in os.listdir(model_dir)
                   if f.startswith("az_epoch_") and f.endswith(".pt")]
    if checkpoints:
        latest = max(checkpoints,
                     key=lambda f: int(f.split("_")[-1].split(".")[0]))
        ckpt = torch.load(
            os.path.join(model_dir, latest),
            map_location=device, weights_only=False,
        )
        network.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")
    else:
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "games", "blue_wins", "red_wins", "ties",
                "avg_moves", "policy_loss", "value_loss", "total_loss",
                "avg_grad_norm", "policy_entropy_nats",
                "avg_root_val_winner", "avg_root_val_loser",
                "buffer_size", "num_workers", "epoch_time_s",
            ])

    # ── Spawn workers + inference server, run training ────────────────────
    with WorkerContext(network.state_dict(), device, num_workers) as ctx:

        for epoch in range(start_epoch, cfg.NUM_EPOCHS):
            epoch_start = time.perf_counter()
            curriculum_opts = _get_curriculum_options(epoch)
            epoch_sims = (cfg.NUM_SIMULATIONS_CURRICULUM
                          if curriculum_opts is not None
                          else cfg.NUM_SIMULATIONS)
            phase = ("phase1" if epoch < cfg.CURRICULUM_PHASE1_END
                     else "phase2" if epoch < cfg.CURRICULUM_PHASE2_END
                     else "full")
            print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS} — "
                  f"self-play ({cfg.GAMES_PER_EPOCH} games, "
                  f"{epoch_sims} sims/move, {phase}, "
                  f"{num_workers} workers)")

            # Push latest weights into inference server before self-play
            ctx.update_model(network.state_dict())

            # ── Self-play phase ──────────────────────────────────────────
            game_tasks = [{"epoch": epoch}
                          for _ in range(cfg.GAMES_PER_EPOCH)]
            all_results = ctx.run_epoch(
                game_tasks, label=f"Epoch {epoch + 1}"
            )

            # ── Populate replay buffer + write detailed game CSV ─────────
            blue_wins = red_wins = ties = total_moves = 0
            total_root_val_winner = 0.0
            total_root_val_loser  = 0.0
            total_entropy         = 0.0

            detailed_csv_path = os.path.join(
                detailed_dir, f"az_games_epoch_{epoch + 1}.csv"
            )
            detailed_headers = [
                "game_id", "epoch", "game_num", "winner", "num_moves",
                "avg_root_val_winner", "avg_root_val_loser",
                "avg_policy_entropy_nats", "move_sequence",
            ]
            with open(detailed_csv_path, mode="w", newline="") as f:
                csv.writer(f).writerow(detailed_headers)

            for game_idx, res in enumerate(all_results):
                winner     = res["winner"]
                game_data  = res["game_data"]
                game_stats = res["game_stats"]
                total_moves += res["num_moves"]

                if winner == BLUE:    blue_wins += 1
                elif winner == RED:   red_wins  += 1
                else:                 ties      += 1

                total_root_val_winner += game_stats["avg_root_val_winner"]
                total_root_val_loser  += game_stats["avg_root_val_loser"]
                total_entropy         += game_stats["avg_policy_entropy"]

                for state, mcts_policy, player_color in game_data:
                    if winner in ("Tie", "None"):
                        outcome = cfg.TIE_OUTCOME_VALUE
                    elif winner == player_color:
                        outcome = 1.0
                    else:
                        outcome = -1.0
                    replay_buffer.append((state, mcts_policy, outcome))

                with open(detailed_csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch * cfg.GAMES_PER_EPOCH + game_idx + 1,
                        epoch + 1,
                        game_idx + 1,
                        winner,
                        res["num_moves"],
                        round(game_stats["avg_root_val_winner"], 4),
                        round(game_stats["avg_root_val_loser"],  4),
                        round(game_stats["avg_policy_entropy"],  4),
                        game_stats["move_sequence"],
                    ])

            # Compress and remove the per-epoch detailed CSV
            epoch_zip_path = os.path.join(
                detailed_dir, f"az_games_epoch_{epoch + 1}.zip"
            )
            with zipfile.ZipFile(epoch_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(detailed_csv_path,
                         arcname=os.path.basename(detailed_csv_path))
            os.remove(detailed_csv_path)

            avg_moves           = total_moves           / cfg.GAMES_PER_EPOCH
            avg_root_val_winner = total_root_val_winner / cfg.GAMES_PER_EPOCH
            avg_root_val_loser  = total_root_val_loser  / cfg.GAMES_PER_EPOCH
            epoch_avg_entropy   = total_entropy         / cfg.GAMES_PER_EPOCH

            # ── Training phase (GPU gradient updates) ────────────────────
            p_loss = v_loss = t_loss = avg_grad_norm = 0.0
            if len(replay_buffer) >= cfg.BATCH_SIZE:
                network.train()
                total_p = total_v = total_t = total_gn = 0.0

                # Snapshot the deque to a list once so that random.sample can
                # use O(1) index access.  deque.__getitem__(i) is O(n) for
                # middle elements; sampling 256 items 200 times from a 500K
                # deque would otherwise do millions of slow pointer walks.
                buffer_snapshot = list(replay_buffer)

                for _ in range(cfg.TRAIN_STEPS_PER_EPOCH):
                    batch = random.sample(buffer_snapshot, cfg.BATCH_SIZE)
                    states, policies, values = zip(*batch)

                    st = torch.FloatTensor(np.array(states)).to(device)
                    tp = torch.FloatTensor(np.array(policies)).to(device)
                    tv = (torch.FloatTensor(np.array(values))
                          .unsqueeze(1).to(device))

                    logits, vals = network(st)
                    log_probs = torch.log_softmax(logits, dim=1)
                    pl   = -(tp * log_probs).sum(dim=1).mean()
                    vl   = nn.MSELoss()(vals, tv)
                    loss = pl + cfg.VALUE_LOSS_WEIGHT * vl

                    optimizer.zero_grad()
                    loss.backward()
                    pre_clip_norm = nn.utils.clip_grad_norm_(
                        network.parameters(), cfg.GRAD_CLIP_NORM
                    )
                    optimizer.step()

                    total_p  += pl.item()
                    total_v  += vl.item()
                    total_t  += loss.item()
                    total_gn += pre_clip_norm.item()

                n = cfg.TRAIN_STEPS_PER_EPOCH
                p_loss       = total_p  / n
                v_loss       = total_v  / n
                t_loss       = total_t  / n
                avg_grad_norm = total_gn / n
                network.eval()

            # Buffer-level policy entropy
            if len(replay_buffer) >= 1:
                sample_n  = min(2048, len(replay_buffer))
                sample    = random.sample(list(replay_buffer), sample_n)
                policies  = np.array([s[1] for s in sample])
                eps       = 1e-10
                buffer_entropy = float(
                    -(policies * np.log(policies + eps)).sum(axis=1).mean()
                )
            else:
                buffer_entropy = 0.0

            scheduler.step()
            elapsed = time.perf_counter() - epoch_start

            # ── Logging ───────────────────────────────────────────────────
            print(f"  Epoch {epoch + 1} complete in {elapsed:.0f}s  "
                  f"({elapsed / cfg.GAMES_PER_EPOCH:.1f}s/game)")
            print(f"  Blue: {blue_wins}  Red: {red_wins}  Ties: {ties}  "
                  f"Avg Moves: {avg_moves:.1f}")
            print(f"  Policy Loss: {p_loss:.4f}  Value Loss: {v_loss:.4f}  "
                  f"Total: {t_loss:.4f}")
            print(f"  Grad Norm (pre-clip): {avg_grad_norm:.4f}  "
                  f"[clip={cfg.GRAD_CLIP_NORM}  "
                  f"{'BINDING' if avg_grad_norm > cfg.GRAD_CLIP_NORM * 0.9 else 'not binding'}]")
            print(f"  Buffer Policy Entropy: {buffer_entropy:.4f} nats  "
                  f"(max uniform ~{np.log(8):.2f} for 8-move branching)")
            print(f"  Value calibration — winner avg: {avg_root_val_winner:+.3f}  "
                  f"loser avg: {avg_root_val_loser:+.3f}  "
                  f"(ideal: +1.0 / -1.0)")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}  "
                  f"Buffer: {len(replay_buffer)}")

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, cfg.GAMES_PER_EPOCH,
                    blue_wins, red_wins, ties, avg_moves,
                    p_loss, v_loss, t_loss,
                    avg_grad_norm, buffer_entropy,
                    round(avg_root_val_winner, 4), round(avg_root_val_loser, 4),
                    len(replay_buffer), num_workers, round(elapsed, 1),
                ])

            # ── Checkpoint ────────────────────────────────────────────────
            if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                path = os.path.join(model_dir, f"az_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch":                epoch + 1,
                    "model_state_dict":     network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "blue_wins":            blue_wins,
                    "red_wins":             red_wins,
                    "ties":                 ties,
                }, path)
                print(f"  Checkpoint saved: {path}")

    print("\nAlphaZero parallel training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel AlphaZero training for checkers"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of CPU worker processes (default: auto-detect)",
    )
    args = parser.parse_args()
    train_alphazero_parallel(num_workers=args.workers)
