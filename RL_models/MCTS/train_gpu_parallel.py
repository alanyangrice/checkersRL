"""GPU-accelerated parallel AlphaZero training for checkers.

Architecture
------------
                        ┌─────────────────────────────────────┐
                        │          Main Process (GPU)         │
                        │                                     │
                        │  AlphaZeroInferenceServer (thread)  │
  Worker 0 ──req──►     │    batches leaf-eval requests       │
  Worker 1 ──req──►     │    runs one forward pass for all    │
    ...      ...        │    sends (logits, value) back       │
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
from RL_models.MCTS.WDLAlphaZeroNetwork import WDLAlphaZeroNetwork
from RL_models.MCTS import training_config as cfg
from RL_models.PPO_Model.Agent import get_device
from checkers_game.constants import BLUE, RED, NUM_ACTIONS

# NumpyCheckersEnv is not used directly here but is imported so it is
# available in worker processes when MCTSSearch.search() calls
# NumpyCheckersEnv.from_env(env) for fast simulation cloning.
from RL_models.numpy_checkers_env import NumpyCheckersEnv  # noqa: F401
from RL_models.MCTS.evaluate import (
    freeze_state_dict, test_mcts_correctness,
    play_vs_random, test_value_head_calibration,
)


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
    """Generate guaranteed-asymmetric board options for the current curriculum phase.

    The weak side draws its piece count first, then the strong side draws from
    [weak+1, strong_max], guaranteeing a strict material advantage on every game.
    Which color is the strong side is re-rolled 50/50 each game so both BLUE and
    RED learn to play from both material situations equally.

    Phase 1: weak ∈ [1, 5],  strong ∈ [weak+1, 6]  — endgame positions
    Phase 2: weak ∈ [5, 8],  strong ∈ [weak+1, 9]  — mid-game positions (≥5 pieces/side)
    """
    if epoch < cfg.CURRICULUM_PHASE1_END:
        weak   = random.randint(cfg.CURRICULUM_PHASE1_WEAK_MIN, cfg.CURRICULUM_PHASE1_WEAK_MAX)
        strong = random.randint(weak + 1, cfg.CURRICULUM_PHASE1_STRONG_MAX)
    elif epoch < cfg.CURRICULUM_PHASE2_END:
        weak   = random.randint(cfg.CURRICULUM_PHASE2_WEAK_MIN, cfg.CURRICULUM_PHASE2_WEAK_MAX)
        strong = random.randint(weak + 1, cfg.CURRICULUM_PHASE2_STRONG_MAX)
    else:
        return None

    if random.random() < 0.5:
        return {"num_blue": strong, "num_red": weak}
    return {"num_blue": weak, "num_red": strong}


def _get_material(env):
    """Return (blue_material, red_material), kings weighted by KING_MATERIAL_VALUE."""
    board = env.game.board.board
    blue_mat = sum(
        cfg.KING_MATERIAL_VALUE if (p != 0 and p.color == BLUE and p.king)
        else (1.0 if p != 0 and p.color == BLUE else 0.0)
        for row in board for p in row
    )
    red_mat = sum(
        cfg.KING_MATERIAL_VALUE if (p != 0 and p.color == RED and p.king)
        else (1.0 if p != 0 and p.color == RED else 0.0)
        for row in board for p in row
    )
    return blue_mat, red_mat


def _adjudicate_move_cap(env):
    """Determine winner at move cap based on material (zero-sum compatible).

    Kings count as KING_MATERIAL_VALUE pieces.  Side with more material wins.
    Equal material → Tie.
    """
    if not cfg.MOVE_CAP_ADJUDICATE:
        return "Tie"
    blue_mat, red_mat = _get_material(env)
    if blue_mat > red_mat:
        return BLUE
    elif red_mat > blue_mat:
        return RED
    return "Tie"


def _play_self_play_game(worker_id, request_queue, response_queue,
                          epoch, state_buf=None, mask_buf=None,
                          start_board=None, start_turn=None, c_puct=None):
    """Play one complete AlphaZero self-play game using the GPU inference server.

    Args:
        start_board: Optional (4,8,8) absolute board state from
                     env.get_absolute_board_state() to start from instead of
                     the standard initial position (regret-buffer diverse starts).
        start_turn:  Color constant (BLUE/RED) for start_board; required when
                     start_board is provided.
        c_puct:      PUCT exploration constant override; defaults to cfg.C_PUCT.
    """
    evaluator = RemoteEvaluator(
        worker_id, request_queue, response_queue, state_buf, mask_buf
    )

    curriculum_opts = _get_curriculum_options(epoch)
    num_sims = cfg.get_num_simulations(epoch)
    temp_threshold, temp_late = cfg.get_temperature_config(epoch)
    effective_c_puct = c_puct if c_puct is not None else cfg.C_PUCT

    mcts = MCTSSearch(
        evaluator=evaluator,
        num_simulations=num_sims,
        c_puct=effective_c_puct,
        dirichlet_alpha=cfg.DIRICHLET_ALPHA,
        dirichlet_epsilon=cfg.DIRICHLET_EPSILON,
        move_cap=cfg.get_max_game_moves(epoch),
    )
    env = CheckersEnv()
    if start_board is not None and start_turn is not None:
        env.load_absolute_board_state(start_board, start_turn)
    else:
        env.reset(options=curriculum_opts)

    game_data         = []   # (state, policy, player, mcts_q_value)
    abs_board_states  = []   # absolute board state at each step (for regret buffer)
    value_log         = []
    entropy_log       = []
    move_count        = 0
    done              = False
    info              = {}
    mcts._root        = None
    cap_terminated    = False   # True when the game ended by hitting the move cap

    while not done:
        if move_count >= cfg.get_max_game_moves(epoch):
            info = {"winner": _adjudicate_move_cap(env)}
            cap_terminated = True
            break

        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            break

        temperature = (
            cfg.TEMPERATURE_EARLY
            if move_count < temp_threshold
            else temp_late
        )

        state          = env.get_board_state()
        abs_state      = env.get_absolute_board_state()   # for regret buffer
        current_player = env.game.turn

        action, mcts_policy, root_value = mcts.select_action(
            env, temperature=temperature, add_noise=True,
            no_progress_count=env.game._no_progress_count,
        )

        # MCTS Q-value: the backed-up mean value after all simulations —
        # a better estimate than the raw network value (root_value) because
        # it incorporates actual tree search.  Used as the training target
        # for non-natural game terminations (Fix 1 + Fix 4).
        mcts_q_value = mcts._root.q_value if mcts._root is not None else root_value

        # AlphaZero policy target: always use the raw visit-count distribution
        # N(s,a)/ΣN — never temperature-sharpened.
        #
        # Temperature only governs *action selection* (which board positions
        # appear in the replay buffer, for state diversity).  Applying the
        # same low temperature (e.g. T=0.5, exponent 2) to the training
        # label too would sharpen even mild visit preferences into near-delta
        # functions, collapsing training distribution entropy and teaching the
        # network to be overconfident regardless of how uncertain the search
        # actually was.  Storing raw N/ΣN preserves the full distributional
        # information from all simulations and matches the AlphaZero paper.
        policy_target = mcts_policy

        # Entropy logged from the raw visit distribution (not action-selection
        # temperature) so it reflects true MCTS uncertainty over moves.
        eps = 1e-10
        entropy = float(-np.sum(mcts_policy * np.log(mcts_policy + eps)))
        value_log.append((current_player, root_value))
        entropy_log.append(entropy)

        game_data.append((state, policy_target, current_player, mcts_q_value))
        abs_board_states.append(abs_state)

        _, _, done, _, info = env.step(action)
        mcts.update_root(action)

        if info.get("turn_complete", True):
            move_count += 1
            # Safety-net no-progress draw: game.check_winner() (called inside
            # env.step()) is the primary handler for the 40-move rule, but
            # this outer check catches any edge case where it returns None
            # despite the threshold being reached (e.g. a decisive check
            # firing first on the same move).  It reads game._no_progress_count
            # directly so there is a single canonical counter with no drift.
            if not done and env.game._no_progress_count >= cfg.NO_PROGRESS_DRAW_MOVES:
                info = {"winner": "Tie"}
                done = True

    winner = info.get("winner", "Tie")

    # Per-player contempt for tie games: the side ahead in material at the
    # time of the draw receives stronger contempt (failed to convert advantage).
    tie_contempts = {BLUE: cfg.CONTEMPT_VALUE, RED: cfg.CONTEMPT_VALUE}
    if winner in ("Tie", "None"):
        blue_mat, red_mat = _get_material(env)
        tie_contempts = {
            BLUE: cfg.get_contempt_value(blue_mat, red_mat),
            RED:  cfg.get_contempt_value(red_mat,  blue_mat),
        }

    is_decisive = winner not in ("Tie", "None")
    winner_vals = [v for p, v in value_log if is_decisive and p == winner]
    loser_vals  = [v for p, v in value_log if is_decisive and p != winner]

    # MCTS Q-values (root.q_value after all sims) grouped by winner/loser.
    # Comparable to avg_root_val_* but reflects search-improved estimates.
    winner_qvals = [q for s, _, p, q in game_data if is_decisive and p == winner]
    loser_qvals  = [q for s, _, p, q in game_data if is_decisive and p != winner]

    # Per-move Q-value sequence — one value per turn, from the mover's perspective.
    # Useful for visualising how value estimates evolve during a game.
    mcts_q_sequence = [q for _, _, _, q in game_data]

    game_stats = {
        "avg_root_val_winner": float(np.mean(winner_vals))  if winner_vals  else 0.0,
        "avg_root_val_loser":  float(np.mean(loser_vals))   if loser_vals   else 0.0,
        "avg_mcts_q_winner":   float(np.mean(winner_qvals)) if winner_qvals else 0.0,
        "avg_mcts_q_loser":    float(np.mean(loser_qvals))  if loser_qvals  else 0.0,
        "avg_policy_entropy":  float(np.mean(entropy_log))  if entropy_log  else 0.0,
        "move_sequence":       ", ".join(env.game.moves),
        "mcts_q_sequence":     ", ".join(f"{q:.4f}" for q in mcts_q_sequence),
    }

    return {
        "game_data":        game_data,
        "abs_board_states": abs_board_states,   # parallel list for regret buffer
        "cap_terminated":   cap_terminated,
        "winner":           winner,
        "num_moves":        move_count,
        "game_stats":       game_stats,
        "tie_contempts":    tie_contempts,
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
            start_board=task.get("start_board"),
            start_turn=task.get("start_turn"),
            c_puct=task.get("c_puct"),
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

    def __init__(self, initial_state_dict, device, num_workers, use_wdl=False):
        self.device      = device
        self.num_workers = num_workers

        # Inference server has its own model copy — never shared with the
        # training model to avoid backward() racing with load_state_dict().
        input_shape  = (4, 8, 8)
        NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork
        self._server_model = NetworkClass(input_shape, NUM_ACTIONS).to(device)
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
# Parallel evaluation infrastructure  (two-network, mirroring WorkerContext)
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_MAX_MOVES         = 150
_EVAL_STOCHASTIC_MOVES  = 6   # first N full-turn moves played at temperature=1


def _play_eval_game_parallel(task, new_eval, old_eval):
    """Play one evaluation game using two remote evaluators.

    new_eval routes to the current-epoch inference server.
    old_eval routes to the reference-model inference server.

    task keys:
        game_idx           int  — determines which color new_net plays
        num_games          int  — total games in this eval run (for color split)
        num_simulations    int
        stochastic_opening int  — moves played at temperature=1
    """
    game_idx          = task["game_idx"]
    num_games         = task["num_games"]
    num_simulations   = task["num_simulations"]
    stoch             = task.get("stochastic_opening", _EVAL_STOCHASTIC_MOVES)

    new_color = BLUE if game_idx < num_games // 2 else RED

    new_mcts = MCTSSearch(evaluator=new_eval, num_simulations=num_simulations,
                          c_puct=cfg.C_PUCT, move_cap=_EVAL_MAX_MOVES)
    old_mcts = MCTSSearch(evaluator=old_eval, num_simulations=num_simulations,
                          c_puct=cfg.C_PUCT, move_cap=_EVAL_MAX_MOVES)

    env = CheckersEnv()
    env.reset()
    new_mcts._root = old_mcts._root = None

    move_count        = 0
    done              = False
    info              = {}

    while not done:
        if move_count >= _EVAL_MAX_MOVES:
            info = {"winner": _adjudicate_move_cap(env)}
            break

        mask = env.get_action_mask()
        if mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            break

        temp         = 1.0 if move_count < stoch else 0.0
        active_mcts  = new_mcts if env.game.turn == new_color else old_mcts
        active_mcts._root = None   # no tree reuse across eval moves (matches sequential eval)

        action, _, _ = active_mcts.select_action(env, temperature=temp,
                                                  add_noise=False,
                                                  no_progress_count=env.game._no_progress_count)
        _, _, done, _, info = env.step(action)
        if info.get("turn_complete", True):
            move_count += 1
            if not done and env.game._no_progress_count >= cfg.NO_PROGRESS_DRAW_MOVES:
                info = {"winner": "Tie"}
                done = True

    winner = info.get("winner", "Tie")
    return {"game_idx": game_idx, "winner": winner,
            "new_color": new_color, "num_moves": move_count}


def eval_worker_fn(worker_id,
                   new_req_q, new_resp_q,
                   old_req_q, old_resp_q,
                   task_queue, results_queue,
                   shm_new_state_name, shm_old_state_name,
                   num_workers):
    """Persistent eval worker process.

    Maintains one RemoteEvaluator per network (new / old) and plays eval games
    until a _WORKER_EXIT sentinel arrives.
    """
    _shm_new      = _shm_module.SharedMemory(name=shm_new_state_name)
    _shm_old      = _shm_module.SharedMemory(name=shm_old_state_name)
    _state_buf_new = np.ndarray((num_workers, 4, 8, 8), dtype=np.float32,
                                buffer=_shm_new.buf)
    _state_buf_old = np.ndarray((num_workers, 4, 8, 8), dtype=np.float32,
                                buffer=_shm_old.buf)

    new_eval = RemoteEvaluator(worker_id, new_req_q, new_resp_q, _state_buf_new)
    old_eval = RemoteEvaluator(worker_id, old_req_q, old_resp_q, _state_buf_old)

    while True:
        item = task_queue.get()
        if item is _WORKER_EXIT:
            break
        if item == _WORKER_BATCH_DONE:
            results_queue.put(_WORKER_BATCH_DONE)
            continue

        task_idx, task = item
        result = _play_eval_game_parallel(task, new_eval, old_eval)
        results_queue.put((task_idx, result))

    _shm_new.close()
    _shm_old.close()


class EvalContext:
    """Two-network GPU inference context for parallel evaluation games.

    Mirrors WorkerContext but serves two networks (new and reference) via
    separate inference server threads.  Workers interleave requests to both
    servers within each game depending on the active player.

    Usage:
        with EvalContext(new_sd, old_sd, device, num_workers) as ectx:
            stats = ectx.run_eval(num_games, num_simulations)
    """

    def __init__(self, new_state_dict, old_state_dict, device, num_workers,
                 use_wdl=False):
        self.device      = device
        self.num_workers = num_workers

        input_shape  = (4, 8, 8)
        NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork

        self._new_model = NetworkClass(input_shape, NUM_ACTIONS).to(device)
        self._new_model.load_state_dict(new_state_dict)
        self._new_model.eval()

        self._old_model = NetworkClass(input_shape, NUM_ACTIONS).to(device)
        self._old_model.load_state_dict(old_state_dict)
        self._old_model.eval()

        # Per-network IPC queues
        self._new_req_q   = Queue()
        self._new_resp_qs = [Queue() for _ in range(num_workers)]
        self._old_req_q   = Queue()
        self._old_resp_qs = [Queue() for _ in range(num_workers)]

        # Task / results queues
        self._task_queue    = Queue()
        self._results_queue = Queue()

        # Shared-memory buffers (one slab per network)
        _state_bytes = num_workers * 4 * 8 * 8 * 4   # float32
        self._shm_new = _shm_module.SharedMemory(create=True, size=_state_bytes)
        self._shm_old = _shm_module.SharedMemory(create=True, size=_state_bytes)
        self._state_buf_new = np.ndarray((num_workers, 4, 8, 8), dtype=np.float32,
                                         buffer=self._shm_new.buf)
        self._state_buf_old = np.ndarray((num_workers, 4, 8, 8), dtype=np.float32,
                                         buffer=self._shm_old.buf)

        # Two inference server threads
        self._stop_new = threading.Event()
        self._stop_old = threading.Event()
        self._new_server = AlphaZeroInferenceServer(
            self._new_model, device,
            self._new_req_q, self._new_resp_qs,
            self._stop_new,
            max_batch=num_workers * 2,
            state_buf=self._state_buf_new,
        )
        self._old_server = AlphaZeroInferenceServer(
            self._old_model, device,
            self._old_req_q, self._old_resp_qs,
            self._stop_old,
            max_batch=num_workers * 2,
            state_buf=self._state_buf_old,
        )
        self._new_server.start()
        self._old_server.start()

        # Worker processes
        self._workers = []
        for wid in range(num_workers):
            p = Process(
                target=eval_worker_fn,
                args=(
                    wid,
                    self._new_req_q, self._new_resp_qs[wid],
                    self._old_req_q, self._old_resp_qs[wid],
                    self._task_queue, self._results_queue,
                    self._shm_new.name, self._shm_old.name,
                    num_workers,
                ),
            )
            p.start()
            self._workers.append(p)

    def run_eval(self, num_games, num_simulations, label="eval"):
        """Distribute eval games across workers; return aggregate stats dict."""
        for i in range(num_games):
            self._task_queue.put((i, {
                "game_idx":          i,
                "num_games":         num_games,
                "num_simulations":   num_simulations,
                "stochastic_opening": _EVAL_STOCHASTIC_MOVES,
            }))
        for _ in range(self.num_workers):
            self._task_queue.put(_WORKER_BATCH_DONE)

        result_map   = {}
        reported     = 0
        log_interval = max(10, num_games // 5)
        idle_workers = 0

        while idle_workers < self.num_workers or len(result_map) < num_games:
            try:
                item = self._results_queue.get(timeout=1.0)
                if item == _WORKER_BATCH_DONE:
                    idle_workers += 1
                    continue
                task_idx, result = item
                result_map[task_idx] = result

                done = len(result_map)
                if done - reported >= log_interval or done == num_games:
                    w = sum(1 for r in result_map.values()
                            if r["winner"] == r["new_color"])
                    l = sum(1 for r in result_map.values()
                            if r["winner"] not in ("Tie", "None")
                            and r["winner"] != r["new_color"])
                    t = sum(1 for r in result_map.values()
                            if r["winner"] in ("Tie", "None"))
                    print(f"    {label}: {done}/{num_games} games  "
                          f"({w}W/{l}L/{t}T so far)", flush=True)
                    reported = done
            except Exception:
                if (not any(p.is_alive() for p in self._workers)
                        and len(result_map) < num_games):
                    raise RuntimeError(
                        f"[{label}] All eval workers died after "
                        f"{len(result_map)}/{num_games} games."
                    )

        results = [result_map[i] for i in range(num_games)]
        wins   = sum(1 for r in results if r["winner"] == r["new_color"])
        losses = sum(1 for r in results
                     if r["winner"] not in ("Tie", "None")
                     and r["winner"] != r["new_color"])
        ties   = sum(1 for r in results if r["winner"] in ("Tie", "None"))
        total_moves = sum(r["num_moves"] for r in results)
        n = max(num_games, 1)
        return {
            "wins": wins, "losses": losses, "ties": ties,
            "win_rate":  wins / n,
            "score":     (wins + 0.5 * ties) / n,
            "games":     num_games,
            "avg_moves": total_moves / n,
        }

    def shutdown(self):
        for _ in self._workers:
            self._task_queue.put(_WORKER_EXIT)
        for p in self._workers:
            p.join(timeout=15)

        self._stop_new.set()
        self._new_req_q.put(_SHUTDOWN)
        self._new_server.join(timeout=5)

        self._stop_old.set()
        self._old_req_q.put(_SHUTDOWN)
        self._old_server.join(timeout=5)

        self._shm_new.close(); self._shm_new.unlink()
        self._shm_old.close(); self._shm_old.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def _outcome_to_wdl(v):
    """Convert a scalar outcome in [-1, 1] to a WDL probability distribution.

    Maps v = P(win) - P(loss) to [P(win), P(draw), P(loss)]:
      v = +1.0  → [1, 0, 0]  (certain win)
      v =  0.0  → [0, 1, 0]  (certain draw)
      v = -1.0  → [0, 0, 1]  (certain loss)
      v = +0.3  → [0.3, 0.7, 0]  (partial win signal)
      v = -0.3  → [0, 0.7, 0.3]  (contempt draw — draw leaning toward loss)

    Used for:
      1. Converting the final game outcome scalar to a WDL training target.
      2. Converting the MCTS Q-value to a WDL distribution for soft-Z blending.
    """
    w = float(max(0.0, v))
    l = float(max(0.0, -v))
    d = max(0.0, 1.0 - w - l)
    return np.array([w, d, l], dtype=np.float32)


def train_alphazero_parallel(num_workers=None, use_wdl=False):
    """GPU-accelerated parallel AlphaZero training loop.

    Self-play runs across `num_workers` CPU processes.  Each MCTS leaf
    evaluation is dispatched to the GPU inference server, which batches
    requests from all workers for maximum GPU utilisation.  After all games
    finish the main process runs the supervised gradient updates on GPU.

    Args:
        num_workers: number of CPU worker processes.
                     None → auto-detect from cfg.get_num_workers_parallel().
        use_wdl:     If True, use WDLAlphaZeroNetwork with cross-entropy WDL
                     value loss and soft-Z value blending (--network-type wdl).
                     If False (default), use AlphaZeroNetwork with MSE scalar
                     value loss — fully backward-compatible with v1 checkpoints.
    """
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if num_workers is None:
        num_workers = cfg.get_num_workers_parallel()
    print(f"Workers: {num_workers}")

    # ── Training network (on GPU — used only for gradient updates) ────────
    input_shape  = (4, 8, 8)
    NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork
    network      = NetworkClass(input_shape, NUM_ACTIONS).to(device)
    c_puct_train = cfg.C_PUCT_WDL if use_wdl else cfg.C_PUCT
    network_type_str = "wdl" if use_wdl else "scalar"
    print(f"Network type: {network_type_str}  C_PUCT: {c_puct_train}")
    optimizer = optim.AdamW(
        network.parameters(),
        lr=cfg.get_lr(0), weight_decay=cfg.WEIGHT_DECAY,
    )

    replay_buffer = deque(maxlen=cfg.BUFFER_SIZE)

    # ── Directories / CSV ─────────────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    model_dir    = os.path.join(base_dir, "alphazero_checkpoints")
    detailed_dir = os.path.join(base_dir, "az_detailed_games_parallel")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    csv_path               = os.path.join(base_dir, "alphazero_training_progress_parallel.csv")
    eval_csv_path          = os.path.join(base_dir, "alphazero_eval_benchmarks.csv")
    buffer_path            = os.path.join(model_dir, "replay_buffer.npz")
    reference_buffer_path  = os.path.join(model_dir, "az_reference_buffer.npz")

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
        # Validate that the checkpoint's network type matches the current mode
        ckpt_network_type = ckpt.get("network_type", "scalar")
        if ckpt_network_type != network_type_str:
            raise RuntimeError(
                f"Checkpoint network_type='{ckpt_network_type}' does not match "
                f"--network-type '{network_type_str}'. Use the correct flag or "
                f"start a new run from scratch."
            )
        network.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # LR is now computed from epoch number — no scheduler state to restore.
        # Re-apply the correct LR for the resumed epoch immediately.
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.get_lr(start_epoch)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

        # ── Restore replay buffer from disk ──────────────────────────────
        if os.path.exists(buffer_path):
            buf_data = np.load(buffer_path)
            states, policies, outcomes = (
                buf_data["states"], buf_data["policies"], buf_data["outcomes"]
            )
            for s, p, o in zip(states, policies, outcomes):
                # WDL buffers store (N,3) outcomes; scalar buffers store (N,)
                target = o if use_wdl else float(o)
                replay_buffer.append((s, p, target))
            print(f"  Replay buffer restored: {len(replay_buffer):,} positions")
    else:
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "games", "blue_wins", "red_wins", "ties",
                "avg_moves", "policy_loss", "value_loss", "total_loss",
                "avg_grad_norm", "policy_entropy_nats",
                "avg_root_val_winner", "avg_root_val_loser",
                "buffer_size", "num_workers", "epoch_time_s",
            ])
        with open(eval_csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "prev_eval_epoch",
                "gate_wins", "gate_losses", "gate_ties",
                "gate_win_rate", "gate_score", "gate_accepted",
                "mcts_test_passed", "mcts_winning_visit_share", "mcts_root_value",
                "vs_random_wins", "vs_random_losses", "vs_random_ties",
                "vs_random_score",
                "val_clear_win", "val_clear_loss", "val_equal", "val_calibrated",
            ])

    # ── Reference model (sliding-window gate) ────────────────────────────
    # Mirrors the PPO pattern: persisted to disk so a resume always loads the
    # correct N-epochs-ago snapshot rather than re-using the latest checkpoint.
    reference_model_path = os.path.join(model_dir, "az_reference_model.pt")
    if os.path.exists(reference_model_path):
        ref_ckpt = torch.load(reference_model_path,
                               map_location=device, weights_only=False)
        prev_eval_state_dict = ref_ckpt["model_state_dict"]
        prev_eval_epoch      = ref_ckpt["epoch"]
        print(f"Loaded reference model from epoch {prev_eval_epoch}")
    else:
        prev_eval_state_dict = freeze_state_dict(network)
        prev_eval_epoch      = start_epoch
        torch.save({
            "model_state_dict":     prev_eval_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                prev_eval_epoch,
        }, reference_model_path)
        # Snapshot the buffer as it stands now (empty on first run, or pre-loaded
        # from replay_buffer.npz on a resume) so a gate rejection can revert to it.
        _init_buf = list(replay_buffer)
        if _init_buf:
            np.savez_compressed(
                reference_buffer_path,
                states   = np.array([x[0] for x in _init_buf], dtype=np.float32),
                policies = np.array([x[1] for x in _init_buf], dtype=np.float32),
                outcomes = np.array([x[2] for x in _init_buf], dtype=np.float32),
            )
        print(f"Saved initial reference model (epoch {prev_eval_epoch})")

    # ── Regret buffer for diverse starting positions ──────────────────────
    # Stores absolute board states from high-regret positions in recent games.
    # Populated after each epoch; sampled for ~REGRET_SAMPLE_PROB of tasks.
    regret_buffer = deque(maxlen=cfg.REGRET_BUFFER_SIZE)

    # ── Spawn workers + inference server, run training ────────────────────
    with WorkerContext(network.state_dict(), device, num_workers, use_wdl=use_wdl) as ctx:

        for epoch in range(start_epoch, cfg.NUM_EPOCHS):
            epoch_start = time.perf_counter()
            # Apply per-epoch LR (warm restarts at curriculum phase boundaries)
            current_lr = cfg.get_lr(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            epoch_sims = cfg.get_num_simulations(epoch)
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
            # Build task list; inject regret-buffer start states for a fraction
            # of games (RGSC-style diverse starting positions).
            game_tasks = []
            regret_list = list(regret_buffer)   # snapshot for consistent sampling
            for _ in range(cfg.GAMES_PER_EPOCH):
                task = {"epoch": epoch, "c_puct": c_puct_train}
                if regret_list and random.random() < cfg.REGRET_SAMPLE_PROB:
                    abs_state, turn = random.choice(regret_list)
                    task["start_board"] = abs_state
                    task["start_turn"]  = turn
                game_tasks.append(task)

            all_results = ctx.run_epoch(
                game_tasks, label=f"Epoch {epoch + 1}"
            )

            # ── Populate replay buffer + write detailed game CSV ─────────
            blue_wins = red_wins = ties = total_moves = new_positions = 0
            total_root_val_winner = 0.0
            total_root_val_loser  = 0.0
            total_entropy         = 0.0

            detailed_csv_path = os.path.join(
                detailed_dir, f"az_games_epoch_{epoch + 1}.csv"
            )
            detailed_headers = [
                "game_id", "epoch", "game_num", "winner", "num_moves",
                "avg_root_val_winner", "avg_root_val_loser",
                "avg_mcts_q_winner", "avg_mcts_q_loser",
                "avg_policy_entropy_nats", "move_sequence", "mcts_q_sequence",
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

                tie_contempts = res.get(
                    "tie_contempts",
                    {BLUE: cfg.CONTEMPT_VALUE, RED: cfg.CONTEMPT_VALUE},
                )
                abs_board_states = res.get("abs_board_states", [])
                for i, (state, mcts_policy, player_color, mcts_qval) in enumerate(game_data):
                    if winner in ("Tie", "None"):
                        # Variable contempt: side ahead in material at draw
                        # receives a stronger penalty for failing to convert.
                        scalar_outcome = tie_contempts.get(player_color, cfg.CONTEMPT_VALUE)
                    elif winner == player_color:
                        scalar_outcome = 1.0
                    else:
                        scalar_outcome = -1.0

                    if use_wdl:
                        # WDL label uses the raw game outcome, NOT the contempt scalar.
                        # Contempt is applied at inference time in WDLAlphaZeroNetwork
                        # .forward() as:  value = P(win) - P(loss) + CONTEMPT * P(draw)
                        # Using the contempt scalar here AND in forward() double-counts
                        # it, producing val_equal ≈ -0.5 instead of the intended ≈ -0.25.
                        #
                        # Draws:  wdl_scalar = 0.0  → _outcome_to_wdl → [0, 1, 0]  (pure draw)
                        # Wins:   wdl_scalar = +1.0 → _outcome_to_wdl → [1, 0, 0]
                        # Losses: wdl_scalar = −1.0 → _outcome_to_wdl → [0, 0, 1]
                        wdl_scalar = 0.0 if winner in ("Tie", "None") else scalar_outcome
                        game_wdl   = _outcome_to_wdl(wdl_scalar)
                        mcts_wdl   = _outcome_to_wdl(float(mcts_qval))
                        target     = (cfg.SOFT_Z_ALPHA * game_wdl
                                      + (1.0 - cfg.SOFT_Z_ALPHA) * mcts_wdl)
                    else:
                        target = scalar_outcome

                    replay_buffer.append((state, mcts_policy, target))
                    new_positions += 1

                    # Populate regret buffer for diverse starting positions
                    # (Trudeau & Bowling 2023 Go-Exploit; Tsai et al. 2026 RGSC)
                    if (i < len(abs_board_states)
                            and abs(float(mcts_qval) - scalar_outcome) > cfg.HIGH_REGRET_THRESHOLD):
                        regret_buffer.append((abs_board_states[i], player_color))

                with open(detailed_csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch * cfg.GAMES_PER_EPOCH + game_idx + 1,
                        epoch + 1,
                        game_idx + 1,
                        winner,
                        res["num_moves"],
                        round(game_stats["avg_root_val_winner"], 4),
                        round(game_stats["avg_root_val_loser"],  4),
                        round(game_stats["avg_mcts_q_winner"],   4),
                        round(game_stats["avg_mcts_q_loser"],    4),
                        round(game_stats["avg_policy_entropy"],  4),
                        game_stats["move_sequence"],
                        game_stats["mcts_q_sequence"],
                    ])


            avg_moves           = total_moves           / cfg.GAMES_PER_EPOCH
            avg_root_val_winner = total_root_val_winner / cfg.GAMES_PER_EPOCH
            avg_root_val_loser  = total_root_val_loser  / cfg.GAMES_PER_EPOCH
            epoch_avg_entropy   = total_entropy         / cfg.GAMES_PER_EPOCH

            # ── Training phase (GPU gradient updates) ────────────────────
            train_steps = cfg.get_train_steps(new_positions)
            p_loss = v_loss = t_loss = avg_grad_norm = 0.0
            if len(replay_buffer) >= cfg.BATCH_SIZE:
                network.train()
                total_p = total_v = total_t = total_gn = 0.0

                buffer_snapshot = list(replay_buffer)

                for _ in range(train_steps):
                    batch = random.sample(buffer_snapshot, cfg.BATCH_SIZE)
                    states, policies, targets = zip(*batch)

                    st = torch.FloatTensor(np.array(states)).to(device)
                    tp = torch.FloatTensor(np.array(policies)).to(device)

                    if use_wdl:
                        # WDL path: cross-entropy loss against 3-element target
                        tv = torch.FloatTensor(np.array(targets)).to(device)  # (B,3)
                        logits, wdl_pred, _ = network.forward_wdl(st)
                        log_probs = torch.log_softmax(logits, dim=1)
                        pl  = -(tp * log_probs).sum(dim=1).mean()
                        vl  = -(tv * torch.log(wdl_pred + 1e-8)).sum(dim=1).mean()

                        # Color-swap value augmentation (batch-time, value head only).
                        # The policy head is NOT exposed to swapped states because the
                        # mcts_policy action indices are orientation-specific and
                        # storing them in the buffer with swapped states caused a 30%
                        # policy loss spike (wrong cross-entropy targets dominated
                        # gradients even at 14% buffer contamination).
                        #
                        # Batch-time augmentation is correct for the value head:
                        # every batch simultaneously trains Blue-perspective and
                        # Red-perspective value predictions, balancing the feedback
                        # loop that drove the Red-dominance color bias.
                        st_aug = torch.flip(st, dims=[2])               # vertical flip
                        st_aug = torch.cat(
                            [st_aug[:, 2:4], st_aug[:, 0:2]], dim=1
                        )                                                # swap ch 0,1 ↔ 2,3
                        tv_aug = torch.stack(
                            [tv[:, 2], tv[:, 1], tv[:, 0]], dim=1
                        )                                                # swap P(win)↔P(loss)
                        _, wdl_aug, _ = network.forward_wdl(st_aug)
                        vl_aug = -(tv_aug * torch.log(wdl_aug + 1e-8)).sum(dim=1).mean()
                        vl = 0.5 * (vl + vl_aug)
                    else:
                        # Scalar path: MSE loss against scalar outcome (original behavior)
                        tv = (torch.FloatTensor(np.array(targets))
                              .unsqueeze(1).to(device))        # (B,1)
                        logits, vals = network(st)
                        log_probs = torch.log_softmax(logits, dim=1)
                        pl  = -(tp * log_probs).sum(dim=1).mean()
                        vl  = nn.MSELoss()(vals, tv)

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

                n = train_steps
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

            elapsed = time.perf_counter() - epoch_start

            # ── Logging ───────────────────────────────────────────────────
            print(f"  Epoch {epoch + 1} complete in {elapsed:.0f}s  "
                  f"({elapsed / cfg.GAMES_PER_EPOCH:.1f}s/game)")
            print(f"  Blue: {blue_wins}  Red: {red_wins}  Ties: {ties}  "
                  f"Avg Moves: {avg_moves:.1f}  "
                  f"New Positions: {new_positions}  "
                  f"Train Steps: {train_steps}")
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
            print(f"  LR: {current_lr:.2e}  "
                  f"Buffer: {len(replay_buffer)}")

            # ── Evaluation ─────────────────────────────────────────────────
            if (epoch + 1) % cfg.EVAL_INTERVAL == 0:
                gate_label = (f"epoch-{prev_eval_epoch} model"
                              if prev_eval_epoch > 0 else "initial model")
                print(f"\n  Running evaluation (gate vs {gate_label}, "
                      f"{cfg.EVAL_GAMES_GATE} games, {num_workers} workers)...",
                      flush=True)

                # MCTS correctness test — single position, fast, stays sequential
                mt = test_mcts_correctness(device)
                mt_status = "PASS" if mt["passed"] else "FAIL"
                print(f"  MCTS correctness: {mt_status}  "
                      f"(winning_visits={mt['winning_visit_share']:.0%}, "
                      f"root_val={mt['root_value']:+.3f})", flush=True)

                # Value head calibration — raw network output on known positions
                vc = test_value_head_calibration(network, device)
                vc_status = "PASS" if vc["val_calibrated"] else "FAIL"
                print(f"  Value calibration: {vc_status}  "
                      f"(4v1={vc['val_clear_win']:+.3f}, "
                      f"1v4={vc['val_clear_loss']:+.3f}, "
                      f"3v3={vc['val_equal']:+.3f})", flush=True)

                # Absolute strength — network vs random opponent (sequential)
                vr = play_vs_random(network, device, num_games=40,
                                    num_simulations=100)
                print(f"  vs Random: {vr['wins']}W / {vr['losses']}L / {vr['ties']}T  "
                      f"(score={vr['score']:.0%})", flush=True)

                # Gate evaluation — fully parallel
                gate_result   = None
                gate_accepted = None
                if cfg.GATE_ENABLED:
                    with EvalContext(network.state_dict(), prev_eval_state_dict,
                                     device, num_workers, use_wdl=use_wdl) as ectx:
                        gate_result = ectx.run_eval(
                            cfg.EVAL_GAMES_GATE, cfg.EVAL_SIMULATIONS,
                            label="vs-prev",
                        )
                    gate_accepted = gate_result["score"] >= cfg.GATE_THRESHOLD
                    g = gate_result
                    verdict = "ACCEPTED" if gate_accepted else "rejected"
                    print(f"  vs Prev: {g['wins']}W / {g['losses']}L / {g['ties']}T  "
                          f"(score={g['score']:.0%}) → {verdict}", flush=True)

                network.eval()
                ref_epoch_for_log = prev_eval_epoch
                should_advance = (not cfg.GATE_ENABLED) or (gate_accepted is True)
                if should_advance:
                    # Gate passed: advance reference model, optimizer state, and buffer.
                    prev_eval_state_dict = freeze_state_dict(network)
                    prev_eval_epoch      = epoch + 1
                    torch.save({
                        "model_state_dict":     prev_eval_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch":                prev_eval_epoch,
                    }, reference_model_path)
                    # Snapshot the buffer so we can revert to it if a future gate fails.
                    _ref_buf = list(replay_buffer)
                    np.savez_compressed(
                        reference_buffer_path,
                        states   = np.array([x[0] for x in _ref_buf], dtype=np.float32),
                        policies = np.array([x[1] for x in _ref_buf], dtype=np.float32),
                        outcomes = np.array([x[2] for x in _ref_buf], dtype=np.float32),
                        # scalar: shape (N,); WDL: shape (N,3) — np infers from data
                    )
                    print(f"  Reference model advanced to epoch {prev_eval_epoch}")
                else:
                    # Gate rejected: revert the network, optimizer, and replay buffer
                    # to the last accepted reference so training resumes from solid ground.
                    print(f"  Gate rejected — reverting to epoch-{prev_eval_epoch} "
                          f"reference...", flush=True)
                    network.load_state_dict(prev_eval_state_dict)
                    _ref_ckpt = torch.load(reference_model_path,
                                           map_location=device, weights_only=False)
                    if "optimizer_state_dict" in _ref_ckpt:
                        optimizer.load_state_dict(_ref_ckpt["optimizer_state_dict"])
                    if os.path.exists(reference_buffer_path):
                        replay_buffer.clear()
                        _bd = np.load(reference_buffer_path)
                        for s, p, o in zip(_bd["states"], _bd["policies"],
                                           _bd["outcomes"]):
                            target = o if use_wdl else float(o)
                            replay_buffer.append((s, p, target))
                        print(f"  Replay buffer reverted: "
                              f"{len(replay_buffer):,} positions")
                    else:
                        replay_buffer.clear()
                        print(f"  Replay buffer cleared (no reference snapshot found)")
                    network.eval()
                    print(f"  Reverted to epoch-{prev_eval_epoch} model")

                g = gate_result or {}
                with open(eval_csv_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, ref_epoch_for_log,
                        g.get("wins", ""), g.get("losses", ""), g.get("ties", ""),
                        round(g["win_rate"], 4) if g else "",
                        round(g["score"],    4) if g else "",
                        gate_accepted if g else "",
                        mt["passed"],
                        mt["winning_visit_share"],
                        mt["root_value"],
                        vr["wins"], vr["losses"], vr["ties"],
                        round(vr["score"], 4),
                        vc["val_clear_win"], vc["val_clear_loss"],
                        vc["val_equal"], vc["val_calibrated"],
                    ])

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, cfg.GAMES_PER_EPOCH,
                    blue_wins, red_wins, ties, avg_moves,
                    p_loss, v_loss, t_loss,
                    avg_grad_norm, buffer_entropy,
                    round(avg_root_val_winner, 4), round(avg_root_val_loser, 4),
                    len(replay_buffer), num_workers, round(elapsed, 1),
                ])

            # ── Replay buffer — save every epoch ─────────────────────────
            if len(replay_buffer) > 0:
                buf_list = list(replay_buffer)
                np.savez_compressed(
                    buffer_path,
                    states   = np.array([x[0] for x in buf_list],
                                        dtype=np.float32),
                    policies = np.array([x[1] for x in buf_list],
                                        dtype=np.float32),
                    outcomes = np.array([x[2] for x in buf_list],
                                        dtype=np.float32),
                    # scalar mode: outcomes shape (N,)
                    # WDL mode:    outcomes shape (N,3) — np infers from data
                )
                print(f"  Replay buffer saved: {len(replay_buffer):,} positions")

            # ── Checkpoint ────────────────────────────────────────────────
            if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                path = os.path.join(model_dir, f"az_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch":                epoch + 1,
                    "network_type":         network_type_str,   # for resume validation
                    "model_state_dict":     network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
    parser.add_argument(
        "--network-type", choices=["scalar", "wdl"], default="scalar",
        dest="network_type",
        help=(
            "scalar (default): AlphaZeroNetwork with tanh value head and MSE loss "
            "— fully backward-compatible with v1 checkpoints. "
            "wdl: WDLAlphaZeroNetwork with [P(win),P(draw),P(loss)] softmax head, "
            "cross-entropy value loss, and soft-Z blending (new run required)."
        ),
    )
    args = parser.parse_args()
    train_alphazero_parallel(
        num_workers=args.workers,
        use_wdl=(args.network_type == "wdl"),
    )
