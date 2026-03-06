"""MCTS parallel training infrastructure: sentinels, inference server, remote evaluator, workers."""

import threading
from multiprocessing import Process, Queue
from multiprocessing import shared_memory as _shm_module

import numpy as np

from checkers_game.constants import BLUE, RED, NUM_ACTIONS

from rl.envs import CheckersEnv
from rl.algorithms.mcts.mcts_search import MCTSSearch
from rl.networks import AlphaZeroNetwork, WDLAlphaZeroNetwork
from rl.algorithms.mcts.utils import (
    adjudicate_move_cap,
    get_curriculum_options,
    get_material,
)
from rl.training_utils.gpu_inference_server import AlphaZeroInferenceServer
from rl.training_utils.worker_pool import BaseWorkerContext
from rl.training_utils.parallel_utils import attach_shm_buffers, WORKER_EXIT as WORKER_EXIT, WORKER_BATCH_DONE as WORKER_BATCH_DONE
from rl.configs.mcts_config import MCTSConfig

default_config = MCTSConfig()

# Import so NumpyCheckersEnv is available in worker processes when MCTSSearch runs
from rl.envs import NumpyCheckersEnv  # noqa: F401


class RemoteEvaluator:
    """Drop-in for MCTSSearch.evaluator — routes inference through the GPU server.

    Called once per MCTS simulation for each leaf evaluation.  Uses the
    shared-memory protocol when buffers are available to eliminate pickling.
    """

    def __init__(self, worker_id, request_queue, response_queue,
                 state_buf=None, mask_buf=None):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._state_buf = state_buf
        self._mask_buf = mask_buf

    def __call__(self, state, action_mask):
        """Send (state, action_mask) to GPU server, return (logits_np, value)."""
        if self._state_buf is not None:
            self._state_buf[self.worker_id][:] = state
            self.request_queue.put(self.worker_id)
        else:
            self.request_queue.put((self.worker_id, state, action_mask))
        return self.response_queue.get()


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side: play one complete self-play game
# ─────────────────────────────────────────────────────────────────────────────


def _play_game(worker_id, request_queue, response_queue,
                        epoch, state_buf=None, mask_buf=None,
                        start_board=None, start_turn=None, c_puct=None, config=None):
    config = config or default_config
    """Play one complete AlphaZero self-play game using the GPU inference server.

    Args:
        start_board: Optional (4,8,8) absolute board state from
                     env.get_absolute_board_state() to start from instead of
                     the standard initial position (regret-buffer diverse starts).
        start_turn:  Color constant (BLUE/RED) for start_board; required when
                     start_board is provided.
        c_puct:      PUCT exploration constant override; defaults to config.C_PUCT.
    """
    evaluator = RemoteEvaluator(
        worker_id, request_queue, response_queue, state_buf, mask_buf
    )

    curriculum_opts = get_curriculum_options(epoch)
    num_sims = config.get_num_simulations(epoch)
    temp_threshold, temp_late = config.get_temperature_config(epoch)
    effective_c_puct = c_puct if c_puct is not None else config.C_PUCT

    mcts = MCTSSearch(
        evaluator=evaluator,
        num_simulations=num_sims,
        c_puct=effective_c_puct,
        dirichlet_alpha=config.DIRICHLET_ALPHA,
        dirichlet_epsilon=config.DIRICHLET_EPSILON,
        move_cap=config.get_max_game_moves(epoch),
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
        if move_count >= config.get_max_game_moves(epoch):
            info = {"winner": adjudicate_move_cap(env)}
            cap_terminated = True
            break

        action_mask = env.get_action_mask()

        if action_mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            break

        temperature = (
            config.TEMPERATURE_EARLY
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

        mcts_q_value = mcts._root.q_value if mcts._root is not None else root_value
        policy_target = mcts_policy

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
            if not done and env.game._no_progress_count >= config.NO_PROGRESS_DRAW_MOVES:
                info = {"winner": "Tie"}
                done = True

    winner = info.get("winner", "Tie")

    tie_contempts = {BLUE: config.CONTEMPT_VALUE, RED: config.CONTEMPT_VALUE}
    if winner in ("Tie", "None"):
        blue_mat, red_mat = get_material(env)
        tie_contempts = {
            BLUE: config.get_contempt_value(blue_mat, red_mat),
            RED:  config.get_contempt_value(red_mat,  blue_mat),
        }

    is_decisive = winner not in ("Tie", "None")
    winner_vals = [v for p, v in value_log if is_decisive and p == winner]
    loser_vals  = [v for p, v in value_log if is_decisive and p != winner]

    winner_qvals = [q for s, _, p, q in game_data if is_decisive and p == winner]
    loser_qvals  = [q for s, _, p, q in game_data if is_decisive and p != winner]

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
        "abs_board_states": abs_board_states,
        "cap_terminated":   cap_terminated,
        "winner":           winner,
        "num_moves":        move_count,
        "game_stats":       game_stats,
        "tie_contempts":    tie_contempts,
    }


def worker_fn(worker_id, request_queue, response_queue, results_queue,
              task_queue, progress_queue,
              shm_state_name=None, shm_mask_name=None, num_workers=None, seed=None):
    if seed is not None:
        from rl.utils.seed_utils import set_seed
        set_seed(seed + worker_id)
    """Persistent worker process.

    Pulls tasks from task_queue in a loop.
    Sentinels: _WORKER_EXIT → terminate; _WORKER_BATCH_DONE → echo back.
    task_dict keys: "epoch", optionally "start_board", "start_turn", "c_puct".
    """
    _shm_state, _shm_mask, _state_buf, _mask_buf = attach_shm_buffers(
        shm_state_name, shm_mask_name, num_workers, obs_shape=(4, 8, 8), num_actions=NUM_ACTIONS
    )

    games_played = 0

    while True:
        item = task_queue.get()

        if item is WORKER_EXIT:
            if _shm_state is not None:
                _shm_state.close()
                _shm_mask.close()
            break

        if item == WORKER_BATCH_DONE:
            results_queue.put(WORKER_BATCH_DONE)
            continue

        task_index, task = item

        result = _play_game(
            worker_id, request_queue, response_queue,
            task["epoch"],
            state_buf=_state_buf,
            mask_buf=_mask_buf,
            start_board=task.get("start_board"),
            start_turn=task.get("start_turn"),
            c_puct=task.get("c_puct"),
            config=task.get("config")
        )
        results_queue.put((task_index, result))

        games_played += 1
        if games_played % 50 == 0:
            progress_queue.put((worker_id, games_played))


class WorkerContext(BaseWorkerContext):
    """Manages the persistent worker pool and GPU inference server.

    Workers are spawned once and reused across all epochs.  Model weights are
    hot-swapped between epochs via update_model() without restarting anything.

    Usage (context manager):
        with WorkerContext(model.state_dict(), device, num_workers) as ctx:
            for epoch in range(NUM_EPOCHS):
                ctx.update_model(model.state_dict())
                results = ctx.run_epoch(game_tasks, label=f"Epoch {epoch}")
    """

    def __init__(self, initial_state_dict, device, num_workers, use_wdl=False, config=None, seed=None):
        super().__init__(device, num_workers, obs_shape=(4, 8, 8), num_actions=NUM_ACTIONS)
        config = config or default_config
        self.config = config

        input_shape  = (4, 8, 8)
        NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork
        self._server_model = NetworkClass(input_shape, NUM_ACTIONS).to(device)
        self._server_model.load_state_dict(initial_state_dict)
        self._server_model.eval()

        self._server = AlphaZeroInferenceServer(
            self._server_model, device,
            self._request_queue, self._response_queues,
            self._stop_event,
            max_batch=num_workers * 4,
            state_buf=self._state_buf,
            mask_buf=self._mask_buf
        )
        self._server.start()

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
                    seed,
                ),
            )
            p.start()
            self._workers.append(p)

    def update_model(self, state_dict):
        """Push new model weights into the inference server (between epochs)."""
        self._server.update_weights(state_dict)

    def run_epoch(self, game_tasks, label="games"):
        return self._distribute_tasks_and_collect(game_tasks, label)


