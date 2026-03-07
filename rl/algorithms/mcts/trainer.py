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
    python -m rl.algorithms.mcts.trainer_parallel
    python -m rl.algorithms.mcts.trainer_parallel --workers 12
"""

import os
import csv
import random
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.configs.mcts_config import MCTSConfig
default_config = MCTSConfig()

from rl.networks import AlphaZeroNetwork, WDLAlphaZeroNetwork
from rl.training_utils.device_utils import get_device
from checkers_game.constants import BLUE, RED, NUM_ACTIONS

from rl.eval.mcts_evaluate import (
    freeze_state_dict, test_mcts_correctness,
    play_vs_random, test_value_head_calibration,
    EvalContext,
)


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure (sentinels, inference server, remote evaluator)
# ─────────────────────────────────────────────────────────────────────────────

from rl.utils.seed_utils import set_seed
from rl.algorithms.mcts.parallel_infra import WorkerContext
from rl.algorithms.mcts.utils import (
    outcome_to_wdl,
    compute_buffer_entropy,
    resume_from_checkpoint,
    init_training_csv_headers,
    populate_replay_buffer,
    run_training_step,
    save_replay_buffer,
)


def train_alphazero_parallel(num_workers=None, use_wdl=False, seed=None, config=None):
    config = config or default_config
    """GPU-accelerated parallel AlphaZero training loop.

    Self-play runs across `num_workers` CPU processes.  Each MCTS leaf
    evaluation is dispatched to the GPU inference server, which batches
    requests from all workers for maximum GPU utilisation.  After all games
    finish the main process runs the supervised gradient updates on GPU.

    Args:
        num_workers: number of CPU worker processes.
                     None → auto-detect from config.get_num_workers_parallel().
        use_wdl:     If True, use WDLAlphaZeroNetwork with cross-entropy WDL
                     value loss and soft-Z value blending (--network-type wdl).
                     If False (default), use AlphaZeroNetwork with MSE scalar
                     value loss — fully backward-compatible with v1 checkpoints.
        seed:        Random seed for reproducibility (optional).
    """
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if num_workers is None:
        num_workers = config.get_num_workers_parallel()
    print(f"Workers: {num_workers}")

    # ── Training network (on GPU — used only for gradient updates) ────────
    input_shape  = (4, 8, 8)
    NetworkClass = WDLAlphaZeroNetwork if use_wdl else AlphaZeroNetwork
    network      = NetworkClass(input_shape, NUM_ACTIONS).to(device)
    c_puct_train = config.C_PUCT_WDL if use_wdl else config.C_PUCT
    network_type_str = "wdl" if use_wdl else "scalar"
    print(f"Network type: {network_type_str}  C_PUCT: {c_puct_train}")
    optimizer = optim.AdamW(
        network.parameters(),
        lr=config.get_lr(0), weight_decay=config.WEIGHT_DECAY,
    )

    replay_buffer = deque(maxlen=config.BUFFER_SIZE)

    # ── Directories / CSV ─────────────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir  = os.path.join(base_dir, "training_results", "mcts")
    model_dir    = os.path.join(results_dir, "alphazero_checkpoints")
    detailed_dir = os.path.join(results_dir, "az_detailed_games_parallel")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    csv_path               = os.path.join(results_dir, "alphazero_training_progress_parallel.csv")
    eval_csv_path          = os.path.join(results_dir, "alphazero_eval_benchmarks.csv")
    buffer_path            = os.path.join(model_dir, "replay_buffer.npz")
    reference_buffer_path  = os.path.join(model_dir, "az_reference_buffer.npz")

    # ── Resume from latest checkpoint ────────────────────────────────────
    start_epoch = resume_from_checkpoint(
        model_dir, buffer_path, network, optimizer, device,
        network_type_str, use_wdl, replay_buffer,
    )
    if start_epoch == 0:
        init_training_csv_headers(csv_path, eval_csv_path)

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
            "rng_state_torch":      torch.random.get_rng_state(),
            "rng_state_numpy":      np.random.get_state(),
            "rng_state_python":     random.getstate(),
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
    regret_buffer = deque(maxlen=config.REGRET_BUFFER_SIZE)

    # ── Spawn workers + inference server, run training ────────────────────
    with WorkerContext(network.state_dict(), device, num_workers, use_wdl=use_wdl, config=config, seed=seed) as ctx:

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            epoch_start = time.perf_counter()
            # Apply per-epoch LR (warm restarts at curriculum phase boundaries)
            current_lr = config.get_lr(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            epoch_sims = config.get_num_simulations(epoch)
            phase = ("phase1" if epoch < config.CURRICULUM_PHASE1_END
                     else "phase2" if epoch < config.CURRICULUM_PHASE2_END
                     else "full")
            print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS} — "
                  f"self-play ({config.GAMES_PER_EPOCH} games, "
                  f"{epoch_sims} sims/move, {phase}, "
                  f"{num_workers} workers)")

            # Push latest weights into inference server before self-play
            ctx.update_model(network.state_dict())

            # ── Self-play phase ──────────────────────────────────────────
            # Build task list; inject regret-buffer start states for a fraction
            # of games (RGSC-style diverse starting positions).
            game_tasks = []
            regret_list = list(regret_buffer)   # snapshot for consistent sampling
            for _ in range(config.GAMES_PER_EPOCH):
                task = {"epoch": epoch, "c_puct": c_puct_train, "config": config}
                if regret_list and random.random() < config.REGRET_SAMPLE_PROB:
                    abs_state, turn = random.choice(regret_list)
                    task["start_board"] = abs_state
                    task["start_turn"]  = turn
                game_tasks.append(task)

            all_results = ctx.run_epoch(
                game_tasks, label=f"Epoch {epoch + 1}"
            )

            # ── Populate replay buffer + write detailed game CSV ─────────
            detailed_csv_path = os.path.join(
                detailed_dir, f"az_games_epoch_{epoch + 1}.csv"
            )
            detailed_headers = [
                "game_id", "epoch", "game_num", "winner", "num_moves",
                "avg_root_val_winner", "avg_root_val_loser",
                "avg_mcts_q_winner", "avg_mcts_q_loser",
                "avg_policy_entropy_nats", "move_sequence", "mcts_q_sequence",
            ]
            new_positions, stats = populate_replay_buffer(
                all_results, replay_buffer, regret_buffer, use_wdl,
                detailed_csv_path, detailed_headers, epoch,
            )
            blue_wins = stats["blue_wins"]
            red_wins = stats["red_wins"]
            ties = stats["ties"]
            avg_moves = stats["avg_moves"]
            avg_root_val_winner = stats["avg_root_val_winner"]
            avg_root_val_loser = stats["avg_root_val_loser"]

            # ── Training phase (GPU gradient updates) ────────────────────
            p_loss, v_loss, t_loss, avg_grad_norm, train_steps = run_training_step(
                network, optimizer, replay_buffer, new_positions, device, use_wdl,
            )

            # Buffer-level policy entropy
            buffer_entropy = compute_buffer_entropy(replay_buffer)

            elapsed = time.perf_counter() - epoch_start

            # ── Logging ───────────────────────────────────────────────────
            print(f"  Epoch {epoch + 1} complete in {elapsed:.0f}s  "
                  f"({elapsed / config.GAMES_PER_EPOCH:.1f}s/game)")
            print(f"  Blue: {blue_wins}  Red: {red_wins}  Ties: {ties}  "
                  f"Avg Moves: {avg_moves:.1f}  "
                  f"New Positions: {new_positions}  "
                  f"Train Steps: {train_steps}")
            print(f"  Policy Loss: {p_loss:.4f}  Value Loss: {v_loss:.4f}  "
                  f"Total: {t_loss:.4f}")
            print(f"  Grad Norm (pre-clip): {avg_grad_norm:.4f}  "
                  f"[clip={config.GRAD_CLIP_NORM}  "
                  f"{'BINDING' if avg_grad_norm > config.GRAD_CLIP_NORM * 0.9 else 'not binding'}]")
            print(f"  Buffer Policy Entropy: {buffer_entropy:.4f} nats  "
                  f"(max uniform ~{np.log(8):.2f} for 8-move branching)")
            print(f"  Value calibration — winner avg: {avg_root_val_winner:+.3f}  "
                  f"loser avg: {avg_root_val_loser:+.3f}  "
                  f"(ideal: +1.0 / -1.0)")
            print(f"  LR: {current_lr:.2e}  "
                  f"Buffer: {len(replay_buffer)}")

            # ── Checkpoint (before eval so az_epoch_N.pt always has epoch N weights) ──
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                path = os.path.join(model_dir, f"az_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch":                epoch + 1,
                    "network_type":         network_type_str,
                    "model_state_dict":     network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "blue_wins":            blue_wins,
                    "red_wins":             red_wins,
                    "ties":                 ties,
                    "rng_state_torch":      torch.random.get_rng_state(),
                    "rng_state_numpy":      np.random.get_state(),
                    "rng_state_python":     random.getstate(),
                }, path)
                print(f"  Checkpoint saved: {path}")

            # ── Evaluation ─────────────────────────────────────────────────
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                gate_label = (f"epoch-{prev_eval_epoch} model"
                              if prev_eval_epoch > 0 else "initial model")
                print(f"\n  Running evaluation (gate vs {gate_label}, "
                      f"{config.EVAL_GAMES_GATE} games, {num_workers} workers)...",
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
                if config.GATE_ENABLED:
                    with EvalContext(network.state_dict(), prev_eval_state_dict,
                                     device, num_workers, use_wdl=use_wdl, seed=seed) as ectx:
                        gate_result = ectx.run_eval(
                            config.EVAL_GAMES_GATE, config.EVAL_SIMULATIONS,
                            label="vs-prev",
                        )
                    gate_accepted = gate_result["score"] >= config.GATE_THRESHOLD
                    g = gate_result
                    verdict = "ACCEPTED" if gate_accepted else "rejected"
                    print(f"  vs Prev: {g['wins']}W / {g['losses']}L / {g['ties']}T  "
                          f"(score={g['score']:.0%}) → {verdict}", flush=True)

                network.eval()
                ref_epoch_for_log = prev_eval_epoch
                should_advance = (not config.GATE_ENABLED) or (gate_accepted is True)
                if should_advance:
                    # Gate passed: advance reference model, optimizer state, and buffer.
                    prev_eval_state_dict = freeze_state_dict(network)
                    prev_eval_epoch      = epoch + 1
                    torch.save({
                        "model_state_dict":     prev_eval_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch":                prev_eval_epoch,
                        "rng_state_torch":      torch.random.get_rng_state(),
                        "rng_state_numpy":      np.random.get_state(),
                        "rng_state_python":     random.getstate(),
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
                          
                    # Handle state dict key mismatches (e.g., 'net.' prefix)
                    model_has_net = any(k.startswith('net.') for k in network.state_dict().keys())
                    ckpt_has_net = any(k.startswith('net.') for k in prev_eval_state_dict.keys())
                    
                    if model_has_net and not ckpt_has_net:
                        prev_eval_state_dict = {f"net.{k}": v for k, v in prev_eval_state_dict.items()}
                    elif ckpt_has_net and not model_has_net:
                        prev_eval_state_dict = {k.replace('net.', '', 1): v for k, v in prev_eval_state_dict.items() if k.startswith('net.')}
                        
                    network.load_state_dict(prev_eval_state_dict)
                    _ref_ckpt = torch.load(reference_model_path,
                                           map_location=device, weights_only=False)
                    if "optimizer_state_dict" in _ref_ckpt:
                        optimizer.load_state_dict(_ref_ckpt["optimizer_state_dict"])
                    if "rng_state_torch" in _ref_ckpt:
                        torch.random.set_rng_state(_ref_ckpt["rng_state_torch"].cpu())
                    if "rng_state_numpy" in _ref_ckpt:
                        np.random.set_state(_ref_ckpt["rng_state_numpy"])
                    if "rng_state_python" in _ref_ckpt:
                        random.setstate(_ref_ckpt["rng_state_python"])
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
                    epoch + 1, config.GAMES_PER_EPOCH,
                    blue_wins, red_wins, ties, avg_moves,
                    p_loss, v_loss, t_loss,
                    avg_grad_norm, buffer_entropy,
                    round(avg_root_val_winner, 4), round(avg_root_val_loser, 4),
                    len(replay_buffer), num_workers, round(elapsed, 1),
                ])

            # ── Replay buffer — save every epoch ─────────────────────────
            if len(replay_buffer) > 0:
                save_replay_buffer(replay_buffer, buffer_path)
                print(f"  Replay buffer saved: {len(replay_buffer):,} positions")

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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fully-deterministic", action="store_true",
        help="Force CuDNN determinism (slower)",
    )
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed, fully_deterministic=args.fully_deterministic)
    train_alphazero_parallel(
        num_workers=args.workers,
        use_wdl=(args.network_type == "wdl"),
        seed=args.seed,
    )
