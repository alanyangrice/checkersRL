"""Benchmark: measures time saved by the key checkersRL optimizations.

Sections
--------
  1. Env clone:      copy.deepcopy vs CheckersEnv.fast_clone vs NumpyEnv.fast_clone
  2. IPC:            pickle serialization round-trip vs shared-memory write/read
  3. GAE:            pure Python backward scan vs torch.jit.script
  4. Batch scaling:  N individual batch-1 calls vs one batch-N forward pass
  5. Observation:    CheckersEnv.get_board_state vs NumpyCheckersEnv.get_board_state
  6. Action mask:    CheckersEnv._update_action_mask vs NumpyCheckersEnv version
  7. GPU transfer:   plain .to(device) vs pinned-memory non_blocking (CUDA only)

Run from repo root:
    python -m rl.scripts.benchmark_optimizations
    python -m rl.scripts.benchmark_optimizations --quick
"""

import argparse
import copy
import pickle
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from checkers_game.constants import NUM_ACTIONS
from rl.envs.checkers_env import CheckersEnv
from rl.envs.numpy_checkers_env import NumpyCheckersEnv
from rl.networks.policy_network import PPOPolicyNetwork
from rl.algorithms.ppo.agent import compute_gae


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _row(label, mean_ms, note=""):
    print(f"  {label:<50s} {mean_ms:8.4f} ms  {note}")


def _timeit(fn, n_warmup, n_iters):
    """Return mean wall-clock time in milliseconds over n_iters calls."""
    if n_iters < 1:
        n_iters = 1
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000


def _advance_game(env, moves=25):
    """Play N random moves to produce a realistic mid-game state with a
    non-trivial board_states repetition dict (~25 entries)."""
    env.reset()
    for _ in range(moves):
        mask = env.get_action_mask()
        if mask.sum() == 0:
            break
        action = int(np.random.choice(np.where(mask > 0)[0]))
        _, _, done, _, _ = env.step(action)
        if done:
            env.reset()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment clone
# ─────────────────────────────────────────────────────────────────────────────

def bench_clones(n_warmup, n_iters):
    _banner("1. Environment Clone Comparison")
    print("  Context: MCTS calls fast_clone() once per simulation.")
    print("  100 simulations per move × many moves per game.\n")

    env = CheckersEnv()
    _advance_game(env, moves=25)
    numpy_env = NumpyCheckersEnv.from_env(env)

    # (a) copy.deepcopy — naive Python deep copy
    #     Traverses the full Game object graph: board (8×8 Piece objects),
    #     board_states dict (~25 nested-tuple keys), moves list, etc.
    t_deepcopy = _timeit(lambda: copy.deepcopy(env), n_warmup, n_iters)

    # (b) CheckersEnv.fast_clone — hand-rolled clone
    #     Skips board_states and moves (irrelevant for short MCTS sims).
    #     Still copies individual Piece objects via Piece.clone().
    t_fast = _timeit(env.fast_clone, n_warmup, n_iters)

    # (c) NumpyCheckersEnv.fast_clone — 64-byte memcpy of the int8 board array
    #     No Python object graph traversal at all.
    t_numpy = _timeit(numpy_env.fast_clone, n_warmup, n_iters)

    # (d) NumpyCheckersEnv.from_env — one-time conversion (paid once per search call,
    #     not per simulation). Shows the fixed overhead of switching representations.
    t_from = _timeit(lambda: NumpyCheckersEnv.from_env(env), n_warmup, max(n_iters // 5, 50))

    _row("copy.deepcopy(CheckersEnv)    [baseline]", t_deepcopy)
    _row("CheckersEnv.fast_clone()      [skips dicts]", t_fast,
         f"{t_deepcopy/t_fast:.1f}x faster than deepcopy")
    _row("NumpyCheckersEnv.fast_clone() [64-byte memcpy]", t_numpy,
         f"{t_deepcopy/t_numpy:.1f}x faster than deepcopy")
    _row("NumpyCheckersEnv.from_env()   [one-time cost]", t_from)

    sims = 100
    cost_naive  = sims * t_deepcopy
    cost_numpy  = t_from + sims * t_numpy
    print(f"\n  Projected cost for {sims} MCTS simulations (per search call):")
    print(f"    Naive  ({sims}× deepcopy)                 : {cost_naive:7.2f} ms")
    print(f"    Numpy  (1× from_env + {sims}× fast_clone) : {cost_numpy:7.2f} ms")
    print(f"    Speedup                                  : {cost_naive/cost_numpy:.1f}x")

    return t_deepcopy, t_numpy


# ─────────────────────────────────────────────────────────────────────────────
# 2. IPC serialization: pickle vs shared memory
# ─────────────────────────────────────────────────────────────────────────────

def bench_ipc(n_warmup, n_iters):
    _banner("2. IPC Serialization: Pickle vs Shared Memory")
    print("  NOTE: measures serialization cost only, not OS pipe latency.")
    print("  multiprocessing.Queue uses pickle internally on every put/get.\n")

    state = np.random.randn(4, 8, 8).astype(np.float32)   # 1 024 bytes of actual data
    mask  = np.zeros(NUM_ACTIONS, dtype=np.float32)         #   680 bytes
    mask[:10] = 1.0
    worker_id  = 3
    num_workers = 16

    # ── Normal (pickle) path ──────────────────────────────────────────────────
    # Worker does: request_queue.put((worker_id, state, mask))
    # Server does: worker_id, state, mask = request_queue.get()
    # Queue.put() calls pickle.dumps; Queue.get() calls pickle.loads.
    def pickle_roundtrip():
        data = pickle.dumps((worker_id, state, mask))
        _wid, _s, _m = pickle.loads(data)

    t_pickle = _timeit(pickle_roundtrip, n_warmup, n_iters)
    payload_bytes = len(pickle.dumps((worker_id, state, mask)))

    # ── Shared memory path ────────────────────────────────────────────────────
    # A single shared-memory block is allocated once at startup (BaseWorkerContext.__init__).
    # The block covers all workers: shape (num_workers, 4, 8, 8) for states,
    # (num_workers, NUM_ACTIONS) for masks — backed by a single OS shared-memory object.
    #
    # Worker does: state_buf[worker_id][:] = state  ← direct write, no pickling
    #              mask_buf[worker_id][:]  = mask
    #              request_queue.put(worker_id)      ← sends just an int
    #
    # Server does: states_np = state_buf[worker_ids].copy()  ← numpy slice + 1 copy
    #              masks_np  = mask_buf[worker_ids].copy()
    state_shm = SharedMemory(create=True, size=num_workers * 4 * 8 * 8 * 4)
    mask_shm  = SharedMemory(create=True, size=num_workers * NUM_ACTIONS * 4)
    state_buf = np.ndarray((num_workers, 4, 8, 8), dtype=np.float32, buffer=state_shm.buf)
    mask_buf  = np.ndarray((num_workers, NUM_ACTIONS), dtype=np.float32, buffer=mask_shm.buf)

    def shm_write_read():
        state_buf[worker_id][:] = state          # worker writes — no serialization
        mask_buf[worker_id][:]  = mask
        _s = state_buf[[worker_id]].copy()       # server reads — one slice copy
        _m = mask_buf[[worker_id]].copy()

    t_shm = _timeit(shm_write_read, n_warmup, n_iters)

    try:
        state_shm.close(); state_shm.unlink()
        mask_shm.close();  mask_shm.unlink()
    except Exception:
        pass

    _row(f"pickle.dumps + pickle.loads  ({payload_bytes} bytes)", t_pickle)
    _row("shm write + numpy slice copy  [no pickle]", t_shm,
         f"{t_pickle/t_shm:.1f}x faster")

    # 5000 games × ~50 agent turns / game = 250 000 IPC calls per epoch
    n_calls = 250_000
    cost_p = n_calls * t_pickle / 1_000
    cost_s = n_calls * t_shm    / 1_000
    print(f"\n  Projected serialization cost per epoch ({n_calls:,} agent turns):")
    print(f"    Pickle path  : {cost_p:.1f} s")
    print(f"    Shared memory: {cost_s:.1f} s")
    print(f"    Time saved   : {cost_p - cost_s:.1f} s/epoch")

    return t_pickle, t_shm


# ─────────────────────────────────────────────────────────────────────────────
# 3. GAE: pure Python vs torch.jit.script
# ─────────────────────────────────────────────────────────────────────────────

def bench_gae(n_warmup, n_iters, rollout_len=5_000):
    _banner("3. GAE Computation: Pure Python vs torch.jit.script")
    print(f"  Rollout length: {rollout_len:,} timesteps (typical PPO epoch)\n")

    gamma, lam = 0.99, 0.95
    deltas_np   = np.random.randn(rollout_len).astype(np.float32)
    notdone_np  = (np.random.rand(rollout_len) > 0.05).astype(np.float32)
    deltas_t    = torch.from_numpy(deltas_np)
    notdone_t   = torch.from_numpy(notdone_np)

    # Reference: plain Python backward scan — one float multiply per step
    def gae_python():
        n = len(deltas_np)
        adv = [0.0] * n
        gae = 0.0
        for t in range(n - 1, -1, -1):
            gae = float(deltas_np[t]) + gamma * lam * float(notdone_np[t]) * gae
            adv[t] = gae
        return adv

    # JIT version (agent.py:compute_gae) — compiled on first call
    def gae_jit():
        return compute_gae(deltas_t, notdone_t, gamma, lam)

    # Ensure JIT is compiled before timing
    for _ in range(5):
        gae_jit()

    t_python = _timeit(gae_python, max(n_warmup // 10, 5), max(n_iters // 10, 20))
    t_jit    = _timeit(gae_jit,    n_warmup, n_iters)

    _row(f"Pure Python loop   (n={rollout_len:,})", t_python)
    _row(f"torch.jit.script   (n={rollout_len:,})", t_jit,
         f"{t_python/t_jit:.1f}x faster")
    print(f"\n  Pure Python runs once per PPO epoch (negligible vs 5 000 games),")
    print(f"  but JIT also removes Python interpreter overhead inside the loop.")

    return t_python, t_jit


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference batch scaling
# ─────────────────────────────────────────────────────────────────────────────

def bench_inference(n_warmup, n_iters):
    _banner("4. Inference Batch Scaling (PPOPolicyNetwork)")
    print("  Key question: is 1×batch-16 faster than 16×batch-1?")
    print("  GPU batch overhead is dominated by kernel launch cost,")
    print("  so large batches amortize it across many samples.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PPOPolicyNetwork((4, 8, 8), NUM_ACTIONS).to(device)
    model.eval()
    print(f"  Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + "\n")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    batch_sizes = [1, 2, 4, 8, 16, 32]
    print(f"  {'Batch':>6}  {'ms/fwd':>9}  {'ms/sample':>10}  {'samples/s':>11}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*11}")

    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, 4, 8, 8, device=device)
        def fwd(d=dummy):
            with torch.no_grad():
                model(d)
            sync()
        t_ms = _timeit(fwd, n_warmup, n_iters)
        per_sample = t_ms / bs
        throughput = bs / (t_ms / 1000)
        results[bs] = (t_ms, per_sample)
        print(f"  {bs:>6d}  {t_ms:>9.3f}  {per_sample:>10.4f}  {throughput:>10.0f}/s")

    # The GPU server batches N workers' requests into one forward pass.
    # Show the speedup of that strategy.
    n_workers = 16
    best_bs   = min(n_workers, max(results.keys()))
    t_serial  = n_workers * results[1][0]
    t_batched = results[best_bs][0]
    print(f"\n  {n_workers} workers, serial (16×batch-1): {t_serial:.2f} ms")
    print(f"  {n_workers} workers, batched (1×batch-{best_bs}): {t_batched:.2f} ms")
    print(f"  GPU server speedup: {t_serial/t_batched:.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Board-state observation generation
# ─────────────────────────────────────────────────────────────────────────────

def bench_board_state(n_warmup, n_iters):
    _banner("5. Board-State Observation: CheckersEnv vs NumpyCheckersEnv")
    print("  Called once per MCTS node expansion — extremely hot path.\n")

    env = CheckersEnv()
    _advance_game(env, moves=25)
    numpy_env = NumpyCheckersEnv.from_env(env)

    # CheckersEnv: nested Python loops over 8×8 grid, accessing Piece objects
    t_cv = _timeit(env.get_board_state, n_warmup, n_iters)
    # NumpyCheckersEnv: vectorized numpy comparisons (b == BLUE_PIECE, etc.)
    t_np = _timeit(numpy_env.get_board_state, n_warmup, n_iters)

    _row("CheckersEnv.get_board_state()      [Python loop + Piece]", t_cv)
    _row("NumpyCheckersEnv.get_board_state() [numpy vectorized]", t_np,
         f"{t_cv/t_np:.1f}x faster")

    return t_cv, t_np


# ─────────────────────────────────────────────────────────────────────────────
# 6. Action mask computation
# ─────────────────────────────────────────────────────────────────────────────

def bench_action_mask(n_warmup, n_iters):
    _banner("6. Action Mask Computation: CheckersEnv vs NumpyCheckersEnv")
    print("  Rebuilt after every env.step() — called on every MCTS node.\n")

    env = CheckersEnv()
    _advance_game(env, moves=25)
    numpy_env = NumpyCheckersEnv.from_env(env)

    # CheckersEnv: iterates Piece objects, calls board.valid_moves_for_piece()
    t_cv = _timeit(env._update_action_mask, n_warmup, n_iters)
    # NumpyCheckersEnv: reads int8 cells directly, no object attribute access
    t_np = _timeit(numpy_env._update_action_mask, n_warmup, n_iters)

    _row("CheckersEnv._update_action_mask()      [Piece objects]", t_cv)
    _row("NumpyCheckersEnv._update_action_mask() [int8 array]", t_np,
         f"{t_cv/t_np:.1f}x faster")

    return t_cv, t_np


# ─────────────────────────────────────────────────────────────────────────────
# 7. CPU→GPU transfer: plain vs pinned memory (CUDA only)
# ─────────────────────────────────────────────────────────────────────────────

def bench_pinned_memory(n_warmup, n_iters, batch_size=16):
    if not torch.cuda.is_available():
        print("\n=== 7. Pinned Memory Transfer === SKIPPED (no CUDA)\n")
        return None, None

    _banner(f"7. CPU->GPU Transfer: plain vs pinned memory (batch={batch_size})")
    print("  agent.py uses pin_memory().to(device, non_blocking=True) for each")
    print("  numpy array passed to the PPO update loop.\n")

    device = torch.device("cuda")
    arr    = np.random.randn(batch_size, 4, 8, 8).astype(np.float32)

    # Plain: synchronous H2D copy through pageable memory
    def plain():
        torch.from_numpy(arr).to(device)
        torch.cuda.synchronize()

    # Pinned: page-lock the host buffer so DMA can transfer without a staging copy
    # non_blocking=True lets the copy overlap with CPU work in the training loop
    def pinned():
        torch.from_numpy(arr).pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize()

    t_plain  = _timeit(plain,  n_warmup, n_iters)
    t_pinned = _timeit(pinned, n_warmup, n_iters)

    _row("torch.from_numpy(arr).to(device)                   ", t_plain)
    _row("torch.from_numpy(arr).pin_memory().to(device, nb)  ", t_pinned,
         f"{t_plain/t_pinned:.2f}x")
    print("\n  NOTE: the pin_memory() call itself has a one-time allocation cost.")
    print("  agent.py re-allocates per batch — the real gain comes from the")
    print("  non_blocking flag enabling CPU/GPU overlap, not raw throughput.")

    return t_plain, t_pinned


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark checkersRL optimizations")
    parser.add_argument("--quick", action="store_true",
                        help="Run 20%% of iterations for a fast sanity check")
    args = parser.parse_args()

    scale = 0.2 if args.quick else 1.0
    def sc(n, minimum=10):
        return max(minimum, int(n * scale))

    print("\n" + "="*62)
    print("  checkersRL optimization benchmarks")
    print("="*62)
    print(f"  Mode : {'quick (20% iterations)' if args.quick else 'full'}")
    print(f"  CUDA : {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" — {torch.cuda.get_device_name(0)}", end="")
    print()

    results = {}
    results["clone"]     = bench_clones(sc(50), sc(2_000))
    results["ipc"]       = bench_ipc(sc(200), sc(5_000))
    results["gae"]       = bench_gae(sc(100), sc(2_000))
    results["inference"] = bench_inference(sc(50), sc(500))
    results["obs"]       = bench_board_state(sc(200), sc(5_000))
    results["mask"]      = bench_action_mask(sc(200), sc(5_000))
    results["gpu_xfer"]  = bench_pinned_memory(sc(100), sc(2_000))

    t_deepcopy, t_numpy_clone = results["clone"]
    t_pickle,   t_shm         = results["ipc"]
    t_gae_py,   t_gae_jit     = results["gae"]
    infer                      = results["inference"]
    t_obs_cv,   t_obs_np      = results["obs"]
    t_mask_cv,  t_mask_np     = results["mask"]

    _banner("SUMMARY — Speedups Over Naive Baseline")
    print(f"  {'Optimization':<50s}  {'Speedup':>8}")
    print(f"  {'-'*50}  {'-'*8}")
    print(f"  {'NumpyEnv.fast_clone  vs copy.deepcopy':<50s}  {t_deepcopy/t_numpy_clone:>7.1f}x")
    print(f"  {'Shared memory write  vs pickle round-trip':<50s}  {t_pickle/t_shm:>7.1f}x")
    print(f"  {'torch.jit.script GAE vs pure Python loop':<50s}  {t_gae_py/t_gae_jit:>7.1f}x")
    if 16 in infer and 1 in infer:
        serial  = 16 * infer[1][0]
        batched = infer[16][0]
        print(f"  {'Batch-16 inference   vs 16x batch-1':<50s}  {serial/batched:>7.1f}x")
    print(f"  {'NumpyEnv.get_board_state vs CheckersEnv':<50s}  {t_obs_cv/t_obs_np:>7.1f}x")
    print(f"  {'NumpyEnv._update_action_mask vs CheckersEnv':<50s}  {t_mask_cv/t_mask_np:>7.1f}x")
    print()


if __name__ == "__main__":
    main()
