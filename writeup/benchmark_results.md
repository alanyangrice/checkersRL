# Infrastructure Benchmark Results

**Hardware:** NVIDIA GeForce RTX 5090 | Windows 11 | Python 3.14  
**Script:** `rl/scripts/benchmark_optimizations.py`  
**Reference writeup:** `writeup/07_infrastructure.md`

---

## Context

The infrastructure writeup (`07_infrastructure.md`) documents the evolution from a naive sequential training loop (~12 min/epoch) to a GPU-accelerated parallel architecture (~4.5 min/epoch — a **2.65x** speedup). The benchmarks below measure the specific micro-optimizations that drove that improvement and, importantly, expose several results that are **counterintuitive** or contradict the original narrative.

---

## 1. Environment Clone

> *Relevant section: 07_infrastructure.md §6.1 — "Model inference dominated execution time"*

MCTS calls `fast_clone()` once per simulation — 75–400 times per move. The clone method is one of the hottest paths in the entire codebase.

| Method | Time (ms) | Speedup vs deepcopy |
|---|---|---|
| `copy.deepcopy(CheckersEnv)` | 0.6919 ms | 1× (baseline) |
| `CheckersEnv.fast_clone()` | 0.0043 ms | **161×** |
| `NumpyCheckersEnv.fast_clone()` | 0.0007 ms | **1,018×** |
| `NumpyCheckersEnv.from_env()` *(one-time)* | 0.0898 ms | — |

**Why deepcopy is slow:** `copy.deepcopy(CheckersEnv)` traverses the full `Game` object graph — 12 `Piece` objects with Python attributes, an `8×8` list-of-lists board, a `board_states` dict accumulating ~25 repetition-tracking entries by mid-game, and a `moves` list. Python's generic copier pays per-object overhead on every node.

**Why `fast_clone()` is 161×:** The hand-rolled `CheckersEnv.fast_clone()` skips `board_states` and `moves` entirely (irrelevant for short MCTS simulations). It still copies individual `Piece` objects via `Piece.clone()`.

**Why `NumpyCheckersEnv.fast_clone()` is 1,018×:** The numpy env represents the board as a flat `(8, 8) int8` array — 64 bytes. `fast_clone()` is a single `ndarray.copy()` (essentially `memcpy`) plus a handful of scalar assignments. No Python object graph traversal at all. `_base_counts` (the seeded repetition history) is shared by reference since it is never written after construction.

**Projected cost for 100 MCTS simulations (per search call):**

| Approach | Cost |
|---|---|
| Naive: 100× `deepcopy` | 69.19 ms |
| Numpy: 1× `from_env` + 100× `fast_clone` | **0.16 ms** |
| **Speedup** | **438×** |

**Interview note:** This is the single largest optimization in the codebase. The one-time `from_env()` conversion (0.09 ms) is negligible amortized across 100 simulations.

---

## 2. IPC Serialization: Pickle vs Shared Memory

> *Relevant section: 07_infrastructure.md §6.3 — "Zero-Copy Shared Memory IPC"*

`multiprocessing.Queue` uses `pickle` internally on every `put()` and `get()`. This benchmark measures serialization cost only — not OS pipe latency.

| Method | Time (ms) | Speedup |
|---|---|---|
| `pickle.dumps` + `pickle.loads` (1,864 bytes) | 0.0064 ms | 1× (baseline) |
| Shared memory write + numpy slice read | 0.0029 ms | **2.2×** |

**Why pickle is slow:** The 1,864-byte payload is already small (1,024 bytes of state + 680 bytes of mask). Even so, pickle must: traverse Python's `__reduce__` protocol, serialize numpy array metadata + raw bytes, write to a pipe, and deserialize on the other side — duplicating all data in memory.

**Why shared memory is faster:** Both the main process (GPU server thread) and child processes (workers) map the same physical memory pages via `multiprocessing.shared_memory`. The worker does `state_buf[worker_id][:] = state` — a direct write, no serialization. The server reads `state_buf[worker_ids].copy()` — a single numpy slice plus one copy to make the batch contiguous. Only the integer `worker_id` travels over the queue.

**Projected per-epoch IPC cost** (5,000 games × ~50 agent turns = 250,000 calls):

| Path | Cost/epoch |
|---|---|
| Pickle | 1.6 s |
| Shared memory | 0.7 s |
| **Saved** | **0.9 s/epoch** |

**Important caveat:** This benchmark does not include OS pipe latency, which the existing `benchmark_inference.py:benchmark_queue_roundtrip()` measured at ~0.3–0.5 ms round-trip (dominating the serialization cost). The shared memory path eliminates the large payload from the pipe — only an `int` travels — which also reduces pipe latency.

---

## 3. GAE Computation: `torch.jit.script` vs Pure Python

> *Relevant section: 07_infrastructure.md §6.5 — "TorchScript GAE: 3–5 seconds → negligible"*

**⚠️ Surprising result: JIT is significantly slower on this hardware.**

| Method | Time (ms, n=5,000) | Speedup |
|---|---|---|
| Pure Python backward scan | 0.5418 ms | 1× (baseline) |
| `torch.jit.script` | 38.5808 ms | **0.014× (71× slower)** |

**Why JIT underperforms here:** The `compute_gae` function is a scalar-indexed loop: `gae = deltas[t] + gamma * gae_lambda * not_done[t] * gae`. In TorchScript, `deltas[t]` returns a 0-dimensional `torch.Tensor`, not a Python float. Each of the 5,000 iterations dispatches through PyTorch's operator dispatch system — creating and destroying scalar tensors on every step. The overhead is multiplicative, not amortized.

The pure Python version accesses numpy arrays directly, producing Python `float` scalars per iteration. For this sequential scalar pattern, the Python interpreter + numpy is faster than PyTorch's dispatch.

**Why the writeup claims 3–5 seconds:** Likely measured under different conditions: a much larger rollout (e.g., 400k transitions from league training), or an older PyTorch/hardware combination where the JIT kernel launch overhead was different. The JIT function also incurs a one-time compilation cost on first call that inflates early measurements.

**Practical takeaway:** For rollout lengths actually used in training (~250k transitions), the pure Python version takes ~27 ms and JIT takes ~1.9 seconds — so the writeup's direction (JIT is better) appears inverted on this hardware. This is worth investigating with profiling before claiming JIT as an optimization.

---

## 4. GPU Inference Batch Scaling

> *Relevant section: 07_infrastructure.md §6.1 — "batch of 16: 0.96 ms GPU vs 7.30 ms CPU (29× speedup)"*

**Device:** NVIDIA GeForce RTX 5090

| Batch | ms/forward | ms/sample | Samples/sec |
|---|---|---|---|
| 1 | 0.993 ms | 1.0998 ms | 909/s |
| 2 | 0.871 ms | 0.4357 ms | 2,295/s |
| 4 | 0.845 ms | 0.2112 ms | 4,734/s |
| 8 | 0.829 ms | 0.1036 ms | 9,649/s |
| 16 | 0.965 ms | 0.0603 ms | 16,587/s |
| 32 | 0.953 ms | 0.0298 ms | 33,566/s |

**GPU inference server speedup:**

| Strategy | Cost |
|---|---|
| 16 workers, serial (16× batch-1) | 15.89 ms |
| 16 workers, batched (1× batch-16) | **0.97 ms** |
| **Speedup** | **16.5×** |

**Key observation:** Forward pass time is essentially **flat** across batch sizes 1–32 (0.83–1.10 ms). The RTX 5090's tensor cores are so fast relative to kernel launch overhead that batch size barely affects total forward pass time — but it massively affects per-sample cost. This is the non-linear scaling the writeup describes: going from batch-1 to batch-16 is essentially "free" from a GPU perspective.

**Confirmed:** The writeup's claim of ~1 ms for a batch-16 GPU forward pass is accurate (we measure 0.97 ms). The CPU baseline (batch-1) is 0.99 ms here, but CPU scales linearly — 16× serial would be ~15.9 ms.

---

## 5. Board-State Observation Generation

> *Relevant section: 07_infrastructure.md §6.1 — NumpyCheckersEnv for fast MCTS*

**⚠️ Counterintuitive result: NumpyCheckersEnv is marginally *slower*.**

| Method | Time (ms) | Relative |
|---|---|---|
| `CheckersEnv.get_board_state()` — Python loop + Piece objects | 0.0067 ms | 1× (baseline) |
| `NumpyCheckersEnv.get_board_state()` — numpy vectorized | 0.0072 ms | **0.9× (7% slower)** |

**Why:** The numpy version computes 4 full-board comparisons (`b == BLUE_PIECE`, etc.) over a `(8,8)` array, creating 4 intermediate boolean arrays. `np.flip()` is called for RED's perspective. For a sparse checkers board (at most 24 pieces across 32 active squares), the Python loop with early-continue logic is competitive because it avoids allocating temporary arrays. At 8×8 = 64 cells, numpy's vectorization overhead is not yet amortized.

**Why the NumpyEnv clone speedup still dominates:** `get_board_state()` takes ~7 µs either way. `fast_clone()` saves ~690 µs per call. Across 100 MCTS simulations, the observation generation adds ~0.7 ms regardless of env type — invisible against the 69 ms deepcopy baseline.

---

## 6. Action Mask Computation

| Method | Time (ms) | Relative |
|---|---|---|
| `CheckersEnv._update_action_mask()` — Piece object iteration | 0.0095 ms | 1× (baseline) |
| `NumpyCheckersEnv._update_action_mask()` — int8 array iteration | 0.0120 ms | **0.8× (26% slower)** |

**Why NumpyEnv is slower here:** Both implementations iterate the board with a Python `for` loop to find pieces and compute valid moves — the bottleneck is the move-generation logic (`_valid_moves_for_piece`), not the board representation. The numpy version adds overhead from converting `int` cells back to Python comparisons (`int(cell) in _BLUE_CELLS`) rather than direct attribute access (`piece.color == BLUE`). The data structure advantage only pays off in `fast_clone()`.

**Takeaway:** The NumpyCheckersEnv's performance advantage is *entirely* in the clone operation, not in step/observe/mask. This is by design — the class doc explicitly says "MCTS converts once per search via `from_env()`, then clones the numpy env for each simulation."

---

## 7. CPU→GPU Transfer: Pinned Memory

| Method | Time (ms, batch=16) | Relative |
|---|---|---|
| `torch.from_numpy(arr).to(device)` | 0.0146 ms | 1× (baseline) |
| `.pin_memory().to(device, non_blocking=True)` | 0.0195 ms | **0.75× (33% slower in isolation)** |

**Why it's slower in isolation:** `pin_memory()` page-locks host memory, which requires a separate allocation and memory registration syscall. Measured in isolation with an immediate `cuda.synchronize()`, the overhead dominates.

**Why it's still used:** The benefit is not raw transfer speed — it's **CPU/GPU overlap**. With `non_blocking=True`, the CUDA transfer runs asynchronously while the CPU continues executing. In the PPO update loop (`agent.py`), multiple arrays are transferred back-to-back (`states`, `actions`, `rewards`, `log_probs`, `masks`, `values`). `non_blocking` lets these overlap, hiding latency behind computation. The synchronization cost is paid once at the end (`torch.cuda.synchronize()` on line 113), not per array.

---

## Summary Table

| Optimization | Measured Speedup | Expected? |
|---|---|---|
| `NumpyEnv.fast_clone` vs `deepcopy` | **1,018×** | ✅ Yes |
| GPU batch-16 vs 16× batch-1 | **16.5×** | ✅ Yes |
| Shared memory vs pickle IPC | **2.2×** | ✅ Yes (serialization cost only) |
| `torch.jit.script` GAE vs Python | **0.014× (71× slower)** | ❌ Unexpected |
| `NumpyEnv.get_board_state` vs CheckersEnv | **0.9× (slightly slower)** | ❌ Unexpected |
| `NumpyEnv._update_action_mask` vs CheckersEnv | **0.8× (slightly slower)** | ❌ Unexpected |
| Pinned memory transfer vs plain | **0.75× (slower in isolation)** | ⚠️ Context-dependent |

---

## Interview Notes: Where to Focus

**Defend confidently (confirmed by benchmarks):**
- The 1,018× clone speedup is the killer number for NumpyCheckersEnv — interview-ready
- GPU batch scaling is flat on the RTX 5090: batch-1 ≈ batch-32 in total time, so the server's 16.5× speedup is essentially free
- The writeup's batch-16 GPU timing (~1 ms) is confirmed accurate

**Be honest about (counterintuitive results):**
- JIT GAE is slower on this hardware — acknowledge that `torch.jit.script` helps when Python interpreter overhead dominates, but for scalar tensor indexing loops it adds dispatch overhead. The observed benefit was likely measured on a different system.
- NumpyEnv's obs/mask methods are not faster — the speedup is localized to `fast_clone()` only, which is exactly what the class was designed for
- Pinned memory's benefit is async overlap in a multi-transfer loop, not isolated transfer speed

**The real win stack (additive reasoning):**
1. NumpyEnv clone → MCTS search goes from 69 ms/call to 0.16 ms/call
2. GPU batching → 16 workers cost 1 ms instead of 16 ms per inference round
3. Shared memory → removes ~0.9s/epoch of serialization overhead + reduces pipe pressure
4. These compound: the epoch wall-clock improvement (12 min → 4.5 min) comes from all three together
