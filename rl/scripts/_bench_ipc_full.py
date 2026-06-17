"""Measure full queue round-trip latency (serialization + OS pipe) for both IPC paths."""
import time
import numpy as np
import pickle
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

NUM_ACTIONS = 170
N_WARMUP    = 100
N_ITERS     = 2000

RESPONSE = (5, -0.693, 0.5, 0.12)   # (action, log_prob, entropy, value) — same either way


def echo_worker(req_q, resp_q, n):
    for _ in range(n):
        req_q.get()
        resp_q.put(RESPONSE)


if __name__ == "__main__":
    state     = np.random.randn(4, 8, 8).astype(np.float32)
    mask      = np.zeros(NUM_ACTIONS, dtype=np.float32); mask[:10] = 1.0
    worker_id = 3
    total     = N_WARMUP + N_ITERS

    # ── 1. Serialization ONLY — no pipe ──────────────────────────────────────
    for _ in range(N_WARMUP):
        d = pickle.dumps((worker_id, state, mask)); pickle.loads(d)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        d = pickle.dumps((worker_id, state, mask)); pickle.loads(d)
    t_ser = (time.perf_counter() - t0) / N_ITERS * 1000
    payload_bytes = len(pickle.dumps((worker_id, state, mask)))
    print(f"Serialization only (no pipe):        {t_ser:.4f} ms  ({payload_bytes} bytes)")

    # ── 2. Full round-trip: large pickle payload through Queue pipe ───────────
    req_q, resp_q = Queue(), Queue()
    p = Process(target=echo_worker, args=(req_q, resp_q, total))
    p.start()
    for _ in range(N_WARMUP):
        req_q.put((worker_id, state, mask)); resp_q.get()
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        req_q.put((worker_id, state, mask))
        resp_q.get()
        times.append(time.perf_counter() - t0)
    p.join(timeout=15)
    t_pickle = float(np.median(times)) * 1000
    print(f"Full round-trip, pickle ({payload_bytes} bytes on pipe): {t_pickle:.4f} ms")

    # ── 3. Full round-trip: shared memory write + int on pipe ────────────────
    num_workers = 16
    state_shm = SharedMemory(create=True, size=num_workers * 4 * 8 * 8 * 4)
    mask_shm  = SharedMemory(create=True, size=num_workers * NUM_ACTIONS * 4)
    state_buf = np.ndarray((num_workers, 4, 8, 8),   dtype=np.float32, buffer=state_shm.buf)
    mask_buf  = np.ndarray((num_workers, NUM_ACTIONS), dtype=np.float32, buffer=mask_shm.buf)

    req_q2, resp_q2 = Queue(), Queue()
    p2 = Process(target=echo_worker, args=(req_q2, resp_q2, total))
    p2.start()
    for _ in range(N_WARMUP):
        state_buf[worker_id][:] = state
        mask_buf[worker_id][:]  = mask
        req_q2.put(worker_id); resp_q2.get()
    times2 = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        state_buf[worker_id][:] = state    # worker writes into shared memory
        mask_buf[worker_id][:]  = mask
        req_q2.put(worker_id)              # sends just an int (~8 bytes)
        resp_q2.get()
        times2.append(time.perf_counter() - t0)
    p2.join(timeout=15)
    t_shm = float(np.median(times2)) * 1000

    try:
        state_shm.close(); state_shm.unlink()
        mask_shm.close();  mask_shm.unlink()
    except Exception:
        pass

    print(f"Full round-trip, shm + int (~8 bytes on pipe):  {t_shm:.4f} ms")
    print()

    saving_per_call   = t_pickle - t_shm
    agent_turns       = 250_000
    total_saving_s    = saving_per_call * agent_turns / 1000
    pipe_overhead     = t_pickle - t_ser
    ser_fraction      = t_ser / t_pickle * 100

    print(f"Speedup (full round-trip):               {t_pickle/t_shm:.2f}x")
    print(f"Per-call saving:                         {saving_per_call:.4f} ms")
    print(f"Projected saving @ 250k turns/epoch:     {total_saving_s:.1f} s  ({total_saving_s/60:.2f} min)")
    print()
    print(f"--- Breakdown of pickle path ---")
    print(f"  Serialization:   {t_ser:.4f} ms  ({ser_fraction:.1f}% of total round-trip)")
    print(f"  Pipe overhead:   {pipe_overhead:.4f} ms  ({100-ser_fraction:.1f}% of total round-trip)")
    print()
    print(f"Our earlier benchmark captured only {ser_fraction:.1f}% of the actual cost.")
    total_mb = payload_bytes * agent_turns / 1_000_000
    print(f"Pipe traffic: {total_mb:.0f} MB/epoch (pickle) -> {4*agent_turns/1e6:.1f} MB/epoch (shm int) = {total_mb/(4*agent_turns/1e6):.0f}x reduction")
