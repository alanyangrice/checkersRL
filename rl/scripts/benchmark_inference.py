"""Benchmark: CPU vs GPU inference latency and multiprocessing.Queue overhead.

Run from repo root:
    python -m rl.algorithms.ppo.benchmark_inference
"""

import time
import numpy as np
import torch
from multiprocessing import Queue, Process
from torch.distributions import Categorical

from rl.networks.policy_network import PPOPolicyNetwork
from rl.envs.checkers_env import CheckersEnv
from checkers_game.constants import NUM_ACTIONS


def benchmark_inference(device_name, model, batch_sizes, n_warmup=50, n_iters=500):
    """Measure forward-pass latency for various batch sizes."""
    try:
        device = model.net.policy_fc.weight.device
    except AttributeError:
        device = model.policy_fc.weight.device
    results = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, 4, 8, 8, device=device)
        # warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iters * 1000  # ms

        per_sample = elapsed / bs
        results[bs] = (elapsed, per_sample)
        print(f"  {device_name} batch={bs:>3d}: {elapsed:7.3f} ms total, {per_sample:7.4f} ms/sample")

    return results


def benchmark_env_step(n_steps=2000):
    """Measure pure game-logic cost (env.step with random actions)."""
    env = CheckersEnv()
    env.reset()
    times = []
    steps = 0

    while steps < n_steps:
        mask = env.get_action_mask()
        if mask.sum() == 0:
            env.reset()
            continue
        action = int(np.random.choice(np.where(mask > 0)[0]))
        t0 = time.perf_counter()
        _, _, done, _, _ = env.step(action)
        times.append(time.perf_counter() - t0)
        steps += 1
        if done:
            env.reset()

    arr = np.array(times) * 1000
    print(f"  env.step(): mean={arr.mean():.3f} ms, median={np.median(arr):.3f} ms, "
          f"p95={np.percentile(arr, 95):.3f} ms  (n={n_steps})")
    return arr.mean()


def queue_echo_worker(req_q, resp_q, n):
    """Worker that echoes items back through a response queue."""
    for _ in range(n):
        item = req_q.get()
        resp_q.put(item)


def benchmark_queue_roundtrip(n_warmup=50, n_iters=2000):
    """Measure round-trip latency of multiprocessing.Queue (main → worker → main)."""
    req_q = Queue()
    resp_q = Queue()
    payload = np.random.randn(4, 8, 8).astype(np.float32)  # same size as a state

    total = n_warmup + n_iters
    p = Process(target=queue_echo_worker, args=(req_q, resp_q, total))
    p.start()

    # warmup
    for _ in range(n_warmup):
        req_q.put(payload)
        resp_q.get()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        req_q.put(payload)
        resp_q.get()
        times.append(time.perf_counter() - t0)

    p.join(timeout=5)
    arr = np.array(times) * 1000
    print(f"  Queue round-trip: mean={arr.mean():.3f} ms, median={np.median(arr):.3f} ms, "
          f"p95={np.percentile(arr, 95):.3f} ms  (n={n_iters})")
    return arr.mean()


def benchmark_full_select_action(device_name, model, device, n_iters=1000):
    """Measure full select_action equivalent: tensor creation + forward + sample."""
    state = np.random.randn(4, 8, 8).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[:10] = 1.0  # 10 valid actions

    # warmup
    for _ in range(50):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        m = torch.FloatTensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(s)
        masked = logits + torch.where(m > 0, 0.0, torch.tensor(-1e10, device=device))
        probs = Categorical(logits=masked)
        probs.sample()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        m = torch.FloatTensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(s)
        masked = logits + torch.where(m > 0, 0.0, torch.tensor(-1e10, device=device))
        probs = Categorical(logits=masked)
        probs.sample()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  {device_name} select_action (batch=1): {elapsed:.3f} ms/call")
    return elapsed


if __name__ == "__main__":
    input_shape = (4, 8, 8)
    n_actions = NUM_ACTIONS

    print(f"Model: PPOPolicyNetwork, NUM_ACTIONS={n_actions}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── 1. CPU inference ──────────────────────────────────────────────
    print("=== CPU Inference ===")
    cpu_model = PPOPolicyNetwork(input_shape, n_actions).to("cpu")
    cpu_model.eval()
    cpu_results = benchmark_inference("CPU", cpu_model, [1, 4, 8, 16, 32])
    print()

    # ── 2. GPU inference ──────────────────────────────────────────────
    if torch.cuda.is_available():
        print("=== GPU Inference ===")
        gpu_model = PPOPolicyNetwork(input_shape, n_actions).to("cuda")
        gpu_model.eval()
        gpu_results = benchmark_inference("GPU", gpu_model, [1, 4, 8, 16, 32, 64])
        print()
    else:
        print("=== GPU Inference === SKIPPED (no CUDA)\n")
        gpu_results = {}

    # ── 3. Full select_action timing ──────────────────────────────────
    print("=== Full select_action (tensor build + forward + sample) ===")
    cpu_sa = benchmark_full_select_action("CPU", cpu_model, torch.device("cpu"))
    if torch.cuda.is_available():
        gpu_sa = benchmark_full_select_action("GPU", gpu_model, torch.device("cuda"))
    print()

    # ── 4. Game env step ──────────────────────────────────────────────
    print("=== Game Environment Step ===")
    env_ms = benchmark_env_step(2000)
    print()

    # ── 5. Queue round-trip ───────────────────────────────────────────
    print("=== Multiprocessing Queue Round-trip ===")
    queue_ms = benchmark_queue_roundtrip(2000)
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    cpu_b1 = cpu_results[1][0]
    print(f"  CPU select_action (batch=1):    {cpu_sa:.3f} ms")
    if torch.cuda.is_available():
        gpu_b16 = gpu_results[16][1]
        print(f"  GPU inference (batch=16, /sample): {gpu_b16:.4f} ms")
        print(f"  GPU select_action (batch=1):    {gpu_sa:.3f} ms")
    print(f"  Game env.step():                {env_ms:.3f} ms")
    print(f"  Queue round-trip:               {queue_ms:.3f} ms")
    print()

    if torch.cuda.is_available():
        # Projected per-step time with GPU server
        gpu_amortized = gpu_results[16][1] + queue_ms  # per-sample GPU + queue overhead
        current = cpu_sa + env_ms
        projected = gpu_amortized + env_ms
        speedup = current / projected
        print(f"  Current per-step (CPU infer + env):  {current:.3f} ms")
        print(f"  Projected per-step (GPU batch + Q):  {projected:.3f} ms")
        print(f"  Projected speedup:                   {speedup:.2f}x")
    print()
