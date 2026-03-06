"""Parallel infrastructure utilities."""

import numpy as np
from multiprocessing import shared_memory as _shm_module

WORKER_EXIT = None
WORKER_BATCH_DONE = "BATCH_DONE"

def attach_shm_buffers(shm_state_name, shm_mask_name, num_workers, obs_shape, num_actions):
    """Attach to existing shared memory blocks created by the main process."""
    _shm_state = _shm_mask = None
    _state_buf = _mask_buf = None
    
    if shm_state_name is not None:
        _shm_state = _shm_module.SharedMemory(name=shm_state_name)
        _state_buf = np.ndarray(
            (num_workers, *obs_shape), dtype=np.float32, buffer=_shm_state.buf
        )
    
    if shm_mask_name is not None:
        _shm_mask = _shm_module.SharedMemory(name=shm_mask_name)
        _mask_buf = np.ndarray(
            (num_workers, num_actions), dtype=np.float32, buffer=_shm_mask.buf
        )
        
    return _shm_state, _shm_mask, _state_buf, _mask_buf

def collect_results(results_queue, num_workers, workers, total, label, log_interval, batch_done_sentinel=WORKER_BATCH_DONE):
    """Drain the results queue until all expected results arrive."""
    result_map = {}
    reported = 0
    idle_workers = 0

    while idle_workers < num_workers or len(result_map) < total:
        try:
            item = results_queue.get(timeout=1.0)
            if item == batch_done_sentinel:
                idle_workers += 1
                continue
            task_idx, result = item
            result_map[task_idx] = result

            done = len(result_map)
            if done - reported >= log_interval or done == total:
                alive = sum(1 for p in workers if p.is_alive())
                print(f"  {label}: {done}/{total} games ({alive} workers active)", flush=True)
                reported = done
        except Exception:
            if not any(p.is_alive() for p in workers) and len(result_map) < total:
                raise RuntimeError(
                    f"[{label}] All workers died after {len(result_map)}/{total} games."
                )

    return [result_map[i] for i in range(total)]

def shutdown_workers(workers, task_queue, stop_event, request_queue, server, shm_blocks, exit_sentinel=None, shutdown_sentinel="SHUTDOWN"):
    """Gracefully terminate workers, stop inference server, and free shared memory."""
    for _ in workers:
        task_queue.put(exit_sentinel)
    for p in workers:
        p.join(timeout=15)

    if stop_event is not None:
        stop_event.set()
    if request_queue is not None:
        request_queue.put(shutdown_sentinel)
    if server is not None:
        server.join(timeout=5)

    for shm in shm_blocks:
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
