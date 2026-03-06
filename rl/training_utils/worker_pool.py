"""Abstract BaseWorkerContext for managing worker pools and GPU inference servers."""

from abc import ABC, abstractmethod
import threading
from multiprocessing import Queue
from multiprocessing import shared_memory as _shm_module

import numpy as np

from rl.training_utils.parallel_utils import collect_results, shutdown_workers

class BaseWorkerContext(ABC):
    """Abstract base class for maintaining a persistent pool of CPU workers and GPU inference servers."""

    def __init__(self, device, num_workers, obs_shape=(4, 8, 8), num_actions=170):
        self.device = device
        self.num_workers = num_workers

        self._request_queue = Queue()
        self._response_queues = [Queue() for _ in range(num_workers)]
        self._results_queue = Queue()
        self._progress_queue = Queue()
        self._task_queue = Queue()
        self._stop_event = threading.Event()

        # Allocate shared memory
        _state_bytes = num_workers * int(np.prod(obs_shape)) * 4
        _mask_bytes = num_workers * num_actions * 4

        self._shm_state = _shm_module.SharedMemory(create=True, size=_state_bytes)
        self._shm_mask = _shm_module.SharedMemory(create=True, size=_mask_bytes)
        
        self._state_buf = np.ndarray(
            (num_workers, *obs_shape), dtype=np.float32, buffer=self._shm_state.buf
        )
        self._mask_buf = np.ndarray(
            (num_workers, num_actions), dtype=np.float32, buffer=self._shm_mask.buf
        )

        self._server = None
        self._workers = []

    def _distribute_tasks_and_collect(self, tasks, label, batch_done_sentinel=None, log_interval=None):
        if batch_done_sentinel is None:
            from rl.training_utils.parallel_utils import WORKER_BATCH_DONE
            batch_done_sentinel = WORKER_BATCH_DONE

        total = len(tasks)
        for i, task in enumerate(tasks):
            self._task_queue.put((i, task))

        for _ in range(self.num_workers):
            self._task_queue.put(batch_done_sentinel)

        if log_interval is None:
            log_interval = max(10, total // 10)

        return collect_results(
            self._results_queue,
            self.num_workers,
            self._workers,
            total,
            label,
            log_interval,
            batch_done_sentinel=batch_done_sentinel
        )

    def shutdown(self):
        """Terminate all workers and the inference server, release shared memory."""
        shutdown_workers(
            self._workers,
            self._task_queue,
            self._stop_event,
            self._request_queue,
            self._server,
            [self._shm_state, self._shm_mask]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
