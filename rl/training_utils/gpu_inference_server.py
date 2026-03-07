"""Shared GPU inference server infrastructure for PPO and AlphaZero training.

Provides abstract BatchedGPUServer base class that batches inference requests
from CPU workers and runs them on GPU. Subclasses implement the forward pass:
  - PPOInferenceServer: policy sampling → (action, log_prob, entropy, value)
  - AlphaZeroInferenceServer: raw network output → (logits_np, value)
"""

import threading
import time
from abc import ABC, abstractmethod
from queue import Empty as QueueEmpty

import numpy as np
import torch
from torch.distributions import Categorical

# Sentinel for shutting down the server thread
SHUTDOWN = "SHUTDOWN"


class BatchedGPUServer(threading.Thread, ABC):
    """Abstract base for batching inference requests from workers onto GPU.

    Handles: queue polling, batch collection, shared-memory vs classic protocol,
    and dispatching results to per-worker response queues.
    Subclasses implement _forward_batch() for the actual model forward pass.
    """

    def __init__(self, model, device, request_queue, response_queues,
                 stop_event, max_batch=64, max_wait_ms=3.0,
                 state_buf=None, mask_buf=None):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.stop_event = stop_event
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self._state_buf = state_buf
        self._mask_buf = mask_buf

    def run(self):
        self.model.eval()
        req_q = self.request_queue

        while not self.stop_event.is_set():
            batch = []

            # Block until at least one request arrives
            try:
                item = req_q.get(timeout=0.01)
                if item == SHUTDOWN:
                    break
                batch.append(item)
            except QueueEmpty:
                continue

            # Drain additional pending requests up to max_batch
            effective_max = max(8, min(req_q.qsize() + len(batch) + 1, self.max_batch))
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < effective_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = req_q.get_nowait()
                    if item == SHUTDOWN:
                        req_q.put(SHUTDOWN)
                        break
                    batch.append(item)
                except QueueEmpty:
                    if remaining > 0.0005:
                        time.sleep(0.0003)
                    else:
                        break

            if not batch:
                continue

            # Extract worker_ids, states, optionally masks
            if self._state_buf is not None:
                worker_ids = list(batch)
                states_np = self._state_buf[worker_ids].copy()
                masks_np = self._mask_buf[worker_ids].copy() if self._mask_buf is not None else None
            else:
                worker_ids = [b[0] for b in batch]
                states_np = np.stack([b[1] for b in batch])
                masks_np = np.stack([b[2] for b in batch]) if len(batch[0]) > 2 else None

            states_t = torch.from_numpy(states_np).to(self.device)
            masks_np_for_subclass = masks_np

            # Subclass does the forward pass and returns responses (one per worker, same order)
            responses = self._forward_batch(states_t, worker_ids, masks_np_for_subclass)

            for i, wid in enumerate(worker_ids):
                self.response_queues[wid].put(responses[i])

    @abstractmethod
    def _forward_batch(self, states_t, worker_ids, masks_np=None):
        """Run model forward pass for the batch. Return list of responses, one per worker.

        Args:
            states_t: (B, C, H, W) tensor on device
            worker_ids: list of worker ints
            masks_np: (B, A) optional action masks; None if not used

        Returns:
            List of response tuples, one per worker (order matches worker_ids).
        """
        pass

    def _load_weights(self, state_dict):
        """Override in subclasses if custom load logic needed (e.g. torch.compile)."""
        # We need to gracefully handle models wrapped in `torch.compile` which adds an `_orig_mod.` prefix
        # and our unified DualHeadResNet which nests layers under `net.`
        
        # Check if the checkpoint has nested 'net.' keys while the model does not (or vice versa)
        model_has_net = any(k.startswith('net.') for k in self.model.state_dict().keys())
        ckpt_has_net = any(k.startswith('net.') for k in state_dict.keys())
        
        if model_has_net and not ckpt_has_net:
            state_dict = {f"net.{k}": v for k, v in state_dict.items()}
        elif ckpt_has_net and not model_has_net:
            state_dict = {k.replace('net.', '', 1): v for k, v in state_dict.items() if k.startswith('net.')}
            
        getattr(self.model, '_orig_mod', self.model).load_state_dict(state_dict)

    def update_weights(self, state_dict):
        """Hot-reload model weights (called from main thread between epochs)."""
        self._load_weights(state_dict)
        self.model.eval()


class PPOInferenceServer(BatchedGPUServer):
    """PPO policy sampling: returns (action, log_prob, entropy, value) per worker."""
    
    def __init__(self, *args, deterministic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic

    def _forward_batch(self, states_t, worker_ids, masks_np=None):
        with torch.no_grad():
            logits, values_t = self.model(states_t)

        if torch.isnan(logits).any():
            # NaN guard: send random valid actions
            responses = []
            for i, wid in enumerate(worker_ids):
                mask = masks_np[i] if masks_np is not None else None
                if mask is not None:
                    valid = np.where(mask > 0)[0]
                    action = int(np.random.choice(valid)) if len(valid) > 0 else 0
                    n = max(int(mask.sum()), 1)
                    log_prob = float(np.log(1.0 / n))
                else:
                    action, log_prob = 0, 0.0
                responses.append((action, log_prob, 0.0, 0.0))
            return responses

        masks_t = torch.from_numpy(masks_np).to(self.device) if masks_np is not None else None
        if masks_t is not None:
            masked_logits = logits + torch.where(
                masks_t > 0,
                torch.zeros_like(logits),
                torch.full_like(logits, -1e10),
            )
        else:
            masked_logits = logits

        probs = Categorical(logits=masked_logits)
        if self.deterministic:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            actions = probs.sample()
            
        log_probs = probs.log_prob(actions)
        entropies = probs.entropy()

        actions_cpu = actions.cpu().numpy()
        log_probs_cpu = log_probs.cpu().numpy()
        entropies_cpu = entropies.cpu().numpy()
        values_cpu = values_t.squeeze(-1).cpu().numpy()

        return [
            (int(actions_cpu[i]), float(log_probs_cpu[i]),
             float(entropies_cpu[i]), float(values_cpu[i]))
            for i in range(len(worker_ids))
        ]

    def _load_weights(self, state_dict):
        super()._load_weights(state_dict)


class AlphaZeroInferenceServer(BatchedGPUServer):
    """AlphaZero leaf evaluation: returns (logits_np, value_float) per worker."""

    def _forward_batch(self, states_t, worker_ids, masks_np=None):
        with torch.no_grad():
            logits_t, values_t = self.model(states_t)

        logits_np = logits_t.cpu().numpy()
        values_np = values_t.squeeze(-1).cpu().numpy()

        return [(logits_np[i], float(values_np[i])) for i in range(len(worker_ids))]
