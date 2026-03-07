import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork
from RL_models.PPO_Model import training_config as cfg


def get_device():
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_compile_available():
    """Return True only when torch.compile can fully execute on CUDA.

    torch.compile with both mode='default' and mode='reduce-overhead' uses
    TorchInductor to generate optimized CUDA kernels, which requires Triton.
    Triton is Linux-only in standard PyTorch releases and is not available on
    Windows.  Attempting to compile without Triton raises TritonMissing on the
    first forward call, not at compile time, causing a silent crash deep in
    training.  This guard prevents that.
    """
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "compile"):
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def get_policy_state_dict(policy):
    """Return a plain state dict from a policy, stripping any torch.compile prefix.

    torch.compile wraps modules in an OptimizedModule whose state_dict() keys
    all start with '_orig_mod.'.  This helper always returns plain keys so that
    saved checkpoints, pool files, and cross-process weight transfers remain
    portable regardless of whether the module was compiled.
    """
    return getattr(policy, '_orig_mod', policy).state_dict()


def load_policy_state_dict(policy, state_dict):
    """Load a state dict into a policy, handling torch.compile transparently.

    Accepts both plain keys and '_orig_mod.*' prefixed keys (backward compat
    with checkpoints saved before torch.compile was added).  Always loads into
    the underlying (uncompiled) module so keys match regardless of compilation.
    """
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    getattr(policy, '_orig_mod', policy).load_state_dict(state_dict)


@torch.jit.script
def _compute_gae(deltas: torch.Tensor, not_done: torch.Tensor,
                 gamma: float, gae_lambda: float) -> torch.Tensor:
    """Compute Generalized Advantage Estimation via backward scan.

    Compiled to C++ via TorchScript — ~10-50x faster than the equivalent
    Python for-loop with .item() calls for large transition counts (400K+).
    """
    n = deltas.shape[0]
    advantages = torch.zeros(n)
    gae: float = 0.0
    for t in range(n - 1, -1, -1):
        gae = deltas[t] + gamma * gae_lambda * not_done[t] * gae
        advantages[t] = gae
    return advantages


def _to_gpu(arr, dtype, device):
    """Convert a numpy array to a pinned-memory tensor and transfer non-blocking."""
    return (torch.from_numpy(np.asarray(arr, dtype=dtype))
            .pin_memory()
            .to(device, non_blocking=True))


class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.95, eps_clip=0.2,
                 K_epochs=4, gae_lambda=0.95, device=None, augment=True,
                 augment_noise=0.05, mini_batch_size=2048, entropy_bonus=0.01):
        self.device = device or get_device()
        self.policy = PPOPolicyNetwork(input_shape, n_actions).to(self.device)

        # Fused Adam: single CUDA kernel per parameter group, ~2-5x faster optimizer step.
        # Falls back silently on CPU/MPS where fused is not supported.
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=lr,
            fused=(self.device.type == "cuda"),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.LR_SCHEDULER_T_MAX,
            eta_min=cfg.LR_SCHEDULER_ETA_MIN,
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.augment = augment
        self.augment_noise = augment_noise
        self.mini_batch_size = mini_batch_size
        self.entropy_bonus = entropy_bonus

        # Mixed precision: bfloat16 on CUDA (same exponent range as FP32,
        # no overflow risk, no GradScaler needed — uses tensor cores on Ampere+)
        self._use_amp = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self._amp_dtype = torch.bfloat16 if self._use_amp else torch.float32

        # torch.compile: fuses ops and replays CUDA graphs for the repeated
        # mini-batch forward/backward passes in update().
        # mode="reduce-overhead" is optimal for repeated same-shape calls.
        # Requires Triton (Linux only in standard PyTorch) — skipped on Windows.
        if _torch_compile_available():
            self.policy = torch.compile(self.policy, mode="reduce-overhead")

    def get_policy_state_dict(self):
        """Convenience wrapper — see module-level get_policy_state_dict()."""
        return get_policy_state_dict(self.policy)

    def select_action(self, state, action_mask):
        """Select an action using the policy network and a semantic action mask.

        Args:
            state: Board observation array (4, 8, 8).
            action_mask: numpy array of shape (NUM_ACTIONS,) where 1.0 = valid, 0.0 = invalid.

        Returns:
            (action_index, log_prob, entropy)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.policy(state_t)

        # Apply semantic mask: -inf for invalid actions, 0 for valid
        masked_logits = logits + torch.where(
            mask_t > 0,
            torch.zeros_like(logits),
            torch.full_like(logits, -1e10),
        )

        probs = Categorical(logits=masked_logits)
        action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        """PPO update with GAE, value function clipping, and optional DrAC augmentation.

        When augment=True, an additional noisy copy of all states is included
        in the PPO batch, forcing robustness to small observation perturbations.

        Value function clipping (PPO paper §3): stores the old value estimates
        from collection time and clips the value loss to prevent unbounded updates
        when the agent encounters new-distribution states after pool strength increases.
        """
        if len(memory.states) == 0:
            return

        # Switch to train mode for BatchNorm to use batch statistics during updates.
        # The InferenceServer calls model.eval() on its own separate copy; this
        # policy object stays in train mode for the duration of update().
        self.policy.train()

        # Pin memory + non-blocking H2D: queue all seven DMA transfers simultaneously
        # so the PCIe bus runs at full bandwidth. The synchronize() call below acts
        # as a single fence ensuring all transfers complete before the first forward pass.
        states        = _to_gpu(memory.states,       np.float32, self.device)
        actions       = _to_gpu(memory.actions,      np.int64,   self.device)
        rewards       = _to_gpu(memory.rewards,      np.float32, self.device)
        log_probs_old = _to_gpu(memory.log_probs,    np.float32, self.device)
        done_flags    = _to_gpu(memory.done,         np.bool_,   self.device)
        action_masks  = _to_gpu(memory.action_masks, np.float32, self.device)
        old_values    = _to_gpu(memory.values,       np.float32, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # fence: all seven non-blocking transfers complete

        # --- GAE on original data -----------------------------------------
        # Forward pass on GPU to get state values (chunked to avoid OOM)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            sv_chunks = []
            for i in range(0, states.size(0), self.mini_batch_size):
                _, sv = self.policy(states[i : i + self.mini_batch_size])
                sv_chunks.append(sv.squeeze(-1))
            state_values = torch.cat(sv_chunks, dim=0).float()  # ensure FP32 for GAE math

        # GAE scan on CPU using TorchScript-compiled loop (~10-50x faster
        # than the Python for-loop it replaces).
        sv_cpu = state_values.cpu()
        rw_cpu = rewards.cpu()
        df_cpu = done_flags.cpu()

        n = len(rw_cpu)
        next_values = torch.zeros(n)
        if n > 1:
            next_values[:-1] = sv_cpu[1:]
        next_values[df_cpu] = 0.0
        next_values[-1] = 0.0

        deltas = rw_cpu + self.gamma * next_values - sv_cpu
        not_done = (~df_cpu).float()

        advantages_cpu = _compute_gae(deltas, not_done, self.gamma, self.gae_lambda)

        # Move results back to GPU
        advantages = advantages_cpu.to(self.device)
        returns = (advantages + state_values)

        # --- Augment with noisy copies ----------------------------------
        if self.augment:
            noise = torch.randn_like(states) * self.augment_noise
            states_noisy = torch.clamp(states + noise, 0.0, 1.0)

            # Compute surrogate old log-probs for noisy states (chunked, AMP)
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                logit_chunks = []
                for i in range(0, states_noisy.size(0), self.mini_batch_size):
                    lg, _ = self.policy(states_noisy[i : i + self.mini_batch_size])
                    logit_chunks.append(lg)
                logits_noisy = torch.cat(logit_chunks, dim=0).float()

            # Skip augmentation if network produced NaN (early sign of instability)
            if torch.isnan(logits_noisy).any():
                print("Warning: NaN detected in augmentation logits, skipping augmentation this batch")
            else:
                logits_noisy = torch.clamp(logits_noisy, -50.0, 50.0)
                # Apply action masks so augmented log_probs match the masked distribution
                masked_logits_noisy = logits_noisy + torch.where(
                    action_masks > 0,
                    torch.zeros_like(logits_noisy),
                    torch.full_like(logits_noisy, -1e10),
                )
                probs_noisy = Categorical(logits=masked_logits_noisy)
                log_probs_old_noisy = probs_noisy.log_prob(actions)

                # Concatenate original + noisy (only when augmentation succeeded).
                # old_values must also be doubled so mini-batch indices into [0, 2N)
                # remain valid after torch.randperm(2N).
                states       = torch.cat([states,       states_noisy],          dim=0)
                actions      = torch.cat([actions,      actions.clone()],        dim=0)
                log_probs_old = torch.cat([log_probs_old, log_probs_old_noisy],  dim=0)
                advantages   = torch.cat([advantages,   advantages.clone()],     dim=0)
                returns      = torch.cat([returns,      returns.clone()],         dim=0)
                action_masks = torch.cat([action_masks, action_masks.clone()],   dim=0)
                old_values   = torch.cat([old_values,   old_values.clone()],     dim=0)

        # Normalize advantages over the full (augmented) batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.unsqueeze(-1)

        # --- PPO update for K epochs with mini-batches --------------------
        total = states.size(0)

        for _ in range(self.K_epochs):
            # Shuffle indices each epoch for stochastic mini-batches
            perm = torch.randperm(total, device=self.device)

            for mb_start in range(0, total, self.mini_batch_size):
                mb_idx = perm[mb_start : mb_start + self.mini_batch_size]
                mb_states        = states[mb_idx]
                mb_actions       = actions[mb_idx]
                mb_log_probs_old = log_probs_old[mb_idx]
                mb_advantages    = advantages[mb_idx]
                mb_returns       = returns[mb_idx]
                mb_masks         = action_masks[mb_idx]
                mb_old_values    = old_values[mb_idx]

                # Mixed-precision forward pass (BF16 on CUDA, FP32 elsewhere)
                with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                    logits, current_values = self.policy(mb_states)

                    # Guard against NaN logits (sign of weight instability)
                    if torch.isnan(logits).any():
                        print("Warning: NaN detected in policy logits, skipping this mini-batch")
                        continue

                    logits = torch.clamp(logits, -50.0, 50.0)

                    # Apply action masks so the distribution matches collection
                    masked_logits = logits + torch.where(
                        mb_masks > 0,
                        torch.zeros_like(logits),
                        torch.full_like(logits, -1e10),
                    )
                    probs = Categorical(logits=masked_logits)
                    log_probs = probs.log_prob(mb_actions)
                    entropy = probs.entropy()
                    ratios = torch.exp(log_probs - mb_log_probs_old)

                    # Clipped policy surrogate loss
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Clipped value loss (PPO paper §3): limits how far the value
                    # head can move per update, preventing unbounded TD errors when
                    # the agent encounters new-distribution states (e.g. stronger
                    # pool opponents).
                    cv = current_values.squeeze(-1)
                    values_clipped = mb_old_values + torch.clamp(
                        cv - mb_old_values, -self.eps_clip, self.eps_clip
                    )
                    value_loss = 0.5 * torch.max(
                        F.mse_loss(cv,             mb_returns.squeeze(-1)),
                        F.mse_loss(values_clipped, mb_returns.squeeze(-1)),
                    )

                    entropy_bonus = -self.entropy_bonus * entropy.mean()

                    loss = policy_loss + value_loss + entropy_bonus

                # set_to_none=True skips the memset pass over gradient tensors — free win
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

    def step_scheduler(self):
        """Step the learning rate scheduler and restore eval mode. Call once per epoch."""
        self.scheduler.step()
        # Restore eval mode after training so the InferenceServer's shared model
        # (if any) uses frozen BatchNorm running stats for inference.
        self.policy.eval()
