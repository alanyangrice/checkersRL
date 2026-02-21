import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork


def get_device():
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.95, eps_clip=0.2,
                 K_epochs=4, gae_lambda=0.95, device=None, augment=True,
                 augment_noise=0.05, mini_batch_size=2048):
        self.device = device or get_device()
        self.policy = PPOPolicyNetwork(input_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=1e-6)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.augment = augment
        self.augment_noise = augment_noise
        self.mini_batch_size = mini_batch_size

        # Mixed precision: bfloat16 on CUDA (same exponent range as FP32,
        # no overflow risk, no GradScaler needed — uses tensor cores on Ampere+)
        self._use_amp = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self._amp_dtype = torch.bfloat16 if self._use_amp else torch.float32

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
        masked_logits = logits + torch.where(mask_t > 0, 0.0, torch.tensor(-1e10, device=self.device))

        probs = Categorical(logits=masked_logits)
        action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        """PPO update with GAE and optional random noise data augmentation.

        When augment=True, an additional noisy copy of all states is included
        in the PPO batch. The noisy copy shares the same actions, rewards, and
        advantages as the originals but forces the network to be robust to small
        observation perturbations (similar to DrAC / RAD in RL literature).
        """
        if len(memory.states) == 0:
            return

        states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        done_flags = torch.tensor(memory.done, dtype=torch.bool).to(self.device)
        action_masks = torch.FloatTensor(np.array(memory.action_masks)).to(self.device)

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
                    action_masks > 0, 0.0, torch.tensor(-1e10, device=self.device)
                )
                probs_noisy = Categorical(logits=masked_logits_noisy)
                log_probs_old_noisy = probs_noisy.log_prob(actions)

            # Concatenate original + noisy
            states = torch.cat([states, states_noisy], dim=0)
            actions = torch.cat([actions, actions.clone()], dim=0)
            log_probs_old = torch.cat([log_probs_old, log_probs_old_noisy], dim=0)
            advantages = torch.cat([advantages, advantages.clone()], dim=0)
            returns = torch.cat([returns, returns.clone()], dim=0)
            action_masks = torch.cat([action_masks, action_masks.clone()], dim=0)

        # Normalize advantages over the full (augmented) batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.unsqueeze(-1)

        # --- PPO update for K epochs with mini-batches --------------------
        total = states.size(0)
        neg_inf = torch.tensor(-1e10, device=self.device)

        for _ in range(self.K_epochs):
            # Shuffle indices each epoch for stochastic mini-batches
            perm = torch.randperm(total, device=self.device)

            for mb_start in range(0, total, self.mini_batch_size):
                mb_idx = perm[mb_start : mb_start + self.mini_batch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs_old = log_probs_old[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_masks = action_masks[mb_idx]

                # Mixed-precision forward pass (BF16 on CUDA, FP32 elsewhere)
                with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                    logits, current_values = self.policy(mb_states)

                    # Guard against NaN logits (sign of weight instability)
                    if torch.isnan(logits).any():
                        print("Warning: NaN detected in policy logits, skipping this mini-batch")
                        continue

                    logits = torch.clamp(logits, -50.0, 50.0)

                    # Apply action masks so the distribution matches collection
                    masked_logits = logits + torch.where(mb_masks > 0, 0.0, neg_inf)
                    probs = Categorical(logits=masked_logits)
                    log_probs = probs.log_prob(mb_actions)
                    entropy = probs.entropy()
                    ratios = torch.exp(log_probs - mb_log_probs_old)

                    # Clipped Surrogate Loss
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * nn.MSELoss()(current_values, mb_returns)
                    entropy_bonus = -0.01 * entropy.mean()

                    loss = policy_loss + value_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

    def step_scheduler(self):
        """Step the learning rate scheduler. Call once per epoch."""
        self.scheduler.step()
