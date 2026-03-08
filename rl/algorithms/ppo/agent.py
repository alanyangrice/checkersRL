from rl.configs.ppo_config import PPOConfig

default_config = PPOConfig()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from rl.networks import PPOPolicyNetwork
from rl.algorithms.ppo.torch_helpers import (
    get_device,
    torch_compile_available,
    get_policy_state_dict,
    load_policy_state_dict,
)

# Re-export for backward compatibility
__all__ = ["PPOAgent", "get_device", "get_policy_state_dict", "load_policy_state_dict"]


@torch.jit.script
def compute_gae(deltas: torch.Tensor, not_done: torch.Tensor,
                 gamma: float, gae_lambda: float) -> torch.Tensor:
    """Compute Generalized Advantage Estimation via backward scan."""
    n = deltas.shape[0]
    advantages = torch.zeros(n)
    gae: float = 0.0
    for t in range(n - 1, -1, -1):
        gae = deltas[t] + gamma * gae_lambda * not_done[t] * gae
        advantages[t] = gae
    return advantages


def to_gpu(arr, dtype, device):
    """Convert a numpy array to a pinned-memory tensor and transfer non-blocking."""
    return (torch.from_numpy(np.asarray(arr, dtype=dtype))
            .pin_memory()
            .to(device, non_blocking=True))


class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.95, eps_clip=0.2,
                 K_epochs=4, gae_lambda=0.95, device=None, augment=True,
                 augment_noise=0.05, mini_batch_size=2048, entropy_bonus=0.01, config=None):
        config = config or default_config
        self.device = device or get_device()
        self.policy = PPOPolicyNetwork(input_shape, n_actions).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=lr,
            fused=(self.device.type == "cuda"),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.LR_SCHEDULER_T_MAX,
            eta_min=config.LR_SCHEDULER_ETA_MIN,
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.augment = augment
        self.augment_noise = augment_noise
        self.mini_batch_size = mini_batch_size
        self.entropy_bonus = entropy_bonus

        self._use_amp = (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self._amp_dtype = torch.bfloat16 if self._use_amp else torch.float32

        if torch_compile_available():
            self.policy = torch.compile(self.policy, mode="reduce-overhead")

    def get_policy_state_dict(self):
        return get_policy_state_dict(self.policy)

    def select_action(self, state, action_mask, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.policy(state_t)

        masked_logits = logits + torch.where(
            mask_t > 0,
            torch.zeros_like(logits),
            torch.full_like(logits, -1e10),
        )

        probs = Categorical(logits=masked_logits)
        
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        if len(memory.states) == 0:
            return

        self.policy.train()

        states        = to_gpu(memory.states,       np.float32, self.device)
        actions       = to_gpu(memory.actions,      np.int64,   self.device)
        rewards       = to_gpu(memory.rewards,      np.float32, self.device)
        log_probs_old = to_gpu(memory.log_probs,    np.float32, self.device)
        done_flags    = to_gpu(memory.done,         np.bool_,   self.device)
        action_masks  = to_gpu(memory.action_masks, np.float32, self.device)
        old_values    = to_gpu(memory.values,        np.float32, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            sv_chunks = []
            for i in range(0, states.size(0), self.mini_batch_size):
                _, sv = self.policy(states[i : i + self.mini_batch_size])
                sv_chunks.append(sv.squeeze(-1))
            state_values = torch.cat(sv_chunks, dim=0).float()

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

        advantages_cpu = compute_gae(deltas, not_done, self.gamma, self.gae_lambda)

        advantages = advantages_cpu.to(self.device)
        returns = (advantages + state_values)

        if self.augment:
            noise = torch.randn_like(states) * self.augment_noise
            states_noisy = torch.clamp(states + noise, 0.0, 1.0)

            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                logit_chunks = []
                for i in range(0, states_noisy.size(0), self.mini_batch_size):
                    lg, _ = self.policy(states_noisy[i : i + self.mini_batch_size])
                    logit_chunks.append(lg)
                logits_noisy = torch.cat(logit_chunks, dim=0).float()

            if torch.isnan(logits_noisy).any():
                print("Warning: NaN detected in augmentation logits, skipping augmentation this batch")
            else:
                logits_noisy = torch.clamp(logits_noisy, -50.0, 50.0)
                masked_logits_noisy = logits_noisy + torch.where(
                    action_masks > 0,
                    torch.zeros_like(logits_noisy),
                    torch.full_like(logits_noisy, -1e10),
                )
                probs_noisy = Categorical(logits=masked_logits_noisy)
                log_probs_old_noisy = probs_noisy.log_prob(actions)

                states       = torch.cat([states,       states_noisy],          dim=0)
                actions      = torch.cat([actions,      actions.clone()],        dim=0)
                log_probs_old = torch.cat([log_probs_old, log_probs_old_noisy],  dim=0)
                advantages   = torch.cat([advantages,   advantages.clone()],     dim=0)
                returns      = torch.cat([returns,      returns.clone()],       dim=0)
                action_masks = torch.cat([action_masks, action_masks.clone()],   dim=0)
                old_values   = torch.cat([old_values,   old_values.clone()],     dim=0)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.unsqueeze(-1)

        total = states.size(0)

        for _ in range(self.K_epochs):
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

                with torch.amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                    logits, current_values = self.policy(mb_states)

                    if torch.isnan(logits).any():
                        print("Warning: NaN detected in policy logits, skipping this mini-batch")
                        continue

                    logits = torch.clamp(logits, -50.0, 50.0)

                    masked_logits = logits + torch.where(
                        mb_masks > 0,
                        torch.zeros_like(logits),
                        torch.full_like(logits, -1e10),
                    )
                    probs = Categorical(logits=masked_logits)
                    log_probs = probs.log_prob(mb_actions)
                    entropy = probs.entropy()
                    ratios = torch.exp(log_probs - mb_log_probs_old)

                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

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

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()
        self.policy.eval()
