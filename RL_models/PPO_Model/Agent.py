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


class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.95, eps_clip=0.2,
                 K_epochs=4, gae_lambda=0.95, device=None, augment=True,
                 augment_noise=0.05):
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

        temp = 0.7
        probs = Categorical(logits=masked_logits / temp)
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

        # --- GAE on original data ----------------------------------------
        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        n = len(rewards)

        for t in reversed(range(n)):
            if done_flags[t] or t == n - 1:
                next_value = 0.0
            else:
                next_value = state_values[t + 1]

            delta = rewards[t] + self.gamma * next_value - state_values[t]
            gae = delta + self.gamma * self.gae_lambda * (0 if done_flags[t] else 1) * gae
            advantages[t] = gae
            returns[t] = gae + state_values[t]

        # --- Augment with noisy copies ----------------------------------
        if self.augment:
            noise = torch.randn_like(states) * self.augment_noise
            states_noisy = torch.clamp(states + noise, 0.0, 1.0)

            # Compute surrogate old log-probs for noisy states
            with torch.no_grad():
                logits_noisy, _ = self.policy(states_noisy)
                probs_noisy = Categorical(logits=logits_noisy)
                log_probs_old_noisy = probs_noisy.log_prob(actions)

            # Concatenate original + noisy
            states = torch.cat([states, states_noisy], dim=0)
            actions = torch.cat([actions, actions.clone()], dim=0)
            log_probs_old = torch.cat([log_probs_old, log_probs_old_noisy], dim=0)
            advantages = torch.cat([advantages, advantages.clone()], dim=0)
            returns = torch.cat([returns, returns.clone()], dim=0)

        # Normalize advantages over the full (augmented) batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.unsqueeze(-1)

        # --- PPO update for K epochs ------------------------------------
        for _ in range(self.K_epochs):
            logits, current_values = self.policy(states)
            probs = Categorical(logits=logits)
            log_probs = probs.log_prob(actions)
            entropy = probs.entropy()
            ratios = torch.exp(log_probs - log_probs_old)

            # Clipped Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * nn.MSELoss()(current_values, returns)
            entropy_bonus = -0.05 * entropy.mean()

            loss = policy_loss + value_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def step_scheduler(self):
        """Step the learning rate scheduler. Call once per epoch."""
        self.scheduler.step()
