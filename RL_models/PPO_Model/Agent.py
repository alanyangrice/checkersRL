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
                 K_epochs=4, gae_lambda=0.95, device=None):
        self.device = device or get_device()
        self.policy = PPOPolicyNetwork(input_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=1e-6)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

    def select_action(self, state, num_legal_moves):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.policy(state)

        # Mask invalid actions
        mask = torch.full(logits.size(), -1e10, device=self.device)
        mask[0, :num_legal_moves] = 0

        temp = 0.7
        masked_logits = logits + mask
        probs = Categorical(logits=masked_logits / temp)
        action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        """PPO update with Generalized Advantage Estimation (GAE)."""
        states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        done_flags = torch.tensor(memory.done, dtype=torch.bool).to(self.device)

        # Compute state values for GAE
        with torch.no_grad():
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)

        # GAE computation
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

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.unsqueeze(-1)

        # PPO update for K epochs
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
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def step_scheduler(self):
        """Step the learning rate scheduler. Call once per epoch."""
        self.scheduler.step()
