import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from PolicyNetwork import PPOPolicyNetwork

class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-5, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PPOPolicyNetwork(input_shape, n_actions).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        logits, _ = self.policy(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        # Convert memory to tensors and move to GPU
        states = torch.FloatTensor(memory.states).cuda()
        actions = torch.LongTensor(memory.actions).cuda()
        rewards = torch.FloatTensor(memory.rewards).cuda()
        log_probs_old = torch.FloatTensor(memory.log_probs).cuda()

        # Calculate discounted rewards
        discounted_rewards = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards).cuda()

        # PPO update for K epochs
        for _ in range(self.K_epochs):
            logits, state_values = self.policy(states)
            probs = Categorical(logits=logits)
            log_probs = probs.log_prob(actions)
            entropy = probs.entropy()
            ratios = torch.exp(log_probs - log_probs_old)

            # Clipped Surrogate Loss
            advantages = discounted_rewards - state_values.squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, discounted_rewards) - 0.01 * entropy

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
