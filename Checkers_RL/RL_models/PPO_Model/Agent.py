import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork

class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=1e-5, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PPOPolicyNetwork(input_shape, n_actions)#.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state, num_legal_moves):
        state = torch.FloatTensor(state).unsqueeze(0)#.cuda()

        # Get logits for all actions from the policy network
        logits, _ = self.policy(state)

        # Create a mask to set all invalid actions to a large negative value
        mask = torch.full(logits.size(), -1e10)#.cuda()
        mask[0, :num_legal_moves] = 0  # Allow only the first `num_legal_moves` actions

        # Apply the mask to logits to zero out invalid action probabilities
        temp = 0.5
        masked_logits = logits + mask
        probs = Categorical(logits=masked_logits / temp)
        action = probs.sample()

        return action.item(), probs.log_prob(action), probs.entropy()

    def update(self, memory):
        # Convert memory to tensors and move to GPU
        states = torch.FloatTensor(np.array(memory.states))#.cuda()
        actions = torch.LongTensor(np.array(memory.actions))#.cuda()
        rewards = torch.FloatTensor(np.array(memory.rewards))#.cuda()
        log_probs_old = torch.FloatTensor(memory.log_probs)#.cuda()

        # Calculate discounted rewards
        discounted_rewards = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards).view(-1, 1)#.cuda()

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
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, discounted_rewards) - 0.05 * entropy

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
