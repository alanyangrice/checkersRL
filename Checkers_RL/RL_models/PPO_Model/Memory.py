# Memory class to store experiences
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def add(self, state, action_index, reward, log_prob):
        self.states.append(state)
        self.actions.append(action_index)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []