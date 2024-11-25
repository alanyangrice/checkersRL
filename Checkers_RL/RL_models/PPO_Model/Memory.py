# Memory class to store experiences
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.done = []

    def add(self, state, action_index, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action_index)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.done.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.done = []