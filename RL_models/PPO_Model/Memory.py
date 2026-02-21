# Memory class to store experiences
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.done = []
        self.action_masks = []

    def add(self, state, action_index, reward, log_prob, done, action_mask=None):
        self.states.append(state)
        self.actions.append(action_index)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.done.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.done = []
        self.action_masks = []
    
    def extend(self, other_memory):
        self.states.extend(other_memory.states)
        self.actions.extend(other_memory.actions)
        self.rewards.extend(other_memory.rewards)
        self.log_probs.extend(other_memory.log_probs)
        self.done.extend(other_memory.done)
        self.action_masks.extend(other_memory.action_masks)

    def update_last_done(self):
        if self.done:
            self.done[-1] = True
