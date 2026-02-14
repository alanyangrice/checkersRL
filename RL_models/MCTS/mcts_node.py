import math
import numpy as np


class MCTSNode:
    """A node in the Monte Carlo Tree Search tree.

    Each node represents a game state after a specific action was taken.
    The root node has action=None and represents the current board position.

    Attributes:
        parent: Parent MCTSNode (None for root).
        action: The action (semantic index) that led to this node.
        prior: P(action) from the policy network at the parent state.
        visit_count: N(s, a) -- number of times this node was visited.
        value_sum: W(s, a) -- total accumulated value from backpropagation.
        children: dict mapping action_index -> MCTSNode.
        is_terminal: Whether this node represents a finished game.
        terminal_value: The game outcome value if terminal (+1/-1/0).
    """

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self):
        """Mean action-value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.5):
        """PUCT score: Q(s,a) + c * P(s,a) * sqrt(N_parent) / (1 + N(s,a)).

        Balances exploitation (Q) with exploration (prior * visit ratio).
        """
        if self.parent is None:
            return 0.0

        parent_visits = self.parent.visit_count
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    @property
    def is_expanded(self):
        """Whether this node has been expanded (children created)."""
        return len(self.children) > 0

    @property
    def is_root(self):
        return self.parent is None

    def best_child(self, c_puct=1.5):
        """Select the child with the highest UCB score."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def best_action_by_visits(self):
        """Select the action with the most visits (used for final move selection)."""
        return max(self.children.keys(), key=lambda a: self.children[a].visit_count)

    def visit_count_distribution(self, num_actions):
        """Return a probability distribution proportional to visit counts.

        Args:
            num_actions: Total size of the action space (NUM_ACTIONS).

        Returns:
            numpy array of shape (num_actions,) summing to 1.0.
        """
        counts = np.zeros(num_actions, dtype=np.float32)
        for action, child in self.children.items():
            counts[action] = child.visit_count

        total = counts.sum()
        if total > 0:
            return counts / total
        return counts

    def __repr__(self):
        return (f"MCTSNode(action={self.action}, prior={self.prior:.3f}, "
                f"visits={self.visit_count}, Q={self.q_value:.3f}, "
                f"children={len(self.children)})")
