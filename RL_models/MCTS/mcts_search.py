import copy
import numpy as np
import torch
from torch.distributions import Categorical

from RL_models.MCTS.mcts_node import MCTSNode
from checkers_game.constants import NUM_ACTIONS, BLUE, RED


class MCTSSearch:
    """Monte Carlo Tree Search using a policy/value network for guidance.

    Implements the AlphaZero-style MCTS:
      1. SELECT:   traverse tree using PUCT (UCB + prior).
      2. EXPAND:   at a leaf, use the policy network to get action priors.
      3. EVALUATE:  use the value network to estimate the leaf value.
      4. BACKUP:   propagate the value back up the tree.

    After N simulations, the root's visit-count distribution is used to select
    the final action (proportional sampling with temperature, or argmax).
    """

    def __init__(self, network, num_simulations=100, c_puct=1.5, device=None):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def _evaluate(self, env):
        """Run the network on the current env state to get (policy_priors, value).

        Returns:
            priors: numpy array of shape (NUM_ACTIONS,) -- masked & normalized action probabilities.
            value: float -- estimated value of the position for the current player.
        """
        state = env.get_board_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask = env.get_action_mask()

        logits, value = self.network(state_t)
        logits = logits.squeeze(0).cpu().numpy()
        value = value.item()

        # Mask invalid actions and compute softmax for priors
        mask = action_mask > 0
        if mask.sum() == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float32), value

        # Apply mask: set invalid to -inf, then softmax
        masked_logits = np.full(NUM_ACTIONS, -1e10, dtype=np.float32)
        masked_logits[mask] = logits[mask]

        # Stable softmax
        exp_logits = np.exp(masked_logits - masked_logits.max())
        priors = exp_logits / exp_logits.sum()

        return priors, value

    def search(self, env):
        """Run MCTS from the current env state.

        Args:
            env: CheckersEnv instance (will be deep-copied for simulations).

        Returns:
            action_probs: numpy array of shape (NUM_ACTIONS,) -- visit-count distribution.
            root_value: float -- estimated value of the root position.
        """
        root = MCTSNode()

        # Expand root
        priors, root_value = self._evaluate(env)
        action_mask = env.get_action_mask()
        self._expand_node(root, priors, action_mask)

        # Check if root is already terminal
        if action_mask.sum() == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float32), root_value

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            env_copy = copy.deepcopy(env)
            search_path = [node]

            # --- SELECT: traverse tree using PUCT ---
            while node.is_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
                search_path.append(node)

                # Apply the action in the copied env
                _, _, done, _, info = env_copy.step(node.action)

                if done:
                    node.is_terminal = True
                    # Determine terminal value from current player's perspective at root
                    winner = info.get("winner", "None")
                    node.terminal_value = self._outcome_value(winner, env.game.turn, env_copy.game.turn)
                    break

            # --- EVALUATE leaf ---
            if node.is_terminal:
                leaf_value = node.terminal_value
            else:
                # Expand the leaf node
                priors, leaf_value = self._evaluate(env_copy)
                action_mask = env_copy.get_action_mask()

                if action_mask.sum() == 0:
                    # No legal moves = loss for the current player at this node
                    node.is_terminal = True
                    node.terminal_value = -1.0
                    leaf_value = -1.0
                else:
                    self._expand_node(node, priors, action_mask)

            # --- BACKUP: propagate value up the tree ---
            self._backup(search_path, leaf_value, env.game.turn)

        # Return visit-count distribution from root
        action_probs = root.visit_count_distribution(NUM_ACTIONS)
        return action_probs, root_value

    def select_action(self, env, temperature=1.0):
        """Run MCTS and select an action.

        Args:
            env: CheckersEnv instance.
            temperature: Controls exploration vs exploitation.
                         1.0 = proportional to visit counts (exploration).
                         0.0 = always pick the most-visited action (exploitation).

        Returns:
            action: Selected action index.
            action_probs: Full visit-count distribution (for training targets).
        """
        action_probs, _ = self.search(env)

        if temperature == 0:
            # Greedy: pick the most-visited action
            action = int(np.argmax(action_probs))
        elif temperature == 1.0:
            # Sample proportional to visit counts
            if action_probs.sum() > 0:
                action = int(np.random.choice(NUM_ACTIONS, p=action_probs))
            else:
                action = 0
        else:
            # Sharpen/soften the distribution with temperature
            log_probs = np.log(action_probs + 1e-10) / temperature
            exp_probs = np.exp(log_probs - log_probs.max())
            tempered = exp_probs / exp_probs.sum()
            action = int(np.random.choice(NUM_ACTIONS, p=tempered))

        return action, action_probs

    def _expand_node(self, node, priors, action_mask):
        """Create child nodes for all valid actions."""
        valid_actions = np.where(action_mask > 0)[0]
        for action in valid_actions:
            if action not in node.children:
                node.children[action] = MCTSNode(
                    parent=node,
                    action=int(action),
                    prior=float(priors[action]),
                )

    def _backup(self, search_path, leaf_value, root_turn):
        """Propagate value back up the search path.

        The value alternates sign at each turn boundary because each player
        wants to maximize their own value. However, within a capture chain
        (multiple actions by the same player), the sign stays the same.

        For simplicity, we negate at every level (since each step in the tree
        corresponds to one env.step(), and turn switching is handled by the env).
        """
        # The leaf_value is from the perspective of the player who is about to
        # move at the leaf. We need to propagate it back, flipping sign at each
        # level since parent and child typically represent different players.
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip perspective going up

    def _outcome_value(self, winner, root_turn, current_turn):
        """Convert a game outcome into a value from the root player's perspective.

        Returns +1 if root player wins, -1 if root player loses, 0 for tie.
        The value is then negated appropriately during backup based on tree depth.
        """
        if winner == "Tie" or winner == "None":
            return 0.0

        # winner is a color tuple (BLUE or RED)
        if winner == root_turn:
            # Root player won. Value from the current node's perspective depends
            # on whether current node's player is the same as root.
            if current_turn == root_turn:
                return 1.0
            else:
                return -1.0
        else:
            if current_turn == root_turn:
                return -1.0
            else:
                return 1.0
