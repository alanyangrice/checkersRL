"""AlphaZero-style Monte Carlo Tree Search for checkers.

Four-phase loop per simulation
--------------------------------
  SELECT   - walk the tree from the root following PUCT scores until reaching
             an unexpanded leaf or a terminal node.
  EXPAND   - run the network on the leaf to get action priors and a value
             estimate; create child nodes for every legal action.
  EVALUATE - use the network value as the leaf estimate (no random rollouts).
  BACKUP   - propagate the value back up the path, flipping sign only when
             the active player changes (handles multi-jump captures correctly).

Key correctness notes
---------------------
* Value perspective: every node's value_sum accumulates values from the
  perspective of the player who is *about to move* in that node's state.
  The sign flip in backup therefore only occurs at player-change boundaries,
  not unconditionally at every level.  Without this, multi-jump capture chains
  -- where the same player acts at several consecutive nodes -- would corrupt
  Q-values throughout the chain.

* Dirichlet noise: added to root priors during self-play so that MCTS always
  considers every legal move at least sometimes, preventing the training data
  from collapsing onto a narrow set of lines.

* Tree reuse: after a move is played the root is advanced to the matching
  child, preserving all accumulated visit counts and Q-values for subtrees
  that will be explored again next turn.
"""

import numpy as np
import torch

from RL_models.numpy_checkers_env import NumpyCheckersEnv
from RL_models.MCTS.mcts_node import MCTSNode
from RL_models.MCTS import training_config as cfg
from checkers_game.constants import NUM_ACTIONS


class MCTSSearch:

    def __init__(
        self,
        network=None,
        num_simulations=cfg.NUM_SIMULATIONS,
        c_puct=cfg.C_PUCT,
        dirichlet_alpha=cfg.DIRICHLET_ALPHA,
        dirichlet_epsilon=cfg.DIRICHLET_EPSILON,
        device=None,
        evaluator=None,
    ):
        """Create an MCTS search object.

        Exactly one of *network* or *evaluator* must be supplied:
          network   — AlphaZeroNetwork instance; inference runs in this process
                      (training process, or CPU-worker non-parallel mode).
          evaluator — callable(state, action_mask) -> (logits_np, value_float);
                      inference is delegated to a remote GPU server
                      (used by worker processes in train_parallel.py).
        """
        if network is None and evaluator is None:
            raise ValueError("Provide either network or evaluator")
        if network is not None and evaluator is not None:
            raise ValueError("Provide network or evaluator, not both")

        self.network   = network
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or torch.device("cpu")

        self._root = None  # cached root node for tree reuse between moves

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def update_root(self, action):
        """Advance the cached root to the child reached by *action*.

        Called after each env.step() during self-play so the next search
        builds on top of previously accumulated statistics rather than
        starting from an empty tree.  If the action was never explored
        (cache miss), the tree is cleared and rebuilt from scratch next call.
        """
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None  # detach old tree for garbage collection
        else:
            self._root = None

    def search(self, env, add_noise=False):
        """Run MCTS from the current env state and return a policy.

        Args:
            env:       CheckersEnv or NumpyCheckersEnv instance.
                       If a CheckersEnv is passed it is converted to
                       NumpyCheckersEnv once here; all 100 simulations then
                       use NumpyCheckersEnv.fast_clone() instead of
                       copy.deepcopy — a single 64-byte memcpy vs a full
                       Python object-graph traversal.
            add_noise: If True, mix Dirichlet noise into root priors.
                       Should be True during self-play, False during evaluation.

        Returns:
            action_probs: (NUM_ACTIONS,) visit-count distribution (sums to 1).
            root_value:   float - network's value estimate at the root.
        """
        # Convert to numpy env once — negligible cost vs 100 simulations.
        if not isinstance(env, NumpyCheckersEnv):
            env = NumpyCheckersEnv.from_env(env)

        # ------------------------------------------------------------------ #
        # Root: reuse cached node or create fresh
        # ------------------------------------------------------------------ #
        root = self._root if self._root is not None else MCTSNode()

        # Always re-evaluate the root so priors are current (+ optional noise).
        # For tree reuse: child priors are updated; visit counts are preserved.
        priors, root_value, action_mask = self._evaluate(env)

        if action_mask.sum() == 0:
            self._root = None
            return np.zeros(NUM_ACTIONS, dtype=np.float32), root_value

        if add_noise:
            priors = self._add_dirichlet_noise(priors, action_mask)

        if root.is_expanded:
            # Tree reuse: refresh priors for existing children
            for action, child in root.children.items():
                child.prior = float(priors[action])
        else:
            self._expand_node(root, priors, action_mask)

        self._root = root
        root_player = env.game.turn

        # ------------------------------------------------------------------ #
        # Simulations
        # ------------------------------------------------------------------ #
        for _ in range(self.num_simulations):
            node = root
            env_copy = env.fast_clone()

            # search_path: list of (MCTSNode, player_about_to_move_at_this_node)
            # The player at a node is env.game.turn *after* applying the action
            # that led to that node (i.e. the player who moves NEXT from here).
            search_path = [(node, root_player)]

            # --- SELECT ---------------------------------------------------
            while node.is_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
                _, _, done, _, info = env_copy.step(node.action)
                node_player = env_copy.game.turn
                search_path.append((node, node_player))

                if done:
                    node.is_terminal = True
                    winner = info.get("winner", "None")
                    # node_player is env_copy.game.turn after the done step.
                    # For a normal game-ending move (all pieces captured):
                    #   _finish_turn calls switch_turn → game.turn = loser
                    # For a no-legal-moves ending:
                    #   step calls switch_turn → game.turn = winner
                    # _outcome_value handles both correctly.
                    node.terminal_value = self._outcome_value(winner, node_player)
                    break

            # --- EVALUATE leaf --------------------------------------------
            if node.is_terminal:
                leaf_value = node.terminal_value
            else:
                priors, leaf_value, action_mask = self._evaluate(env_copy)

                if action_mask.sum() == 0:
                    # Current player has no legal moves → they lose
                    node.is_terminal = True
                    node.terminal_value = -1.0
                    leaf_value = -1.0
                else:
                    self._expand_node(node, priors, action_mask)

            # --- BACKUP ---------------------------------------------------
            self._backup(search_path, leaf_value)

        action_probs = root.visit_count_distribution(NUM_ACTIONS)
        return action_probs, root_value

    def select_action(self, env, temperature=1.0, add_noise=False):
        """Run MCTS and select an action.

        Args:
            env:         CheckersEnv instance.
            temperature: Controls exploration in the final action selection.
                         1.0 = sample proportional to visit counts (exploration).
                         0.0 = pick the most-visited action (exploitation).
                         Other values rescale the distribution via temperature.
            add_noise:   Whether to add Dirichlet noise at the root.
                         Pass True during self-play, False for evaluation.

        Returns:
            action:       int - selected action index.
            action_probs: (NUM_ACTIONS,) visit-count distribution (training target).
        """
        action_probs, _ = self.search(env, add_noise=add_noise)

        if temperature == 0:
            action = int(np.argmax(action_probs))
        elif temperature == 1.0:
            action = (
                int(np.random.choice(NUM_ACTIONS, p=action_probs))
                if action_probs.sum() > 0
                else 0
            )
        else:
            log_probs = np.log(action_probs + 1e-10) / temperature
            exp_probs = np.exp(log_probs - log_probs.max())
            tempered = exp_probs / exp_probs.sum()
            action = int(np.random.choice(NUM_ACTIONS, p=tempered))

        return action, action_probs

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _evaluate(self, env):
        """Evaluate a board position: return (priors, value, action_mask).

        Dispatches to either:
          • Local GPU/CPU inference via self.network  (training / single-process)
          • Remote GPU server via self.evaluator       (worker processes)

        Returns:
            priors:      (NUM_ACTIONS,) float32 - masked & normalised action probs.
            value:       float - estimated win probability for the current player.
            action_mask: (NUM_ACTIONS,) float32 - 1.0 for legal actions.
        """
        state       = env.get_board_state()
        action_mask = env.get_action_mask()

        if self.evaluator is not None:
            # Remote path — IPC call to GPU inference server in main process.
            # Returns (logits_np, value) already as numpy / float.
            logits, value = self.evaluator(state, action_mask)
        else:
            # Local path — direct network forward pass.
            state_t      = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits_t, v  = self.network(state_t)
            logits       = logits_t.squeeze(0).cpu().numpy()
            value        = v.item()

        mask = action_mask > 0
        if mask.sum() == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float32), value, action_mask

        masked_logits = np.full(NUM_ACTIONS, -1e10, dtype=np.float32)
        masked_logits[mask] = logits[mask]
        exp_logits = np.exp(masked_logits - masked_logits.max())
        priors = exp_logits / exp_logits.sum()

        return priors, value, action_mask

    def _add_dirichlet_noise(self, priors, action_mask):
        """Mix Dirichlet noise into root priors to ensure self-play exploration.

        P(root, a) = (1 - ε) * p_network(a) + ε * η_a,   η ~ Dir(α)

        Noise is only applied to legal actions; illegal actions stay at 0.
        """
        valid = np.where(action_mask > 0)[0]
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid))
        noisy = priors.copy()
        noisy[valid] = (
            (1 - self.dirichlet_epsilon) * priors[valid]
            + self.dirichlet_epsilon * noise
        )
        return noisy

    def _expand_node(self, node, priors, action_mask):
        """Create child nodes for all legal actions."""
        for action in np.where(action_mask > 0)[0]:
            if action not in node.children:
                node.children[int(action)] = MCTSNode(
                    parent=node,
                    action=int(action),
                    prior=float(priors[action]),
                )

    def _backup(self, search_path, leaf_value):
        """Propagate leaf value back up the search path.

        Each node's value_sum accumulates the value from *its own player's*
        perspective.  The sign flips only when consecutive nodes belong to
        different players.  This correctly handles checkers multi-jump captures
        where the same player acts at several consecutive tree levels.

        Args:
            search_path: list of (MCTSNode, player_color) from root to leaf.
            leaf_value:  float in [-1, 1] from the leaf player's perspective.
        """
        value = leaf_value
        for i in range(len(search_path) - 1, -1, -1):
            node, player = search_path[i]
            node.visit_count += 1
            node.value_sum += value

            if i > 0:
                _, parent_player = search_path[i - 1]
                if parent_player != player:
                    value = -value
                # Same player (multi-jump continuation): value sign unchanged

    def _outcome_value(self, winner, current_player):
        """Convert a game outcome to a value from *current_player*'s perspective.

        Returns +1 if current_player won, -1 if current_player lost, 0 for tie.
        """
        if winner == "Tie" or winner == "None":
            return 0.0
        return 1.0 if winner == current_player else -1.0
