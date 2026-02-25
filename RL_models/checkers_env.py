import numpy as np
import copy

import gymnasium as gym
from gymnasium import spaces

from checkers_game.game import Game
from checkers_game.constants import (
    RED, BLUE, ROWS, COLS,
    NUM_ACTIONS, ALL_ACTIONS, ACTION_TO_INDEX,
    encode_action, decode_action,
    board_number_to_position, position_to_board_number,
)


class CheckersEnv(gym.Env):
    """Checkers environment with a semantic single-step action space.

    Each action is a fixed (from_square, to_square) pair representing one
    diagonal step (regular move) or one capture jump. Multi-captures are
    decomposed into sequential single-step actions within the same turn.

    The env handles turn switching internally. Training loops should NOT
    call game.switch_turn() -- just keep calling step().
    """

    def __init__(self, reward_config=None):
        """Create a CheckersEnv.

        Args:
            reward_config: Optional dict overriding default reward parameters.
                Keys (all optional, defaults shown):
                    capture_bonus      (10.0)  — reward per capture hop
                    capture_penalty    (-5.0)  — retroactive penalty to victim per piece
                    king_bonus         (15.0)  — one-time king-promotion reward
                    tie_base           (-200)  — base tie penalty
                    shaping_scale      (0.5)   — multiplier on all shaped rewards
                    time_penalty_scale (1.0)   — multiplier on per-move time penalty
        """
        super(CheckersEnv, self).__init__()

        self.game = Game()

        # Observation: 4 channels from current player's perspective
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32)

        # Semantic action space: fixed (from_sq, to_sq) single-step pairs
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Capture chain tracking
        self._capture_in_progress = False
        self._capturing_piece_sq = None      # board number of the piece mid-chain
        self._visited_squares = set()        # (row, col) visited during chain
        self._current_move_chain = []        # list of board numbers in this turn
        self._is_capture_turn = False        # whether this turn involves captures
        self._turn_start_board = None        # board snapshot before the turn began

        # Action mask (updated after reset/step)
        self._action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Reward parameters (configurable for multi-agent league play)
        cfg = reward_config or {}
        self._capture_bonus      = cfg.get("capture_bonus",      10.0)
        self._capture_penalty    = cfg.get("capture_penalty",    -5.0)
        self._king_bonus         = cfg.get("king_bonus",         15.0)
        self._tie_base           = cfg.get("tie_base",          -200)
        self._shaping_scale      = cfg.get("shaping_scale",       0.5)
        self._time_penalty_scale = cfg.get("time_penalty_scale",  1.0)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the game to initial state.

        Options (via the `options` dict):
            num_pieces (int): Number of pieces per side for curriculum learning.
                              If omitted or 12, uses the standard starting position.
                              Values 1-11 create a random board with that many pieces per side.
            num_blue (int):   Override piece count for BLUE (asymmetric boards).
            num_red  (int):   Override piece count for RED  (asymmetric boards).
            king_prob (float): Probability of promoting pieces placed in mid-board (default 0.15).
        """
        super().reset(seed=seed)
        self.game = Game()

        if options:
            num_pieces = options.get("num_pieces")
            num_blue = options.get("num_blue")
            num_red = options.get("num_red")
            needs_random = (
                (num_pieces is not None and 1 <= num_pieces < 12)
                or num_blue is not None
                or num_red is not None
            )
            if needs_random:
                king_prob = options.get("king_prob", 0.15)
                from checkers_game.board import Board
                self.game.board = Board()
                self.game.board.create_random_board(
                    num_pieces if num_pieces is not None else 6,
                    king_prob=king_prob,
                    num_blue=num_blue,
                    num_red=num_red,
                )

        self._capture_in_progress = False
        self._capturing_piece_sq = None
        self._visited_squares = set()
        self._current_move_chain = []
        self._is_capture_turn = False
        self._turn_start_board = None

        self._update_action_mask()
        return self.get_board_state(), {}

    def step(self, action):
        """Apply a single-step semantic action.

        Returns standard Gymnasium 5-tuple: (obs, reward, terminated, truncated, info).
        info["turn_complete"] indicates whether the current player's full turn is done.
        """

        # Handle no-legal-moves (current player loses)
        if self._action_mask.sum() == 0:
            loser_color = self.game.turn
            winner = BLUE if loser_color == RED else RED

            # No action was taken, so step reward is 0.  All terminal signals
            # go through per-color adjustments applied to each side's last
            # memory entry: loser gets -100, winner gets +100.
            reward = 0.0
            blue_adj = 0.0
            red_adj = 0.0
            if loser_color == BLUE:
                blue_adj = -100.0
                red_adj = 100.0
            else:
                red_adj = -100.0
                blue_adj = 100.0

            self.game.switch_turn()
            self._update_action_mask()
            obs = self.get_board_state()
            return obs, reward, True, False, {
                "turn": self.game.turn,
                "turn_complete": True,
                "winner": copy.deepcopy(winner),
                "blue_reward_adjustment": blue_adj,
                "red_reward_adjustment": red_adj,
            }

        # Validate / fallback for invalid action
        if self._action_mask[action] == 0:
            valid_indices = np.where(self._action_mask > 0)[0]
            action = int(np.random.choice(valid_indices))

        # Decode the semantic action
        from_sq, to_sq = decode_action(action)
        from_row, from_col = board_number_to_position(from_sq)
        to_row, to_col = board_number_to_position(to_sq)

        is_capture = abs(from_row - to_row) == 2

        # --- First hop of a new turn: snapshot the board -----------------
        if not self._capture_in_progress:
            self._turn_start_board = copy.deepcopy(self.game.board)
            self._current_move_chain = [from_sq]
            self._visited_squares = {(from_row, from_col)}
            self._is_capture_turn = is_capture

        # --- Apply the single-step move/capture -------------------------
        if is_capture:
            self.game.board.capture_piece(from_row, from_col, to_row, to_col)
        else:
            self.game.board.move_piece(from_row, from_col, to_row, to_col)

        self._current_move_chain.append(to_sq)
        self._visited_squares.add((to_row, to_col))

        # --- Check for continuation captures -----------------------------
        has_continuations = False
        if is_capture:
            piece = self.game.board.get_piece(to_row, to_col)
            if piece != 0:
                continuations = self.game.board.valid_moves_for_piece(
                    piece, to_row, to_col, capture_only=True
                )
                # Filter out previously visited landing squares
                continuations = {
                    k: v for k, v in continuations.items()
                    if k not in self._visited_squares
                }
                if continuations:
                    has_continuations = True

        # --- Branch: intermediate hop vs turn complete -------------------
        if has_continuations:
            # Capture chain continues -- same player acts again
            self._capture_in_progress = True
            self._capturing_piece_sq = to_sq

            self._update_action_mask()
            obs = self.get_board_state()

            return obs, self._capture_bonus, False, False, {
                "turn": self.game.turn,
                "turn_complete": False,
                "winner": "None",
                "blue_reward_adjustment": 0.0,
                "red_reward_adjustment": 0.0,
            }
        else:
            # Turn is complete
            return self._finish_turn()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finish_turn(self):
        """Compute full rewards, log the move, switch turns, return step tuple."""
        old_board = self._turn_start_board
        new_board = copy.deepcopy(self.game.board)

        # Build move string for logging
        delimiter = "x" if self._is_capture_turn else "-"
        move_str = delimiter.join(map(str, self._current_move_chain))
        self.game.moves.append(move_str)

        # --- Reward computation ------------------------------------------
        # Shaped rewards are scaled down so terminal outcomes (win/loss/tie)
        # have more relative impact on the learning signal.
        # Uses instance variables so each env instance can have a different
        # reward profile (league play with diverse agents).

        # Check game-over first (need 'done' flag for opponent penalty guard)
        end_reward, done, winner = self.reward_end_game()

        shaped_reward = 0.0

        # King promotion: one-time bonus for advancing a piece
        shaped_reward += self.reward_king_promotion(old_board)

        # capture_bonus for the final capture hop (intermediates already got it)
        if self._is_capture_turn:
            shaped_reward += self._capture_bonus

        # NOTE: penalize_undefended_pieces removed — the retroactive capture
        # penalty (-2.5 per piece taken) already punishes the victim when a
        # capture actually happens, and the pre-emptive version was discouraging
        # tactical positions where pieces are in contact.
        #
        # Static board-feature rewards (center control, rear protection,
        # balance, king count) were also removed because they reward
        # *maintaining* a position rather than *making progress*.

        # Combine: scaled shaping + full-strength terminal reward
        reward = shaped_reward * self._shaping_scale + end_reward

        # Note: Immediate per-step repetition penalties were tried but disrupted
        # training (blue/red asymmetry, catastrophic reward scale).  The
        # combination of 5-fold repetition threshold + higher base tie penalty
        # (-200) + gamma=0.98 (in training_config) is the preferred approach.

        # --- Per-color reward adjustments for the training loop ----------
        # These are retroactive penalties applied to the *opponent's* last
        # memory entry, returned via info so the env owns all reward math.
        acting_color = self.game.turn
        opponent_color = RED if acting_color == BLUE else BLUE
        blue_adj, red_adj = 0.0, 0.0

        # Capture penalty: penalise the opponent whose piece(s) were taken.
        if self._is_capture_turn:
            num_captured = len(self._current_move_chain) - 1
            capture_penalty = self._capture_penalty * num_captured * self._shaping_scale
            if opponent_color == BLUE:
                blue_adj += capture_penalty
            else:
                red_adj += capture_penalty

        # Terminal loser penalty: when the acting player wins, the opponent
        # (loser) needs -100.  (If the acting player loses, their -100 is
        # already in `reward` from reward_end_game.)
        if done and winner is not None and winner != "Tie":
            loser_color = RED if winner == BLUE else BLUE
            if loser_color != acting_color:
                if loser_color == BLUE:
                    blue_adj -= 100.0
                else:
                    red_adj -= 100.0

        # Tie penalty for the non-acting player: the acting player already
        # receives the tie penalty via `reward` (from reward_end_game), but
        # the opponent gets nothing.  Apply the same tie penalty retroactively,
        # computed from the *opponent's* perspective so the advantage penalty
        # is correct for their side.
        if done and winner == "Tie":
            tie_pen = self._tie_reward(color=opponent_color)
            if opponent_color == BLUE:
                blue_adj += tie_pen
            else:
                red_adj += tie_pen

        # --- Switch turn ------------------------------------------------
        self.game.switch_turn()

        # Reset capture chain state
        self._capture_in_progress = False
        self._capturing_piece_sq = None
        self._visited_squares = set()
        self._current_move_chain = []
        self._is_capture_turn = False
        self._turn_start_board = None

        # Update mask for the next player (or empty if done)
        if not done:
            self._update_action_mask()
        else:
            self._action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        obs = self.get_board_state()
        info = {
            "turn": self.game.turn,
            "turn_complete": True,
            "winner": copy.deepcopy(winner) if winner else "None",
            "blue_reward_adjustment": blue_adj,
            "red_reward_adjustment": red_adj,
        }
        return obs, reward, done, False, info

    def _update_action_mask(self):
        """Rebuild the action mask for the current board state."""
        self._action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if self._capture_in_progress:
            # Only continuation captures from the mid-chain piece
            row, col = board_number_to_position(self._capturing_piece_sq)
            piece = self.game.board.get_piece(row, col)
            if piece != 0:
                captures = self.game.board.valid_moves_for_piece(
                    piece, row, col, capture_only=True
                )
                for (to_row, to_col) in captures:
                    if (to_row, to_col) not in self._visited_squares:
                        to_sq = position_to_board_number(to_row, to_col)
                        if to_sq is not None:
                            idx = encode_action(self._capturing_piece_sq, to_sq)
                            if idx >= 0:
                                self._action_mask[idx] = 1.0
        else:
            # All legal single-step moves for current player
            capture_mandatory = self.game.board.is_capture_possible(self.game.turn)
            for row in range(ROWS):
                for col in range(COLS):
                    piece = self.game.board.get_piece(row, col)
                    if piece != 0 and piece.color == self.game.turn:
                        if capture_mandatory:
                            moves = self.game.board.valid_moves_for_piece(
                                piece, row, col, capture_only=True
                            )
                        else:
                            moves = self.game.board.valid_moves_for_piece(
                                piece, row, col, capture_only=False
                            )
                        from_sq = position_to_board_number(row, col)
                        if from_sq is None:
                            continue
                        for (to_row, to_col) in moves:
                            to_sq = position_to_board_number(to_row, to_col)
                            if to_sq is not None:
                                idx = encode_action(from_sq, to_sq)
                                if idx >= 0:
                                    self._action_mask[idx] = 1.0

    def get_action_mask(self):
        """Return the current action mask (NUM_ACTIONS float array, 1.0 = valid)."""
        return self._action_mask.copy()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_board_state(self):
        """Return a normalized 4-channel board state from the current player's perspective.

        Channels:
          0: current player regular pieces
          1: current player king pieces
          2: opponent regular pieces
          3: opponent king pieces

        Board is flipped vertically for Red so the agent always sees pieces
        moving in the same direction.
        """
        board_state = np.zeros((4, 8, 8), dtype=np.float32)

        current_color = self.game.turn
        opponent_color = RED if current_color == BLUE else BLUE

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == current_color:
                        if piece.king:
                            board_state[1, row, col] = 1
                        else:
                            board_state[0, row, col] = 1
                    elif piece.color == opponent_color:
                        if piece.king:
                            board_state[3, row, col] = 1
                        else:
                            board_state[2, row, col] = 1

        if current_color == RED:
            board_state = np.flip(board_state, axis=1).copy()

        return board_state

    def fast_clone(self):
        """Create a lightweight clone of this environment for MCTS simulations.

        A full copy.deepcopy(env) also copies game.board_states (a dict with
        50+ nested-tuple entries by mid-game) and game.moves (50+ strings).
        Neither is needed for short MCTS simulations, but both are expensive
        to copy.  This method skips them and uses Piece.clone() / Board.clone()
        instead of generic deepcopy machinery.

        What is copied independently:
          - board.board grid  — each Piece gets its own clone (mutations in the
                                simulation cannot corrupt the real game state)
          - _action_mask      — numpy array (mutated in-place each step)
          - _visited_squares  — set (elements added during capture chains)
          - _current_move_chain — list (appended to during captures)

        What is intentionally reset / shared:
          - board_states → {}    (tie-by-repetition; MCTS sims too short to need it)
          - moves        → []    (logging only; not used for game logic)
          - num_moves    → copied as int; 250-move limit still tracked via
                           max(num_moves, len(moves)) in check_winner()
          - _turn_start_board → shared reference (read-only in _finish_turn();
                                replaced on the first step of any new turn)
          - reward config floats → shared (immutable scalars)
          - observation/action spaces → shared (stateless gym objects)
        """
        env = CheckersEnv.__new__(CheckersEnv)

        # ── Game object ────────────────────────────────────────────────────
        g = Game.__new__(Game)
        g.board         = self.game.board.clone()
        g.turn          = self.game.turn        # immutable tuple — safe to share

        # Skipped: board_states, moves — expensive and irrelevant for MCTS
        g.board_states  = {}
        g.moves         = []
        g.num_moves     = self.game.num_moves   # int — copy by value

        # GUI / move-tree state — not used by env.step(); zero cost to reset
        g.selected_piece     = None
        g.current_node       = None
        g.rootNode           = None
        g.move_in_progress   = True
        g.capture_in_progress = False
        g.capture_possible   = False
        g.move_chain         = []

        env.game = g

        # ── Capture-chain tracking ─────────────────────────────────────────
        env._capture_in_progress  = self._capture_in_progress
        env._capturing_piece_sq   = self._capturing_piece_sq
        env._visited_squares      = self._visited_squares.copy()
        env._current_move_chain   = self._current_move_chain[:]
        env._is_capture_turn      = self._is_capture_turn

        # _turn_start_board is only read (never mutated) in _finish_turn().
        # When _capture_in_progress is False, env.step() overwrites it on the
        # very first call, so the current value is never used.
        env._turn_start_board = self._turn_start_board

        # ── Action mask ───────────────────────────────────────────────────
        env._action_mask = self._action_mask.copy()

        # ── Reward config (immutable scalars) ─────────────────────────────
        env._capture_bonus      = self._capture_bonus
        env._capture_penalty    = self._capture_penalty
        env._king_bonus         = self._king_bonus
        env._tie_base           = self._tie_base
        env._shaping_scale      = self._shaping_scale
        env._time_penalty_scale = self._time_penalty_scale

        # ── Gymnasium spaces (stateless — safe to share) ──────────────────
        env.observation_space = self.observation_space
        env.action_space      = self.action_space

        return env

    def render(self):
        print("Board State")
        print(self.game.board)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def analyze_board(self):
        """Precompute useful board statistics."""
        stats = {
            "blue_pieces": 0, "red_pieces": 0,
            "blue_kings": 0, "red_kings": 0,
            "blue_positions": [], "red_positions": [],
            "blue_back_row": 0, "red_back_row": 0,
            "center_control": 0,
        }
        central_positions = {(3, 4), (4, 3)}

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == BLUE:
                        stats["blue_pieces"] += 1
                        stats["blue_positions"].append((row, col))
                        if piece.king:
                            stats["blue_kings"] += 1
                        if row == 0:
                            stats["blue_back_row"] += 1
                    elif piece.color == RED:
                        stats["red_pieces"] += 1
                        stats["red_positions"].append((row, col))
                        if piece.king:
                            stats["red_kings"] += 1
                        if row == 7:
                            stats["red_back_row"] += 1
                    if (row, col) in central_positions:
                        stats["center_control"] += 1 if piece.color == self.game.turn else 0
        return stats

    def reward_control_center(self, board_stats):
        return 0.5 * board_stats["center_control"]

    def reward_protect_rear(self, board_stats):
        if self.game.turn == BLUE:
            return 0.5 * board_stats["blue_back_row"]
        return 0.5 * board_stats["red_back_row"]

    def reward_balance(self, board_stats):
        positions = board_stats["blue_positions"] if self.game.turn == BLUE else board_stats["red_positions"]
        left = sum(1 for _, c in positions if c < COLS // 2)
        right = sum(1 for _, c in positions if c >= COLS // 2)
        return -0.1 * abs(left - right)

    def reward_for_kings(self, board_stats):
        if self.game.turn == BLUE:
            return board_stats["blue_kings"] * 0.2
        return board_stats["red_kings"] * 0.2

    def reward_king_promotion(self, old_board):
        if self._king_promoted(old_board.board):
            return self._king_bonus
        return 0

    def penalize_undefended_pieces(self, old_board, new_board):
        self.game.board = old_board
        old_undefended = self._enemy_capture_count()
        # new_board is already a deep copy (created in _finish_turn),
        # so no need to deep copy again.
        self.game.board = new_board
        new_undefended = self._enemy_capture_count()
        return max(old_undefended - new_undefended, 0) * 5 - new_undefended * 5

    def reward_end_game(self):
        winner = self.game.check_winner()
        if winner == self.game.turn:
            return 100, True, copy.deepcopy(self.game.turn)
        elif winner == "Tie":
            return self._tie_reward(), True, "Tie"
        elif winner is not None:
            return -100, True, copy.deepcopy(winner)
        return -np.sqrt(len(self.game.moves)) / 10 * self._time_penalty_scale, False, None
        # return 0, False, None

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------
    
    def _tie_reward(self, color=None):
        """Tie penalty that scales with material advantage and total pieces.

        Three components (all negative):
        1. Base penalty       – every tie is bad.
        2. Advantage penalty  – if *you* had more material and still drew,
           you failed to convert; extra penalty proportional to your edge.
        3. Stalling penalty   – more total pieces on the board means the game
           ended prematurely (repetition / move limit) instead of being
           played out.  Discourages passive play that leads to draws.

        Example outcomes (kings count as 1.5 pieces):
          Equal material, few pieces  (1v1):   -80 +  0 + -4  = -84
          Equal material, many pieces (10v10): -80 +  0 + -40 = -120
          Advantage, many pieces      (10v5):  -80 + -40 + -30 = -150
          Disadvantage, few pieces    (1v3):   -80 +  0 + -8  = -88

        Args:
            color: The color to compute the penalty for. Defaults to
                   self.game.turn (the acting player).
        """
        board = self.game.board.board
        my_color = color if color is not None else self.game.turn
        opp_color = RED if my_color == BLUE else BLUE

        my_reg = sum(1 for row in board for p in row
                     if p != 0 and p.color == my_color and not p.king)
        my_kings = sum(1 for row in board for p in row
                       if p != 0 and p.color == my_color and p.king)
        opp_reg = sum(1 for row in board for p in row
                      if p != 0 and p.color == opp_color and not p.king)
        opp_kings = sum(1 for row in board for p in row
                        if p != 0 and p.color == opp_color and p.king)

        my_material = my_reg + my_kings * 1.5
        opp_material = opp_reg + opp_kings * 1.5
        total_material = my_material + opp_material
        advantage = my_material - opp_material   # positive = I had more

        base = self._tie_base
        adv_penalty = -8 * max(advantage, 0)     # only the stronger side pays
        stall_penalty = -2 * total_material       # more pieces left = worse

        return base + adv_penalty + stall_penalty

    def _remaining_diff(self, board):
        blue_king = sum(1 for row in board for p in row if p != 0 and p.color == BLUE and p.king)
        blue_reg = sum(1 for row in board for p in row if p != 0 and p.color == BLUE and not p.king)
        red_king = sum(1 for row in board for p in row if p != 0 and p.color == RED and p.king)
        red_reg = sum(1 for row in board for p in row if p != 0 and p.color == RED and not p.king)
        diff = (blue_king - red_king) * 3 + (blue_reg - red_reg)
        return diff if self.game.turn == BLUE else -diff

    def _king_promoted(self, old_board):
        prev = sum(1 for row in old_board for p in row if p != 0 and p.color == self.game.turn and p.king)
        curr = sum(1 for row in self.game.board.board for p in row if p != 0 and p.color == self.game.turn and p.king)
        return curr > prev

    def _enemy_capture_count(self):
        """Count how many of the current player's pieces the opponent can capture."""
        count = 0
        seen = set()
        self.game.switch_turn()
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0 and piece.color == self.game.turn:
                    captures = self.game.board.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures and piece not in seen:
                        seen.add(piece)
                        count += 1
        self.game.switch_turn()
        return count
