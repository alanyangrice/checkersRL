"""Fast numpy-backed checkers environment for MCTS simulations.

Why a separate class?
---------------------
CheckersEnv keeps pieces as Python objects (Piece class) in a list-of-lists.
Cloning it means traversing and deep-copying that object graph plus the
game.board_states dict (50+ entries mid-game) and game.moves list.

NumpyCheckersEnv represents the board as a single (8, 8) int8 numpy array.
fast_clone() is therefore a single 64-byte memcpy plus a handful of scalar
copies — no object graph traversal at all.

PPO is unaffected: it continues to use CheckersEnv as-is.  MCTS converts
once per search via from_env(), then clones the numpy env for each of the
100 simulations.

Board encoding (int8)
---------------------
    0  — empty
    1  — BLUE regular piece
    2  — BLUE king
    3  — RED regular piece
    4  — RED king

BLUE pieces move in the +row direction (0 → 7).
RED  pieces move in the -row direction (7 → 0).
Kings move in both directions.

Interface
---------
The class exposes the same subset of CheckersEnv that MCTSSearch needs:
    step(action)       → (obs, 0.0, done, False, info)
    get_board_state()  → (4, 8, 8) float32  (identical output to CheckersEnv)
    get_action_mask()  → (NUM_ACTIONS,) float32
    game.turn          → current player colour (BLUE or RED tuple)
    fast_clone()       → independent NumpyCheckersEnv copy

Rewards are always 0.0 — MCTS ignores them entirely.
Board-repetition tie detection is omitted; the move cap is configurable.
"""

import types

import numpy as np

from checkers_game.constants import (
    BLUE, RED, ROWS, COLS,
    NUM_ACTIONS,
    encode_action, decode_action,
    board_number_to_position, position_to_board_number,
)

# ── Board cell constants ──────────────────────────────────────────────────────
EMPTY      = np.int8(0)
BLUE_PIECE = np.int8(1)
BLUE_KING  = np.int8(2)
RED_PIECE  = np.int8(3)
RED_KING   = np.int8(4)

_BLUE_CELLS = frozenset({1, 2})
_RED_CELLS  = frozenset({3, 4})
_KING_CELLS = frozenset({2, 4})


def _is_blue(cell):  return int(cell) in _BLUE_CELLS
def _is_red(cell):   return int(cell) in _RED_CELLS
def _is_king(cell):  return int(cell) in _KING_CELLS


def _board_hash_to_bytes(board_hash_tuple):
    """Convert Board.get_board_hash() tuple-of-tuples to NumpyCheckersEnv bytes.

    Board.get_board_hash() encodes each cell as:
        0                    — empty square
        (color_rgb, is_king) — piece (color is one of the RGB tuples from constants)

    We map to the same int8 encoding used by NumpyCheckersEnv:
        0=EMPTY, 1=BLUE_PIECE, 2=BLUE_KING, 3=RED_PIECE, 4=RED_KING

    Returns only the board bytes; callers pair this with the turn to form the
    full (bytes, turn) key used in _base_counts / _delta_counts.

    Used in from_env() to convert the full env.game.board_states history into
    the (bytes, turn)-keyed _base_counts dict so MCTS simulation sees complete history.
    """
    arr = np.zeros((ROWS, COLS), dtype=np.int8)
    for r, row in enumerate(board_hash_tuple):
        for c, cell in enumerate(row):
            if cell != 0:
                color, king = cell
                if color == BLUE:
                    arr[r, c] = BLUE_KING if king else BLUE_PIECE
                else:
                    arr[r, c] = RED_KING if king else RED_PIECE
    return arr.tobytes()


class NumpyCheckersEnv:
    """Lightweight checkers env for MCTS — numpy board, ultra-fast clone."""

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, env, move_cap=250, adjudicate_cap=False,
                 no_progress_count=0, no_progress_draw_moves=40):
        """Convert a running CheckersEnv snapshot to NumpyCheckersEnv.

        Called once per MCTS search call — the conversion cost is negligible
        compared with the 100 simulations that follow.

        Repetition seeding: the full env.game.board_states history is converted
        to bytes-keyed _base_counts so MCTS correctly handles any position that
        is one step from a 5th repetition — not just the root.  _base_counts is
        shared read-only across all simulation branches spawned by fast_clone().

        Args:
            move_cap:       Stop simulations at this many completed full turns
                            (should match the training-phase cap so MCTS sees
                            the same termination condition as the outer game loop).
            adjudicate_cap: When True, resolve cap terminations by material
                            count (matching _adjudicate_move_cap in training).
                            When False, cap terminations become Ties.
        """
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        for row in range(ROWS):
            for col in range(COLS):
                piece = env.game.board.board[row][col]
                if piece != 0:
                    if piece.color == BLUE:
                        board[row, col] = BLUE_KING if piece.king else BLUE_PIECE
                    else:
                        board[row, col] = RED_KING if piece.king else RED_PIECE

        # Seed base_counts from the full real-game repetition history.
        # game.board_states keys are (Board.get_board_hash() tuple, turn) pairs
        # so that positions with the same pieces but different sides to move are
        # counted separately (matching standard repetition rules).  We convert
        # the board portion to bytes and keep the turn, producing (bytes, turn)
        # keys that match _check_winner()'s (board.tobytes(), self._turn) keys.
        # Converting the entire board_states dict is O(history * 64) — negligible
        # compared with the simulations ahead.
        base_counts = {
            (_board_hash_to_bytes(board_hash), turn): cnt
            for (board_hash, turn), cnt in env.game.board_states.items()
            if cnt > 0
        }

        return cls._construct(
            board=board,
            turn=env.game.turn,
            move_count=len(env.game.moves),
            capture_in_progress=env._capture_in_progress,
            capturing_piece_sq=env._capturing_piece_sq,
            visited_squares=env._visited_squares.copy(),
            current_move_chain=env._current_move_chain[:],
            is_capture_turn=env._is_capture_turn,
            action_mask=env._action_mask.copy(),
            base_counts=base_counts,
            move_cap=move_cap,
            adjudicate_cap=adjudicate_cap,
            no_progress_count=no_progress_count,
            no_progress_draw_moves=no_progress_draw_moves,
        )

    @classmethod
    def _construct(cls, board, turn, move_count=0,
                   capture_in_progress=False, capturing_piece_sq=None,
                   visited_squares=None, current_move_chain=None,
                   is_capture_turn=False, action_mask=None,
                   base_counts=None, delta_counts=None,
                   move_cap=250, adjudicate_cap=False,
                   no_progress_count=0, no_progress_draw_moves=40):
        """Internal factory used by from_env() and fast_clone().

        Repetition tracking uses two dicts:
          _base_counts  — seeded from the real game; shared across all branches;
                          never written after construction (safe as shared ref).
          _delta_counts — positions visited within this simulation path only;
                          copied in fast_clone() so branches stay independent.
        """
        env = cls.__new__(cls)
        env._board               = board
        env._turn                = turn
        env._move_count          = move_count
        env._capture_in_progress = capture_in_progress
        env._capturing_piece_sq  = capturing_piece_sq
        env._visited_squares     = visited_squares or set()
        env._current_move_chain  = current_move_chain or []
        env._is_capture_turn     = is_capture_turn
        env._base_counts         = base_counts  if base_counts  is not None else {}
        env._delta_counts        = delta_counts if delta_counts is not None else {}
        env._move_cap               = move_cap
        env._adjudicate_cap         = adjudicate_cap
        env._no_progress_count      = no_progress_count
        env._no_progress_draw_moves = no_progress_draw_moves
        env._had_promotion          = False
        if action_mask is not None:
            env._action_mask = action_mask
        else:
            env._action_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            env._update_action_mask()
        env._game_proxy = types.SimpleNamespace(turn=turn)
        return env

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def game(self):
        """Proxy object so env.game.turn works identically to CheckersEnv."""
        self._game_proxy.turn = self._turn
        return self._game_proxy

    def get_action_mask(self):
        return self._action_mask.copy()

    def get_board_state(self):
        """Return the same (4, 8, 8) float32 observation as CheckersEnv.

        Channels (always from current player's perspective):
            0: current player regular pieces
            1: current player king pieces
            2: opponent regular pieces
            3: opponent king pieces

        The board is flipped vertically for RED so both players always see
        their pieces advancing upward — identical to CheckersEnv convention.
        """
        b = self._board
        state = np.zeros((4, ROWS, COLS), dtype=np.float32)
        if self._turn == BLUE:
            state[0] = (b == BLUE_PIECE)
            state[1] = (b == BLUE_KING)
            state[2] = (b == RED_PIECE)
            state[3] = (b == RED_KING)
        else:
            state[0] = np.flip(b == RED_PIECE,  axis=0)
            state[1] = np.flip(b == RED_KING,   axis=0)
            state[2] = np.flip(b == BLUE_PIECE, axis=0)
            state[3] = np.flip(b == BLUE_KING,  axis=0)
        return state.astype(np.float32)

    def step(self, action):
        """Apply one semantic action, mirroring CheckersEnv.step().

        Returns: (obs, 0.0, done, False, info)

        info keys used by MCTSSearch:
            "winner"       — colour tuple, "Tie", or "None"
            "turn_complete"— False during a multi-jump chain, True otherwise
        """
        # ── No legal moves: current player loses ──────────────────────────
        if self._action_mask.sum() == 0:
            loser  = self._turn
            winner = BLUE if loser == RED else RED
            self._turn = winner
            # Update mask for the winner (mirrors CheckersEnv.step() which
            # calls _update_action_mask() after switching to the winner's turn)
            self._update_action_mask()
            return (
                self.get_board_state(), 0.0, True, False,
                {"turn": self._turn, "turn_complete": True, "winner": winner},
            )

        # ── Validate action ─────────────────────────────────────────────────
        # NumpyCheckersEnv is used inside MCTS simulations where actions come
        # from expanded children — an illegal action here means a tree bug.
        assert self._action_mask[action] != 0, (
            f"MCTS selected illegal action {action} "
            f"(mask sum={self._action_mask.sum()}, turn={self._turn})"
        )

        from_sq, to_sq         = decode_action(action)
        from_row, from_col     = board_number_to_position(from_sq)
        to_row,   to_col       = board_number_to_position(to_sq)
        is_capture             = abs(from_row - to_row) == 2

        # ── Snapshot start-of-turn state ──────────────────────────────────
        if not self._capture_in_progress:
            self._current_move_chain = [from_sq]
            self._visited_squares    = {(from_row, from_col)}
            self._is_capture_turn    = is_capture

        # ── Apply the move ────────────────────────────────────────────────
        cell = self._board[from_row, from_col]
        self._board[to_row,   to_col]   = cell
        self._board[from_row, from_col] = EMPTY

        if is_capture:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            self._board[mid_row, mid_col] = EMPTY

        # King promotion
        if cell == BLUE_PIECE and to_row == ROWS - 1:
            self._board[to_row, to_col] = BLUE_KING
            self._had_promotion = True
        elif cell == RED_PIECE and to_row == 0:
            self._board[to_row, to_col] = RED_KING
            self._had_promotion = True

        self._current_move_chain.append(to_sq)
        self._visited_squares.add((to_row, to_col))

        # ── Check for capture-chain continuation ──────────────────────────
        if is_capture:
            cont = self._valid_moves_for_piece(to_row, to_col, capture_only=True)
            cont = {k: v for k, v in cont.items()
                    if k not in self._visited_squares}
            if cont:
                self._capture_in_progress = True
                self._capturing_piece_sq  = to_sq
                self._update_action_mask()
                return (
                    self.get_board_state(), 0.0, False, False,
                    {"turn": self._turn, "turn_complete": False,
                     "winner": "None"},
                )

        return self._finish_turn()

    def fast_clone(self):
        """Return an independent copy — board is a single 64-byte numpy memcpy.

        This is the core performance gain over copy.deepcopy(CheckersEnv):
        no Python object graph traversal, no deep dict copy.

        Repetition dicts:
          _base_counts  — shared reference; never written, so no copy needed.
          _delta_counts — per-branch; shallow-copied so branches accumulate
                          counts independently (values are plain ints).
        """
        c = NumpyCheckersEnv.__new__(NumpyCheckersEnv)
        c._board               = self._board.copy()       # 64-byte memcpy
        c._turn                = self._turn
        c._move_count          = self._move_count
        c._capture_in_progress = self._capture_in_progress
        c._capturing_piece_sq  = self._capturing_piece_sq
        c._visited_squares     = self._visited_squares.copy()
        c._current_move_chain  = self._current_move_chain[:]
        c._is_capture_turn     = self._is_capture_turn
        c._action_mask         = self._action_mask.copy()
        c._base_counts         = self._base_counts          # shared, read-only
        c._delta_counts        = self._delta_counts.copy()  # per-branch
        c._move_cap               = self._move_cap
        c._adjudicate_cap         = self._adjudicate_cap
        c._no_progress_count      = self._no_progress_count
        c._no_progress_draw_moves = self._no_progress_draw_moves
        c._had_promotion          = self._had_promotion
        c._game_proxy             = types.SimpleNamespace(turn=c._turn)
        return c

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finish_turn(self):
        self._move_count += 1

        # No-progress tracking: reset on any capture or promotion, else increment.
        # Must be evaluated before _is_capture_turn and _had_promotion are cleared.
        if self._is_capture_turn or self._had_promotion:
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1
        self._had_promotion = False

        self._capture_in_progress  = False
        self._capturing_piece_sq   = None
        self._visited_squares      = set()
        self._current_move_chain   = []
        self._is_capture_turn      = False

        winner = self._check_winner()
        done   = winner is not None

        self._turn = BLUE if self._turn == RED else RED

        if done:
            self._action_mask[:] = 0.0
        else:
            self._update_action_mask()

        return (
            self.get_board_state(), 0.0, done, False,
            {"turn": self._turn, "turn_complete": True,
             "winner": winner if winner is not None else "None"},
        )

    def _check_winner(self):
        """Return winner colour, 'Tie', or None (game continues).

        Check order — decisive results first, draws last:
          1. Move cap (_move_cap) → material adjudication or Tie.
          2. One side has no pieces → other side wins.
          3. Next player has no legal moves → current player wins.
          4. 5-fold repetition → Tie.

        Wins before repetition ensures a terminal win is never downgraded to a
        draw.  (In practice pieces can only disappear via captures, so a
        piece-count win is always a fresh board state and can never be a
        repetition.  The legal-moves win could theoretically coincide with a
        repeated position, but checking wins first matches standard rules.)

        Hash key: (board bytes, turn) — includes the side-to-move so that
        "position P with Blue to move" and "position P with Red to move" are
        counted as separate states (matching standard repetition rules and
        game.check_winner()'s (Board.get_board_hash(), turn) keys).

        Two-dict design (base + delta):
          _base_counts  — seeded from the real game's count for the root
                          position; shared across all branches, never written.
          _delta_counts — counts added within this simulation path only;
                          copied in fast_clone() so branches stay independent.
        """
        if self._move_count >= self._move_cap:
            return self._adjudicate_by_material() if self._adjudicate_cap else "Tie"

        # Decisive checks before draw claims — a player cornering or eliminating
        # the opponent always wins, regardless of how long the game has taken.
        has_blue = bool(np.any((self._board == BLUE_PIECE) |
                               (self._board == BLUE_KING)))
        has_red  = bool(np.any((self._board == RED_PIECE)  |
                               (self._board == RED_KING)))

        if not has_blue:
            return RED
        if not has_red:
            return BLUE

        next_player = BLUE if self._turn == RED else RED
        if not self._board_has_legal_moves(next_player):
            return self._turn

        # Draw claims — only reached when neither side has a decisive advantage.

        # No-progress draw: N consecutive turns without a capture or promotion.
        # Mirrors the WCDF 40-move rule; eliminates king-oscillation endgames.
        if self._no_progress_count >= self._no_progress_draw_moves:
            return "Tie"

        # Repetition draw ──────────────────────────────────────────────────
        board_hash  = (self._board.tobytes(), self._turn)
        delta_after = self._delta_counts.get(board_hash, 0) + 1
        self._delta_counts[board_hash] = delta_after
        if self._base_counts.get(board_hash, 0) + delta_after >= 5:
            return "Tie"

        return None

    def _adjudicate_by_material(self):
        """Resolve a cap termination by piece count, mirroring _adjudicate_move_cap.

        Kings count as KING_MATERIAL_VALUE regular pieces.
        Equal material → "Tie".
        """
        from RL_models.MCTS import training_config as cfg  # local import avoids circular dep
        b = self._board
        blue_mat = (float(np.sum(b == BLUE_PIECE)) +
                    cfg.KING_MATERIAL_VALUE * float(np.sum(b == BLUE_KING)))
        red_mat  = (float(np.sum(b == RED_PIECE)) +
                    cfg.KING_MATERIAL_VALUE * float(np.sum(b == RED_KING)))
        if blue_mat > red_mat:
            return BLUE
        elif red_mat > blue_mat:
            return RED
        return "Tie"

    def _board_has_legal_moves(self, turn):
        capture_mandatory = self._is_capture_possible(turn)
        for row in range(ROWS):
            for col in range(COLS):
                cell = int(self._board[row, col])
                if cell == 0:
                    continue
                if turn == BLUE and not _is_blue(cell):
                    continue
                if turn == RED and not _is_red(cell):
                    continue
                if self._valid_moves_for_piece(row, col,
                                               capture_only=capture_mandatory):
                    return True
        return False

    def _is_capture_possible(self, turn):
        for row in range(ROWS):
            for col in range(COLS):
                cell = int(self._board[row, col])
                if cell == 0:
                    continue
                if turn == BLUE and not _is_blue(cell):
                    continue
                if turn == RED and not _is_red(cell):
                    continue
                if self._valid_moves_for_piece(row, col, capture_only=True):
                    return True
        return False

    def _valid_moves_for_piece(self, row, col, capture_only=False):
        """Return {(to_row, to_col): captured_pos_or_None} for piece at (row, col).

        Mirrors Board.valid_moves_for_piece(), operating on the int8 array.
        """
        cell = int(self._board[row, col])
        if cell == 0:
            return {}

        is_blue      = _is_blue(cell)
        is_king_cell = _is_king(cell)

        # Pieces can only move in their direction; kings move both ways.
        if is_king_cell:
            row_dirs = (-1, 1)
        elif is_blue:
            row_dirs = (1,)    # BLUE moves down (increasing row)
        else:
            row_dirs = (-1,)   # RED moves up (decreasing row)

        moves = {}
        for dr in row_dirs:
            for dc in (-1, 1):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                target = int(self._board[nr, nc])
                if target == 0:
                    if not capture_only:
                        moves[(nr, nc)] = None           # regular move
                else:
                    target_is_blue = _is_blue(target)
                    if target_is_blue == is_blue:
                        continue                          # friendly piece
                    lr, lc = row + 2 * dr, col + 2 * dc
                    if (0 <= lr < ROWS and 0 <= lc < COLS
                            and self._board[lr, lc] == 0):
                        moves[(lr, lc)] = (nr, nc)       # capture
        return moves

    def _update_action_mask(self):
        """Rebuild the action mask — identical logic to CheckersEnv."""
        self._action_mask[:] = 0.0

        if self._capture_in_progress:
            row, col = board_number_to_position(self._capturing_piece_sq)
            if self._board[row, col] != EMPTY:
                moves = self._valid_moves_for_piece(row, col, capture_only=True)
                for (to_row, to_col) in moves:
                    if (to_row, to_col) not in self._visited_squares:
                        to_sq = position_to_board_number(to_row, to_col)
                        if to_sq is not None:
                            idx = encode_action(self._capturing_piece_sq, to_sq)
                            if idx >= 0:
                                self._action_mask[idx] = 1.0
        else:
            capture_mandatory = self._is_capture_possible(self._turn)
            for row in range(ROWS):
                for col in range(COLS):
                    cell = int(self._board[row, col])
                    if cell == 0:
                        continue
                    if self._turn == BLUE and not _is_blue(cell):
                        continue
                    if self._turn == RED and not _is_red(cell):
                        continue
                    moves = self._valid_moves_for_piece(
                        row, col, capture_only=capture_mandatory
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
