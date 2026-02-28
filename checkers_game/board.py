import random as _random

from checkers_game.piece import Piece
from checkers_game.constants import ROWS, COLS, SQUARE_SIZE, RED, BLUE, WHITE, GREEN, BLACK, font
import pygame


class Board:
    def __init__(self):
        self.board = self.create_board()

    def create_board(self):
        """Create the board."""
        board = []
        for row in range(ROWS):
            board.append([])
            for col in range(COLS):
                if (row + col) % 2 == 1:  # Only place pieces on dark squares
                    if row < 3:
                        board[row].append(Piece(row, col, BLUE))  # Blue pieces on top
                    elif row > 4:
                        board[row].append(Piece(row, col, RED))  # Red pieces on bottom
                    else:
                        board[row].append(0)
                else:
                    board[row].append(0)
        return board

    def draw_squares(self, screen):
        """Draw squares on the checkers board."""
        square_number = 1
        for row in range(ROWS):
            for col in range(COLS):
                color = GREEN if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                if (row + col) % 2 == 1:
                    text = font.render(str(square_number), True, WHITE)
                    screen.blit(text, (col * SQUARE_SIZE + 10, row * SQUARE_SIZE + 10))
                    square_number += 1

    def draw(self, screen, update=True):
        """Draw the checkers board.

        Args:
            screen: The pygame surface to draw on.
            update: If True (default), call pygame.display.update() after drawing.
        """
        screen.fill(BLACK)
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                if piece != 0:
                    piece.draw(screen)
        if update:
            pygame.display.update()

    def __str__(self):
        """Returns a string representation of the current board state."""
        board_str = ""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                if piece == 0:
                    board_str += " . "
                else:
                    board_str += " " + str(piece) + " "
            board_str += "\n"
        return board_str

    def get_board_hash(self):
        """Converts the board into a hashable format (tuple of tuples).
        Includes both piece color and king status to avoid false tie detection."""
        def piece_key(piece):
            if piece == 0:
                return 0
            return (piece.color, piece.king)
        return tuple(tuple(piece_key(piece) for piece in row) for row in self.board)

    def get_piece(self, row, col):
        """Returns the piece at the specified row and column."""
        return self.board[row][col]

    def remove_piece(self, row, col):
        """Removes a piece from the board."""
        self.board[row][col] = 0

    def move_piece(self, from_row, from_col, to_row, to_col):
        """Move piece from one location to another on the board."""
        piece = self.get_piece(from_row, from_col)
        piece.move(to_row, to_col)
        self.board[to_row][to_col] = piece
        self.remove_piece(from_row, from_col)

        if self.should_become_king(piece, to_row):
            piece.make_king()

    def capture_piece(self, from_row, from_col, to_row, to_col):
        """Perform a capture and update the board."""
        piece = self.get_piece(from_row, from_col)
        self.move_piece(from_row, from_col, to_row, to_col)
        self.remove_piece((from_row + to_row) // 2, (from_col + to_col) // 2)

        if self.should_become_king(piece, to_row):
            piece.make_king()

    def should_become_king(self, piece, row):
        """Check if a piece should be promoted to a king."""
        if piece.color == RED and row == 0 and not piece.king:
            return True
        elif piece.color == BLUE and row == ROWS - 1 and not piece.king:
            return True
        return False

    def is_capture_possible(self, turn):
        """Returns True if any piece of the current player can capture."""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0 and piece.color == turn:
                    captures = self.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures:
                        return True
        return False

    def valid_moves_for_piece(self, piece, row, col, capture_only=False):
        """Generates valid moves (including captures) for a given piece at a specific position."""
        moves = {}

        if piece.king:
            directions = [-1, 1]
        else:
            directions = [piece.direction]

        for d in directions:
            for dc in [-1, 1]:  # Check left and right diagonal
                r, c = row + d, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    if not capture_only and self.get_piece(r, c) == 0:
                        moves[(r, c)] = None  # Normal move
                    elif 0 <= r + d < ROWS and 0 <= c + dc < COLS:
                        target_piece = self.get_piece(r, c)
                        landing_spot = self.get_piece(r + d, c + dc)
                        if target_piece != 0 and target_piece.color != piece.color and landing_spot == 0:
                            moves[(r + d, c + dc)] = (r, c)  # Capture move
        return moves

    # King probability by row for each colour.
    # Indexed by row 0-7; values reflect how likely a piece at that depth is to
    # be a king.  Pieces at the promotion row are FORCED kings (probability 1.0)
    # to prevent stuck non-king pieces with no legal moves.
    #
    # BLUE promotes at row 7 (moves downward: 0 → 7).
    # Deeper in RED territory → higher probability.
    _BLUE_KING_PROB = [0.05, 0.08, 0.12, 0.20, 0.35, 0.60, 0.80, 1.00]
    #                 row0  row1  row2  row3  row4  row5  row6  row7
    #                 own  ←──── home ──────center──── deep enemy ──→ promo

    # RED promotes at row 0 (moves upward: 7 → 0).
    _RED_KING_PROB  = [1.00, 0.80, 0.60, 0.35, 0.20, 0.12, 0.08, 0.05]
    #                 row0  row1  row2  row3  row4  row5  row6  row7
    #                 promo ←── deep enemy ────center──── home ────→ own

    def create_random_board(self, num_pieces_per_side, king_prob=None,
                            num_blue=None, num_red=None):
        """Create a random board for curriculum learning.

        Pieces are placed on any dark square across the full 8×8 board for
        both colours.  King status is assigned with a probability that scales
        with depth into opponent territory so the distribution resembles real
        mid-game / endgame positions:

          • A BLUE piece at its promotion row (row 7) is **always** a king —
            a non-king there would have no legal forward moves and freeze the
            game.
          • A BLUE piece at rows 5-6 (deep in RED's half) is likely a king
            (~60-80 %) since reaching that depth without promoting is rare.
          • Pieces in the centre or home rows are mostly non-kings.

        Args:
            num_pieces_per_side: Default piece count for both sides (1-12).
                Used when num_blue/num_red are not specified.
            king_prob: Ignored (kept for API compatibility).  King probability
                is now position-dependent; see _BLUE_KING_PROB / _RED_KING_PROB.
            num_blue: Override piece count for BLUE (None → use num_pieces_per_side).
            num_red:  Override piece count for RED  (None → use num_pieces_per_side).
        """
        n_blue = max(1, min(12, num_blue if num_blue is not None else num_pieces_per_side))
        n_red  = max(1, min(12, num_red  if num_red  is not None else num_pieces_per_side))

        self.board = [[0] * COLS for _ in range(ROWS)]

        # All 32 dark squares are available to both sides.
        all_dark = [(r, c) for r in range(ROWS) for c in range(COLS)
                    if (r + c) % 2 == 1]
        _random.shuffle(all_dark)

        # Assign squares: first n_blue to BLUE, then n_red from the remainder.
        # This guarantees no square overlap between the two sides.
        blue_positions = all_dark[:n_blue]
        remainder      = all_dark[n_blue:]
        _random.shuffle(remainder)
        red_positions  = remainder[:n_red]

        for row, col in blue_positions:
            piece = Piece(row, col, BLUE)
            if _random.random() < self._BLUE_KING_PROB[row]:
                piece.make_king()
            self.board[row][col] = piece

        for row, col in red_positions:
            piece = Piece(row, col, RED)
            if _random.random() < self._RED_KING_PROB[row]:
                piece.make_king()
            self.board[row][col] = piece

    def clone(self):
        """Return a fast independent copy of the board for MCTS simulations.

        Uses Piece.clone() on every occupied cell instead of Python's generic
        deepcopy machinery, skipping the attribute-dict traversal overhead.
        The result is a fully independent Board whose pieces can be mutated
        without affecting the original.
        """
        b = Board.__new__(Board)
        b.board = [
            [cell.clone() if cell != 0 else 0 for cell in row]
            for row in self.board
        ]
        return b

    def has_legal_moves(self, turn):
        """Returns True if the given player has any legal moves available."""
        capture_possible = self.is_capture_possible(turn)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0 and piece.color == turn:
                    moves = self.valid_moves_for_piece(piece, row, col, capture_only=capture_possible)
                    if moves:
                        return True
        return False
