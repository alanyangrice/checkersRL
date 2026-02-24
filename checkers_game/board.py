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

    def create_random_board(self, num_pieces_per_side, king_prob=0.15):
        """Create a board with random piece placement for curriculum learning.

        Args:
            num_pieces_per_side: Number of pieces each side gets (1-12).
            king_prob: Probability that a piece placed outside its home rows is a king.

        Blue pieces are placed on rows 0-4 (top half + middle).
        Red pieces are placed on rows 3-7 (bottom half + middle).
        Overlap zone (rows 3-4) can contain either color.
        """
        num_pieces_per_side = max(1, min(12, num_pieces_per_side))

        # Clear the board
        self.board = []
        for row in range(ROWS):
            self.board.append([0] * COLS)

        # Collect all dark squares
        all_dark = [(r, c) for r in range(ROWS) for c in range(COLS) if (r + c) % 2 == 1]

        # Blue placement candidates: rows 0-4
        blue_candidates = [(r, c) for r, c in all_dark if r <= 4]
        # Red placement candidates: rows 3-7
        red_candidates = [(r, c) for r, c in all_dark if r >= 3]

        # Place Blue pieces
        _random.shuffle(blue_candidates)
        blue_positions = blue_candidates[:num_pieces_per_side]

        # Place Red pieces (avoid squares already taken by Blue)
        blue_set = set(blue_positions)
        red_available = [pos for pos in red_candidates if pos not in blue_set]
        _random.shuffle(red_available)
        red_positions = red_available[:num_pieces_per_side]

        for row, col in blue_positions:
            piece = Piece(row, col, BLUE)
            # Promote to king if in opponent's territory or middle with some probability
            if row >= 3 and _random.random() < king_prob:
                piece.make_king()
            self.board[row][col] = piece

        for row, col in red_positions:
            piece = Piece(row, col, RED)
            # Promote to king if in opponent's territory or middle with some probability
            if row <= 4 and _random.random() < king_prob:
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
