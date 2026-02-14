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

    def draw(self, screen):
        """Draw the checkers board."""
        screen.fill(BLACK)
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                if piece != 0:
                    piece.draw(screen)
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
