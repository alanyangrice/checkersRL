import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

from checkers_game.piece import Piece  # Checkers Pieces
from checkers_game.constants import ROWS, COLS, SQUARE_SIZE, RED, BLUE, WHITE, GREEN, BLACK, font
import pygame

class Board:
    def __init__(self):
        self.board = self.create_board()

    def create_board(self):
        """Create the board"""
        board = []
        for row in range(ROWS):
            board.append([])
            for col in range(COLS):
                if (row + col) % 2 == 1:  # Only place pieces on black squares
                    if row < 3:
                        board[row].append(Piece(row, col, BLUE))  # Blue (black) pieces on top
                    elif row > 4:
                        board[row].append(Piece(row, col, RED))  # Red pieces on bottom
                    else:
                        board[row].append(0)
                else:
                    board[row].append(0)
        return board
    
    def draw_squares(self, screen):
        """Draw squares on the checkers board"""
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
        """Draw the checkers board"""
        screen.fill(BLACK)
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                if piece != 0:
                    piece.draw(screen)
        pygame.display.update()

    def __str__(self):
        """Returns a string representation of the current board state"""
        board_str = ""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                if piece == 0:
                    board_str += " . "  # Use a dot to represent an empty square
                else:
                    board_str += " " + str(piece) + " "  # Use the piece's string representation
            board_str += "\n"  # Newline after each row
        return board_str
    
    def get_board_hash(self):
        """Converts the board into a hashable format (tuple of tuples)."""
        return tuple(tuple(piece.color if piece != 0 else 0 for piece in row) for row in self.board)

    def get_piece(self, row, col):
        """Returns the piece at the specified row and column"""
        return self.board[row][col]
    
    def remove_piece(self, row, col):
        """Removes a piece from the board."""
        self.board[row][col] = 0
    
    def move_piece(self, from_row, from_col, to_row, to_col):
        """Move piece from one location to another on the board"""
        piece = self.get_piece(from_row, from_col)
        piece.move(to_row, to_col)
        self.board[to_row][to_col] = piece
        self.remove_piece(from_row, from_col)
        
        if self.should_become_king(piece, to_row):  # Update the piece if it promotes to king
            piece.make_king()

    def capture_piece(self, from_row, from_col, to_row, to_col):
        """Perform a capture and update the board."""
        piece = self.get_piece(from_row, from_col)
        self.move_piece(from_row, from_col, to_row, to_col)  # Move piece
        self.remove_piece((from_row + to_row) // 2, (from_col + to_col) // 2)  # Remove captured piece

        if self.should_become_king(piece, to_row):  # Update the piece if it promotes to king
            piece.make_king()

    def should_become_king(self, piece, row):
        """Check if a piece should be promoted to a king."""
        if piece.color == RED and row == 0 and piece.king != True:  # Red pieces become kings when reaching row 0
            return True
        elif piece.color == BLUE and row == ROWS - 1 and piece.king != True:  # Blue pieces become kings when reaching the last row
            return True
        return False
    
    def is_capture_possible(self, turn):
        """Returns True if any piece of the current player can capture; otherwise, returns False."""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0 and piece.color == turn:  # Check only the pieces of the current player
                    # Get valid capture moves for this piece
                    captures = self.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures:  # If any valid captures exist
                        return True
        return False
    
    def valid_moves_for_piece(self, piece, row, col, capture_only=False):
        """Generates valid moves (including captures) for a given piece at a specific position. When capture_only is True, it will only return capture moves."""
        moves = {}
        
        if piece.king:
            directions = [-1, 1]  # Kings can move both forward and backward
        else:
            directions = [piece.direction]  # Regular pieces move in one direction (forward)

        for d in directions:
            for dc in [-1, 1]:  # Check left and right diagonal
                r, c = row + d, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    # If capture_only is True, we ignore normal moves and only look for captures
                    if not capture_only and self.get_piece(r, c) == 0:
                        moves[(r, c)] = None  # Normal move
                    # Check capture move (opponent's piece followed by an empty square)
                    elif 0 <= r + d < ROWS and 0 <= c + dc < COLS:
                        target_piece = self.get_piece(r, c)
                        landing_spot = self.get_piece(r + d, c + dc)
                        if target_piece != 0 and target_piece.color != piece.color and landing_spot == 0:
                            moves[(r + d, c + dc)] = (r, c)  # Capture move
        return moves