from piece import Piece # Checkers Pieces
from MoveTree import MoveTree # Tree to build move tree
from MoveNode import MoveNode
from constants import ROWS, COLS, SQUARE_SIZE, RED, BLUE, WHITE, GREEN, BLACK, font, board_number_to_position, position_to_board_number
import pygame
import copy

class Board:
    def __init__(self):
        self.board = self.create_board()
        self.turn = BLUE  # Blue goes first

    def create_board(self):
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
        print(self.get_board_hash())
        screen.fill(BLACK)
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(screen)
        pygame.display.flip()

    def __str__(self):
        """Returns a string representation of the current board state."""
        board_str = ""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece == 0:
                    board_str += " . "  # Use a dot to represent an empty square
                else:
                    board_str += " " + str(piece) + " "  # Use the piece's string representation
            board_str += "\n"  # Newline after each row
        return board_str

    def move_piece(self, input_moves):
        """Handles normal piece moves without capturing."""
        if len(input_moves) != 2: # Invalid input move
            return False
        
        move_from = input_moves[0]
        move_to = input_moves[1]

        from_row, from_col = board_number_to_position(move_from)
        to_row, to_col = board_number_to_position(move_to)

        piece = self.board[from_row][from_col]
        if piece and piece.color == self.turn:
            print("Checking move piece")
            valid_moves = self.valid_moves_for_piece(piece, from_row, from_col)
            if (to_row, to_col) in valid_moves and valid_moves[(to_row, to_col)] is None:
                self.board[from_row][from_col] = 0
                piece.move(to_row, to_col)
                self.board[to_row][to_col] = piece
                if self.should_become_king(piece, to_row):
                    piece.make_king()
                #self.switch_turn()
                return True
            else:
                print("Invalid move.")
        return False
    
    def capture_piece(self, capture_moves):
        """Handles capturing pieces, including multi-captures."""
        from_row, from_col = board_number_to_position(capture_moves[0])
        piece = self.board[from_row][from_col]  # Get the piece to move

        if piece and piece.color == self.turn:
            visited_squares = set()  # Keep track of visited squares to prevent revisiting
            visited_squares.add((from_row, from_col))  # Add the starting position

            # Validate the entire capture sequence
            if not self.validate_capture_sequence(piece, capture_moves, visited_squares):
                return False

            # Switch turns after completing all captures
            #self.switch_turn()
            return True
        else:
            print("Invalid move: no piece selected or wrong turn.")
            return False

    def validate_capture_sequence(self, piece, capture_moves, visited_squares):
        """Validate the entire capture sequence before making any changes."""
        # Create a deep copy of the current state to simulate the moves
        original_board = copy.deepcopy(self.board)

        try:
            for i in range(len(capture_moves) - 1):
                from_square = capture_moves[i]
                to_square = capture_moves[i + 1]

                from_row, from_col = board_number_to_position(from_square)
                to_row, to_col = board_number_to_position(to_square)

                # Ensure we don't revisit a square
                if (to_row, to_col) in visited_squares:
                    raise Exception(f"Invalid move: revisited square at {to_row}, {to_col}.")

                # Validate the capture move
                if not self.valid_captures(piece, from_row, from_col, to_row, to_col):
                    raise Exception(f"Not a valid capture move.")

                # Simulate the capture on the original board
                self.perform_capture(piece, from_square, to_square)

                # Mark for promotion
                if self.should_become_king(piece, to_row):
                    piece.make_king()  # Promote the piece immediately for validation

                visited_squares.add((to_row, to_col))

            # After completing the capture sequence, check for more available captures
            final_row, final_col = board_number_to_position(capture_moves[-1])
            if self.all_captures_available(piece, final_row, final_col):
                raise Exception("More captures are available.")
            
        except Exception as e:
            # Restore the original board state if an error occurs
            self.board = original_board
            print(f"{e}")
            return False

        return True  # Valid sequence

    def all_captures_available(self, piece, row, col):
        """Check if more captures are available for the piece at the given position."""
        print("Checking all captures available")
        further_valid_moves = self.valid_moves_for_piece(piece, row, col, capture_only=True)
        return any(captured_pos is not None for captured_pos in further_valid_moves.values())

    def perform_capture(self, piece, from_square, to_square):
        """Perform a capture and update the board."""
        from_row, from_col = board_number_to_position(from_square)
        to_row, to_col = board_number_to_position(to_square)
        print("Checking perform capture")
        captured_pos = self.valid_moves_for_piece(piece, from_row, from_col).get((to_row, to_col))

        # Remove the piece from the old position
        self.board[from_row][from_col] = 0

        # Move the piece to the new position
        piece.move(to_row, to_col)
        self.board[to_row][to_col] = piece

        # Update the piece if it promotes to king
        if self.should_become_king(piece, to_row):
            piece.make_king()

        # Remove the captured piece right away to prevent recapture
        if captured_pos:
            self.remove_piece(captured_pos[0], captured_pos[1])

    def valid_captures(self, piece, from_row, from_col, to_row, to_col):
        """Check if a capture move is valid from the current piece's position."""
        print("Checking valid captures")
        valid_moves = self.valid_moves_for_piece(piece, from_row, from_col, capture_only=True)
        captured_pos = valid_moves.get((to_row, to_col))
        return captured_pos is not None  # Return True if a capture is valid

    def valid_moves_for_piece(self, piece, row, col, capture_only=False):
        """Generates valid moves (including captures) for a given piece at a specific position. When capture_only is True, it will only return capture moves."""
        moves = {}
        
        # Kings can move in all four directions; regular pieces move in one direction
        if piece.king:
            directions = [-1, 1]  # Kings can move both forward and backward
        else:
            directions = [piece.direction]  # Regular pieces move in one direction (forward)

        for d in directions:
            for dc in [-1, 1]:  # Check left and right diagonal
                r, c = row + d, col + dc
                # If capture_only is True, we ignore normal moves and only look for captures
                if not capture_only and 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == 0:
                    moves[(r, c)] = None  # Normal move
                # Check capture move (opponent's piece followed by an empty square)
                elif 0 <= r + d < ROWS and 0 <= c + dc < COLS and self.board[r][c] != 0 and self.board[r][c].color != piece.color and self.board[r + d][c + dc] == 0:
                    moves[(r + d, c + dc)] = (r, c)  # Capture move
        print(f"Moves: {moves}")
        return moves
    
    def is_capture_possible(self):
        """Returns True if any piece of the current player can capture; otherwise, returns False."""
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0 and piece.color == self.turn:  # Check only the pieces of the current player
                    # Get valid capture moves for this piece
                    print("Checking is capture possible")
                    captures = self.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures:  # If any valid captures exist
                        return True
        return False
    
    def should_become_king(self, piece, row):
        """Check if a piece should be promoted to a king."""
        if piece.color == RED and row == 0 and piece.king != True:  # Red pieces become kings when reaching row 0
            return True
        elif piece.color == BLUE and row == ROWS - 1 and piece.king != True:  # Blue pieces become kings when reaching the last row
            return True
        return False

    def remove_piece(self, row, col):
        """Removes a piece from the board."""
        self.board[row][col] = 0

    def switch_turn(self):
        """Switches turn between RED and BLUE."""
        self.turn = BLUE if self.turn == RED else RED

    def check_winner(self):
        """Checks for a winner by counting remaining pieces."""
        red_pieces = sum([1 for row in self.board for piece in row if isinstance(piece, Piece) and piece.color == RED])
        blue_pieces = sum([1 for row in self.board for piece in row if isinstance(piece, Piece) and piece.color == BLUE])

        if red_pieces == 0:
            return "Blue Wins!"
        elif blue_pieces == 0:
            return "Red Wins!"
        return None
    
    def get_board_hash(self):
        """Converts the board into a hashable format (tuple of tuples)."""
        return tuple(tuple(piece.color if piece != 0 else 0 for piece in row) for row in self.board)
    




    def get_all_possible_moves(self):
        """Returns a dictionary of all possible moves for each piece on the board and resulting board state."""
        print("In get_all_possible_moves")

        all_moves = set()
        original_board = copy.deepcopy(self.board)  # Make copy of original board
        capture_possible = self.is_capture_possible()  # Check if copy is possible

        # Iterate through the board to find all pieces
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0 and piece.color == self.turn:  # Only check pieces of the current player
                    print(f"Looking through tile: {position_to_board_number(row, col)}")

                    moveTree = MoveTree(position_to_board_number(row, col), original_board)  # Initialize tree

                    valid_moves = self.valid_moves_for_piece(piece, row, col, capture_possible)  # Get valid moves for this piece

                    print(f"Valid_moves: {valid_moves}")

                    visited_squares = set((row, col))  # Make sure recursion doesn't revisit the same square twice

                    if valid_moves and not capture_possible:  # Regular Move
                        for moves in valid_moves.keys():  # Simulate each move
                            self.move_piece((position_to_board_number(row, col), position_to_board_number(moves[0], moves[1])))
                            moveTree.add_node(
                                moveTree.root,  # Parent Node
                                position_to_board_number(moves[0], moves[1]),  # New Node
                                copy.deepcopy(self.board)  # Add board state
                            )
                            self.board = copy.deepcopy(original_board)  # Revert to original boardparent_node, child_value, board
                            print(self)
                        print("Possible Sequences:")
                        for sequence in moveTree.get_leaf_sequences(moveTree.root):
                            all_moves.add(sequence)
                            print(sequence)

                    elif valid_moves and capture_possible: # Capture Move
                        self._get_all_captures(piece, row, col, moveTree.root, valid_moves, visited_squares) # Recursively generate tree of all possible capture moves
                        print("Possible Sequences:")
                        for sequence in moveTree.get_leaf_sequences(moveTree.root, delimiter="x"):
                            all_moves.add(sequence)
                            print(sequence)
        print(all_moves)
        return all_moves
    
    def _get_all_captures(self, piece, row, col, parent_node, valid_moves, visited_squares):
        """Recursively go through all possible capture sequences"""
        if valid_moves: # Recursive condition
            original_board = copy.deepcopy(self.board) # Make copy of original board

            for key, val in valid_moves.items():
                if val != None and key not in visited_squares: # Make sure you don't visit same square twice
                    self.perform_capture(piece, position_to_board_number(row, col), position_to_board_number(key[0], key[1])) # Capture piece
                    
                    if self.should_become_king(piece, key[0]):
                        piece.make_king()  # Promote the piece immediately for validation

                    child_node = MoveNode(position_to_board_number(key[0], key[1]), copy.deepcopy(self.board))  # Create new node
                    parent_node.add_child(child_node)  # Add node to tree

                    new_visited_squares = copy.deepcopy(visited_squares) # Make copy of visited squares to make unmutable
                    new_visited_squares.add((key[0], key[1])) # Add visited square
                    new_piece = self.board[key[0]][key[1]] # Update piece location
                    new_valid_moves = self.valid_moves_for_piece(piece, key[0], key[1], capture_only=True)
                    self._get_all_captures(new_piece, key[0], key[1], child_node, new_valid_moves, new_visited_squares)
                    self.board = copy.deepcopy(original_board) # Revert back to board state before capture

        else: # Base condition
            return
