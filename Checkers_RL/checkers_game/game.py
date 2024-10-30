import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

from checkers_game.MoveNode import MoveNode  # Tree to build move tree
from checkers_game.board import Board
from checkers_game.piece import Piece
from checkers_game.constants import ROWS, COLS, SQUARE_SIZE, RED, BLUE, WHITE, GREEN, BLACK, font, board_number_to_position, position_to_board_number
import pygame
import copy

class Game:
    def __init__(self):
        self.board = Board()  # Create new board

        self.turn = BLUE  # Blue starts first

        self.selected_piece = None  # Track selected piece for the current turn
        self.current_node = None  # Track current node for move tree navigation in capture sequences
        self.rootNode = None  # Root node for the move tree of the selected piece
        self.move_in_progress = True  # Track move success
        self.capture_in_progress = False  # Track if a capture sequence is ongoing
        self.capture_possible = False  # Indicates if a capture is available
        self.move_chain = []  # Track specific moves in a sequence

        self.board_states = {}  # Track board states for tie checking
        self.num_moves = 0  # Track total moves to check for ties
        self.moves = []  # Store all moves in the game for history/logging

    def switch_turn(self):
        """Switches the player's turn."""
        self.turn = BLUE if self.turn == RED else RED

    def update_board(self, screen):
        """Clears the board of previously highlighted selected piece and possible moves."""
        # Redraw squares to clear highlights
        self.board.draw(screen)
        pygame.display.update()

    def highlight_piece(self, screen):
        """Highlights the selected piece by drawing a yellow border around it."""
        if self.selected_piece:
            print("In highlight piece")
            row, col = board_number_to_position(self.selected_piece)
            piece = self.board.get_piece(row, col)

            # Only proceed if a piece is selected and exists on the board
            if piece:
                # Determine the center position and radius of the piece
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                radius = SQUARE_SIZE // 2 - piece.PADDING + piece.OUTLINE  # Slightly larger than the piece radius

                # Draw the yellow highlight circle around the piece
                pygame.draw.circle(screen, (255, 255, 0), (center_x, center_y), radius + 1, 3)  # Slightly larger for the outline

    def show_piece_moves(self, screen, moves):
        """Shows all possible moves for a piece by highlighting target positions."""
        print("in show piece moves")
        print(f"moves: {moves}")
        for move in moves:
            row, col = board_number_to_position(move)
            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(screen, (50, 50, 50), (center_x, center_y), SQUARE_SIZE // 4)  # Highlight possible move

    def handle_click(self, row, col):
        """Handles piece selection based on user input."""
        piece = self.board.get_piece(row, col)  # Check for a piece at the clicked position

        board_number = position_to_board_number(row, col)  # Get board number

        print(f"Attempting to select piece at Row: {row}, Col: {col}. Found piece: {piece}. Board number: {board_number}")  # Debug info
        print(f"Current turn: {'BLUE' if self.turn == BLUE else 'RED'}")  # Display current turn

        if board_number:
            if self.selected_piece is None:  # No piece currently selected, so select this one if it’s the current player’s turn
                if piece != 0 and piece.color == self.turn:
                    return board_number, False
            else:  # If a piece is already selected
                if piece != 0 and piece.color == self.turn and board_number:
                    return board_number, False  # User clicked another piece of the same color, so switch selection
                else:
                    return board_number, True  # User clicked on a square to try and move
        
        return 0, False
    
    def select_piece(self, screen, board_number):
        """Selects a piece on the board and shows possible moves."""        
        if board_number != 0:
            # Clear previous highlights and reset possible moves
            self.update_board(screen)
            
            self.selected_piece = board_number

            row, col = board_number_to_position(board_number)
            self.rootNode, self.capture_possible = self.get_all_piece_moves(row, col)
            self.current_node = self.rootNode  # Start at the root node of the move tree

            if self.capture_in_progress:
                if board_number not in self.move_chain:
                    self.move_chain.append(board_number)
            else:
                self.move_chain = [board_number]

            self.capture_in_progress = self.capture_possible  # Start capture chain if required

            self.highlight_piece(screen)
            self.show_piece_moves(screen, [child.position for child in self.current_node.children])
            pygame.display.update()
    
    def player_action(self, screen):
        """Main method to handle player actions on mouse click."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                row, col = mouse_pos[1] // SQUARE_SIZE, mouse_pos[0] // SQUARE_SIZE

                print(f"Row: {row}, Col: {col}")

                board_number, success_move = self.handle_click(row, col)

                print(f"board number: {board_number}, success_move: {success_move}")

                if board_number == 0:
                    print("Invalid selection: No valid piece or move at the clicked position.")
                    return False  # Ignore the click as it is not on a valid position
                
                # Process the click as either a move, capture, or piece selection
                if self.capture_in_progress and success_move:
                    print("In capture chain")
                    # Handle capture chain if it's in progress
                    valid_move = self.make_move(screen, board_number)
                    self.update_board(screen)
                    if not valid_move:
                        print("Invalid move: Please select a valid capture.")
                else:
                    print("In move or capture")
                    # Either select a new piece or make the initial move/capture
                    if not success_move:
                        print("In select_piece")
                        self.select_piece(screen, board_number)
                    else:
                        valid_move = self.make_move(screen, board_number)
                        self.update_board(screen)
                        if not valid_move:
                            print("Invalid move: Please select a valid move.")

                # If the player’s turn is complete, return True to indicate completion
                if not self.capture_in_progress and not self.move_in_progress:
                    self.moves.append(self.format_move_chain())  # Log the completed move chain
                    print(f"Move: {self.format_move_chain()}")
                    self.move_chain = []  # Reset for the next turn
                    self.move_in_progress = True
                    return True  # Player's turn is complete

    def make_move(self, screen, board_number):
        """Handles making a move or capture and checks for additional captures in a chain."""
        # Find the child node that matches the selected board position

        # Find the child node that matches the selected board position
        print(f"Attempting move to board number: {board_number}")
        print(f"Available moves in current node: {[child.position for child in self.current_node.children]}")  # Debug info
        
        next_node = next((child for child in self.current_node.children if child.position == board_number), None)

        if next_node:
            # Log the move to self.move_chain
            self.move_chain.append(board_number)
            print(f"Move chain: {self.move_chain}")
            from_row, from_col = board_number_to_position(self.selected_piece)
            to_row, to_col = board_number_to_position(board_number)

            print(f"Selected location: {self.selected_piece}, Move location: {board_number}")

            # Execute the move or capture based on tree structure
            if self.capture_possible:
                print("entered capture_piece")
                self.board.capture_piece(from_row, from_col, to_row, to_col)
                self.capture_in_progress = True  # Continue capture chain if more captures exist
            else:
                print("Entered move_piece")
                self.board.move_piece(from_row, from_col, to_row, to_col)
                self.capture_in_progress = False  # End turn if it's a regular move

            print("exited drawing function")

            # Update the selected piece and current node
            self.selected_piece = board_number
            self.current_node = next_node

            # Check if there are more capture moves to continue the chain
            if self.capture_in_progress and self.current_node.children:
                print("Further capture actions")
                # Clear previous highlights and reset possible moves
                self.update_board(screen)
                self.highlight_piece(screen)
                self.show_piece_moves(screen, [child.position for child in self.current_node.children])
                pygame.display.update()
            else:
                print("No further actions")
                # No further captures; end the capture chain
                self.rootNode = None
                self.current_node = None
                self.capture_in_progress = False
                self.move_in_progress = False
                self.selected_piece = None

                self.update_board(screen)

            #return True  # Move was valid
        else:
            print("Invalid move: No matching move in the move tree.")
        
        return False  # Move was invalid

    def format_move_chain(self):
        """Formats the move chain into a string using '-' for regular moves and 'x' for captures."""
        if len(self.move_chain) < 2:
            return ""  # No move to format if there's less than two positions
        
        is_capture = self.capture_possible  # Check if this turn involves captures
        
        # Choose delimiter based on capture status
        delimiter = "x" if is_capture else "-"

        return delimiter.join(map(str, self.move_chain))

    def check_winner(self):
        """Checks for a winner or tie."""
        # Move countx
        if len(self.move_chain) > 1:
            self.num_moves += 1

        if self.num_moves > 250:  # Tie by game length
            return "Tie"

        # Check board state
        board_hash = self.board.get_board_hash()
        self.board_states[board_hash] = self.board_states.get(board_hash, 0) + 1
        if self.board_states[board_hash] >= 3:  # Tie by repitition
            return "Tie"

        # Count pieces
        red_pieces = sum([1 for row in self.board.board for piece in row if isinstance(piece, Piece) and piece.color == RED])
        blue_pieces = sum([1 for row in self.board.board for piece in row if isinstance(piece, Piece) and piece.color == BLUE])

        if red_pieces == 0:
            return BLUE
        elif blue_pieces == 0:
            return RED
        return None
    
    def get_all_piece_moves(self, row, col):
        """Returns a list of all possible moves for a specific piece on the board and resulting board states."""
        original_board = copy.deepcopy(self.board)  # Make copy of original board
        capture_possible = self.board.is_capture_possible(self.turn)  # Check if copy is possible

        piece = self.board.get_piece(row, col)

        if piece != 0 and piece.color == self.turn:  # Only check pieces of the current player
            rootNode = MoveNode(position_to_board_number(row, col), original_board)  # Initialize tree

            valid_moves = self.board.valid_moves_for_piece(piece, row, col, capture_possible)  # Get valid moves for this piece
            visited_squares = set((row, col))  # Make sure recursion doesn't revisit the same square twice

            if valid_moves and not capture_possible:  # Regular Move
                for moves in valid_moves.keys():  # Simulate each move
                    self.board.move_piece(row, col, moves[0], moves[1])

                    newNode = MoveNode(position_to_board_number(moves[0], moves[1]), copy.deepcopy(self.board))  # New Node with board state
                    rootNode.add_child(newNode)  # Add node to root node

                    self.board = copy.deepcopy(original_board)  # Revert to original boardparent_node, child_value, board

                return rootNode, capture_possible

            elif valid_moves and capture_possible:  # Capture Move
                self._get_all_captures(piece, row, col, rootNode, valid_moves, visited_squares)  # Recursively generate tree of all possible capture moves
                return rootNode, capture_possible
            
        return rootNode, capture_possible
    
    def get_all_possible_moves(self):
        """Returns a list of all possible moves for each piece on the board and resulting board states."""
        all_moves = []

        # Iterate through the board to find all pieces
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board.get_piece(row, col)
                if piece != 0 and piece.color == self.turn:
                    rootNode, capture_possible = self.get_all_piece_moves(row, col)  # Get all moves for a specific piece

                    if rootNode:
                        delimiter = "x" if capture_possible else "-"
                        for sequence in rootNode.get_leaf_sequences(rootNode, delimiter=delimiter):
                            all_moves.append(sequence)  # Add piece moves to the end of all moves
        return all_moves

    def _get_all_captures(self, piece, row, col, parent_node, valid_moves, visited_squares):
        """Recursively go through all possible capture sequences"""
        if valid_moves:  # Recursive condition
            original_board = copy.deepcopy(self.board)  # Make copy of original board

            for key, val in valid_moves.items():
                if val != None and key not in visited_squares:  # Make sure you don't visit same square twice
                    self.board.capture_piece(row, col, key[0], key[1])  # Capture piece

                    child_node = MoveNode(position_to_board_number(key[0], key[1]), copy.deepcopy(self.board))  # Create new node
                    parent_node.add_child(child_node)  # Add node to tree

                    new_visited_squares = copy.deepcopy(visited_squares)  # Make copy of visited squares to make unmutable
                    new_visited_squares.add((key[0], key[1]))  # Add visited square
                    new_piece = self.board.get_piece(key[0], key[1])  # Update piece location
                    new_valid_moves = self.board.valid_moves_for_piece(piece, key[0], key[1], capture_only=True)
                    self._get_all_captures(new_piece, key[0], key[1], child_node, new_valid_moves, new_visited_squares)  # Recursive call

                    self.board = copy.deepcopy(original_board)  # Revert back to board state before capture

        else:  # Base condition
            return

    