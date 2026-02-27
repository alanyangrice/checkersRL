import logging
import copy
import sys

import pygame

from checkers_game.MoveNode import MoveNode
from checkers_game.board import Board
from checkers_game.piece import Piece
from checkers_game.constants import (
    ROWS, COLS, SQUARE_SIZE, RED, BLUE, WHITE, GREEN, BLACK, YELLOW, GRAY,
    font, board_number_to_position, position_to_board_number
)

logger = logging.getLogger(__name__)


class Game:
    def __init__(self, no_progress_draw_moves=40):
        self.board = Board()

        self.turn = BLUE  # Blue starts first

        self.selected_piece = None
        self.current_node = None
        self.rootNode = None
        self.move_in_progress = True
        self.capture_in_progress = False
        self.capture_possible = False
        self.move_chain = []

        self.board_states = {}
        self.num_moves = 0
        self.moves = []

        # No-progress draw: declare a tie after this many consecutive full turns
        # without a capture or king promotion (WCDF 40-move rule).
        self._no_progress_draw_moves = no_progress_draw_moves
        self._no_progress_count      = 0   # turns since last capture/promotion
        self._prev_king_count        = 0   # for promotion detection each turn

    def switch_turn(self):
        """Switches the player's turn."""
        self.turn = BLUE if self.turn == RED else RED

    # Colours for highlighting the last completed move
    _HIGHLIGHT_FROM = (255, 215, 0)   # Gold  — source square
    _HIGHLIGHT_TO   = (0, 200, 100)   # Green — destination square

    def update_board(self, screen):
        """Redraws the board, highlights the last move, then updates the display."""
        self.board.draw(screen, update=False)
        self._draw_last_move_highlight(screen)
        pygame.display.update()

    def _draw_last_move_highlight(self, screen):
        """Draw border highlights on the from/to squares of the most recent move."""
        if not self.moves:
            return

        move_str = self.moves[-1]
        if "x" in move_str:
            squares = list(map(int, move_str.split("x")))
        elif "-" in move_str:
            squares = list(map(int, move_str.split("-")))
        else:
            return

        # Highlight source square (gold)
        fr, fc = board_number_to_position(squares[0])
        pygame.draw.rect(
            screen, self._HIGHLIGHT_FROM,
            (fc * SQUARE_SIZE, fr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4,
        )

        # Highlight destination square (green)
        tr, tc = board_number_to_position(squares[-1])
        pygame.draw.rect(
            screen, self._HIGHLIGHT_TO,
            (tc * SQUARE_SIZE, tr * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4,
        )

    def highlight_piece(self, screen):
        """Highlights the selected piece by drawing a yellow border around it."""
        if self.selected_piece:
            row, col = board_number_to_position(self.selected_piece)
            piece = self.board.get_piece(row, col)

            if piece:
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                radius = SQUARE_SIZE // 2 - piece.PADDING + piece.OUTLINE
                pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius - 3, 3)

    def show_piece_moves(self, screen, moves):
        """Shows all possible moves for a piece by highlighting target positions."""
        for move in moves:
            row, col = board_number_to_position(move)
            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(screen, GRAY, (center_x, center_y), SQUARE_SIZE // 4)

    def handle_click(self, row, col):
        """Handles piece selection based on user input."""
        piece = self.board.get_piece(row, col)
        board_number = position_to_board_number(row, col)

        logger.debug(f"Click at row={row}, col={col}, piece={piece}, board_number={board_number}")

        if board_number:
            if self.selected_piece is None:
                if piece != 0 and piece.color == self.turn:
                    return board_number, False
            else:
                if piece != 0 and piece.color == self.turn and board_number:
                    return board_number, False
                else:
                    return board_number, True

        return 0, False

    def select_piece(self, screen, board_number):
        """Selects a piece on the board and shows possible moves."""
        if board_number != 0:
            self.update_board(screen)

            self.selected_piece = board_number

            row, col = board_number_to_position(board_number)
            self.rootNode, self.capture_possible = self.get_all_piece_moves(row, col)
            self.current_node = self.rootNode

            if self.capture_in_progress:
                if self.rootNode and board_number not in self.move_chain and board_number in [child.position for child in self.rootNode.children]:
                    self.move_chain.append(board_number)
            else:
                if self.rootNode and self.capture_possible and board_number in [child.position for child in self.rootNode.children]:
                    self.move_chain.append(board_number)
                    self.capture_in_progress = self.capture_possible
                elif self.rootNode and self.capture_possible:
                    self.move_chain = [board_number]
                else:
                    self.move_chain = [board_number]

            self.highlight_piece(screen)
            if self.current_node:
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

                board_number, success_move = self.handle_click(row, col)

                if board_number == 0:
                    logger.debug("Invalid selection: No valid piece or move at clicked position.")
                    return False

                if self.capture_in_progress and success_move:
                    valid_move = self.make_move(screen, board_number)
                    self.update_board(screen)
                    if not valid_move:
                        logger.debug("Invalid capture move.")
                else:
                    if not success_move:
                        self.select_piece(screen, board_number)
                    else:
                        valid_move = self.make_move(screen, board_number)
                        self.update_board(screen)
                        if not valid_move:
                            logger.debug("Invalid move.")

                if not self.capture_in_progress and not self.move_in_progress:
                    self.moves.append(self.format_move_chain())
                    logger.debug(f"Move: {self.format_move_chain()}")
                    self.move_chain = []
                    self.move_in_progress = True
                    return True

    def make_move(self, screen, board_number):
        """Handles making a move or capture and checks for additional captures in a chain."""
        next_node = None

        if self.current_node:
            next_node = next((child for child in self.current_node.children if child.position == board_number), None)

        if next_node:
            self.move_chain.append(board_number)
            from_row, from_col = board_number_to_position(self.selected_piece)
            to_row, to_col = board_number_to_position(board_number)

            if self.capture_possible:
                self.board.capture_piece(from_row, from_col, to_row, to_col)
                self.capture_in_progress = True
            else:
                self.board.move_piece(from_row, from_col, to_row, to_col)
                self.capture_in_progress = False

            self.selected_piece = board_number
            self.current_node = next_node

            if self.capture_in_progress and self.current_node.children:
                self.update_board(screen)
                self.highlight_piece(screen)
                self.show_piece_moves(screen, [child.position for child in self.current_node.children])
                pygame.display.update()
            else:
                self.rootNode = None
                self.current_node = None
                self.capture_in_progress = False
                self.move_in_progress = False
                self.selected_piece = None
                self.update_board(screen)
        else:
            logger.debug("Invalid move: No matching move in the move tree.")

        return False

    def format_move_chain(self):
        """Formats the move chain into a string using '-' for regular moves and 'x' for captures."""
        if len(self.move_chain) < 2:
            return ""

        delimiter = "x" if self.capture_possible else "-"
        return delimiter.join(map(str, self.move_chain))

    def check_winner(self):
        """Checks for a winner or tie. Also checks if the current player has no legal moves."""
        if len(self.move_chain) > 1:
            self.num_moves += 1

        # Use the larger of num_moves (GUI) and len(self.moves) (RL env)
        # so the 250-move limit works in both contexts.
        move_count = max(self.num_moves, len(self.moves))
        if move_count >= 250:
            return "Tie"

        # No-progress tracking — always update the counter first so draw claims
        # below have the correct count, regardless of which condition fires.
        #
        # Captures: detected by 'x' in the last move string (appended to
        #   self.moves by both the GUI path and CheckersEnv before this call).
        # Promotions: detected by an increase in total king count vs last turn.
        #
        # Note: a capture *of* a king reduces the king count; that is already
        # caught by the 'x' check, so we only look for king-count *increases*.
        last_move     = self.moves[-1] if self.moves else ""
        curr_kings    = sum(1 for row in self.board.board
                            for p in row if isinstance(p, Piece) and p.king)
        was_progress  = ('x' in last_move) or (curr_kings > self._prev_king_count)
        self._prev_king_count = curr_kings
        if was_progress:
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1

        # Decisive checks before draw claims — a player cornering or eliminating
        # the opponent always wins, regardless of how long the game has taken.

        # Check for presence of pieces for each color
        red_pieces_found, blue_pieces_found = False, False
        for row in self.board.board:
            for piece in row:
                if isinstance(piece, Piece):
                    if piece.color == RED:
                        red_pieces_found = True
                    elif piece.color == BLUE:
                        blue_pieces_found = True
                if red_pieces_found and blue_pieces_found:
                    break
            if red_pieces_found and blue_pieces_found:
                break

        if not red_pieces_found:
            return BLUE
        if not blue_pieces_found:
            return RED

        # Check if the next player (opponent) has no legal moves — they lose.
        # check_winner() is always called BEFORE switch_turn(), so self.turn
        # is the player who just moved. The player who cannot move NEXT is the
        # opponent, not the current mover.
        next_player = RED if self.turn == BLUE else BLUE
        if not self.board.has_legal_moves(next_player):
            return self.turn  # current player wins; opponent is stuck

        # Draw claims — only reached when neither side has a decisive advantage.

        # No-progress draw (WCDF 40-move rule)
        if self._no_progress_count >= self._no_progress_draw_moves:
            return "Tie"

        # Board state repetition — key includes the side-to-move so that
        # "position P with Blue to move" and "position P with Red to move"
        # are counted as separate states (matching standard repetition rules).
        board_hash = (self.board.get_board_hash(), self.turn)
        self.board_states[board_hash] = self.board_states.get(board_hash, 0) + 1
        if self.board_states[board_hash] >= 5:
            return "Tie"

        return None

    def get_all_piece_moves(self, row, col):
        """Returns a move tree for a specific piece and whether captures are possible."""
        original_board = copy.deepcopy(self.board)
        capture_possible = self.board.is_capture_possible(self.turn)

        piece = self.board.get_piece(row, col)

        if piece != 0 and piece.color == self.turn:
            rootNode = MoveNode(position_to_board_number(row, col), original_board)

            valid_moves = self.board.valid_moves_for_piece(piece, row, col, capture_possible)
            visited_squares = set()
            visited_squares.add((row, col))

            if valid_moves and not capture_possible:  # Regular move
                for moves in valid_moves.keys():
                    self.board.move_piece(row, col, moves[0], moves[1])
                    newNode = MoveNode(position_to_board_number(moves[0], moves[1]), copy.deepcopy(self.board))
                    rootNode.add_child(newNode)
                    self.board = copy.deepcopy(original_board)

                return rootNode, capture_possible

            elif valid_moves and capture_possible:  # Capture move
                self._get_all_captures(piece, row, col, rootNode, valid_moves, visited_squares)
                return rootNode, capture_possible

        return None, False

    def get_all_possible_moves(self):
        """Returns a list of all possible moves for each piece on the board and resulting board states."""
        all_moves = []

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board.get_piece(row, col)
                if piece != 0 and piece.color == self.turn:
                    rootNode, capture_possible = self.get_all_piece_moves(row, col)

                    if rootNode:
                        delimiter = "x" if capture_possible else "-"
                        for sequence, board in rootNode.get_leaf_sequences(rootNode, delimiter=delimiter):
                            all_moves.append([sequence, board])
        return all_moves

    def _get_all_captures(self, piece, row, col, parent_node, valid_moves, visited_squares):
        """Recursively go through all possible capture sequences."""
        if valid_moves:
            original_board = copy.deepcopy(self.board)

            for key, val in valid_moves.items():
                if val is not None and key not in visited_squares:
                    self.board.capture_piece(row, col, key[0], key[1])

                    child_node = MoveNode(position_to_board_number(key[0], key[1]), copy.deepcopy(self.board))
                    parent_node.add_child(child_node)

                    new_visited_squares = copy.deepcopy(visited_squares)
                    new_visited_squares.add((key[0], key[1]))
                    new_piece = self.board.get_piece(key[0], key[1])
                    new_valid_moves = self.board.valid_moves_for_piece(piece, key[0], key[1], capture_only=True)
                    self._get_all_captures(new_piece, key[0], key[1], child_node, new_valid_moves, new_visited_squares)

                    self.board = copy.deepcopy(original_board)
        else:
            return
