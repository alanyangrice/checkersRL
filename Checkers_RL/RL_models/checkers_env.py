import numpy as np
import gym
import copy
from gym import spaces
from checkers_game.board import Board
from checkers_game.piece import Piece
from checkers_game.constants import RED, BLUE, ROWS, COLS, board_number_to_position, position_to_board_number

class CheckersEnv(gym.Env):
    def __init__(self):
        super(CheckersEnv, self).__init__()

        # Intialize game in environment
        self.checkers = Board()

        # Check if tie by repitition
        self.board_state = {}
        
        # Define the observation space
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8, 8), dtype=np.int8)

        # Define action space with fixed max number of actions
        self.max_actions = 50
        self.action_space = spaces.Discrete(self.max_actions)

    def reset(self):
        """Resets the game to the initial state."""
        self.checkers = Board()
        return self._get_observation()

    def _get_observation(self):
        """Returns the current state of the board as an 8x8 array."""
        state = np.zeros((8, 8), dtype=np.int8)
        for row in range(8):
            for col in range(8):
                piece = self.checkers.board[row][col]
                if piece != 0:
                    if piece.color == BLUE:
                        state[row][col] = 1 if not piece.king else 2
                    elif piece.color == RED:
                        state[row][col] = -1 if not piece.king else -2
        return state

    def step(self, action):
        # Reward
        reward = 0

        # Save original board for future reward calculations
        original_board = copy.deepcopy(self.checkers.board)

        # Get number of pieces under attack before the turn for future reward calculations
        old_undefended = self.enemy_capture()

        legal_moves = self.get_legal_moves()
        if action >= len(legal_moves) or action < 0:
            raise ValueError("Invalid action. Action index isn't in bounds of legal moves array.")
        
        chosen_move = legal_moves[action]

        # Apply the chosen move on the board (this can be a regular move or a capture)
        if "-" in chosen_move:  # Move piece
            move_success = self.checkers.move_piece([int(pos) for pos in chosen_move.split('-')])
        elif "x" in chosen_move:  # Capture piece
            move_success = self.checkers.capture_piece([int(pos) for pos in chosen_move.split('x')])
        else:
            raise TypeError("Invalid move type.")

        # Reward or penalty for a successful move
        if not move_success:
            reward -= 10  # Penalty for invalid moves
        else:
            reward += 1  # Reward for a successful move

            # Reward for capturing pieces
            if "x" in chosen_move:
                num_captures = len(chosen_move.split('x')) - 1
                reward += 5 ** num_captures  # Increasing reward for multiple captures

            # Reward for promoting a piece to a king
            if self.king_promoted(original_board):
                reward += 20  # Reward for becoming a king

            new_undefended = self.enemy_capture()
            reward -= new_undefended * 3  # Punishment for leaving pieces undefended

            # Reward for defending pieces under attack
            reward += max(old_undefended - new_undefended, 0) * 3
        
        # Update game outcome: Check for winner or draw
        winner = self.checkers.check_winner()
        if winner == self.checkers.turn:
            done = True
            reward += 500 if winner == self.checkers.turn else -500  # Large reward for winning, penalty for losing
        elif winner == "Tie!":
            reward += 0  # Can change this later on
            done = True
        else:
            done = False

        # Get the updated observation
        observation = self._get_observation()

        # Additional info for debugging or logging purposes
        info = {
            "legal_moves": legal_moves,
            "turn": self.checkers.turn,
            "move_success": move_success,
            "winner": winner if winner else "None"
        }

        return observation, reward, done, info

    def king_promoted(self, original_board):
        prev_num_king = sum(1 for row in original_board for piece in row if piece != 0 and piece.color == self.checkers.turn and piece.king)
        new_num_king = sum(1 for row in self.checkers.board for piece in row if piece != 0 and piece.color == self.checkers.turn and piece.king)
        return new_num_king - prev_num_king > 0  # Return boolean if number of kings increases

    def enemy_capture(self):
        """Returns the number of pieces that can be captured."""
        numCaptures = 0
        pieceCapture = set()
        self.checkers.switch_turn()  # Switch turn temporarily

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.checkers.board[row][col]
                if piece != 0 and piece.color == self.checkers.turn:  # Check only the pieces of the opposite player
                    captures = self.checkers.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures and piece not in pieceCapture:  # If any valid captures exist and not already capturable
                        pieceCapture.add(piece)
                        numCaptures += 1
        self.checkers.switch_turn()  # Switch back turn to original
        return numCaptures

    def render(self):
        """Prints the board to the console."""
        print(self.checkers)

    def get_legal_moves(self):
        """Returns all possible moves for the current player."""
        all_moves = self.checkers.get_all_possible_moves()
        return all_moves

    def close(self):
        """Cleanup if necessary."""
        pass
