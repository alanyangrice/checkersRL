import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

import numpy as np
import gym
import copy
from gym import spaces
from checkers_game.game import Game
from checkers_game.constants import RED, BLUE, ROWS, COLS, board_number_to_position, position_to_board_number

class CheckersEnv(gym.Env):
    def __init__(self):
        super(CheckersEnv, self).__init__()

        # Intialize game in environment
        self.game = Game()
        
        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32)

        # Define action space with fixed max number of actions
        self.max_actions = 50
        self.action_space = spaces.Discrete(self.max_actions)

    def reset(self):
        """Resets the game to the initial state."""
        self.game = Game()
        return self.get_board_state()

    def get_board_state(self):
        # Assuming you have a way to access piece information in the environment:
        board_state = np.zeros((4, 8, 8), dtype=np.float32)  # 4 channels, 8x8 board

        # Example filling in channels (adjust according to actual data structure)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == RED:
                        if piece.king:
                            board_state[1, row, col] = 1  # Red king
                        else:
                            board_state[0, row, col] = 1  # Red regular
                    elif piece.color == BLUE:
                        if piece.king:
                            board_state[3, row, col] = 1  # Blue king
                        else:
                            board_state[2, row, col] = 1  # Blue regular
        return board_state

    def step(self, action):
        reward = 0  # Reward
        done = False  # Set done to False by default

        legal_moves = self.game.get_all_possible_moves()

        if action >= len(legal_moves):
            raise ValueError("Invalid action. Action index out of bounds of legal moves array.")

        # Get number of pieces under attack before the turn for future reward calculations
        old_undefended = self.enemy_capture()

        # Apply the chosen move on the board
        chosen_move = legal_moves[action][0]
        new_board = legal_moves[action][1]

        print(f"newboard: {type(new_board)}")

        old_board = copy.deepcopy(self.game.board.board)
        self.game.board = new_board

        # Reward for promoting a piece to a king
        if self.king_promoted(old_board):
            reward += 20  # Reward for becoming a king
            
        # Reward for capturing pieces
        if "x" in chosen_move:
            num_captures = len(chosen_move.split('x')) - 1
            reward += 5 ** num_captures  # Increasing reward for multiple captures

        # Get new undefended
        new_undefended = self.enemy_capture()
        reward -= new_undefended * 3  # Punishment for leaving pieces undefended

        # Reward for defending pieces under attack
        reward += max(old_undefended - new_undefended, 0) * 3

        # Update game outcome: Check for winner or draw
        winner = self.game.check_winner()
        if winner == self.game.turn:
            done = True
            reward += 500 if winner == self.game.turn else -500  # Large reward for winning, penalty for losing
        elif winner == "Tie":
            done = True
            reward += 0  # Add reward for tie

        # Get the updated observation
        observation = self.get_board_state()

        # Additional info for debugging or logging purposes
        info = {
            "legal_moves": legal_moves,
            "turn": self.game.turn,
            "winner": winner if winner else "None"
        }

        return observation, reward, done, info

    def king_promoted(self, old_board):
        prev_num_king = sum(1 for row in old_board for piece in row if piece != 0 and piece.color == self.game.turn and piece.king)
        new_num_king = sum(1 for row in self.game.board.board for piece in row if piece != 0 and piece.color == self.game.turn and piece.king)
        return new_num_king - prev_num_king > 0  # Return boolean if number of kings increases

    def enemy_capture(self):
        """Returns the number of pieces that can be captured."""
        numCaptures = 0
        pieceCapture = set()
        self.game.switch_turn()  # Switch turn temporarily

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0 and piece.color == self.game.turn:  # Check only the pieces of the opposite player
                    captures = self.game.board.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures and piece not in pieceCapture:  # If any valid captures exist and not already capturable
                        pieceCapture.add(piece)
                        numCaptures += 1

        self.game.switch_turn()  # Switch back turn to original
        return numCaptures

    def render(self):
        """Prints the board to the console."""
        print("Board State")
        print(self.game.board)

    def close(self):
        """Cleanup if necessary."""
        pass
