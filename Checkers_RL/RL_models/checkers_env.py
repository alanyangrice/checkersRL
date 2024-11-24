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
        reward = 0  # Reward for this step
        done = False  # Whether the game is over

        legal_moves = self.game.get_all_possible_moves()

        if len(legal_moves) == 0:  # No legal moves, the game ends
            done = True
            reward += 200
            info = {
                "legal_moves": legal_moves,
                "turn": self.game.turn,
                "winner": "Tie"
            }
            return self.get_board_state(), reward, done, info

        # Record the current game state for comparison
        old_board = copy.deepcopy(self.game.board.board)

        # Get number of pieces under attack before the move
        old_undefended = self.enemy_capture()

        # Apply the chosen move
        chosen_move = legal_moves[action][0]
        new_board = legal_moves[action][1]
        self.game.board = copy.deepcopy(new_board)

        # Reward for promoting a king
        if self.king_promoted(old_board):
            reward += 40

        # Reward for capturing opponent's pieces
        if "x" in chosen_move:
            num_captures = len(chosen_move.split('x')) - 1
            reward += 30 * num_captures

        # Punish leaving pieces undefended
        new_undefended = self.enemy_capture()
        reward -= new_undefended * 30  # Penalize undefended pieces

        # Reward defending pieces
        reward += max(old_undefended - new_undefended, 0) * 30

        # End-of-game outcomes
        winner = self.game.check_winner()
        if winner == self.game.turn:  # Current player wins
            done = True
            reward += 500
        elif winner == "Tie":
            done = True
            reward += 0  # Neutral reward for tie

        # Calculate reward for best opponent move
        if not done:
            # Switch turns to play as the opponent
            self.game.switch_turn()
            opponent_legal_moves = self.game.get_all_possible_moves()

            # Evaluate the opponent's best response
            if opponent_legal_moves:
                # Simulate the opponent's best move (greedy evaluation)
                best_opponent_reward = float('-inf')
                opp_old_board = copy.deepcopy(new_board)

                # Get number of pieces under attack before the move
                opp_old_undefended = self.enemy_capture()

                for opp_move in opponent_legal_moves:
                    # Update board
                    self.game.board = opp_move[1]

                    # Opponent's reward logic
                    opp_reward = 0

                    # Reward for promoting a king
                    if self.king_promoted(opp_old_board):
                        opp_reward += 40

                    # Reward for capturing opponent's pieces
                    if "x" in opp_move[0]:
                        opp_num_captures = len(opp_move[0].split('x')) - 1
                        opp_reward += 30 * opp_num_captures

                    # Punish leaving pieces undefended
                    opp_new_undefended = self.enemy_capture()
                    opp_reward -= opp_new_undefended * 40  # Penalize undefended pieces

                    # Reward defending pieces
                    opp_reward += max(opp_old_undefended - opp_new_undefended, 0) * 30

                    # End-of-game outcomes
                    winner = self.game.check_winner()
                    if winner == self.game.turn:
                        opp_reward += 500
                    elif winner == "Tie":
                        opp_reward += 0  # Neutral reward for tie

                    # Update best reward
                    best_opponent_reward = max(best_opponent_reward, opp_reward)
                
                # Revert board to original
                self.game.board = copy.deepcopy(new_board)
            else:
                best_opponent_reward = 200

            # Penalize the current player if it leads to a strong opponent move
            reward -= 0.7 * best_opponent_reward

            # Switch turn back to the current player
            self.game.switch_turn()

        # Update observation and return results
        observation = self.get_board_state()
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
