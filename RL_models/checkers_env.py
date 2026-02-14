import numpy as np
import copy

import gymnasium as gym
from gymnasium import spaces

from checkers_game.game import Game
from checkers_game.constants import RED, BLUE, ROWS, COLS


class CheckersEnv(gym.Env):
    def __init__(self):
        super(CheckersEnv, self).__init__()

        self.game = Game()

        # Observation: 4 channels (current_regular, current_king, opponent_regular, opponent_king)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32)

        # Action space with fixed max number of actions
        self.max_actions = 50
        self.action_space = spaces.Discrete(self.max_actions)

        # Cache legal moves so step() doesn't need them as a parameter
        self._legal_moves = []

    def reset(self, seed=None, options=None):
        """Resets the game to the initial state."""
        super().reset(seed=seed)
        self.game = Game()
        self._legal_moves = self.game.get_all_possible_moves()
        obs = self.get_board_state()
        return obs, {}

    @property
    def legal_moves(self):
        """Returns the cached list of legal moves for the current player."""
        return self._legal_moves

    def get_board_state(self):
        """Returns a normalized 4-channel board state.
        Channels are always from the current player's perspective:
          0: current player regular pieces
          1: current player king pieces
          2: opponent regular pieces
          3: opponent king pieces
        The board is flipped vertically when it's Red's turn so the agent
        always sees pieces moving in the same direction."""
        board_state = np.zeros((4, 8, 8), dtype=np.float32)

        current_color = self.game.turn
        opponent_color = RED if current_color == BLUE else BLUE

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == current_color:
                        if piece.king:
                            board_state[1, row, col] = 1  # Current player king
                        else:
                            board_state[0, row, col] = 1  # Current player regular
                    elif piece.color == opponent_color:
                        if piece.king:
                            board_state[3, row, col] = 1  # Opponent king
                        else:
                            board_state[2, row, col] = 1  # Opponent regular

        # Flip board for Red so the agent always sees from the same perspective
        if current_color == RED:
            board_state = np.flip(board_state, axis=1).copy()

        return board_state

    def render(self):
        """Prints the board to the console."""
        print("Board State")
        print(self.game.board)

    def close(self):
        """Cleanup if necessary."""
        pass

    def step(self, action):
        """Standard Gym step interface. Uses cached legal_moves."""
        legal_moves = self._legal_moves

        if len(legal_moves) == 0:
            done = True
            reward = 20 + self.remaining_diff(self.game.board.board)
            info = {
                "legal_moves": legal_moves,
                "turn": self.game.turn,
                "winner": "Tie"
            }
            return self.get_board_state(), reward, done, False, info

        # Clamp action to valid range
        if action >= len(legal_moves):
            action = np.random.randint(0, len(legal_moves))

        # Apply the action and update the state
        chosen_move, old_board, new_board = self.update_state(action, legal_moves)

        # Analyze the board once
        board_stats = self.analyze_board()

        # Add move to move list
        self.game.moves.append(chosen_move)

        # Calculate reward for this move
        reward, done, winner = self.calculate_reward(chosen_move, old_board, new_board, board_stats)

        # Update legal moves cache for next turn
        if not done:
            self._legal_moves = self.game.get_all_possible_moves()
        else:
            self._legal_moves = []

        observation = self.get_board_state()
        info = {
            "legal_moves": self._legal_moves,
            "turn": self.game.turn,
            "winner": winner if winner else "None"
        }

        return observation, reward, done, False, info

    def update_state(self, action, legal_moves):
        chosen_move = legal_moves[action][0]
        new_board = legal_moves[action][1]

        old_board = copy.deepcopy(self.game.board)
        self.game.board = copy.deepcopy(new_board)

        return chosen_move, old_board, new_board

    def calculate_reward(self, chosen_move, old_board, new_board, board_stats):
        """Calculate the total reward for the current step."""
        reward = 0

        reward += self.reward_control_center(board_stats)
        reward += self.reward_protect_rear(board_stats)
        reward += self.reward_balance(board_stats)
        reward += self.reward_for_kings(board_stats)
        reward += self.reward_king_promotion(old_board)
        reward += self.reward_piece_capture(chosen_move)
        reward += self.penalize_undefended_pieces(old_board, new_board)

        end_reward, done, winner = self.reward_end_game()
        reward += end_reward

        if done:
            return reward, done, winner
        else:
            reward += self.penalize_opponent_advantage(new_board)

        return reward, done, winner

    def analyze_board(self):
        """Precompute useful board statistics."""
        stats = {
            "blue_pieces": 0,
            "red_pieces": 0,
            "blue_kings": 0,
            "red_kings": 0,
            "blue_positions": [],
            "red_positions": [],
            "blue_back_row": 0,
            "red_back_row": 0,
            "center_control": 0,
        }

        back_row_blue = 0
        back_row_red = 7
        central_positions = {(3, 4), (4, 3)}

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == BLUE:
                        stats["blue_pieces"] += 1
                        stats["blue_positions"].append((row, col))
                        if piece.king:
                            stats["blue_kings"] += 1
                        if row == back_row_blue:
                            stats["blue_back_row"] += 1
                    elif piece.color == RED:
                        stats["red_pieces"] += 1
                        stats["red_positions"].append((row, col))
                        if piece.king:
                            stats["red_kings"] += 1
                        if row == back_row_red:
                            stats["red_back_row"] += 1

                    if (row, col) in central_positions:
                        stats["center_control"] += 1 if piece.color == self.game.turn else 0

        return stats

    def reward_control_center(self, board_stats):
        """Reward for controlling central positions."""
        return 0.5 * board_stats["center_control"]

    def reward_protect_rear(self, board_stats):
        """Reward for protecting the back row."""
        if self.game.turn == BLUE:
            return 0.5 * board_stats["blue_back_row"]
        else:
            return 0.5 * board_stats["red_back_row"]

    def reward_balance(self, board_stats):
        """Reward for maintaining a balanced distribution of pieces."""
        if self.game.turn == BLUE:
            positions = board_stats["blue_positions"]
        else:
            positions = board_stats["red_positions"]

        left_half = sum(1 for _, col in positions if col < COLS // 2)
        right_half = sum(1 for _, col in positions if col >= COLS // 2)
        imbalance = abs(left_half - right_half)
        return -0.1 * imbalance

    def reward_for_kings(self, board_stats):
        """Reward for maintaining kings during the game."""
        if self.game.turn == BLUE:
            return board_stats["blue_kings"] * 0.2
        else:
            return board_stats["red_kings"] * 0.2

    def reward_king_promotion(self, old_board):
        """Reward for promoting a piece to a king."""
        if self.king_promoted(old_board.board):
            return 15
        return 0

    def reward_piece_capture(self, chosen_move):
        """Reward for capturing opponent's pieces."""
        if "x" in chosen_move:
            num_captures = len(chosen_move.split('x')) - 1
            return 10 * num_captures
        return 0

    def penalize_undefended_pieces(self, old_board, new_board):
        """Penalize for leaving pieces undefended."""
        self.game.board = old_board
        old_undefended = self.enemy_capture()
        self.game.board = copy.deepcopy(new_board)
        new_undefended = self.enemy_capture()
        return max(old_undefended - new_undefended, 0) * 5 - new_undefended * 5

    def reward_end_game(self):
        """Reward or penalize based on game-ending conditions."""
        winner = self.game.check_winner()
        if winner == self.game.turn:
            return 100, True, copy.deepcopy(self.game.turn)
        elif winner == "Tie":
            return -20 - self.remaining_diff(self.game.board.board), True, "Tie"
        elif winner is not None:
            # Opponent wins (current player has no legal moves or no pieces)
            return -100, True, copy.deepcopy(winner)
        else:
            return -np.sqrt(len(self.game.moves)) / 10, False, None

    def penalize_opponent_advantage(self, new_board):
        """Penalize the agent if it leads to a strong opponent move."""
        self.game.switch_turn()
        opponent_legal_moves = self.game.get_all_possible_moves()
        best_opponent_reward = float('-inf')

        if opponent_legal_moves:
            opp_old_board = copy.deepcopy(new_board)
            for opp_move in opponent_legal_moves:
                self.game.board = opp_move[1]

                opp_board_stats = self.analyze_board()

                opp_reward = 0
                opp_reward += self.reward_control_center(opp_board_stats)
                opp_reward += self.reward_protect_rear(opp_board_stats)
                opp_reward += self.reward_balance(opp_board_stats)
                opp_reward += self.reward_for_kings(opp_board_stats)
                opp_reward += self.reward_king_promotion(opp_old_board)
                opp_reward += self.reward_piece_capture(opp_move[0])
                opp_reward += self.penalize_undefended_pieces(opp_old_board, self.game.board)

                end_reward, _, _ = self.reward_end_game()
                opp_reward += end_reward

                # Undo the board hash increment from check_winner
                board_hash = self.game.board.get_board_hash()
                if board_hash in self.game.board_states:
                    self.game.board_states[board_hash] -= 1

                best_opponent_reward = max(best_opponent_reward, opp_reward)

                self.game.board = copy.deepcopy(new_board)

            self.game.board = new_board
        else:
            best_opponent_reward = 20 + self.remaining_diff(self.game.board.board)

        self.game.switch_turn()
        return -0.5 * best_opponent_reward

    def remaining_diff(self, board):
        blue_king = sum(1 for row in board for piece in row if piece != 0 and piece.color == BLUE and piece.king)
        blue_piece = sum(1 for row in board for piece in row if piece != 0 and piece.color == BLUE and not piece.king)
        red_king = sum(1 for row in board for piece in row if piece != 0 and piece.color == RED and piece.king)
        red_piece = sum(1 for row in board for piece in row if piece != 0 and piece.color == RED and not piece.king)

        point_diff = (blue_king - red_king) * 3 + (blue_piece - red_piece) * 1
        if self.game.turn == BLUE:
            return point_diff
        else:
            return -point_diff

    def king_promoted(self, old_board):
        prev_num_king = sum(1 for row in old_board for piece in row if piece != 0 and piece.color == self.game.turn and piece.king)
        new_num_king = sum(1 for row in self.game.board.board for piece in row if piece != 0 and piece.color == self.game.turn and piece.king)
        return new_num_king - prev_num_king > 0

    def enemy_capture(self):
        """Returns the number of pieces that can be captured."""
        numCaptures = 0
        pieceCapture = set()
        self.game.switch_turn()

        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.get_piece(row, col)
                if piece != 0 and piece.color == self.game.turn:
                    captures = self.game.board.valid_moves_for_piece(piece, row, col, capture_only=True)
                    if captures and piece not in pieceCapture:
                        pieceCapture.add(piece)
                        numCaptures += 1

        self.game.switch_turn()
        return numCaptures
