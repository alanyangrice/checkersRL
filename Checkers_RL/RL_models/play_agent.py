import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

import torch
import pygame
import random
from checkers_game.constants import WIDTH, HEIGHT, BLUE, RED
from RL_models.PPO_Model.Agent import PPOAgent  # Assuming Agent class is in this file
from RL_models.checkers_env import CheckersEnv

# Initialize pygame
pygame.init()

# Load the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers Game - Play Against AI")

# Path to the saved model
model_path = "Checkers_RL/RL_models/PPO_Model/PPO_saved_models/agent2_epoch_21.pt"

# Load the trained agent
input_shape = (4, 8, 8)  # 4 channels, 8x8 board
n_actions = 50

# Create Agents 1 and 2 to play checkers
agent = PPOAgent(input_shape, n_actions)
agent.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load model parameters
agent.policy.eval()  # Set model to evaluation mode

def play_agent():
    env = CheckersEnv()
    state = env.reset()  # Start a new game
    done = False

    # Randomly choose player side
    player_color = BLUE if random.random() < 0.5 else RED
    ai_color = RED if player_color == BLUE else BLUE

    print(f"Player Color: {'BLUE' if player_color == BLUE else 'RED'}")
    print(f"AI Color: {'BLUE' if ai_color == BLUE else 'RED'}")

    env.game.update_board(screen)  # Draw the current board state

    while not done:
        if env.game.turn == player_color:
            # Handle player turn
            turn_complete = env.game.player_action(screen)

            if turn_complete:
                env.game.update_board(screen)

                winner = env.game.check_winner()
                if winner:
                    done = True
                    display_winner(winner, player_color, ai_color)
                    break

                state = env.get_board_state()

                env.game.switch_turn()
        else:
            # AI's turn
            legal_moves = env.game.get_all_possible_moves()
            action, _, _ = agent.select_action(state, len(legal_moves))  # Get AI action

            next_state, reward, done, info = env.step(action)  # Apply AI move
            env.game.moves.append(legal_moves[action][0])  # Record AI move
            
            pygame.time.delay(500)
            env.game.update_board(screen)

            if done:
                display_winner(winner, player_color, ai_color)
                break

            state = next_state  # Update state
            env.game.switch_turn()  # Switch back to player's turn

    print(f"Game moves: {env.game.moves}")  # Print all moves played in the game

def display_winner(winner, player_color, ai_color):
    """Displays the winner or draw on the console and ends the game."""
    if winner == player_color:
        print("Player Won!")
    elif winner == ai_color:
        print("AI Won!")
    else:
        print("Tie!")

if __name__ == "__main__":
    play_agent()