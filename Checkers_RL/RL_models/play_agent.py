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
model_path = "Checkers_RL/RL_models/PPO_Model/PPO_saved_models/agent_model.pt"

# Load the trained agent
agent = PPOAgent()
agent.policy.load_state_dict(torch.load(model_path))  # Loading the model parameters
agent.policy.eval()  # Set model to evaluation mode

def play_agent():
    env = CheckersEnv()
    state = env.reset()  # Start a new game
    moves = []
    done = False

    # Randomly choose side to play
    if random.random() < 0.5:
        player = BLUE
    else:
        player = RED

    while not done:
        env.checkers.draw(screen)

        pygame.time.delay(500)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                sys.exit()

        legal_moves = env.checkers.get_all_possible_moves()
        print(f"Legal Moves: {legal_moves}")

        if player == env.checkers.turn:
            user_input = input(f"Enter your move ({'Red' if env.checkers.turn == player else 'Blue'}), e.g., '11-15' for move, '11x18' or '11x18x22' for capture: ")
        else:
            # Get action and log probabilities from the current agent
            action, _, _ = agent.select_action(state, len(legal_moves))

        if player == env.checkers.turn:
            next_state, _, done, info = env.step(legal_moves.index(user_input))
        else:
            next_state, _, done, info = env.step(action)

        # Flag to check if move is successful
        move_successful = info["move_success"]
            
        if move_successful:
            # Add move to move history
            if player == env.checkers.turn:
                moves.append()
            else:
                moves.append(info["legal_moves"][action])
            
            # Update state for next step
            state = next_state

            # Switch sides
            env.checkers.switch_turn()
        else:
            print("Invalid input.")
    
    print(f"Game moves: {moves}") # Print moves played in entire game
