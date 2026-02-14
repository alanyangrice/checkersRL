import os
import sys
import random

import torch
import pygame

from checkers_game.constants import WIDTH, HEIGHT, BLUE, RED
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.checkers_env import CheckersEnv


def play_agent():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Game - Play Against AI")

    # Load the trained agent
    input_shape = (4, 8, 8)
    n_actions = 50
    agent = PPOAgent(input_shape, n_actions)

    # Look for the latest saved model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "PPO_Model", "PPO_saved_models_parallel")

    if not os.path.exists(model_dir):
        # Fall back to sequential training models
        model_dir = os.path.join(base_dir, "PPO_Model", "PPO_saved_models")

    if os.path.exists(model_dir):
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("agent_epoch_") and f.endswith(".pt")]
        if checkpoints:
            latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            model_path = os.path.join(model_dir, latest)
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No model checkpoints found. The AI will play randomly.")
    else:
        print(f"Model directory not found: {model_dir}. The AI will play randomly.")

    agent.policy.eval()

    env = CheckersEnv()
    state, _ = env.reset()
    done = False

    # Randomly choose player side
    player_color = BLUE if random.random() < 0.5 else RED
    ai_color = RED if player_color == BLUE else BLUE

    print(f"Player Color: {'BLUE' if player_color == BLUE else 'RED'}")
    print(f"AI Color: {'BLUE' if ai_color == BLUE else 'RED'}")

    env.game.update_board(screen)

    while not done:
        if env.game.turn == player_color:
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
                env._legal_moves = env.game.get_all_possible_moves()
        else:
            # AI's turn
            legal_moves = env.legal_moves

            if len(legal_moves) == 0:
                done = True
                display_winner(player_color, player_color, ai_color)
                break

            action, _, _ = agent.select_action(state, len(legal_moves))

            if action >= len(legal_moves):
                action = random.choice(range(len(legal_moves)))

            next_state, reward, done, _, info = env.step(action)
            env.game.update_board(screen)

            if done:
                winner = info.get("winner", "Tie")
                display_winner(winner, player_color, ai_color)
                break

            state = next_state
            env.game.switch_turn()
            env._legal_moves = env.game.get_all_possible_moves()

    print(f"Game moves: {env.game.moves}")

    # Keep window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    pygame.quit()


def display_winner(winner, player_color, ai_color):
    """Displays the winner or draw on the console."""
    if winner == player_color:
        print("Player Won!")
    elif winner == ai_color:
        print("AI Won!")
    else:
        print("Tie!")


if __name__ == "__main__":
    play_agent()
