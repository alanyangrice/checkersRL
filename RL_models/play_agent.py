import os
import sys
import random
import argparse

import torch
import numpy as np
import pygame

from checkers_game.constants import WIDTH, HEIGHT, BLUE, RED, NUM_ACTIONS
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.checkers_env import CheckersEnv


def load_network(device, epoch=None):
    """Load a model checkpoint.

    If *epoch* is given, loads that specific epoch from the PPO parallel folder.
    Otherwise, loads the latest available model (AlphaZero > PPO parallel > PPO sequential).

    Returns (network, mode_name).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_shape = (4, 8, 8)
    network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(device)

    # If a specific epoch was requested, go straight to it
    if epoch is not None:
        model_path = os.path.join(
            base_dir, "PPO_Model", "PPO_saved_models_parallel", f"agent_epoch_{epoch}.pt"
        )
        if not os.path.exists(model_path):
            print(f"ERROR: No checkpoint found at {model_path}")
            sys.exit(1)
        print(f"Loading PPO-parallel model from epoch {epoch}: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["model_state_dict"])
        return network, f"PPO-parallel (epoch {epoch})"

    # Priority order for loading checkpoints
    search_dirs = [
        (os.path.join(base_dir, "MCTS", "alphazero_checkpoints"), "az_epoch_", "AlphaZero"),
        (os.path.join(base_dir, "PPO_Model", "PPO_saved_models_parallel"), "agent_epoch_", "PPO-parallel"),
        (os.path.join(base_dir, "PPO_Model", "PPO_saved_models"), "agent_epoch_", "PPO"),
    ]

    for model_dir, prefix, mode_name in search_dirs:
        if os.path.exists(model_dir):
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
            if checkpoints:
                latest = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
                model_path = os.path.join(model_dir, latest)
                print(f"Loading {mode_name} model: {model_path}")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                network.load_state_dict(checkpoint["model_state_dict"])
                return network, mode_name

    print("No model checkpoints found. The AI will play randomly.")
    return network, "random"


def play_agent(use_mcts=False, num_simulations=100, epoch=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Game - Play Against AI")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, mode_name = load_network(device, epoch=epoch)
    network.eval()

    # Set up the AI action selector
    mcts = None
    if use_mcts:
        mcts = MCTSSearch(network, num_simulations=num_simulations, device=device)
        print(f"AI mode: MCTS ({num_simulations} simulations/move)")
    else:
        print(f"AI mode: direct policy ({mode_name})")

    # Wrap in PPOAgent for direct policy mode
    agent = None
    if not use_mcts:
        agent = PPOAgent((4, 8, 8), NUM_ACTIONS, device=device)
        agent.policy = network

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
            # Human player's turn
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
                env._update_action_mask()

        else:
            # AI's turn
            action_mask = env.get_action_mask()

            if action_mask.sum() == 0:
                done = True
                display_winner(player_color, player_color, ai_color)
                break

            if use_mcts:
                # MCTS: run full search (handles capture chains internally via env copies)
                action, _ = mcts.select_action(env, temperature=0.1)
            else:
                # Direct policy
                action, _, _ = agent.select_action(state, action_mask)

            next_state, reward, done, _, info = env.step(action)
            env.game.update_board(screen)

            turn_complete = info.get("turn_complete", True)

            if done:
                winner = info.get("winner", "Tie")
                display_winner(winner, player_color, ai_color)
                break

            if not turn_complete:
                state = next_state
                continue

            state = next_state

    print(f"Game moves: {env.game.moves}")

    # Keep window open until user closes
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
    parser = argparse.ArgumentParser(description="Play checkers against the AI")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for AI moves (stronger but slower)")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move (default: 100)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Load a specific training epoch (e.g. --epoch 230). "
                             "If omitted, loads the latest checkpoint.")
    args = parser.parse_args()

    play_agent(use_mcts=args.mcts, num_simulations=args.simulations, epoch=args.epoch)
