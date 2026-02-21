import os
import sys
import copy
import csv
import random
import argparse
from datetime import datetime

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


def compute_turn_reward(env, old_board):
    """Compute the reward for a player's completed turn, mirroring env._finish_turn().

    Must be called BEFORE env.game.switch_turn() so that env.game.turn is still
    the color of the player who just moved.

    Args:
        env: The CheckersEnv instance (game.turn == player who just moved).
        old_board: Deep copy of env.game.board taken BEFORE the move.

    Returns:
        float: The shaped reward for this turn (excluding terminal win/loss/tie).
    """
    SHAPING_SCALE = 0.5

    move_str = env.game.moves[-1] if env.game.moves else ""
    is_capture = 'x' in move_str
    num_hops = move_str.count('x') if is_capture else 0

    # Board analysis (uses current board + env.game.turn)
    board_stats = env.analyze_board()
    shaped = 0.0
    shaped += env.reward_control_center(board_stats)
    shaped += env.reward_protect_rear(board_stats)
    shaped += env.reward_balance(board_stats)
    shaped += env.reward_for_kings(board_stats)
    shaped += env.reward_king_promotion(old_board)

    if is_capture:
        shaped += 10.0  # capture bonus (same as _finish_turn)

    # penalize_undefended_pieces temporarily modifies env.game.board
    new_board = copy.deepcopy(env.game.board)
    shaped += env.penalize_undefended_pieces(old_board, new_board)
    env.game.board = new_board  # restore after side-effect

    # Per-move time penalty (same formula as reward_end_game when game isn't over)
    time_penalty = -np.sqrt(len(env.game.moves)) / 5

    # Intermediate capture bonuses (each intermediate hop gives a flat +10 in env.step)
    intermediate_bonus = max(0, num_hops - 1) * 10.0

    return intermediate_bonus + shaped * SHAPING_SCALE + time_penalty


def save_game_csv(env, winner, player_color, ai_color, mode_name,
                  blue_rewards, red_rewards, all_reward_list):
    """Append game results to play_agent_games.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "play_agent_games.csv")

    file_exists = os.path.exists(csv_path)

    # Count existing games for game_number
    game_number = 1
    if file_exists:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                game_number += 1

    blue_win = 1 if winner == BLUE else 0
    red_win = 1 if winner == RED else 0
    total_blue = sum(blue_rewards)
    total_red = sum(red_rewards)
    total_reward = total_blue + total_red

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "game_number", "player_color", "ai_model", "blue_win", "red_win",
                "total_reward", "blue_reward", "red_reward", "time",
                "moves", "reward_list"
            ])
        writer.writerow([
            game_number,
            "BLUE" if player_color == BLUE else "RED",
            mode_name,
            blue_win, red_win,
            f"{total_reward:.4f}", f"{total_blue:.4f}", f"{total_red:.4f}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ", ".join(env.game.moves),
            ", ".join(f"{r:.4f}" for r in all_reward_list),
        ])

    print(f"\nGame saved to {csv_path}")


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
    print(f"{'='*60}")

    # Reward tracking
    blue_rewards = []
    red_rewards = []
    all_reward_list = []
    ai_turn_reward_acc = 0.0   # accumulate across multi-step capture chains
    winner = None

    env.game.update_board(screen)

    while not done:
        current_color = env.game.turn

        if env.game.turn == player_color:
            # ── Player's turn ──────────────────────────────────────────
            old_board = copy.deepcopy(env.game.board)
            turn_complete = env.game.player_action(screen)

            if turn_complete:
                env.game.update_board(screen)

                # Compute shaped reward for the player's move
                reward = compute_turn_reward(env, old_board)

                # Check for game over
                winner = env.game.check_winner()
                if winner:
                    # Add terminal reward
                    if winner == current_color:
                        reward += 100
                    elif winner == "Tie":
                        reward += env._tie_reward()
                    else:
                        reward += -100
                    done = True

                # Track reward
                if current_color == BLUE:
                    blue_rewards.append(reward)
                else:
                    red_rewards.append(reward)
                all_reward_list.append(reward)

                label = "BLUE" if current_color == BLUE else "RED"
                move_str = env.game.moves[-1] if env.game.moves else "?"
                print(f"  Player ({label}) [{move_str}]  reward: {reward:+.2f}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                # Capture penalty: player captured → AI's last move gets penalised
                if "x" in move_str:
                    num_cap = move_str.count("x")
                    cap_pen = -10.0 * num_cap
                    ai_label = "BLUE" if ai_color == BLUE else "RED"
                    if ai_color == BLUE:
                        blue_rewards.append(cap_pen)
                    else:
                        red_rewards.append(cap_pen)
                    all_reward_list.append(cap_pen)
                    print(f"  AI     ({ai_label}) [piece lost ×{num_cap}]  penalty: {cap_pen:+.2f}  "
                          f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                if done:
                    display_winner(winner, player_color, ai_color)
                    break

                state = env.get_board_state()
                env.game.switch_turn()
                env._update_action_mask()

        else:
            # ── AI's turn ──────────────────────────────────────────────
            action_mask = env.get_action_mask()

            if action_mask.sum() == 0:
                # AI has no legal moves → loses
                next_state, reward, done, _, info = env.step(0)
                ai_turn_reward_acc += reward

                if current_color == BLUE:
                    blue_rewards.append(ai_turn_reward_acc)
                else:
                    red_rewards.append(ai_turn_reward_acc)
                all_reward_list.append(ai_turn_reward_acc)

                label = "BLUE" if current_color == BLUE else "RED"
                print(f"  AI    ({label}) [no moves]  reward: {ai_turn_reward_acc:+.2f}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                # Apply terminal adjustments (loser -100, winner +100)
                blue_adj = info.get("blue_reward_adjustment", 0.0)
                red_adj = info.get("red_reward_adjustment", 0.0)
                if blue_adj != 0.0:
                    blue_rewards.append(blue_adj)
                    all_reward_list.append(blue_adj)
                    print(f"  BLUE  [terminal]  adjustment: {blue_adj:+.2f}  "
                          f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")
                if red_adj != 0.0:
                    red_rewards.append(red_adj)
                    all_reward_list.append(red_adj)
                    print(f"  RED   [terminal]  adjustment: {red_adj:+.2f}  "
                          f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                winner = info.get("winner", player_color)
                display_winner(winner, player_color, ai_color)
                ai_turn_reward_acc = 0.0
                break

            if use_mcts:
                action, _ = mcts.select_action(env, temperature=0.1)
            else:
                action, _, _ = agent.select_action(state, action_mask)

            next_state, reward, done, _, info = env.step(action)
            ai_turn_reward_acc += reward
            env.game.update_board(screen)

            turn_complete = info.get("turn_complete", True)

            if done:
                # Game ended on AI's move — log the accumulated turn reward
                if current_color == BLUE:
                    blue_rewards.append(ai_turn_reward_acc)
                else:
                    red_rewards.append(ai_turn_reward_acc)
                all_reward_list.append(ai_turn_reward_acc)

                label = "BLUE" if current_color == BLUE else "RED"
                move_str = env.game.moves[-1] if env.game.moves else "?"
                print(f"  AI    ({label}) [{move_str}]  reward: {ai_turn_reward_acc:+.2f}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                # Capture penalty: AI captured → player's last move gets penalised
                if "x" in move_str:
                    num_cap = move_str.count("x")
                    cap_pen = -10.0 * num_cap
                    pl_label = "BLUE" if player_color == BLUE else "RED"
                    if player_color == BLUE:
                        blue_rewards.append(cap_pen)
                    else:
                        red_rewards.append(cap_pen)
                    all_reward_list.append(cap_pen)
                    print(f"  Player ({pl_label}) [piece lost ×{num_cap}]  penalty: {cap_pen:+.2f}  "
                          f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                winner = info.get("winner", "Tie")
                display_winner(winner, player_color, ai_color)
                ai_turn_reward_acc = 0.0
                break

            if not turn_complete:
                # Capture chain continues — accumulate and loop
                state = next_state
                continue

            # Turn complete — log the full turn reward
            if current_color == BLUE:
                blue_rewards.append(ai_turn_reward_acc)
            else:
                red_rewards.append(ai_turn_reward_acc)
            all_reward_list.append(ai_turn_reward_acc)

            label = "BLUE" if current_color == BLUE else "RED"
            move_str = env.game.moves[-1] if env.game.moves else "?"
            print(f"  AI    ({label}) [{move_str}]  reward: {ai_turn_reward_acc:+.2f}  "
                  f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

            # Capture penalty: AI captured → player's last move gets penalised
            if "x" in move_str:
                num_cap = move_str.count("x")
                cap_pen = -10.0 * num_cap
                pl_label = "BLUE" if player_color == BLUE else "RED"
                if player_color == BLUE:
                    blue_rewards.append(cap_pen)
                else:
                    red_rewards.append(cap_pen)
                all_reward_list.append(cap_pen)
                print(f"  Player ({pl_label}) [piece lost ×{num_cap}]  penalty: {cap_pen:+.2f}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

            ai_turn_reward_acc = 0.0
            state = next_state

    # ── Terminal reward for the losing side ────────────────────────────
    # When one side's move ends the game, only that side gets the
    # terminal reward in their total (e.g. +100 for winning).  The
    # OTHER side never receives -100 for losing.  Fix: add it now.
    if winner is not None and winner != "Tie":
        losing_color = RED if winner == BLUE else BLUE
        # Check if the loser already received -100 (happens when the
        # loser's own turn triggered the game-over, e.g. no legal moves).
        loser_made_last_move = (losing_color == current_color)
        if not loser_made_last_move:
            if losing_color == BLUE:
                blue_rewards.append(-100)
            else:
                red_rewards.append(-100)
            all_reward_list.append(-100)
            label = "BLUE" if losing_color == BLUE else "RED"
            role = "Player" if losing_color == player_color else "AI    "
            print(f"  {role} ({label}) [game over]  loss penalty: -100.00  "
                  f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

    # ── Game summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GAME OVER")
    print(f"  Moves played:      {len(env.game.moves)}")
    print(f"  Blue total reward: {sum(blue_rewards):+.2f}")
    print(f"  Red total reward:  {sum(red_rewards):+.2f}")
    print(f"  Combined reward:   {sum(blue_rewards) + sum(red_rewards):+.2f}")
    print(f"{'='*60}")

    # Save to CSV
    save_game_csv(env, winner, player_color, ai_color, mode_name,
                  blue_rewards, red_rewards, all_reward_list)

    # Keep window open until user closes
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    pygame.quit()


def display_winner(winner, player_color, ai_color):
    """Displays the winner or draw on the console."""
    print()
    if winner == player_color:
        print("  >>> PLAYER WON! <<<")
    elif winner == ai_color:
        print("  >>> AI WON! <<<")
    else:
        print("  >>> TIE! <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play checkers against the AI")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for AI moves (stronger but slower)")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move (default: 100)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Load a specific training epoch (e.g. --epoch 230). "
                             "If omitted, loads the latest checkpoint.")
    args = parser.parse_args()

    play_agent(use_mcts=args.mcts, num_simulations=args.simulations, epoch=args.epoch)
