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
from rl.algorithms.ppo.agent import PPOAgent, load_policy_state_dict
from rl.networks import PPOPolicyNetwork, AlphaZeroNetwork, WDLAlphaZeroNetwork
from rl.algorithms.ppo import utils as ppo_utils
from rl.algorithms.mcts.mcts_search import MCTSSearch

find_latest_checkpoint_path = ppo_utils.find_latest_checkpoint_path
from rl.envs import CheckersEnv


def load_az_checkpoint(path, device):
    """Load an AlphaZero checkpoint, auto-detecting WDL vs scalar architecture.

    Detects by inspecting value_fc2.weight shape:
        shape[0] == 3  → WDLAlphaZeroNetwork
        shape[0] == 1  → AlphaZeroNetwork (scalar)

    Returns (network, is_wdl).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]
    is_wdl = sd["value_fc2.weight"].shape[0] == 3
    NetworkClass = WDLAlphaZeroNetwork if is_wdl else AlphaZeroNetwork
    network = NetworkClass((4, 8, 8), n_actions=NUM_ACTIONS).to(device)
    network.load_state_dict(sd)
    return network, is_wdl


def load_network(device, epoch=None, az_epoch=None, az_version="v2", agent_type=None):
    """Load a model checkpoint.

    Args:
        epoch:      Load a specific PPO epoch (works with or without agent_type).
        az_epoch:   Load a specific AlphaZero epoch. If None, loads the latest.
        az_version: "v2" (default) uses alphazero_checkpoints/;
                    "v1" uses alphazero_checkpoints_v1/.
        agent_type: One of "tactical", "terminal", "aggressive" for a league agent.

    Returns (network, mode_name, is_wdl).
        is_wdl: True if the network is WDLAlphaZeroNetwork (v2 WDL training run).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_shape = (4, 8, 8)

    az_dir_name = "alphazero_checkpoints_v1" if az_version == "v1" else "alphazero_checkpoints"

    # AlphaZero requested (explicit epoch or auto-latest for AZ)
    if az_epoch is not None or (agent_type is None and epoch is None):
        az_dir = os.path.join(base_dir, "training_results", "mcts", az_dir_name)

        if az_epoch is not None:
            model_path = os.path.join(az_dir, f"az_epoch_{az_epoch}.pt")
            if not os.path.exists(model_path):
                print(f"ERROR: No AlphaZero checkpoint found at {model_path}")
                sys.exit(1)
            ep_num = az_epoch
        else:
            if not os.path.exists(az_dir):
                az_dir = None
            else:
                model_path = find_latest_checkpoint_path(az_dir, prefix="az_epoch_")
                if model_path:
                    ep_num = int(os.path.basename(model_path).split("_")[-1].split(".")[0])
                else:
                    az_dir = None

        if az_dir is not None:
            print(f"Loading AlphaZero {az_version} epoch {ep_num}: {model_path}")
            network, is_wdl = load_az_checkpoint(model_path, device)
            arch = "WDL" if is_wdl else "scalar"
            return network, f"AlphaZero-{az_version} ({arch}, epoch {ep_num})", is_wdl

        # Fall through to PPO if no AZ checkpoints found
        if az_epoch is not None:
            sys.exit(1)  # explicit epoch requested but not found — already errored above

    # League agent
    if agent_type is not None:
        league_dir = os.path.join(base_dir, "training_results", "ppo", f"ppo_saved_models_{agent_type}")
        if not os.path.exists(league_dir):
            print(f"ERROR: League model directory not found: {league_dir}")
            sys.exit(1)

        if epoch is not None:
            model_path = os.path.join(league_dir, f"agent_epoch_{epoch}.pt")
        else:
            model_path = find_latest_checkpoint_path(league_dir)
            if model_path is None:
                print(f"ERROR: No checkpoints found in {league_dir}")
                sys.exit(1)

        if not os.path.exists(model_path):
            print(f"ERROR: No checkpoint found at {model_path}")
            sys.exit(1)

        ep_num = model_path.split("_")[-1].split(".")[0]
        mode_name = f"League-{agent_type} (epoch {ep_num})"
        print(f"Loading {mode_name}: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(device)
        load_policy_state_dict(network, checkpoint["model_state_dict"])
        return network, mode_name, False

    # Specific PPO-parallel epoch
    if epoch is not None:
        model_path = os.path.join(
            base_dir, "training_results", "ppo", "ppo_saved_models_parallel", f"agent_epoch_{epoch}.pt"
        )
        if not os.path.exists(model_path):
            print(f"ERROR: No checkpoint found at {model_path}")
            sys.exit(1)
        print(f"Loading PPO-parallel model from epoch {epoch}: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(device)
        load_policy_state_dict(network, checkpoint["model_state_dict"])
        return network, f"PPO-parallel (epoch {epoch})", False

    # Fallback PPO
    ppo_dirs = [
        (os.path.join(base_dir, "training_results", "ppo", "ppo_saved_models_parallel"), "agent_epoch_", "PPO-parallel"),
        (os.path.join(base_dir, "legacy", "ppo_saved_models"), "agent_epoch_", "PPO"),
    ]
    for model_dir, prefix, mode_name in ppo_dirs:
        if os.path.exists(model_dir):
            model_path = find_latest_checkpoint_path(model_dir, prefix=prefix)
            if model_path is not None:
                print(f"Loading {mode_name} model: {model_path}")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(device)
                load_policy_state_dict(network, checkpoint["model_state_dict"])
                return network, mode_name, False

    print("No model checkpoints found. The AI will play randomly.")
    network = PPOPolicyNetwork(input_shape, NUM_ACTIONS).to(device)
    return network, "random", False


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


def apply_capture_penalty(move_str, penalized_color, penalized_role,
                           blue_rewards, red_rewards, all_reward_list):
    """Apply -10 per captured piece to the penalized side (opponent of capturer)."""
    if "x" not in move_str:
        return
    num_cap = move_str.count("x")
    cap_pen = -10.0 * num_cap
    color_label = "BLUE" if penalized_color == BLUE else "RED"
    if penalized_color == BLUE:
        blue_rewards.append(cap_pen)
    else:
        red_rewards.append(cap_pen)
    all_reward_list.append(cap_pen)
    print(f"  {penalized_role:6} ({color_label}) [piece lost ×{num_cap}]  penalty: {cap_pen:+.2f}  "
          f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")


def save_game_csv(env, winner, player_color, ai_color, mode_name,
                  blue_rewards, red_rewards, all_reward_list):
    """Append game results to play_agent_games.csv."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def eval_position(network, env, device, is_wdl, current_color):
    """Run a single network forward pass and return display strings.

    For WDL networks returns (val_str, wdl_blue_str, wdl_red_str).
    For scalar networks returns (val_str, None, None).

    The WDL output is always from the *current player's* perspective:
        wdl[0] = P(win for current player)
        wdl[1] = P(draw)
        wdl[2] = P(loss for current player)
    The opponent's probabilities are obtained by swapping W and L.
    """
    state_t = torch.FloatTensor(env.get_board_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        if is_wdl:
            _, wdl_t, v_t = network.forward_wdl(state_t)
            val = v_t.item()
            w, d, l = wdl_t[0].tolist()          # current player perspective
            ow, od, ol = l, d, w                  # opponent perspective
            if current_color == BLUE:
                blue_w, blue_d, blue_l = w, d, l
                red_w,  red_d,  red_l  = ow, od, ol
            else:
                red_w,  red_d,  red_l  = w, d, l
                blue_w, blue_d, blue_l = ow, od, ol
            val_str      = f"  val: {val:+.3f}"
            wdl_blue_str = f"B[W{blue_w*100:.0f}% D{blue_d*100:.0f}% L{blue_l*100:.0f}%]"
            wdl_red_str  = f"R[W{red_w*100:.0f}% D{red_d*100:.0f}% L{red_l*100:.0f}%]"
            return val_str, wdl_blue_str, wdl_red_str
        else:
            _, v_t = network(state_t)
            return f"  val: {v_t.item():+.3f}", None, None


def play_agent(use_mcts=False, num_simulations=100, epoch=None, az_epoch=None,
               az_version="v2", agent_type=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Game - Play Against AI")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, mode_name, is_wdl = load_network(
        device, epoch=epoch, az_epoch=az_epoch, az_version=az_version, agent_type=agent_type
    )
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
            # Pre-move value estimate (from the player's perspective)
            if use_mcts:
                p_val_str, p_wdl_blue, p_wdl_red = eval_position(
                    mcts.network, env, device, is_wdl, current_color
                )
            else:
                p_val_str, p_wdl_blue, p_wdl_red = None, None, None

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
                val_str  = p_val_str if p_val_str is not None else ""
                wdl_str  = f"  {p_wdl_blue} {p_wdl_red}" if p_wdl_blue is not None else ""
                print(f"  Player ({label}) [{move_str}]  reward: {reward:+.2f}{val_str}{wdl_str}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                # Capture penalty: player captured → AI's last move gets penalised
                apply_capture_penalty(
                    move_str, ai_color, "AI",
                    blue_rewards, red_rewards, all_reward_list,
                )

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
                # Always start from a fresh tree — tree reuse is a self-play
                # training optimisation and causes stale-child assertion errors
                # in human-vs-AI mode (both across human moves and within
                # multi-step capture chains).
                mcts._root = None
                ai_val_str, ai_wdl_blue, ai_wdl_red = eval_position(
                    mcts.network, env, device, is_wdl, current_color
                )
                action, _, _ = mcts.select_action(env, temperature=0.1)
            else:
                ai_val_str, ai_wdl_blue, ai_wdl_red = None, None, None
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
                wdl_str  = f"  {ai_wdl_blue} {ai_wdl_red}" if ai_wdl_blue is not None else ""
                print(f"  AI    ({label}) [{move_str}]  reward: {ai_turn_reward_acc:+.2f}"
                      f"{ai_val_str or ''}{wdl_str}  "
                      f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

                # Capture penalty: AI captured → player's last move gets penalised
                apply_capture_penalty(
                    move_str, player_color, "Player",
                    blue_rewards, red_rewards, all_reward_list,
                )

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
            wdl_str  = f"  {ai_wdl_blue} {ai_wdl_red}" if ai_wdl_blue is not None else ""
            print(f"  AI    ({label}) [{move_str}]  reward: {ai_turn_reward_acc:+.2f}"
                  f"{ai_val_str or ''}{wdl_str}  "
                  f"| Blue: {sum(blue_rewards):+.2f}  Red: {sum(red_rewards):+.2f}")

            # Capture penalty: AI captured → player's last move gets penalised
            apply_capture_penalty(
                move_str, player_color, "Player",
                blue_rewards, red_rewards, all_reward_list,
            )

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
                        help="Load a specific PPO epoch. Works for both the default "
                             "PPO-parallel model (e.g. --epoch 72) and league agents "
                             "(e.g. --agent aggressive --epoch 50). "
                             "If omitted, loads the latest checkpoint.")
    parser.add_argument("--az-epoch", type=int, default=None, dest="az_epoch",
                        help="Load a specific AlphaZero epoch (e.g. --az-epoch 82). "
                             "Automatically enables MCTS mode.")
    parser.add_argument("--az-version", type=str, default="v2", dest="az_version",
                        choices=["v1", "v2"],
                        help="Which AlphaZero checkpoint directory to use: "
                             "v2 (default) = alphazero_checkpoints/ (current training run); "
                             "v1 = alphazero_checkpoints_v1/ (previous run). "
                             "Architecture (scalar vs WDL) is auto-detected from the checkpoint.")
    parser.add_argument("--agent", type=str, default=None,
                        choices=["tactical", "terminal", "aggressive"],
                        help="Play against a league agent (e.g. --agent aggressive). "
                             "If omitted, loads the default PPO-parallel model.")
    args = parser.parse_args()

    # AlphaZero always plays with MCTS
    use_mcts = args.mcts or (args.az_epoch is not None)

    play_agent(use_mcts=use_mcts, num_simulations=args.simulations,
               epoch=args.epoch, az_epoch=args.az_epoch,
               az_version=args.az_version, agent_type=args.agent)
