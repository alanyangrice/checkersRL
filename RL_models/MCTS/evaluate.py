"""Evaluation utilities for AlphaZero checkers training.

Provides three capabilities:
  1. play_vs_random()   — absolute strength: win rate against a random opponent.
  2. play_vs_network()  — relative strength (gating): new net vs previous best.
  3. test_mcts_forced_win() — correctness: MCTS must pick the winning move in
                              a trivially solvable position.

All evaluation games use temperature=0 (greedy) and no Dirichlet noise, so
results reflect the network's actual policy rather than exploration noise.
"""

import numpy as np
import torch

from RL_models.checkers_env import CheckersEnv
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.MCTS.AlphaZeroNetwork import AlphaZeroNetwork
from RL_models.MCTS import training_config as cfg
from checkers_game.constants import (
    BLUE, RED, NUM_ACTIONS, ROWS, COLS,
    position_to_board_number, encode_action,
)
from checkers_game.piece import Piece


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation game runner (shared by all modes)
# ─────────────────────────────────────────────────────────────────────────────

def _play_eval_game(blue_agent, red_agent, max_moves=150):
    """Play one evaluation game between two agents.

    Each agent is a callable: agent(env) -> action.
    Returns (winner, num_moves).
    """
    env = CheckersEnv()
    env.reset()

    move_count = 0
    done = False
    info = {}

    while not done:
        if move_count >= max_moves:
            return "Tie", move_count

        mask = env.get_action_mask()
        if mask.sum() == 0:
            _, _, done, _, info = env.step(0)
            break

        if env.game.turn == BLUE:
            action = blue_agent(env)
        else:
            action = red_agent(env)

        _, _, done, _, info = env.step(action)
        if info.get("turn_complete", True):
            move_count += 1

    return info.get("winner", "Tie"), move_count


def _make_mcts_agent(network, device, num_simulations=None):
    """Create an MCTS agent function: agent(env) -> action."""
    sims = num_simulations or cfg.NUM_SIMULATIONS
    mcts = MCTSSearch(
        network=network, num_simulations=sims,
        c_puct=cfg.C_PUCT, device=device,
    )

    def agent(env):
        mcts._root = None
        action, _, _ = mcts.select_action(env, temperature=0, add_noise=False)
        return action

    return agent


def _random_agent(env):
    """Select a uniformly random legal action."""
    mask = env.get_action_mask()
    valid = np.where(mask > 0)[0]
    return int(np.random.choice(valid))


# ─────────────────────────────────────────────────────────────────────────────
# 1) Absolute strength: network vs random
# ─────────────────────────────────────────────────────────────────────────────

def play_vs_random(network, device, num_games=40, num_simulations=100):
    """Play games against a random opponent and return win/tie/loss stats.

    Half the games are played as BLUE, half as RED, to remove first-move bias.

    Args:
        network:         AlphaZeroNetwork in eval mode.
        device:          torch device for inference.
        num_games:       total games to play (split evenly BLUE/RED).
        num_simulations: MCTS simulations per move during eval.

    Returns:
        dict with keys: wins, losses, ties, win_rate, games, avg_moves
    """
    network.eval()
    mcts_agent = _make_mcts_agent(network, device, num_simulations)
    half = num_games // 2

    wins = losses = ties = 0
    total_moves = 0

    for i in range(num_games):
        if i < half:
            winner, moves = _play_eval_game(mcts_agent, _random_agent)
            net_color = BLUE
        else:
            winner, moves = _play_eval_game(_random_agent, mcts_agent)
            net_color = RED

        total_moves += moves
        if winner == net_color:
            wins += 1
        elif winner == "Tie":
            ties += 1
        else:
            losses += 1

    return {
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / num_games if num_games > 0 else 0.0,
        "games": num_games,
        "avg_moves": total_moves / num_games if num_games > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2) Gating: new network vs previous best
# ─────────────────────────────────────────────────────────────────────────────

def play_vs_network(new_net, old_net, device, num_games=40,
                    num_simulations=100):
    """Play games between two networks and return the new net's stats.

    Half the games have new_net as BLUE, half as RED.

    Args:
        new_net:         candidate AlphaZeroNetwork (eval mode).
        old_net:         current best AlphaZeroNetwork (eval mode).
        device:          torch device.
        num_games:       total games.
        num_simulations: MCTS sims per move.

    Returns:
        dict with keys: wins, losses, ties, win_rate, games, avg_moves
    """
    new_net.eval()
    old_net.eval()
    new_agent = _make_mcts_agent(new_net, device, num_simulations)
    old_agent = _make_mcts_agent(old_net, device, num_simulations)
    half = num_games // 2

    wins = losses = ties = 0
    total_moves = 0

    for i in range(num_games):
        if i < half:
            winner, moves = _play_eval_game(new_agent, old_agent)
            new_color = BLUE
        else:
            winner, moves = _play_eval_game(old_agent, new_agent)
            new_color = RED

        total_moves += moves
        if winner == new_color:
            wins += 1
        elif winner == "Tie":
            ties += 1
        else:
            losses += 1

    return {
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / num_games if num_games > 0 else 0.0,
        "games": num_games,
        "avg_moves": total_moves / num_games if num_games > 0 else 0,
    }


def gate_checkpoint(new_net, old_net, device, num_games=40,
                    num_simulations=100, threshold=0.55):
    """Gating test: accept new_net if it beats old_net above threshold.

    Returns:
        (accepted: bool, stats: dict)
    """
    stats = play_vs_network(new_net, old_net, device, num_games,
                            num_simulations)
    accepted = stats["win_rate"] >= threshold
    return accepted, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3) MCTS correctness: forced-win test
# ─────────────────────────────────────────────────────────────────────────────

def _setup_forced_win_env():
    """Create a trivial position: BLUE has one capture that wins the game.

    Board:
        BLUE piece at (5, 0) = square 21
        RED  piece at (6, 1) = square 25

    BLUE must capture (mandatory captures): 21 → 30, jumping over RED at 25.
    This removes RED's last piece → BLUE wins immediately.

    With correct backup, MCTS should assign ~100% visits to the winning
    action.  With inverted backup (the old bug), it would avoid it.
    """
    from checkers_game.board import Board

    env = CheckersEnv()
    env.reset()

    board = Board.__new__(Board)
    board.board = [[0] * COLS for _ in range(ROWS)]

    board.board[5][0] = Piece(5, 0, BLUE)
    board.board[6][1] = Piece(6, 1, RED)

    env.game.board = board
    env.game.turn = BLUE
    env.game.board_states = {}
    env.game.moves = []
    env.game.num_moves = 0
    env._capture_in_progress = False
    env._capturing_piece_sq = None
    env._visited_squares = set()
    env._current_move_chain = []
    env._is_capture_turn = False
    env._turn_start_board = None
    env._update_action_mask()

    winning_action = encode_action(
        position_to_board_number(5, 0),  # sq 21
        position_to_board_number(7, 2),  # sq 30
    )

    return env, winning_action


def test_mcts_forced_win(network, device, num_simulations=50):
    """Verify MCTS selects the winning action in a forced-win position.

    This is a correctness test for the backup sign convention.
    With inverted backup, MCTS would pick a non-winning action.

    Returns:
        dict with passed (bool), action_selected, winning_action,
        winning_visit_share (fraction of visits on the winning action)
    """
    network.eval()
    env, winning_action = _setup_forced_win_env()

    mcts = MCTSSearch(
        network=network, num_simulations=num_simulations,
        c_puct=cfg.C_PUCT, device=device,
    )
    mcts._root = None

    action_probs, root_value = mcts.search(env, add_noise=False)
    selected = int(np.argmax(action_probs))
    winning_share = float(action_probs[winning_action])

    passed = selected == winning_action

    return {
        "passed": passed,
        "action_selected": selected,
        "winning_action": winning_action,
        "winning_visit_share": round(winning_share, 4),
        "root_value": round(root_value, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation runner (called from training loop)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(network, device, best_state_dict=None,
                   num_games_random=40, num_games_gate=40,
                   eval_simulations=100):
    """Run all evaluation suites and return a summary dict.

    Args:
        network:          current AlphaZeroNetwork (eval mode).
        device:           torch device.
        best_state_dict:  state_dict of the current best model for gating.
                          If None, gating is skipped.
        num_games_random: games to play vs random.
        num_games_gate:   games to play for gating.
        eval_simulations: MCTS simulations per move during eval.

    Returns:
        dict with keys:
            vs_random     — play_vs_random result dict
            gate          — gate_checkpoint result dict (or None)
            gate_accepted — bool (or None)
            mcts_test     — test_mcts_forced_win result dict
    """
    network.eval()

    # Correctness test
    mcts_result = test_mcts_forced_win(network, device)

    # Absolute strength
    random_result = play_vs_random(
        network, device, num_games_random, eval_simulations
    )

    # Gating
    gate_result = None
    gate_accepted = None
    if best_state_dict is not None:
        input_shape = (4, 8, 8)
        best_net = AlphaZeroNetwork(input_shape, NUM_ACTIONS).to(device)
        best_net.load_state_dict(best_state_dict)
        best_net.eval()
        gate_accepted, gate_result = gate_checkpoint(
            network, best_net, device, num_games_gate, eval_simulations
        )

    return {
        "vs_random": random_result,
        "gate": gate_result,
        "gate_accepted": gate_accepted,
        "mcts_test": mcts_result,
    }


def print_evaluation(eval_result, epoch):
    """Pretty-print evaluation results."""
    print(f"\n  ── Evaluation (epoch {epoch}) ──")

    mt = eval_result["mcts_test"]
    status = "PASS" if mt["passed"] else "FAIL"
    print(f"  MCTS correctness: {status}  "
          f"(winning_visits={mt['winning_visit_share']:.0%}, "
          f"root_val={mt['root_value']:+.3f})")

    vr = eval_result["vs_random"]
    print(f"  vs Random: {vr['wins']}W / {vr['losses']}L / {vr['ties']}T  "
          f"({vr['win_rate']:.0%} win rate, "
          f"avg {vr['avg_moves']:.0f} moves)")

    if eval_result["gate"] is not None:
        g = eval_result["gate"]
        accepted = "ACCEPTED" if eval_result["gate_accepted"] else "rejected"
        print(f"  vs Best: {g['wins']}W / {g['losses']}L / {g['ties']}T  "
              f"({g['win_rate']:.0%}) → {accepted}")
