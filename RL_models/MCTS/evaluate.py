"""Evaluation utilities for AlphaZero checkers training.

Provides:
  1. play_vs_random()          — absolute strength vs a random opponent.
  2. play_vs_network() / gate  — relative strength (gating) vs previous best.
  3. test_mcts_correctness()   — backup sign-convention sanity check using a
                                  dummy (uniform) network on a 2-move position.

Gating games use stochastic openings (temperature=1 for the first K moves)
to produce diverse game lines even when policies are sharp.
"""

import random as _random

import numpy as np
import torch
import torch.nn as nn

from RL_models.checkers_env import CheckersEnv
from RL_models.MCTS.mcts_search import MCTSSearch
from RL_models.MCTS.AlphaZeroNetwork import AlphaZeroNetwork
from RL_models.MCTS import training_config as cfg
from checkers_game.constants import (
    BLUE, RED, NUM_ACTIONS, ROWS, COLS,
    board_number_to_position, position_to_board_number, encode_action,
)
from checkers_game.piece import Piece


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def freeze_state_dict(model):
    """Deep-copy a model's state_dict so it is fully detached from the live model.

    PyTorch's state_dict() returns references to the live parameter tensors.
    A plain dict .copy() only copies the dict structure, not the tensors.
    After freeze_state_dict(), no tensor in the returned dict shares storage
    with the model — safe to keep across training epochs.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _adjudicate_move_cap(env):
    """Determine winner at move cap based on material (same rule as training)."""
    if not cfg.MOVE_CAP_ADJUDICATE:
        return "Tie"
    board = env.game.board.board
    blue_mat = sum(
        cfg.KING_MATERIAL_VALUE if (p != 0 and p.color == BLUE and p.king)
        else (1.0 if p != 0 and p.color == BLUE else 0.0)
        for row in board for p in row
    )
    red_mat = sum(
        cfg.KING_MATERIAL_VALUE if (p != 0 and p.color == RED and p.king)
        else (1.0 if p != 0 and p.color == RED else 0.0)
        for row in board for p in row
    )
    if blue_mat > red_mat:
        return BLUE
    elif red_mat > blue_mat:
        return RED
    return "Tie"


# ─────────────────────────────────────────────────────────────────────────────
# Game runner
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_OPENING_MOVES = 6  # stochastic opening depth for gating diversity

def _play_eval_game(blue_agent, red_agent, max_moves=150,
                    stochastic_opening=0):
    """Play one evaluation game between two agents.

    Args:
        blue_agent / red_agent: callable(env) -> action.
        max_moves: move cap (uses training adjudication rule, not always Tie).
        stochastic_opening: number of initial full moves played with
            temperature=1 sampling (for gating diversity).  0 = fully greedy.

    Returns (winner, num_moves).
    """
    env = CheckersEnv()
    env.reset()

    move_count = 0
    done = False
    info = {}

    while not done:
        if move_count >= max_moves:
            return _adjudicate_move_cap(env), move_count

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


def _make_mcts_agent(network, device, num_simulations=None,
                     stochastic_opening_moves=0):
    """Create an MCTS agent: greedy after opening, stochastic for first K moves.

    Args:
        stochastic_opening_moves: number of full turns to use temperature=1.
            After that, temperature=0 (greedy).  Provides game diversity for
            gating without injecting Dirichlet noise.
    """
    sims = num_simulations or cfg.NUM_SIMULATIONS
    mcts = MCTSSearch(
        network=network, num_simulations=sims,
        c_puct=cfg.C_PUCT, device=device,
    )
    move_counter = [0]

    def agent(env):
        mcts._root = None
        temp = 1.0 if move_counter[0] < stochastic_opening_moves else 0
        action, _, _ = mcts.select_action(
            env, temperature=temp, add_noise=False,
            no_progress_count=env.game._no_progress_count,
        )
        if env.get_action_mask().sum() > 0:
            info_peek = {}
            move_counter[0] += 1
        return action

    def reset():
        move_counter[0] = 0

    agent.reset = reset
    return agent


def _random_agent(env):
    """Select a uniformly random legal action."""
    mask = env.get_action_mask()
    valid = np.where(mask > 0)[0]
    return int(np.random.choice(valid))


# ─────────────────────────────────────────────────────────────────────────────
# 1) Absolute strength: network vs random
# ─────────────────────────────────────────────────────────────────────────────

def play_vs_random(network, device, num_games=40, num_simulations=100,
                   verbose=False):
    """Play games against a random opponent.  Half as BLUE, half as RED.

    Returns dict: wins, losses, ties, win_rate, score, games, avg_moves.
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

        if verbose and (i + 1) % 10 == 0:
            print(f"    vs-random: {i + 1}/{num_games} games  "
                  f"({wins}W/{losses}L/{ties}T so far)", flush=True)

    n = max(num_games, 1)
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / n,
        "score": (wins + 0.5 * ties) / n,
        "games": num_games,
        "avg_moves": total_moves / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2) Gating: new network vs previous best
# ─────────────────────────────────────────────────────────────────────────────

def play_vs_network(new_net, old_net, device, num_games=40,
                    num_simulations=100, verbose=False):
    """Play games between two networks.  Half as BLUE, half as RED.

    Uses stochastic openings (first K moves at temperature=1) to produce
    diverse games even when both policies are sharp/deterministic.

    Returns dict: wins, losses, ties, win_rate, score, games, avg_moves.
    """
    new_net.eval()
    old_net.eval()
    new_agent = _make_mcts_agent(
        new_net, device, num_simulations,
        stochastic_opening_moves=_EVAL_OPENING_MOVES,
    )
    old_agent = _make_mcts_agent(
        old_net, device, num_simulations,
        stochastic_opening_moves=_EVAL_OPENING_MOVES,
    )
    half = num_games // 2

    wins = losses = ties = 0
    total_moves = 0

    for i in range(num_games):
        new_agent.reset()
        old_agent.reset()
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

        if verbose and (i + 1) % 10 == 0:
            print(f"    vs-best:   {i + 1}/{num_games} games  "
                  f"({wins}W/{losses}L/{ties}T so far)", flush=True)

    n = max(num_games, 1)
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / n,
        "score": (wins + 0.5 * ties) / n,
        "games": num_games,
        "avg_moves": total_moves / n,
    }


def gate_checkpoint(new_net, old_net, device, num_games=40,
                    num_simulations=100, threshold=None, verbose=False):
    """Gating test: accept new_net if its score exceeds threshold.

    score = (wins + 0.5 * ties) / games
    This counts ties as half a win rather than punishing both sides.

    Returns (accepted: bool, stats: dict).
    """
    if threshold is None:
        threshold = cfg.GATE_THRESHOLD
    stats = play_vs_network(new_net, old_net, device, num_games,
                            num_simulations, verbose=verbose)
    accepted = stats["score"] >= threshold
    return accepted, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3) MCTS correctness: backup sign-convention test
# ─────────────────────────────────────────────────────────────────────────────

class _DummyNetwork(nn.Module):
    """Uniform priors (logits=0) and value=0 for all states.

    Strips out all network influence so the ONLY thing driving MCTS visit
    counts is terminal backup math — exactly what we're testing.
    """
    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, x):
        b = x.size(0)
        logits = torch.zeros(b, NUM_ACTIONS, device=x.device)
        value = torch.zeros(b, 1, device=x.device)
        return logits, value


def _setup_blocking_win_position():
    """Construct a position with 4 legal regular moves, exactly 1 winning.

    Mandatory captures in checkers make it nearly impossible to have 2+ legal
    actions at depth 1 where exactly one wins via piece removal:
      - 1 RED piece → all captures of it win (no discrimination)
      - 2 RED pieces → capturing either leaves one alive (neither wins)

    Instead, we use a "stalemate" win: BLUE's move BLOCKS the opponent's
    only piece, leaving RED with no legal moves.

    Board:
        BLUE king at (1,2) = sq 6      (king — moves in all 4 directions)
        RED  piece at (1,0) = sq 5      (regular — can only move to row 0)

    No captures available (not diag-adjacent for a jump).  BLUE king has
    4 legal regular moves:
        (0,1)=sq1   → BLOCKS RED.  RED at (1,0) tries (0,1): occupied.
                      (0,-1): off board.  No moves → BLUE wins.  [WIN]
        (0,3)=sq2   → RED can still move (1,0)→(0,1).  Game continues.
        (2,1)=sq9   → RED can still move (1,0)→(0,1).  Game continues.
        (2,3)=sq10  → RED can still move (1,0)→(0,1).  Game continues.

    With a dummy network (uniform priors, value=0), correct backup should
    direct >50% of visits to the winning move; inverted backup would avoid it.
    """
    from checkers_game.board import Board

    env = CheckersEnv()
    env.reset()

    board = Board.__new__(Board)
    board.board = [[0] * COLS for _ in range(ROWS)]

    blue_king = Piece(1, 2, BLUE)
    blue_king.make_king()
    board.board[1][2] = blue_king

    board.board[1][0] = Piece(1, 0, RED)

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
        position_to_board_number(1, 2),  # sq 6  (BLUE king)
        position_to_board_number(0, 1),  # sq 1  (blocking square)
    )

    return env, winning_action


def test_mcts_correctness(device, num_simulations=80):
    """Verify MCTS selects the winning action using a dummy (uniform) network.

    Uses a hand-constructed position with 4 legal regular moves, exactly 1
    of which wins by blocking the opponent's only piece (stalemate).  The
    dummy network outputs uniform priors and value=0, so ONLY the terminal
    backup math determines visit allocation.

    With correct backup:  winning move gets majority of visits (>50%).
    With inverted backup: winning move gets the FEWEST visits.

    Returns dict: passed, action_selected, winning_action, winning_visit_share,
                  root_value, num_legal_actions.
    """
    env, winning_action = _setup_blocking_win_position()

    mask = env.get_action_mask()
    num_legal = int(mask.sum())

    dummy_net = _DummyNetwork().to(device)
    dummy_net.eval()

    mcts = MCTSSearch(
        network=dummy_net, num_simulations=num_simulations,
        c_puct=cfg.C_PUCT, device=device,
    )
    mcts._root = None

    action_probs, root_value = mcts.search(env, add_noise=False)
    selected = int(np.argmax(action_probs))
    winning_share = float(action_probs[winning_action])

    passed = selected == winning_action and winning_share > 0.5

    return {
        "passed": passed,
        "action_selected": selected,
        "winning_action": winning_action,
        "winning_visit_share": round(winning_share, 4),
        "root_value": round(root_value, 4),
        "num_legal_actions": num_legal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(network, device, best_state_dict=None,
                   num_games_gate=40, eval_simulations=100, verbose=True):
    """Run gate evaluation and MCTS correctness test; return a summary dict.

    Args:
        network:          current AlphaZeroNetwork (eval mode).
        device:           torch device.
        best_state_dict:  deep-frozen state_dict of the reference model.
                          If None, gating is skipped.
        num_games_gate:   games to play for gating.
        eval_simulations: MCTS simulations per move during eval.
        verbose:          print per-game progress (default True).

    Returns dict: gate, gate_accepted, mcts_test.
    """
    network.eval()

    if verbose:
        print("  MCTS correctness test...", flush=True)
    mcts_result = test_mcts_correctness(device)

    gate_result = None
    gate_accepted = None
    if best_state_dict is not None:
        if verbose:
            print(f"  vs prev ({num_games_gate} games, "
                  f"{eval_simulations} sims/move)...", flush=True)
        input_shape = (4, 8, 8)
        best_net = AlphaZeroNetwork(input_shape, NUM_ACTIONS).to(device)
        best_net.load_state_dict(best_state_dict)
        best_net.eval()
        gate_accepted, gate_result = gate_checkpoint(
            network, best_net, device, num_games_gate, eval_simulations,
            verbose=verbose,
        )

    return {
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
          f"root_val={mt['root_value']:+.3f}, "
          f"legal={mt.get('num_legal_actions', '?')})")

    if eval_result["gate"] is not None:
        g = eval_result["gate"]
        accepted = "ACCEPTED" if eval_result["gate_accepted"] else "rejected"
        print(f"  vs Prev: {g['wins']}W / {g['losses']}L / {g['ties']}T  "
              f"(score={g['score']:.0%}) → {accepted}")
