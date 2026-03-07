"""Run a match between two AlphaZero checkpoints, or an AlphaZero vs PPO checkpoint.

Usage examples
--------------
# Latest AZ vs latest PPO-parallel (10 games):
    python -m RL_models.az_vs_ppo

# Specific epochs, more games, more MCTS sims:
    python -m RL_models.az_vs_ppo --az-epoch 100 --ppo-epoch 200 --games 20 --simulations 200

# AZ vs a league agent:
    python -m RL_models.az_vs_ppo --az-epoch 100 --agent aggressive --games 20

# AZ vs AZ  (v2 epoch 160 vs v1 epoch 135, 20 games, 200 sims each):
    python -m RL_models.az_vs_ppo --az2-epoch 135 --az2-version v1 --az-epoch 160 --games 20 --simulations 200

# AZ vs AZ with independent sim budgets:
    python -m RL_models.az_vs_ppo --az2-epoch 135 --az2-version v1 --az2-simulations 100 --simulations 200
"""

import os
import sys
import argparse

import numpy as np
import torch

from checkers_game.constants import BLUE, RED, NUM_ACTIONS
from RL_models.MCTS.AlphaZeroNetwork import AlphaZeroNetwork
from RL_models.MCTS.WDLAlphaZeroNetwork import WDLAlphaZeroNetwork
from RL_models.PPO_Model.PolicyNetwork import PPOPolicyNetwork
from RL_models.PPO_Model.Agent import PPOAgent
from RL_models.MCTS.evaluate import _play_eval_game, _make_mcts_agent
from RL_models.MCTS import training_config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_az(device, epoch=None, version="v2"):
    base = os.path.dirname(os.path.abspath(__file__))
    dir_name = "alphazero_checkpoints_v1" if version == "v1" else "alphazero_checkpoints"
    az_dir = os.path.join(base, "MCTS", dir_name)

    if epoch is not None:
        path = os.path.join(az_dir, f"az_epoch_{epoch}.pt")
    else:
        ckpts = [f for f in os.listdir(az_dir) if f.startswith("az_epoch_") and f.endswith(".pt")]
        if not ckpts:
            print(f"ERROR: no AlphaZero checkpoints in {az_dir}")
            sys.exit(1)
        latest = max(ckpts, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        path = os.path.join(az_dir, latest)

    if not os.path.exists(path):
        print(f"ERROR: AlphaZero checkpoint not found: {path}")
        sys.exit(1)

    ep = int(os.path.basename(path).split("_")[-1].split(".")[0])
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Auto-detect WDL vs scalar from value_fc2.weight shape
    is_wdl = ckpt["model_state_dict"]["value_fc2.weight"].shape[0] == 3
    NetworkClass = WDLAlphaZeroNetwork if is_wdl else AlphaZeroNetwork
    arch = "WDL" if is_wdl else "scalar"
    print(f"Loading AlphaZero {version} epoch {ep} ({arch}): {path}")
    net = NetworkClass((4, 8, 8), n_actions=NUM_ACTIONS).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, f"AlphaZero-{version}-ep{ep}({arch})"


def _load_ppo(device, epoch=None, agent_type=None):
    base = os.path.dirname(os.path.abspath(__file__))

    if agent_type is not None:
        model_dir = os.path.join(base, "PPO_Model", f"PPO_saved_models_{agent_type}")
        prefix = "agent_epoch_"
        label_prefix = f"PPO-{agent_type}"
    else:
        model_dir = os.path.join(base, "PPO_Model", "PPO_saved_models_parallel")
        prefix = "agent_epoch_"
        label_prefix = "PPO-parallel"

    if not os.path.exists(model_dir):
        print(f"ERROR: PPO model directory not found: {model_dir}")
        sys.exit(1)

    if epoch is not None:
        path = os.path.join(model_dir, f"agent_epoch_{epoch}.pt")
    else:
        ckpts = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not ckpts:
            print(f"ERROR: no PPO checkpoints in {model_dir}")
            sys.exit(1)
        latest = max(ckpts, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        path = os.path.join(model_dir, latest)

    if not os.path.exists(path):
        print(f"ERROR: PPO checkpoint not found: {path}")
        sys.exit(1)

    ep = int(os.path.basename(path).split("_")[-1].split(".")[0])
    print(f"Loading {label_prefix} epoch {ep}: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PPOPolicyNetwork((4, 8, 8), NUM_ACTIONS).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, f"{label_prefix}-ep{ep}"


# ──────────────────────────────────────────────────────────────────────────────
# PPO agent wrapper  (same interface as _make_mcts_agent: callable(env)->action)
# ──────────────────────────────────────────────────────────────────────────────

def _make_ppo_agent(network, device):
    ppo = PPOAgent((4, 8, 8), NUM_ACTIONS, device=device)
    ppo.policy = network

    def agent(env):
        state = env.get_board_state()
        mask  = env.get_action_mask()
        action, _, _ = ppo.select_action(state, mask)
        return action

    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Match runners
# ──────────────────────────────────────────────────────────────────────────────

def _print_match_header(label_a, label_b, num_games, sims_a, sims_b=None):
    sims_str = (f"{sims_a} sims/move" if sims_b is None or sims_b == sims_a
                else f"{sims_a} vs {sims_b} sims/move")
    col_w = max(len(label_a), len(label_b), 22)
    print(f"\n{'='*70}")
    print(f"  {label_a}")
    print(f"    vs  {label_b}")
    print(f"  {num_games} games  |  {sims_str}")
    print(f"{'='*70}")
    print(f"  {'#':>4}  {'A color':<6}  {'Winner':<{col_w}}  {'Moves':>5}  {'Score':>6}")
    print(f"  {'-'*(col_w + 26)}")
    return col_w


def run_match(az_net, az_label, ppo_net, ppo_label, device,
              num_games, num_simulations, stochastic_opening):
    """AlphaZero (MCTS) vs PPO match."""
    az_agent  = _make_mcts_agent(az_net,  device, num_simulations,
                                 stochastic_opening_moves=stochastic_opening)
    ppo_agent = _make_ppo_agent(ppo_net, device)

    half = num_games // 2
    az_wins = az_losses = ties = 0
    total_moves = 0

    col_w = _print_match_header(az_label, ppo_label, num_games, num_simulations)

    for i in range(num_games):
        az_color = BLUE if i < half else RED

        if az_color == BLUE:
            winner, moves = _play_eval_game(az_agent, ppo_agent,
                                            max_moves=cfg.MAX_GAME_MOVES_FULL,
                                            stochastic_opening=stochastic_opening)
        else:
            winner, moves = _play_eval_game(ppo_agent, az_agent,
                                            max_moves=cfg.MAX_GAME_MOVES_FULL,
                                            stochastic_opening=stochastic_opening)

        total_moves += moves

        if winner == az_color:
            az_wins += 1
            result = f"{az_label} wins"
        elif winner == "Tie":
            ties += 1
            result = "Tie"
        else:
            az_losses += 1
            result = f"{ppo_label} wins"

        az_color_str = "BLUE" if az_color == BLUE else "RED"
        score = (az_wins + 0.5 * ties) / (i + 1)
        print(f"  {i+1:>4}  {az_color_str:<6}  {result:<{col_w}}  {moves:>5}  {score:>6.3f}")

        if hasattr(az_agent, "reset"):
            az_agent.reset()

    n = max(num_games, 1)
    print(f"  {'-'*(col_w + 26)}")
    print(f"\n  Results for {az_label}:")
    print(f"    Wins:    {az_wins:>3}  ({100*az_wins/n:.1f}%)")
    print(f"    Losses:  {az_losses:>3}  ({100*az_losses/n:.1f}%)")
    print(f"    Ties:    {ties:>3}  ({100*ties/n:.1f}%)")
    print(f"    Score:   {(az_wins + 0.5*ties)/n:.3f}  (0.5 = even)")
    print(f"    Avg moves per game: {total_moves/n:.1f}")
    print(f"{'='*70}\n")


def run_az_vs_az_match(net_a, label_a, net_b, label_b, device,
                       num_games, sims_a, sims_b, stochastic_opening):
    """AlphaZero vs AlphaZero match with independent MCTS budgets."""
    agent_a = _make_mcts_agent(net_a, device, sims_a,
                               stochastic_opening_moves=stochastic_opening)
    agent_b = _make_mcts_agent(net_b, device, sims_b,
                               stochastic_opening_moves=stochastic_opening)

    half = num_games // 2
    a_wins = a_losses = ties = 0
    total_moves = 0

    col_w = _print_match_header(label_a, label_b, num_games, sims_a, sims_b)

    for i in range(num_games):
        # Alternate colors every game; A plays BLUE for the first half
        a_color = BLUE if i < half else RED

        if a_color == BLUE:
            winner, moves = _play_eval_game(agent_a, agent_b,
                                            max_moves=cfg.MAX_GAME_MOVES_FULL,
                                            stochastic_opening=stochastic_opening)
        else:
            winner, moves = _play_eval_game(agent_b, agent_a,
                                            max_moves=cfg.MAX_GAME_MOVES_FULL,
                                            stochastic_opening=stochastic_opening)

        total_moves += moves

        if winner == a_color:
            a_wins += 1
            result = f"{label_a} wins"
        elif winner == "Tie":
            ties += 1
            result = "Tie"
        else:
            a_losses += 1
            result = f"{label_b} wins"

        a_color_str = "BLUE" if a_color == BLUE else "RED"
        score = (a_wins + 0.5 * ties) / (i + 1)
        print(f"  {i+1:>4}  {a_color_str:<6}  {result:<{col_w}}  {moves:>5}  {score:>6.3f}")

        if hasattr(agent_a, "reset"):
            agent_a.reset()
        if hasattr(agent_b, "reset"):
            agent_b.reset()

    n = max(num_games, 1)
    print(f"  {'-'*(col_w + 26)}")
    print(f"\n  Results for {label_a}:")
    print(f"    Wins:    {a_wins:>3}  ({100*a_wins/n:.1f}%)")
    print(f"    Losses:  {a_losses:>3}  ({100*a_losses/n:.1f}%)")
    print(f"    Ties:    {ties:>3}  ({100*ties/n:.1f}%)")
    print(f"    Score:   {(a_wins + 0.5*ties)/n:.3f}  (0.5 = even)")
    print(f"    Avg moves per game: {total_moves/n:.1f}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaZero vs PPO match, or AlphaZero vs AlphaZero match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model A (primary AZ) ──────────────────────────────────────────────────
    parser.add_argument("--az-epoch", type=int, default=None, dest="az_epoch",
                        help="AlphaZero (model A) epoch to load (default: latest)")
    parser.add_argument("--az-version", type=str, default="v2", dest="az_version",
                        choices=["v1", "v2"],
                        help="v2 (default) = alphazero_checkpoints/; "
                             "v1 = alphazero_checkpoints_v1/. "
                             "Architecture (scalar vs WDL) is auto-detected.")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move for model A (default: 100)")

    # ── Model B: either a second AZ or a PPO ─────────────────────────────────
    az2_group = parser.add_argument_group(
        "AZ vs AZ",
        "Supply --az2-epoch to run an AlphaZero vs AlphaZero match instead of AZ vs PPO.",
    )
    az2_group.add_argument("--az2-epoch", type=int, default=None, dest="az2_epoch",
                           help="Second AlphaZero epoch (triggers AZ-vs-AZ mode)")
    az2_group.add_argument("--az2-version", type=str, default="v2", dest="az2_version",
                           choices=["v1", "v2"],
                           help="Checkpoint directory for model B (default: v2)")
    az2_group.add_argument("--az2-simulations", type=int, default=None, dest="az2_simulations",
                           help="MCTS simulations per move for model B "
                                "(default: same as --simulations)")

    ppo_group = parser.add_argument_group(
        "AZ vs PPO",
        "Options used only in AZ-vs-PPO mode (when --az2-epoch is NOT given).",
    )
    ppo_group.add_argument("--ppo-epoch", type=int, default=None, dest="ppo_epoch",
                           help="PPO epoch to load (default: latest)")
    ppo_group.add_argument("--agent", type=str, default=None,
                           choices=["tactical", "terminal", "aggressive"],
                           help="Use a league PPO agent instead of PPO-parallel")

    # ── Shared ───────────────────────────────────────────────────────────────
    parser.add_argument("--games", type=int, default=10,
                        help="Total games to play (split evenly BLUE/RED, default: 10)")
    parser.add_argument("--opening", type=int, default=0,
                        help="Stochastic opening moves per game (default: 0)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    az_net, az_label = _load_az(device, epoch=args.az_epoch, version=args.az_version)

    if args.az2_epoch is not None:
        # ── AZ vs AZ mode ────────────────────────────────────────────────────
        az2_net, az2_label = _load_az(device, epoch=args.az2_epoch, version=args.az2_version)
        sims_b = args.az2_simulations if args.az2_simulations is not None else args.simulations

        run_az_vs_az_match(
            az_net, az_label, az2_net, az2_label, device,
            num_games=args.games,
            sims_a=args.simulations,
            sims_b=sims_b,
            stochastic_opening=args.opening,
        )
    else:
        # ── AZ vs PPO mode ───────────────────────────────────────────────────
        ppo_net, ppo_label = _load_ppo(device, epoch=args.ppo_epoch, agent_type=args.agent)

        run_match(
            az_net, az_label, ppo_net, ppo_label, device,
            num_games=args.games,
            num_simulations=args.simulations,
            stochastic_opening=args.opening,
        )
