"""Profile MCTS phase timing to quantify where wall-clock time is spent.

Measures cumulative time across all simulations in each phase:
  ROOT-EVAL  — single network call before the sim loop (root priors).
  CLONE      — env.fast_clone() per simulation.
  SELECT     — PUCT traversal of existing tree nodes + env.step calls.
  EVALUATE   — _evaluate() at the leaf node (network forward pass + softmax).
  EXPAND     — creating child MCTSNode objects from legal actions.
  BACKUP     — propagating the leaf value back up the search path.

Usage:
  python -m rl.scripts.profile_mcts                                  # DummyNetwork, CPU only
  python -m rl.scripts.profile_mcts --checkpoint path/to/model.pt   # real network, CPU only
  python -m rl.scripts.profile_mcts --checkpoint path/to/model.pt --compare  # CPU vs CUDA side-by-side
  python -m rl.scripts.profile_mcts --sims 400 --moves 10 --device cuda
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from checkers_game.constants import NUM_ACTIONS
from rl.envs import CheckersEnv, NumpyCheckersEnv
from rl.algorithms.mcts.mcts_search import MCTSSearch
from rl.algorithms.mcts.mcts_node import MCTSNode
from rl.configs.mcts_config import MCTSConfig

PHASES = ["ROOT-EVAL", "CLONE", "SELECT", "EVALUATE", "EXPAND", "BACKUP"]


# --------------------------------------------------------------------------- #
# Dummy network
# --------------------------------------------------------------------------- #

class DummyNetwork(nn.Module):
    """Uniform priors, value=0 — removes real NN cost to baseline tree ops."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, x):
        b = x.size(0)
        return (
            torch.zeros(b, NUM_ACTIONS, device=x.device),
            torch.zeros(b, 1, device=x.device),
        )


# --------------------------------------------------------------------------- #
# Instrumented MCTS
# --------------------------------------------------------------------------- #

class ProfiledMCTSSearch(MCTSSearch):
    """MCTSSearch with per-phase wall-clock timers injected into search()."""

    def reset_timers(self):
        self.t_root_eval  = 0.0
        self.t_clone      = 0.0
        self.t_select     = 0.0
        self.t_evaluate   = 0.0
        self.t_expand     = 0.0
        self.t_backup     = 0.0
        self.n_searches      = 0
        self.n_sims          = 0
        self.n_leaf_evals    = 0
        self.n_terminal_hits = 0

    def timings(self):
        """Return {phase: total_seconds} dict."""
        return {
            "ROOT-EVAL": self.t_root_eval,
            "CLONE":     self.t_clone,
            "SELECT":    self.t_select,
            "EVALUATE":  self.t_evaluate,
            "EXPAND":    self.t_expand,
            "BACKUP":    self.t_backup,
        }

    def counts(self):
        return {
            "ROOT-EVAL": self.n_searches,
            "CLONE":     self.n_sims,
            "SELECT":    self.n_sims,
            "EVALUATE":  self.n_leaf_evals,
            "EXPAND":    max(self.n_sims - self.n_terminal_hits, 1),
            "BACKUP":    self.n_sims,
        }

    def search(self, env, add_noise=False, no_progress_count=0):
        self.n_searches += 1

        if not isinstance(env, NumpyCheckersEnv):
            env = NumpyCheckersEnv.from_env(
                env,
                move_cap=self.move_cap,
                adjudicate_cap=self.config.MOVE_CAP_ADJUDICATE,
                no_progress_count=no_progress_count,
                no_progress_draw_moves=self.config.NO_PROGRESS_DRAW_MOVES,
            )

        root = self._root if self._root is not None else MCTSNode()

        t0 = time.perf_counter()
        priors, root_value, action_mask = self._evaluate(env)
        self.t_root_eval += time.perf_counter() - t0

        if action_mask.sum() == 0:
            self._root = None
            return np.zeros(NUM_ACTIONS, dtype=np.float32), -1.0

        if add_noise:
            priors = self._add_dirichlet_noise(priors, action_mask)

        if root.is_expanded:
            for action, child in root.children.items():
                child.prior = float(priors[action])
        else:
            t0 = time.perf_counter()
            self._expand_node(root, priors, action_mask)
            self.t_expand += time.perf_counter() - t0

        self._root = root
        root_player = env.game.turn

        for _ in range(self.num_simulations):
            self.n_sims += 1
            node = root

            t0 = time.perf_counter()
            env_copy = env.fast_clone()
            self.t_clone += time.perf_counter() - t0

            search_path = [(node, root_player)]

            t0 = time.perf_counter()
            while node.is_expanded and not node.is_terminal:
                best_c = node.best_child(self.c_puct)
                if best_c is None:
                    break
                node = best_c
                _, _, done, _, info = env_copy.step(node.action)
                node_player = env_copy.game.turn
                search_path.append((node, node_player))
                if done:
                    node.is_terminal = True
                    winner = info.get("winner", "None")
                    node.terminal_value = self._outcome_value(winner, node_player)
                    break
            self.t_select += time.perf_counter() - t0

            if node.is_terminal:
                leaf_value = node.terminal_value
                self.n_terminal_hits += 1
            else:
                t0 = time.perf_counter()
                priors, leaf_value, action_mask = self._evaluate(env_copy)
                self.t_evaluate += time.perf_counter() - t0
                self.n_leaf_evals += 1

                if action_mask.sum() == 0:
                    node.is_terminal = True
                    node.terminal_value = -1.0
                    leaf_value = -1.0
                else:
                    t0 = time.perf_counter()
                    self._expand_node(node, priors, action_mask)
                    self.t_expand += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._backup(search_path, leaf_value)
            self.t_backup += time.perf_counter() - t0

        action_probs = root.visit_count_distribution(NUM_ACTIONS)
        return action_probs, root_value


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_network(checkpoint, device):
    from rl.networks import AlphaZeroNetwork, WDLAlphaZeroNetwork
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    for cls, label in [(WDLAlphaZeroNetwork, "WDLAlphaZeroNetwork"),
                       (AlphaZeroNetwork,    "AlphaZeroNetwork")]:
        try:
            net = cls().to(device)
            net.load_state_dict(state)
            net.eval()
            return net, label
        except Exception:
            continue
    raise RuntimeError("Could not load checkpoint into any known network class.")


def run_profiled_moves(net, device, sims, moves, config, warmup_moves=5):
    """Warmup then profile; returns (ProfiledMCTSSearch, wall_seconds)."""
    env = CheckersEnv()

    # Warmup
    w = ProfiledMCTSSearch(network=net, num_simulations=sims, device=device, config=config)
    w.reset_timers()
    env.reset()
    for _ in range(min(warmup_moves, moves)):
        probs, _ = w.search(env, add_noise=False)
        action = int(np.argmax(probs))
        _, _, done, _, _ = env.step(action)
        w.update_root(action)
        if done:
            env.reset()
            w._root = None

    # Profiled run
    mcts = ProfiledMCTSSearch(network=net, num_simulations=sims, device=device, config=config)
    mcts.reset_timers()
    env.reset()
    t0 = time.perf_counter()
    for _ in range(moves):
        probs, _ = mcts.search(env, add_noise=False)
        action = int(np.argmax(probs))
        _, _, done, _, _ = env.step(action)
        mcts.update_root(action)
        if done:
            env.reset()
            mcts._root = None
    wall = time.perf_counter() - t0

    return mcts, wall


def print_single(mcts, wall, device_label, net_label, sims, moves):
    t = mcts.timings()
    c = mcts.counts()
    total = sum(t.values())

    print(f"\nMCTS Phase Profile  [{device_label}]")
    print(f"  Network : {net_label}")
    print(f"  Sims    : {sims}/move  |  Moves: {moves}")
    print(f"  Wall    : {wall:.3f}s  ({wall/moves*1000:.1f} ms/move)")
    print(f"\n{'Phase':<12} {'Total(ms)':>11} {'Pct':>7} {'Calls':>8} {'Per-call(µs)':>14}")
    print("-" * 58)
    for phase in PHASES:
        ms = t[phase] * 1000
        pct = 100.0 * t[phase] / total if total > 0 else 0.0
        n = c[phase]
        us = 1e6 * t[phase] / n if n > 0 else 0.0
        print(f"{phase:<12} {ms:>11.2f} {pct:>6.1f}% {n:>8} {us:>13.1f}")
    print("-" * 58)
    print(f"{'TOTAL':<12} {total*1000:>11.2f} {'100.0%':>7}")
    print(
        f"\nSims: {mcts.n_sims}  |  "
        f"Leaf evals: {mcts.n_leaf_evals}  |  "
        f"Terminal hits: {mcts.n_terminal_hits}"
    )


def print_comparison(cpu_mcts, cpu_wall, cuda_mcts, cuda_wall, net_label, sims, moves):
    ct = cpu_mcts.timings()
    gt = cuda_mcts.timings()
    cpu_total  = sum(ct.values())
    cuda_total = sum(gt.values())

    print(f"\nMCTS Phase Profile  [CPU vs CUDA]")
    print(f"  Network : {net_label}")
    print(f"  Sims    : {sims}/move  |  Moves: {moves}")
    print(f"  Wall    :  CPU {cpu_wall:.3f}s ({cpu_wall/moves*1000:.1f} ms/move)"
          f"   CUDA {cuda_wall:.3f}s ({cuda_wall/moves*1000:.1f} ms/move)"
          f"   Speedup {cpu_wall/cuda_wall:.2f}x")

    w1, w2, w3, w4, w5 = 12, 13, 9, 13, 9
    hdr = (f"\n{'Phase':<{w1}} {'CPU Total(ms)':>{w2}} {'CPU %':>{w3}}"
           f" {'CUDA Total(ms)':>{w4}} {'CUDA %':>{w5}} {'Speedup':>9}")
    sep = "-" * (w1 + w2 + w3 + w4 + w5 + 9 + 5)
    print(hdr)
    print(sep)
    for phase in PHASES:
        c_ms  = ct[phase] * 1000
        g_ms  = gt[phase] * 1000
        c_pct = 100.0 * ct[phase] / cpu_total  if cpu_total  > 0 else 0.0
        g_pct = 100.0 * gt[phase] / cuda_total if cuda_total > 0 else 0.0
        spd   = ct[phase] / gt[phase] if gt[phase] > 0 else float("inf")
        print(f"{phase:<{w1}} {c_ms:>{w2}.2f} {c_pct:>{w3}.1f}%"
              f" {g_ms:>{w4}.2f} {g_pct:>{w5}.1f}% {spd:>8.2f}x")
    print(sep)
    c_tot_ms = cpu_total  * 1000
    g_tot_ms = cuda_total * 1000
    tot_spd  = cpu_total / cuda_total if cuda_total > 0 else float("inf")
    print(f"{'TOTAL':<{w1}} {c_tot_ms:>{w2}.2f} {'100.0%':>{w3}}"
          f" {g_tot_ms:>{w4}.2f} {'100.0%':>{w5}} {tot_spd:>8.2f}x")
    print(
        f"\nCPU  — Sims: {cpu_mcts.n_sims}  Leaf evals: {cpu_mcts.n_leaf_evals}"
        f"  Terminal hits: {cpu_mcts.n_terminal_hits}"
    )
    print(
        f"CUDA — Sims: {cuda_mcts.n_sims}  Leaf evals: {cuda_mcts.n_leaf_evals}"
        f"  Terminal hits: {cuda_mcts.n_terminal_hits}"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Profile MCTS phase timing")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to AlphaZero checkpoint .pt file")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--moves", type=int, default=20,
                        help="Number of moves to profile (default: 20)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device when not using --compare (default: cpu)")
    parser.add_argument("--compare", action="store_true",
                        help="Run on both CPU and CUDA and print a side-by-side comparison")
    args = parser.parse_args()

    config = MCTSConfig()

    if args.compare:
        if not torch.cuda.is_available():
            print("ERROR: --compare requires CUDA, but torch.cuda.is_available() is False.")
            return
        if not args.checkpoint:
            print("NOTE: no --checkpoint supplied; using DummyNetwork (inference cost will be trivial).")

        # Load network twice — once per device
        if args.checkpoint:
            cpu_net,  net_label = load_network(args.checkpoint, torch.device("cpu"))
            cuda_net, _         = load_network(args.checkpoint, torch.device("cuda"))
        else:
            cpu_net  = DummyNetwork().cpu()
            cuda_net = DummyNetwork().cuda()
            net_label = "DummyNetwork"

        print(f"Profiling CPU...  ", end="", flush=True)
        cpu_mcts, cpu_wall = run_profiled_moves(
            cpu_net, torch.device("cpu"), args.sims, args.moves, config
        )
        print("done.")

        print(f"Profiling CUDA... ", end="", flush=True)
        cuda_mcts, cuda_wall = run_profiled_moves(
            cuda_net, torch.device("cuda"), args.sims, args.moves, config
        )
        print("done.")

        print_comparison(cpu_mcts, cpu_wall, cuda_mcts, cuda_wall,
                         net_label, args.sims, args.moves)

    else:
        device = torch.device(args.device)
        if args.checkpoint:
            net, net_label = load_network(args.checkpoint, device)
        else:
            net = DummyNetwork().to(device)
            net_label = "DummyNetwork (uniform priors, value=0)"

        print(f"Profiling {device}... ", end="", flush=True)
        mcts, wall = run_profiled_moves(net, device, args.sims, args.moves, config)
        print("done.")

        print_single(mcts, wall, str(device), net_label, args.sims, args.moves)


if __name__ == "__main__":
    main()
