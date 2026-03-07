"""Shared MCTS training utilities: outcome conversion, adjudication, curriculum, buffer stats, training loop helpers."""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

from checkers_game.constants import BLUE, RED

from rl.training_utils.checkpoint_utils import find_latest_checkpoint_path
from rl.configs.mcts_config import MCTSConfig

default_config = MCTSConfig()



def outcome_to_wdl(v):
    """Convert a scalar outcome in [-1, 1] to a WDL probability distribution.

    Maps v = P(win) - P(loss) to [P(win), P(draw), P(loss)]:
      v = +1.0  → [1, 0, 0]  (certain win)
      v =  0.0  → [0, 1, 0]  (certain draw)
      v = -1.0  → [0, 0, 1]  (certain loss)
      v = +0.3  → [0.3, 0.7, 0]  (partial win signal)
      v = -0.3  → [0, 0.7, 0.3]  (contempt draw — draw leaning toward loss)

    Used for:
      1. Converting the final game outcome scalar to a WDL training target.
      2. Converting the MCTS Q-value to a WDL distribution for soft-Z blending.
    """
    w = float(max(0.0, v))
    l = float(max(0.0, -v))
    d = max(0.0, 1.0 - w - l)
    return np.array([w, d, l], dtype=np.float32)


def get_material(env, config=None):
    config = config or default_config
    """Return (blue_material, red_material), kings weighted by KING_MATERIAL_VALUE."""
    board = env.game.board.board
    blue_mat = sum(
        config.KING_MATERIAL_VALUE if (p != 0 and p.color == BLUE and p.king)
        else (1.0 if p != 0 and p.color == BLUE else 0.0)
        for row in board for p in row
    )
    red_mat = sum(
        config.KING_MATERIAL_VALUE if (p != 0 and p.color == RED and p.king)
        else (1.0 if p != 0 and p.color == RED else 0.0)
        for row in board for p in row
    )
    return blue_mat, red_mat


def adjudicate_move_cap(env, config=None):
    config = config or default_config
    """Determine winner at move cap based on material (zero-sum compatible).

    Kings count as KING_MATERIAL_VALUE pieces.  Side with more material wins.
    Equal material → Tie.
    """
    if not config.MOVE_CAP_ADJUDICATE:
        return "Tie"
    blue_mat, red_mat = get_material(env)
    if blue_mat > red_mat:
        return BLUE
    elif red_mat > blue_mat:
        return RED
    return "Tie"


def get_curriculum_options(epoch, config=None):
    """Generate guaranteed-asymmetric board options for the current curriculum phase.

    The weak side draws its piece count first, then the strong side draws from
    [weak+1, strong_max], guaranteeing a strict material advantage on every game.
    Which color is the strong side is re-rolled 50/50 each game so both BLUE and
    RED learn to play from both material situations equally.

    Phase 1: weak ∈ [WEAK_MIN, WEAK_MAX], strong ∈ [weak+1, STRONG_MAX] — endgame
    Phase 2: weak ∈ [WEAK_MIN, WEAK_MAX], strong ∈ [weak+1, STRONG_MAX] — mid-game
    Phase 3 (full): returns None for standard 12v12.
    """
    config = config or default_config
    if epoch < config.CURRICULUM_PHASE1_END:
        weak = random.randint(config.CURRICULUM_PHASE1_WEAK_MIN, config.CURRICULUM_PHASE1_WEAK_MAX)
        strong = random.randint(weak + 1, config.CURRICULUM_PHASE1_STRONG_MAX)
    elif epoch < config.CURRICULUM_PHASE2_END:
        weak = random.randint(config.CURRICULUM_PHASE2_WEAK_MIN, config.CURRICULUM_PHASE2_WEAK_MAX)
        strong = random.randint(weak + 1, config.CURRICULUM_PHASE2_STRONG_MAX)
    else:
        return None

    if random.random() < 0.5:
        return {"num_blue": strong, "num_red": weak}
    return {"num_blue": weak, "num_red": strong}


def compute_buffer_entropy(replay_buffer, sample_size=2048):
    """Compute mean policy entropy (nats) over a random sample of the replay buffer."""
    if len(replay_buffer) < 1:
        return 0.0
    sample_n = min(sample_size, len(replay_buffer))
    sample = random.sample(list(replay_buffer), sample_n)
    policies = np.array([s[1] for s in sample])
    eps = 1e-10
    return float(-(policies * np.log(policies + eps)).sum(axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop helpers (used by train_gpu_parallel)
# ─────────────────────────────────────────────────────────────────────────────


def resume_from_checkpoint(model_dir, buffer_path, network, optimizer, device,
                           network_type_str, use_wdl, replay_buffer, config=None):
    """Load latest checkpoint and replay buffer. Returns start_epoch or 0 if none."""
    config = config or default_config
    
    # If gating is enabled, the true "latest state" to resume from is the reference model,
    # because standard checkpoints might contain weights that were rejected by the gate.
    if config.GATE_ENABLED:
        ref_path = os.path.join(model_dir, "az_reference_model.pt")
        latest_path = find_latest_checkpoint_path(model_dir, prefix="az_epoch_")
        
        # If there's an az_epoch_N.pt that is strictly newer than the reference model,
        # it means the last run was killed *before* a gating test happened for that epoch.
        # So we should resume from the latest epoch, not the older reference model.
        if latest_path is not None and os.path.exists(ref_path):
            latest_epoch = int(os.path.basename(latest_path).split('_')[-1].split('.')[0])
            
            ref_ckpt = torch.load(ref_path, map_location=device, weights_only=False)
            ref_epoch = ref_ckpt.get("epoch", 0)
            
            # If the latest checkpoint is exactly the same epoch as the reference, we use the reference.
            # If the latest checkpoint is newer (e.g. 140 vs ref 105), we use the latest checkpoint.
            if latest_epoch > ref_epoch:
                path = latest_path
            else:
                path = ref_path
        elif os.path.exists(ref_path):
            path = ref_path
        else:
            path = latest_path
    else:
        path = find_latest_checkpoint_path(model_dir, prefix="az_epoch_")

    if path is None or not os.path.exists(path):
        return 0

    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # Validation checks
    if "network_type" in ckpt: # Reference models might not have this key, standard checkpoints do
        ckpt_network_type = ckpt.get("network_type", "scalar")
        if ckpt_network_type != network_type_str:
            raise RuntimeError(
                f"Checkpoint network_type='{ckpt_network_type}' does not match "
                f"--network-type '{network_type_str}'. Use the correct flag or "
                f"start a new run from scratch."
            )
            
    # Backward compatibility with unified DualHeadResNet
    state_dict = ckpt["model_state_dict"]
    
    model_has_net = any(k.startswith('net.') for k in network.state_dict().keys())
    ckpt_has_net = any(k.startswith('net.') for k in state_dict.keys())
    
    if model_has_net and not ckpt_has_net:
        state_dict = {f"net.{k}": v for k, v in state_dict.items()}
    elif ckpt_has_net and not model_has_net:
        state_dict = {k.replace('net.', '', 1): v for k, v in state_dict.items() if k.startswith('net.')}
        
    network.load_state_dict(state_dict)
    
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    start_epoch = ckpt.get("epoch", 0)
    for pg in optimizer.param_groups:
        pg["lr"] = config.get_lr(start_epoch)
        
    if "rng_state_torch" in ckpt:
        torch.random.set_rng_state(ckpt["rng_state_torch"].cpu())
    if "rng_state_numpy" in ckpt:
        np.random.set_state(ckpt["rng_state_numpy"])
    if "rng_state_python" in ckpt:
        import random
        random.setstate(ckpt["rng_state_python"])
        
    print(f"Resumed from epoch {start_epoch} (using {os.path.basename(path)})")

    # When resuming from a reference model, load the reference buffer instead of the live buffer
    if config.GATE_ENABLED and os.path.basename(path) == "az_reference_model.pt":
        target_buffer_path = os.path.join(model_dir, "az_reference_buffer.npz")
    else:
        target_buffer_path = buffer_path

    if os.path.exists(target_buffer_path):
        buf_data = np.load(target_buffer_path)
        states, policies, outcomes = (
            buf_data["states"], buf_data["policies"], buf_data["outcomes"]
        )
        for s, p, o in zip(states, policies, outcomes):
            target = o if use_wdl else float(o)
            replay_buffer.append((s, p, target))
        print(f"  Replay buffer restored from {os.path.basename(target_buffer_path)}: {len(replay_buffer):,} positions")
    elif os.path.exists(os.path.join(model_dir, "az_reference_buffer.npz")): # Fallback to reference buffer if live buffer is missing
        fallback_path = os.path.join(model_dir, "az_reference_buffer.npz")
        buf_data = np.load(fallback_path)
        states, policies, outcomes = (
            buf_data["states"], buf_data["policies"], buf_data["outcomes"]
        )
        for s, p, o in zip(states, policies, outcomes):
            target = o if use_wdl else float(o)
            replay_buffer.append((s, p, target))
        print(f"  Replay buffer restored from {os.path.basename(fallback_path)} (fallback): {len(replay_buffer):,} positions")
        
    return start_epoch


def init_training_csv_headers(csv_path, eval_csv_path):
    """Write CSV headers for a fresh run."""
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "games", "blue_wins", "red_wins", "ties",
            "avg_moves", "policy_loss", "value_loss", "total_loss",
            "avg_grad_norm", "policy_entropy_nats",
            "avg_root_val_winner", "avg_root_val_loser",
            "buffer_size", "num_workers", "epoch_time_s",
        ])
    with open(eval_csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "prev_eval_epoch",
            "gate_wins", "gate_losses", "gate_ties",
            "gate_win_rate", "gate_score", "gate_accepted",
            "mcts_test_passed", "mcts_winning_visit_share", "mcts_root_value",
            "vs_random_wins", "vs_random_losses", "vs_random_ties",
            "vs_random_score",
            "val_clear_win", "val_clear_loss", "val_equal", "val_calibrated",
        ])


def populate_replay_buffer(all_results, replay_buffer, regret_buffer, use_wdl,
                          detailed_csv_path, detailed_headers, epoch, config=None):
    """Process game results into replay and regret buffers; write detailed CSV.
    Returns (new_positions, stats_dict) with blue_wins, red_wins, ties, etc.
    """
    config = config or default_config
    blue_wins = red_wins = ties = total_moves = new_positions = 0
    total_root_val_winner = 0.0
    total_root_val_loser = 0.0
    total_entropy = 0.0

    with open(detailed_csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(detailed_headers)

    for game_idx, res in enumerate(all_results):
        winner = res["winner"]
        game_data = res["game_data"]
        game_stats = res["game_stats"]
        total_moves += res["num_moves"]

        if winner == BLUE:
            blue_wins += 1
        elif winner == RED:
            red_wins += 1
        else:
            ties += 1

        total_root_val_winner += game_stats["avg_root_val_winner"]
        total_root_val_loser += game_stats["avg_root_val_loser"]
        total_entropy += game_stats["avg_policy_entropy"]

        tie_contempts = res.get(
            "tie_contempts",
            {BLUE: config.CONTEMPT_VALUE, RED: config.CONTEMPT_VALUE},
        )
        abs_board_states = res.get("abs_board_states", [])
        for i, (state, mcts_policy, player_color, mcts_qval) in enumerate(game_data):
            if winner in ("Tie", "None"):
                scalar_outcome = tie_contempts.get(player_color, config.CONTEMPT_VALUE)
            elif winner == player_color:
                scalar_outcome = 1.0
            else:
                scalar_outcome = -1.0

            if use_wdl:
                wdl_scalar = 0.0 if winner in ("Tie", "None") else scalar_outcome
                game_wdl = outcome_to_wdl(wdl_scalar)
                mcts_wdl = outcome_to_wdl(float(mcts_qval))
                target = (config.SOFT_Z_ALPHA * game_wdl
                          + (1.0 - config.SOFT_Z_ALPHA) * mcts_wdl)
            else:
                target = scalar_outcome

            replay_buffer.append((state, mcts_policy, target))
            new_positions += 1

            if (i < len(abs_board_states)
                    and abs(float(mcts_qval) - scalar_outcome) > config.HIGH_REGRET_THRESHOLD):
                regret_buffer.append((abs_board_states[i], player_color))

        with open(detailed_csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                epoch * config.GAMES_PER_EPOCH + game_idx + 1,
                epoch + 1,
                game_idx + 1,
                winner,
                res["num_moves"],
                round(game_stats["avg_root_val_winner"], 4),
                round(game_stats["avg_root_val_loser"],  4),
                round(game_stats["avg_mcts_q_winner"],   4),
                round(game_stats["avg_mcts_q_loser"],    4),
                round(game_stats["avg_policy_entropy"],  4),
                game_stats["move_sequence"],
                game_stats["mcts_q_sequence"],
            ])

    n = config.GAMES_PER_EPOCH
    return new_positions, {
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "ties": ties,
        "total_moves": total_moves,
        "avg_moves": total_moves / n,
        "avg_root_val_winner": total_root_val_winner / n,
        "avg_root_val_loser": total_root_val_loser / n,
        "epoch_avg_entropy": total_entropy / n,
    }


def run_training_step(network, optimizer, replay_buffer, new_positions,
                     device, use_wdl, batch_size=None, config=None):
    """Run gradient updates on replay buffer. Returns (p_loss, v_loss, t_loss, avg_grad_norm, train_steps)."""
    config = config or default_config
    batch_size = batch_size or config.BATCH_SIZE
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0, 0.0, config.get_train_steps(new_positions)

    train_steps = config.get_train_steps(new_positions)
    network.train()
    total_p = total_v = total_t = total_gn = 0.0
    buffer_snapshot = list(replay_buffer)

    for _ in range(train_steps):
        batch = random.sample(buffer_snapshot, batch_size)
        states, policies, targets = zip(*batch)

        st = torch.FloatTensor(np.array(states)).to(device)
        tp = torch.FloatTensor(np.array(policies)).to(device)

        if use_wdl:
            tv = torch.FloatTensor(np.array(targets)).to(device)
            logits, wdl_pred, _ = network.forward_wdl(st)
            log_probs = torch.log_softmax(logits, dim=1)
            pl = -(tp * log_probs).sum(dim=1).mean()
            vl = -(tv * torch.log(wdl_pred + 1e-8)).sum(dim=1).mean()
            st_aug = torch.flip(st, dims=[2])
            st_aug = torch.cat([st_aug[:, 2:4], st_aug[:, 0:2]], dim=1)
            tv_aug = torch.stack([tv[:, 2], tv[:, 1], tv[:, 0]], dim=1)
            _, wdl_aug, _ = network.forward_wdl(st_aug)
            vl_aug = -(tv_aug * torch.log(wdl_aug + 1e-8)).sum(dim=1).mean()
            vl = 0.5 * (vl + vl_aug)
        else:
            tv = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(device)
            logits, vals = network(st)
            log_probs = torch.log_softmax(logits, dim=1)
            pl = -(tp * log_probs).sum(dim=1).mean()
            vl = nn.MSELoss()(vals, tv)

        loss = pl + config.VALUE_LOSS_WEIGHT * vl
        optimizer.zero_grad()
        loss.backward()
        pre_clip_norm = nn.utils.clip_grad_norm_(
            network.parameters(), config.GRAD_CLIP_NORM
        )
        optimizer.step()

        total_p += pl.item()
        total_v += vl.item()
        total_t += loss.item()
        total_gn += pre_clip_norm.item()

    n = train_steps
    network.eval()
    return total_p / n, total_v / n, total_t / n, total_gn / n, train_steps


def save_replay_buffer(replay_buffer, buffer_path):
    """Save replay buffer to compressed npz."""
    if len(replay_buffer) == 0:
        return
    buf_list = list(replay_buffer)
    np.savez_compressed(
        buffer_path,
        states=np.array([x[0] for x in buf_list], dtype=np.float32),
        policies=np.array([x[1] for x in buf_list], dtype=np.float32),
        outcomes=np.array([x[2] for x in buf_list], dtype=np.float32),
    )
