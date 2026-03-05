"""
Research paper figure generation for CheckersRL training results.

Run from the project root:
    python figures/generate_figures.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {
    "az_scalar_train":  os.path.join(ROOT, "RL_models/MCTS/alphazero_training_progress_parallel_scalar.csv"),
    "az_scalar_eval":   os.path.join(ROOT, "RL_models/MCTS/alphazero_eval_benchmarks_scalar.csv"),
    "az_wdl_train":     os.path.join(ROOT, "RL_models/MCTS/alphazero_training_progress_parallel_wdl.csv"),
    "az_wdl_eval":      os.path.join(ROOT, "RL_models/MCTS/alphazero_eval_benchmarks_wdl.csv"),
    "ppo_cs_train":     os.path.join(ROOT, "RL_models/PPO_Model/training_progress_parallel_cs.csv"),
    "ppo_cs_bench":     os.path.join(ROOT, "RL_models/PPO_Model/benchmark_parallel_cs.csv"),
    "ppo_tactical":     os.path.join(ROOT, "RL_models/PPO_Model/training_progress_tactical.csv"),
    "ppo_terminal":     os.path.join(ROOT, "RL_models/PPO_Model/training_progress_terminal.csv"),
    "ppo_aggressive":   os.path.join(ROOT, "RL_models/PPO_Model/training_progress_aggressive.csv"),
    "ppo_league_bench": os.path.join(ROOT, "RL_models/PPO_Model/benchmark_league.csv"),
}

OUT = {
    "az_scalar":     os.path.join(ROOT, "figures/output/az_scalar"),
    "az_wdl":        os.path.join(ROOT, "figures/output/az_wdl"),
    "az_comparison": os.path.join(ROOT, "figures/output/az_comparison"),
    "ppo_curriculum":os.path.join(ROOT, "figures/output/ppo_curriculum"),
    "ppo_league":    os.path.join(ROOT, "figures/output/ppo_league"),
    "ppo_comparison": os.path.join(ROOT, "figures/output/ppo_comparison"),
}

# AlphaZero training phase boundaries (observed from avg_moves / epoch_time jumps)
AZ_PHASE1 = 16   # MCTS search depth increased; avg_moves ~56 -> ~80, time ~85s -> ~280s
AZ_PHASE2 = 66   # Second MCTS config change; avg_moves ~73 -> ~135, time ~250s -> ~875s

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.8,
    "axes.linewidth":     0.8,
})

# Colorblind-safe palette
C = {
    "scalar":     "#2166AC",   # blue  — AZ scalar / PPO curriculum
    "wdl":        "#D6604D",   # red   — AZ WDL
    "tactical":   "#2166AC",   # blue
    "terminal":   "#E08214",   # orange
    "aggressive": "#D6604D",   # red
    "winner":     "#1A9641",   # green
    "loser":      "#D7191C",   # red
    "loss_ref":   "#762A83",   # purple — reference loss rate
    "tie_ref":    "#4DAC26",   # teal   — tie rate
    "raw_alpha":  0.18,
    "fill_alpha": 0.12,
    "annot_alpha":0.85,
}

# Phase annotation text (short enough to fit rotated in margin)
PHASE_LABELS = {
    AZ_PHASE1: "Phase 2",
    AZ_PHASE2: "Phase 3",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def rolling_avg(arr, window=5):
    """Symmetric rolling average; smaller windows near edges."""
    arr = np.asarray(arr, dtype=float)
    result = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        result[i] = np.nanmean(arr[lo:hi])
    return result


def smooth_line(ax, x, y, color, label=None, window=5, zorder=3, lw=None):
    """Plot raw data as faint background, smoothed line on top."""
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, color=color, alpha=C["raw_alpha"], lw=0.9, zorder=zorder - 1)
    sm = rolling_avg(y, window)
    kw = dict(color=color, zorder=zorder)
    if label:
        kw["label"] = label
    if lw:
        kw["lw"] = lw
    ax.plot(x, sm, **kw)


def add_phase_vlines(ax, phases, ylim_frac=0.97, color="0.45"):
    """Draw vertical dashed lines with rotated text labels for phase boundaries."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for epoch, label in phases.items():
        ax.axvline(epoch, color=color, lw=1.1, ls="--", zorder=1)
        ax.text(epoch + 0.8, ylim_frac, label, fontsize=7.5, color=color,
                va="top", rotation=90, clip_on=True, transform=trans)


def set_epoch_ticks(ax, step=10):
    """Force x-axis major ticks every `step` epochs."""
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))


def save_fig(fig, outdir, name):
    """Save as PNG and PDF."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print(f"  saved {name}  ->  {os.path.basename(outdir)}/")
    plt.close(fig)


def build_heatmap_matrix(row, agents=("Tactical", "Terminal", "Aggressive")):
    """
    Build a 3x3 win-rate matrix from a benchmark_league row.
    Entry [i, j] = win rate of agent i when playing against agent j.
    Diagonal = win rate vs own checkpoint from 10 epochs prior.
    """
    cross_keys = {
        ("Tactical",   "Terminal"):   "tactical_vs_terminal_win",
        ("Tactical",   "Aggressive"): "tactical_vs_aggressive_win",
        ("Terminal",   "Tactical"):   "terminal_vs_tactical_win",
        ("Terminal",   "Aggressive"): "terminal_vs_aggressive_win",
        ("Aggressive", "Tactical"):   "aggressive_vs_tactical_win",
        ("Aggressive", "Terminal"):   "aggressive_vs_terminal_win",
    }
    self_keys = {
        "Tactical":   "tactical_vs_self_win",
        "Terminal":   "terminal_vs_self_win",
        "Aggressive": "aggressive_vs_self_win",
    }
    mat = np.full((3, 3), np.nan)
    for i, att in enumerate(agents):
        for j, dfn in enumerate(agents):
            if i != j:
                mat[i, j] = row[cross_keys[(att, dfn)]]
            else:
                mat[i, j] = row[self_keys[att]]
    return mat

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    dfs = {}
    for key, path in DATA.items():
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
        else:
            print(f"  [WARN] missing: {path}")
            dfs[key] = pd.DataFrame()
    return dfs

# ---------------------------------------------------------------------------
# AZ SCALAR FIGURES  (AZ-S-1 through AZ-S-5)
# ---------------------------------------------------------------------------

def az_scalar_figures(dfs):
    tr = dfs["az_scalar_train"]
    ev = dfs["az_scalar_eval"]
    outdir = OUT["az_scalar"]
    color = C["scalar"]
    phases = PHASE_LABELS  # both phase 1 (16) and phase 2 (66)

    # ---- AZ-S-1: Training Loss Curves ----------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    for ax, (col, ylabel, use_log) in zip(axes, [
        ("policy_loss", "Policy Loss", False),
        ("value_loss",  "Value Loss",  True),
        ("total_loss",  "Total Loss",  False),
    ]):
        smooth_line(ax, tr["epoch"], tr[col], color=color, window=5)
        for ep, lbl in phases.items():
            ax.axvline(ep, color="0.45", lw=1.1, ls="--", zorder=1)
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        set_epoch_ticks(ax)

    # Phase labels only on top panel to avoid clutter
    for ep, lbl in phases.items():
        ylim = axes[0].get_ylim()
        axes[0].text(ep + 0.8, ylim[0] + (ylim[1] - ylim[0]) * 0.96,
                     lbl, fontsize=7.5, color="0.4",
                     va="top", rotation=90, clip_on=True)

    axes[-1].set_xlabel("Epoch")
    axes[0].set_title("AlphaZero Scalar — Training Loss Curves")
    fig.align_ylabels(axes)
    save_fig(fig, outdir, "AZ-S-1_loss_curves")

    # ---- AZ-S-2: Value Head Polarization ------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(+1, color="0.7", lw=0.8, ls=":", label="_ideal")
    ax.axhline(-1, color="0.7", lw=0.8, ls=":")
    ax.axhline(0,  color="0.88", lw=0.7)

    ep = tr["epoch"].values
    win_sm = rolling_avg(tr["avg_root_val_winner"].values)
    los_sm = rolling_avg(tr["avg_root_val_loser"].values)

    ax.plot(ep, tr["avg_root_val_winner"], color=C["winner"], alpha=C["raw_alpha"], lw=0.9)
    ax.plot(ep, tr["avg_root_val_loser"],  color=C["loser"],  alpha=C["raw_alpha"], lw=0.9)
    ax.plot(ep, win_sm, color=C["winner"], label="Winner position value", lw=1.9)
    ax.plot(ep, los_sm, color=C["loser"],  label="Loser position value",  lw=1.9)
    ax.fill_between(ep, win_sm, los_sm, alpha=C["fill_alpha"], color="#888888")

    for ep_p, lbl in phases.items():
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Root Value")
    ax.set_title("AlphaZero Scalar — Value Head Polarization")
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="lower right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-S-2_value_polarization")

    # ---- AZ-S-3: Game Complexity --------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    smooth_line(ax, tr["epoch"], tr["avg_moves"], color=color, window=5,
                label="Avg moves per game")

    for ep_p, lbl in phases.items():
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.plot(tr["epoch"], tr["epoch_time_s"] / 60,
             color="#9970AB", alpha=0.55, lw=1.2, ls="-.")
    ax2.set_ylabel("Epoch time (min)", color="#9970AB", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#9970AB", labelsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Moves per Game")
    ax.set_title("AlphaZero Scalar — Game Complexity")
    ax.legend(loc="lower right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-S-3_game_complexity")

    # ---- AZ-S-4: Policy Entropy ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    start_entropy = float(tr["policy_entropy_nats"].iloc[0])
    ax.axhline(start_entropy, color="0.7", lw=0.9, ls=":",
               label=f"Initial entropy ({start_entropy:.2f} nats)")
    smooth_line(ax, tr["epoch"], tr["policy_entropy_nats"], color=color, window=5,
                label="Policy entropy")

    for ep_p in phases:
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("AlphaZero Scalar — Policy Entropy Decay")
    ax.legend(loc="upper right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-S-4_policy_entropy")

    # ---- AZ-S-5: Evaluation Benchmarks --------------------------------------
    if ev.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    fig.subplots_adjust(wspace=0.38)

    # Panel A: gate win rate with accepted/rejected markers
    ax1.plot(ev["epoch"], ev["gate_win_rate"], color=color, lw=1.9,
             label="Gate win rate", zorder=3)
    accepted = ev[ev["gate_accepted"] == True]
    rejected = ev[ev["gate_accepted"] == False]
    ax1.scatter(accepted["epoch"], accepted["gate_win_rate"],
                color="#1A9641", zorder=5, s=55, marker="^", label="Accepted")
    ax1.scatter(rejected["epoch"], rejected["gate_win_rate"],
                color="#D7191C", zorder=5, s=55, marker="v", label="Rejected")
    for ep_p in phases:
        ax1.axvline(ep_p, color="0.45", lw=1.0, ls="--", zorder=1)
    add_phase_vlines(ax1, phases)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gate Win Rate")
    ax1.set_title("(A) Model Gating")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 0.80)
    set_epoch_ticks(ax1, step=20)

    # Panel B: value head calibration — more informative than vs_random (which saturates)
    val_colors = {
        "val_clear_win":  C["winner"],
        "val_clear_loss": C["loser"],
        "val_equal":      "#888888",
    }
    val_labels = {
        "val_clear_win":  "Clear-win positions",
        "val_clear_loss": "Clear-loss positions",
        "val_equal":      "Equal positions",
    }
    ax2.axhline(0, color="0.85", lw=0.7)
    for col, c in val_colors.items():
        ax2.plot(ev["epoch"], ev[col], color=c, lw=1.9,
                 marker="o", ms=4, label=val_labels[col])
    ax2.axhline(+1, color="0.72", lw=0.8, ls=":", label="_ideal +1")
    ax2.axhline(-1, color="0.72", lw=0.8, ls=":")
    for ep_p in phases:
        ax2.axvline(ep_p, color="0.45", lw=1.0, ls="--", zorder=1)
    add_phase_vlines(ax2, phases)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Predicted Value")
    ax2.set_title("(B) Value Head Calibration")
    ax2.set_ylim(-1.25, 1.25)
    ax2.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.15))
    set_epoch_ticks(ax2, step=20)

    fig.suptitle("AlphaZero Scalar — Evaluation Benchmarks", fontsize=11, fontweight="bold")
    save_fig(fig, outdir, "AZ-S-5_eval_benchmarks")


# ---------------------------------------------------------------------------
# AZ WDL FIGURES  (AZ-W-1 through AZ-W-5)
# ---------------------------------------------------------------------------

def az_wdl_figures(dfs):
    tr = dfs["az_wdl_train"]
    ev = dfs["az_wdl_eval"]
    outdir = OUT["az_wdl"]
    color = C["wdl"]
    phases = PHASE_LABELS  # both phase 1 (16) and phase 2 (66)

    # ---- AZ-W-1: Training Loss Curves ----------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    for ax, (col, ylabel, use_log) in zip(axes, [
        ("policy_loss", "Policy Loss", False),
        ("value_loss",  "Value Loss",  True),
        ("total_loss",  "Total Loss",  False),
    ]):
        smooth_line(ax, tr["epoch"], tr[col], color=color, window=5)
        for ep, lbl in phases.items():
            ax.axvline(ep, color="0.45", lw=1.1, ls="--", zorder=1)
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        set_epoch_ticks(ax)

    # Phase labels only on top panel to avoid clutter
    for ep, lbl in phases.items():
        ylim = axes[0].get_ylim()
        axes[0].text(ep + 0.8, ylim[0] + (ylim[1] - ylim[0]) * 0.96,
                     lbl, fontsize=7.5, color="0.4",
                     va="top", rotation=90, clip_on=True)

    axes[-1].set_xlabel("Epoch")
    axes[0].set_title("AlphaZero WDL — Training Loss Curves")
    fig.align_ylabels(axes)
    save_fig(fig, outdir, "AZ-W-1_loss_curves")

    # ---- AZ-W-2: Value Head Polarization ------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(+1, color="0.7", lw=0.8, ls=":")
    ax.axhline(-1, color="0.7", lw=0.8, ls=":")
    ax.axhline(0,  color="0.88", lw=0.7)

    ep = tr["epoch"].values
    win_sm = rolling_avg(tr["avg_root_val_winner"].values)
    los_sm = rolling_avg(tr["avg_root_val_loser"].values)

    ax.plot(ep, tr["avg_root_val_winner"], color=C["winner"], alpha=C["raw_alpha"], lw=0.9)
    ax.plot(ep, tr["avg_root_val_loser"],  color=C["loser"],  alpha=C["raw_alpha"], lw=0.9)
    ax.plot(ep, win_sm, color=C["winner"], label="Winner position value", lw=1.9)
    ax.plot(ep, los_sm, color=C["loser"],  label="Loser position value",  lw=1.9)
    ax.fill_between(ep, win_sm, los_sm, alpha=C["fill_alpha"], color="#888888")

    for ep_p, lbl in phases.items():
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Root Value")
    ax.set_title("AlphaZero WDL — Value Head Polarization")
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="lower right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-W-2_value_polarization")

    # ---- AZ-W-3: Game Complexity --------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    smooth_line(ax, tr["epoch"], tr["avg_moves"], color=color, window=5,
                label="Avg moves per game")

    for ep_p, lbl in phases.items():
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.plot(tr["epoch"], tr["epoch_time_s"] / 60,
             color="#9970AB", alpha=0.55, lw=1.2, ls="-.")
    ax2.set_ylabel("Epoch time (min)", color="#9970AB", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#9970AB", labelsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Moves per Game")
    ax.set_title("AlphaZero WDL — Game Complexity")
    ax.legend(loc="upper left")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-W-3_game_complexity")

    # ---- AZ-W-4: Policy Entropy ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    start_entropy = float(tr["policy_entropy_nats"].iloc[0])
    ax.axhline(start_entropy, color="0.7", lw=0.9, ls=":",
               label=f"Initial entropy ({start_entropy:.2f} nats)")
    smooth_line(ax, tr["epoch"], tr["policy_entropy_nats"], color=color, window=5,
                label="Policy entropy")

    for ep_p in phases:
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, phases)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("AlphaZero WDL — Policy Entropy Decay")
    ax.legend(loc="upper right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-W-4_policy_entropy")

    # ---- AZ-W-5: Evaluation Benchmarks --------------------------------------
    if ev.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    fig.subplots_adjust(wspace=0.38)

    ax1.plot(ev["epoch"], ev["gate_win_rate"], color=color, lw=1.9,
             label="Gate win rate", zorder=3)
    accepted = ev[ev["gate_accepted"] == True]
    rejected = ev[ev["gate_accepted"] == False]
    ax1.scatter(accepted["epoch"], accepted["gate_win_rate"],
                color="#1A9641", zorder=5, s=55, marker="^", label="Accepted")
    ax1.scatter(rejected["epoch"], rejected["gate_win_rate"],
                color="#D7191C", zorder=5, s=55, marker="v", label="Rejected")
    for ep_p in phases:
        ax1.axvline(ep_p, color="0.45", lw=1.0, ls="--", zorder=1)
    add_phase_vlines(ax1, phases)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gate Win Rate")
    ax1.set_title("(A) Model Gating")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 0.65)
    set_epoch_ticks(ax1, step=10)

    # Panel B: value head calibration
    val_colors = {
        "val_clear_win":  C["winner"],
        "val_clear_loss": C["loser"],
        "val_equal":      "#888888",
    }
    val_labels = {
        "val_clear_win":  "Clear-win positions",
        "val_clear_loss": "Clear-loss positions",
        "val_equal":      "Equal positions",
    }
    ax2.axhline(0, color="0.85", lw=0.7)
    for col, c in val_colors.items():
        ax2.plot(ev["epoch"], ev[col], color=c, lw=1.9,
                 marker="o", ms=4, label=val_labels[col])
    ax2.axhline(+1, color="0.72", lw=0.8, ls=":")
    ax2.axhline(-1, color="0.72", lw=0.8, ls=":")
    for ep_p in phases:
        ax2.axvline(ep_p, color="0.45", lw=1.0, ls="--", zorder=1)
    add_phase_vlines(ax2, phases)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Predicted Value")
    ax2.set_title("(B) Value Head Calibration")
    ax2.set_ylim(-1.25, 1.25)
    ax2.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.15))
    set_epoch_ticks(ax2, step=10)

    fig.suptitle("AlphaZero WDL — Evaluation Benchmarks", fontsize=11, fontweight="bold")
    save_fig(fig, outdir, "AZ-W-5_eval_benchmarks")


# ---------------------------------------------------------------------------
# AZ COMPARISON FIGURES  (AZ-C-1, AZ-C-2)
# ---------------------------------------------------------------------------

def az_comparison_figures(dfs):
    s = dfs["az_scalar_train"]
    w = dfs["az_wdl_train"]
    outdir = OUT["az_comparison"]

    max_epoch = min(s["epoch"].max(), w["epoch"].max())
    s_trim = s[s["epoch"] <= max_epoch]
    w_trim = w[w["epoch"] <= max_epoch]

    # ---- AZ-C-1: Loss Comparison -------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    fig.subplots_adjust(wspace=0.35)

    for ax, col, ylabel, title in [
        (ax1, "policy_loss", "Policy Loss", "(A) Policy Loss"),
        (ax2, "value_loss",  "Value Loss",  "(B) Value Loss"),
    ]:
        sm_s = rolling_avg(s_trim[col].values)
        ax.plot(s_trim["epoch"], s_trim[col], color=C["scalar"], alpha=C["raw_alpha"], lw=0.9)
        ax.plot(s_trim["epoch"], sm_s, color=C["scalar"], lw=1.9, ls="--")

        sm_w = rolling_avg(w_trim[col].values)
        ax.plot(w_trim["epoch"], w_trim[col], color=C["wdl"], alpha=C["raw_alpha"], lw=0.9)
        ax.plot(w_trim["epoch"], sm_w, color=C["wdl"], lw=1.9, ls="-")

        for ep_p, lbl in PHASE_LABELS.items():
            ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
        add_phase_vlines(ax, PHASE_LABELS)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0.02)
        set_epoch_ticks(ax)

    ax2.set_yscale("log")

    # Single shared legend at figure level — one entry per model
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=C["scalar"], lw=1.9, ls="--", label="Scalar"),
        Line2D([0], [0], color=C["wdl"],    lw=1.9, ls="-",  label="WDL"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(f"AlphaZero: Scalar vs. WDL — Loss Comparison (Epochs 1\u2013{max_epoch})",
                 fontsize=11, fontweight="bold", y=1.07)
    save_fig(fig, outdir, "AZ-C-1_loss_comparison")

    # ---- AZ-C-2: Value Polarization Comparison ------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(+1, color="0.75", lw=0.8, ls=":")
    ax.axhline(-1, color="0.75", lw=0.8, ls=":")
    ax.axhline(0,  color="0.88", lw=0.7)

    ep_s = s_trim["epoch"].values
    ep_w = w_trim["epoch"].values
    sw_s = rolling_avg(s_trim["avg_root_val_winner"].values)
    sl_s = rolling_avg(s_trim["avg_root_val_loser"].values)
    sw_w = rolling_avg(w_trim["avg_root_val_winner"].values)
    sl_w = rolling_avg(w_trim["avg_root_val_loser"].values)

    ax.plot(ep_s, sw_s, color=C["scalar"], lw=1.9, ls="--", label="Scalar winner")
    ax.plot(ep_s, sl_s, color=C["scalar"], lw=1.4, ls=":",  label="Scalar loser")
    ax.fill_between(ep_s, sw_s, sl_s, alpha=0.08, color=C["scalar"])

    ax.plot(ep_w, sw_w, color=C["wdl"], lw=1.9, ls="-",  label="WDL winner")
    ax.plot(ep_w, sl_w, color=C["wdl"], lw=1.4, ls="-.", label="WDL loser")
    ax.fill_between(ep_w, sw_w, sl_w, alpha=0.08, color=C["wdl"])

    for ep_p, lbl in PHASE_LABELS.items():
        ax.axvline(ep_p, color="0.45", lw=1.1, ls="--", zorder=1)
    add_phase_vlines(ax, PHASE_LABELS)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Root Value")
    ax.set_title("AlphaZero: Scalar vs. WDL — Value Polarization")
    ax.set_ylim(-1.25, 1.25)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "AZ-C-2_value_polarization_comparison")


# ---------------------------------------------------------------------------
# PPO CURRICULUM FIGURES  (PPO-CS-1 through PPO-CS-4)
# ---------------------------------------------------------------------------

def ppo_curriculum_figures(dfs):
    tr = dfs["ppo_cs_train"]
    bm = dfs["ppo_cs_bench"]
    outdir = OUT["ppo_curriculum"]
    color = C["scalar"]
    SWITCH = 81   # curriculum switch: mid-game positions -> full 12v12 board

    # ---- PPO-CS-1: vs-Random Win Rate ----------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(bm["epoch"], bm["vs_random_win"] * 100, color=color, lw=1.9,
            marker="o", ms=5, label="vs. Random win rate")
    ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")
    ax.fill_between(bm["epoch"], 50, bm["vs_random_win"] * 100,
                    where=bm["vs_random_win"] > 0.5, alpha=0.1, color=color)
    ax.axhline(50, color="0.6", lw=0.8, ls=":", label="Random baseline (50%)")

    ylim = ax.get_ylim()
    ax.text(SWITCH + 1, 107,
            "Curriculum switch",
            fontsize=7.5, color="0.35", va="top", rotation=90, clip_on=True)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Win Rate vs. Random (%)")
    ax.set_title("PPO Curriculum — Performance vs. Random Agent")
    ax.set_ylim(35, 112)
    ax.legend(fontsize=8)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-CS-1_vs_random_win_rate")

    # ---- PPO-CS-2: vs-Reference Performance (win + loss + tie) ---------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(bm["epoch"], bm["vs_reference_win"]  * 100,
            color=color,          lw=1.9, marker="o", ms=5, label="Win rate")
    ax.plot(bm["epoch"], bm["vs_reference_loss"] * 100,
            color=C["loser"],     lw=1.6, marker="s", ms=4, ls="-.", label="Loss rate")
    ax.plot(bm["epoch"], bm["vs_reference_tie"]  * 100,
            color=C["loss_ref"],  lw=1.6, marker="^", ms=4, ls="--", label="Tie rate")

    ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")
    ax.text(SWITCH + 1, ax.get_ylim()[1] * 0.95,
            "Curriculum switch", fontsize=7.5, color="0.4",
            va="top", rotation=90, clip_on=True)

    # Annotation for passive co-evolution at epoch 110
    last = bm.iloc[-1]
    ax.annotate(f"37% tie, 22% loss\n(passive co-evolution)",
                xy=(last["epoch"], last["vs_reference_tie"] * 100),
                xytext=(last["epoch"] - 28, 42),
                fontsize=8, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", alpha=C["annot_alpha"]))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate vs. Reference Agent (%)")
    ax.set_title("PPO Curriculum — vs. Reference Agent Performance")
    ax.legend(fontsize=8, loc="upper left")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-CS-2_vs_reference")

    # ---- PPO-CS-3: Episode Length -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    smooth_line(ax, tr["epoch"], tr["average_episode_length"], color=color,
                window=7, label="Episode length")
    ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")

    ylim_cs3 = ax.get_ylim()
    ax.text(SWITCH + 1, ylim_cs3[0] + (ylim_cs3[1] - ylim_cs3[0]) * 0.95,
            "Curriculum switch", fontsize=7.5, color="0.4",
            va="top", rotation=90, clip_on=True)

    ax.annotate("Growing length\nsignals passive play",
                xy=(100, float(tr[tr["epoch"] == 100]["average_episode_length"].values[0])),
                xytext=(78, 65),
                fontsize=8, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", alpha=C["annot_alpha"]))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Episode Length (moves)")
    ax.set_title("PPO Curriculum — Episode Length Over Training")
    ax.legend(fontsize=8)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-CS-3_episode_length")

    # ---- PPO-CS-4: Tie Rate -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(5, color="0.6", lw=0.9, ls=":", label="5% reference")
    smooth_line(ax, tr["epoch"], tr["tie_rate"] * 100, color=color,
                window=7, label="Tie rate")
    ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")

    tie100 = float(tr[tr["epoch"] == 100]["tie_rate"].values[0]) * 100
    ax.annotate(f"Epoch 100: {tie100:.1f}%",
                xy=(100, tie100),
                xytext=(82, tie100 + 3.5),
                fontsize=8, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", alpha=C["annot_alpha"]))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tie Rate (%)")
    ax.set_title("PPO Curriculum — Tie Rate Over Training")
    ax.legend(fontsize=8)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-CS-4_tie_rate")


# ---------------------------------------------------------------------------
# PPO LEAGUE FIGURES  (PPO-L-1 through PPO-L-5)
# ---------------------------------------------------------------------------

def ppo_league_figures(dfs):
    tac = dfs["ppo_tactical"]
    ter = dfs["ppo_terminal"]
    agg = dfs["ppo_aggressive"]
    bm  = dfs["ppo_league_bench"]
    outdir = OUT["ppo_league"]
    SWITCH = 21   # league phase 2: mid-game -> full board

    agents = [
        ("Tactical",   tac, C["tactical"],   "o"),
        ("Terminal",   ter, C["terminal"],   "s"),
        ("Aggressive", agg, C["aggressive"], "^"),
    ]

    # ---- PPO-L-1: vs-Random Win Rate (All Agents) ---------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    cols = {
        "Tactical":   "tactical_vs_random_win",
        "Terminal":   "terminal_vs_random_win",
        "Aggressive": "aggressive_vs_random_win",
    }
    for name, _, col_color, marker in agents:
        ax.plot(bm["epoch"], bm[cols[name]] * 100,
                color=col_color, lw=1.9, marker=marker, ms=5, label=name)

    ax.axhline(50, color="0.6", lw=0.8, ls=":", label="Random baseline (50%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Win Rate vs. Random (%)")
    ax.set_title("PPO League — Performance vs. Random Agent")
    ax.set_ylim(35, 110)
    ax.legend(fontsize=8)
    set_epoch_ticks(ax, step=10)
    save_fig(fig, outdir, "PPO-L-1_vs_random_win_rate")

    # ---- PPO-L-2: Head-to-Head Heatmap Matrix (2×2) -------------------------
    # 4 key snapshots: early / mid / aggressive-peak / final
    EPOCHS = [10, 40, 70, 100]
    N_AGENTS = 3
    agent_labels = ["Tactical", "Terminal", "Aggressive"]
    cmap = sns.diverging_palette(10, 145, s=90, l=45, sep=1, as_cmap=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.8))
    fig.subplots_adjust(wspace=0.38, hspace=0.48, top=0.88, bottom=0.08,
                        left=0.07, right=0.89)

    axes_flat = axes.flatten()

    for ax, ep in zip(axes_flat, EPOCHS):
        bm_row = bm[bm["epoch"] == ep].iloc[0]
        mat = build_heatmap_matrix(bm_row, agent_labels)

        sns.heatmap(mat, ax=ax,
                    annot=True, fmt=".2f",
                    cmap=cmap, center=0.5,
                    vmin=0.30, vmax=0.75,
                    xticklabels=agent_labels,
                    yticklabels=agent_labels,
                    linewidths=0.5,
                    cbar=False,
                    annot_kws={"size": 9.5, "weight": "bold"})
        ax.set_title(f"Epoch {ep}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Opponent", fontsize=9)
        # Show y-label only on left column
        ax.set_ylabel("Agent" if ax in (axes[0, 0], axes[1, 0]) else "", fontsize=9)
        ax.tick_params(axis="both", labelsize=8.5)

        # Diagonal borders: seaborn places row i at data y∈[i, i+1] then
        # inverts y, so Rectangle((i, i), 1, 1) correctly outlines mat[i,i].
        for i in range(N_AGENTS):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor="black", lw=2.0,
                zorder=5, clip_on=False
            ))

    # Single shared colorbar on the right
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.30, vmax=0.75)
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.10, 0.016, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Win Rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "PPO League — Head-to-Head Win Rate Matrix\n"
        r"$\it{Diagonal\ (outlined):\ win\ rate\ vs.\ own\ checkpoint\ from\ 10\ epochs\ prior}$",
        fontsize=11, fontweight="bold", y=0.97
    )
    save_fig(fig, outdir, "PPO-L-2_heatmap_matrix")

    # ---- PPO-L-3: Episode Length (All Agents) --------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for name, df, col_color, _ in agents:
        smooth_line(ax, df["epoch"], df["average_episode_length"],
                    color=col_color, window=7, label=name)
    ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")
    ylim = ax.get_ylim()
    ax.text(SWITCH + 0.5, ylim[0] + (ylim[1] - ylim[0]) * 0.96,
            "Phase 2", fontsize=7.5, color="0.4", va="top", rotation=90, clip_on=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Episode Length (moves)")
    ax.set_title("PPO League — Episode Length by Agent Type")
    ax.legend(fontsize=8)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-L-3_episode_length")

    # ---- PPO-L-4: Tie Rate (All Agents) -------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axhline(5, color="0.6", lw=0.9, ls=":", label="5% reference")
    for name, df, col_color, _ in agents:
        smooth_line(ax, df["epoch"], df["tie_rate"] * 100,
                    color=col_color, window=7, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tie Rate (%)")
    ax.set_title("PPO League — Tie Rate by Agent Type")
    ax.legend(fontsize=8)
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-L-4_tie_rate")

    # ---- PPO-L-5: Reward Trajectories (3 stacked subplots) ------------------
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
    fig.subplots_adjust(hspace=0.38)

    agent_titles = ["Tactical Agent", "Terminal Agent", "Aggressive Agent"]
    for ax, (name, df, col_color, _), title in zip(axes, agents, agent_titles):
        smooth_line(ax, df["epoch"], df["average_epoch_reward"],
                    color=col_color, window=7)
        ax.axvline(SWITCH, color="0.45", lw=1.1, ls="--")
        ylim = ax.get_ylim()
        ax.text(SWITCH + 0.5, ylim[0] + (ylim[1] - ylim[0]) * 0.95,
                "Phase 2", fontsize=7.5, color="0.4",
                va="top", rotation=90, clip_on=True)
        ax.set_ylabel("Avg Reward")
        ax.set_title(title, fontsize=10, fontweight="bold")
        set_epoch_ticks(ax)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("PPO League — Reward Trajectories by Agent Type",
                 fontsize=11, fontweight="bold")
    fig.align_ylabels(axes)
    save_fig(fig, outdir, "PPO-L-5_reward_trajectories")


# ---------------------------------------------------------------------------
# PPO COMPARISON FIGURES  (PPO-C-1)
# ---------------------------------------------------------------------------

def ppo_comparison_figures(dfs):
    bm_cs = dfs["ppo_cs_bench"]
    bm_lg = dfs["ppo_league_bench"]
    outdir = OUT["ppo_comparison"]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Curriculum run (dashed) — no run number in label
    ax.plot(bm_cs["epoch"], bm_cs["vs_random_win"] * 100,
            color=C["scalar"], lw=2.0, ls="--", marker="o", ms=5,
            label="Curriculum")

    # League agents (solid)
    league_cols = [
        ("tactical_vs_random_win",   "Tactical (League)",   C["tactical"],   "o"),
        ("terminal_vs_random_win",   "Terminal (League)",   C["terminal"],   "s"),
        ("aggressive_vs_random_win", "Aggressive (League)", C["aggressive"], "^"),
    ]
    for col, label, col_color, marker in league_cols:
        ax.plot(bm_lg["epoch"], bm_lg[col] * 100,
                color=col_color, lw=1.9, ls="-", marker=marker, ms=5, label=label)

    ax.axhline(50,  color="0.6", lw=0.8, ls=":", label="Random baseline (50%)")
    ax.axhline(100, color="0.7", lw=0.7, ls=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Win Rate vs. Random (%)")
    ax.set_title("PPO Training Comparison — Win Rate vs. Random Agent",
                 fontweight="bold")
    ax.set_ylim(35, 112)
    ax.legend(fontsize=8.5, loc="lower right")
    set_epoch_ticks(ax)
    save_fig(fig, outdir, "PPO-C-1_ppo_comparison")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    dfs = load_data()

    print("\n[1/6] AlphaZero Scalar figures...")
    az_scalar_figures(dfs)

    print("\n[2/6] AlphaZero WDL figures...")
    az_wdl_figures(dfs)

    print("\n[3/6] AlphaZero Comparison figures...")
    az_comparison_figures(dfs)

    print("\n[4/6] PPO Curriculum figures...")
    ppo_curriculum_figures(dfs)

    print("\n[5/6] PPO League figures...")
    ppo_league_figures(dfs)

    print("\n[6/6] PPO Comparison figures...")
    ppo_comparison_figures(dfs)

    print("\nDone. All figures saved to figures/output/")


if __name__ == "__main__":
    main()
