"""WDL (Win/Draw/Loss) AlphaZero-style policy/value network for checkers MCTS training.

Architecture v2 — extends AlphaZeroNetwork with a three-headed value output:
  - Shared residual backbone (convolutional feature extractor) — identical to v1
  - Policy head  → action logits (used as MCTS priors after softmax) — identical to v1
  - WDL value head → [P(win), P(draw), P(loss)] via softmax, in [0, 1] each
                     Scalar value = P(win) - P(loss) ∈ [-1, 1] for MCTS/PUCT

Key design principles
----------------------
1. Backward-compatible forward() interface: returns (logits, value) 2-tuple
   identical to AlphaZeroNetwork, so mcts_search.py and evaluate.py need
   zero changes.

2. forward_wdl() training path: returns (logits, wdl, value) 3-tuple used
   only during gradient updates in train_gpu_parallel.py (--network-type wdl).

3. WDL vs scalar advantages (Lc0 blog, Apr 2020; Lc0 v0.30.0 blog, Jul 2023):
   - Draw probability is predicted explicitly rather than folded into expected
     score, giving more calibrated evaluations in draw-heavy games.
   - Contempt is principled: P(draw) for the material-ahead side can be
     directly targeted rather than applying a scalar penalty.
   - MSE against {-1, 0, +1} collapses draw and near-equal positions; cross-
     entropy against [0,1,0] preserves the draw signal separately.

4. Value target quality (Willemsen, Baier & Kaisers 2022 — "Value targets in
   off-policy AlphaZero: a new greedy backup", Neural Computing and Applications
   34(3):1801-1814): paired with soft-Z blending (SOFT_Z_ALPHA in training_config),
   the WDL target is blended with the per-position MCTS Q-value to reduce the
   on-policy bias of training on exploratory game outcomes.

Usage
-----
    # training
    from RL_models.MCTS.WDLAlphaZeroNetwork import WDLAlphaZeroNetwork
    net = WDLAlphaZeroNetwork(input_shape=(4,8,8), n_actions=NUM_ACTIONS)
    logits, wdl, value = net.forward_wdl(states)  # training step

    # inference (MCTS / evaluate) — identical to AlphaZeroNetwork
    logits, value = net(states)

Checkpoint compatibility
------------------------
    WDLAlphaZeroNetwork checkpoints are NOT compatible with AlphaZeroNetwork
    (value_fc2 weight shape differs: [1, 256] vs [3, 256]).
    Use --network-type wdl consistently for a full run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_models.MCTS import training_config as cfg


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection and batch normalisation.

    Identical to the ResidualBlock in AlphaZeroNetwork — defined here so
    WDLAlphaZeroNetwork is a self-contained module with no cross-import.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class WDLAlphaZeroNetwork(nn.Module):
    """Dual-headed residual network with a WDL value head for AlphaZero training.

    Args:
        input_shape:       (C, H, W) board representation. Default (4, 8, 8).
        n_actions:         Size of the flat action space (NUM_ACTIONS).
        num_res_blocks:    Depth of the residual tower. Default cfg.NUM_RES_BLOCKS.
        backbone_channels: Width of the residual tower. Default cfg.BACKBONE_CHANNELS.

    Outputs (forward — backward-compatible with AlphaZeroNetwork):
        logits: (B, n_actions) — raw policy scores.
        value:  (B, 1)         — scalar win probability P(win) - P(loss) in [-1, 1].

    Outputs (forward_wdl — training path only):
        logits: (B, n_actions) — raw policy scores.
        wdl:    (B, 3)         — [P(win), P(draw), P(loss)] via softmax.
        value:  (B, 1)         — scalar P(win) - P(loss), derived from wdl.
    """

    def __init__(
        self,
        input_shape=(4, 8, 8),
        n_actions=None,
        num_res_blocks=cfg.NUM_RES_BLOCKS,
        backbone_channels=cfg.BACKBONE_CHANNELS,
    ):
        super().__init__()
        if n_actions is None:
            raise ValueError("n_actions must be specified")

        in_channels = input_shape[0]
        board_h, board_w = input_shape[1], input_shape[2]

        # ------------------------------------------------------------------ #
        # Shared backbone — identical to AlphaZeroNetwork
        # ------------------------------------------------------------------ #
        self.initial_conv = nn.Conv2d(
            in_channels, backbone_channels, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm2d(backbone_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(backbone_channels) for _ in range(num_res_blocks)]
        )

        # ------------------------------------------------------------------ #
        # Policy head — identical to AlphaZeroNetwork
        # 1×1 conv (backbone → POLICY_HEAD_CHANNELS) + flatten + linear → logits
        # ------------------------------------------------------------------ #
        policy_flat = cfg.POLICY_HEAD_CHANNELS * board_h * board_w
        self.policy_conv = nn.Conv2d(backbone_channels, cfg.POLICY_HEAD_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(cfg.POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(policy_flat, n_actions)

        # ------------------------------------------------------------------ #
        # WDL value head
        # 1×1 conv (backbone → VALUE_HEAD_CHANNELS) + flatten + FC(backbone_channels)
        # → FC(3) → softmax → [P(win), P(draw), P(loss)]
        #
        # value_fc2 outputs 3 logits instead of v1's single scalar.
        # No tanh — softmax is applied inline in both forward paths.
        # ------------------------------------------------------------------ #
        value_flat = cfg.VALUE_HEAD_CHANNELS * board_h * board_w
        self.value_conv = nn.Conv2d(backbone_channels, cfg.VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(cfg.VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(value_flat, backbone_channels)
        self.value_fc2 = nn.Linear(backbone_channels, 3)   # 3 outputs: W, D, L

    # ---------------------------------------------------------------------- #
    # Shared backbone + heads computation (factored out to avoid duplication)
    # ---------------------------------------------------------------------- #

    def _backbone(self, x):
        """Run shared backbone + both heads, return (logits, wdl, value)."""
        # Shared backbone
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_blocks(out)

        # Policy head → logits (no activation; caller applies softmax/log-softmax)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        # WDL value head → softmax over [W, D, L]
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        wdl = torch.softmax(self.value_fc2(v), dim=1)        # (B, 3)
        value = wdl[:, 0:1] - wdl[:, 2:3]                   # (B, 1) in [-1, 1]

        return logits, wdl, value

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def forward(self, x):
        """Backward-compatible interface: returns (logits, value) 2-tuple.

        Identical signature to AlphaZeroNetwork.forward() so mcts_search.py,
        evaluate.py, and the AlphaZeroInferenceServer need zero changes.

        value = P(win) - P(loss) ∈ [-1, 1], equivalent in range and meaning
        to AlphaZeroNetwork's tanh output.
        """
        logits, _, value = self._backbone(x)
        return logits, value

    def forward_wdl(self, x):
        """Training-only path: returns (logits, wdl, value) 3-tuple.

        Used exclusively in the training step of train_gpu_parallel.py
        (--network-type wdl) to compute cross-entropy loss against WDL targets.

        wdl shape: (B, 3) — [P(win), P(draw), P(loss)], sums to 1 per row.
        """
        return self._backbone(x)
