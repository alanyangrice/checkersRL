"""AlphaZero-style policy/value network for checkers MCTS training.

Architecture mirrors the original AlphaZero paper:
  - Shared residual backbone (convolutional feature extractor)
  - Policy head  → action logits (used as MCTS priors after softmax)
  - Value head   → win probability in [-1, 1] via tanh
                   (represents the current player's chance of winning)

The critical difference from PPOPolicyNetwork is the tanh activation on the
value head. PPO's critic estimates an unbounded cumulative return V(s), but
AlphaZero's value must stay in [-1, 1] so that:
  1. MCTS Q-values (averaged backed-up values) remain on the same [-1, 1] scale.
  2. The PUCT exploration bonus is not dwarfed or overwhelmed by mis-scaled Q.
  3. MSE training against outcomes in {-1, 0, +1} is well-conditioned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_models.MCTS import training_config as cfg


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection and batch normalisation."""

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


class AlphaZeroNetwork(nn.Module):
    """Dual-headed residual network for AlphaZero self-play training.

    Args:
        input_shape: (C, H, W) board representation. Default (4, 8, 8).
        n_actions:   Size of the flat action space (NUM_ACTIONS).
        num_res_blocks:    Depth of the residual tower. Default 5.
        backbone_channels: Width of the residual tower. Default 256.

    Outputs (forward):
        logits: (B, n_actions) — raw policy scores, passed to softmax / log-softmax.
        value:  (B, 1)         — win probability in [-1, 1] via tanh.
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
        # Shared backbone
        # ------------------------------------------------------------------ #
        self.initial_conv = nn.Conv2d(
            in_channels, backbone_channels, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm2d(backbone_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(backbone_channels) for _ in range(num_res_blocks)]
        )

        # ------------------------------------------------------------------ #
        # Policy head
        # 1×1 conv (backbone → POLICY_HEAD_CHANNELS) + flatten + linear → logits
        # ------------------------------------------------------------------ #
        policy_flat = cfg.POLICY_HEAD_CHANNELS * board_h * board_w
        self.policy_conv = nn.Conv2d(backbone_channels, cfg.POLICY_HEAD_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(cfg.POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(policy_flat, n_actions)

        # ------------------------------------------------------------------ #
        # Value head
        # 1×1 conv (backbone → VALUE_HEAD_CHANNELS) + flatten + FC → tanh → scalar
        # ------------------------------------------------------------------ #
        value_flat = cfg.VALUE_HEAD_CHANNELS * board_h * board_w
        self.value_conv = nn.Conv2d(backbone_channels, cfg.VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(cfg.VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(value_flat, backbone_channels)
        self.value_fc2 = nn.Linear(backbone_channels, 1)
        # tanh applied inline in forward — bounds output to [-1, 1]

    def forward(self, x):
        # Shared backbone
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_blocks(out)

        # Policy head → logits (no activation; caller applies softmax/log-softmax)
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        # Value head → tanh-bounded win probability in [-1, 1]
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return logits, value
