"""AlphaZero policy/value networks: scalar and WDL value heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.networks.core import DualHeadResNet
from rl.configs.mcts_config import MCTSConfig
from rl.configs.network_config import (
    BACKBONE_CHANNELS, NUM_RES_BLOCKS, POLICY_HEAD_CHANNELS, MCTS_VALUE_HEAD_CHANNELS
)

default_config = MCTSConfig()


class AlphaZeroNetwork(nn.Module):
    """Dual-headed residual network for AlphaZero self-play training.

    Value head outputs scalar in [-1, 1] via tanh.
    """

    def __init__(
        self,
        input_shape=(4, 8, 8),
        n_actions=None,
        num_res_blocks=None,
        backbone_channels=None,
        config=None
    ):
        super().__init__()
        config = config or default_config
        
        self.net = DualHeadResNet(
            input_shape=input_shape,
            n_actions=n_actions,
            num_res_blocks=num_res_blocks or NUM_RES_BLOCKS,
            backbone_channels=backbone_channels or BACKBONE_CHANNELS,
            policy_head_channels=POLICY_HEAD_CHANNELS,
            value_head_channels=MCTS_VALUE_HEAD_CHANNELS,
            value_out_features=1
        )

    def forward(self, x):
        logits, raw_value = self.net(x)
        return logits, torch.tanh(raw_value)


class WDLAlphaZeroNetwork(nn.Module):
    """Dual-headed residual network with WDL value head.

    Value head outputs [P(win), P(draw), P(loss)] via softmax.
    forward() returns (logits, value) for compatibility; forward_wdl() for training.
    """

    def __init__(
        self,
        input_shape=(4, 8, 8),
        n_actions=None,
        num_res_blocks=None,
        backbone_channels=None,
        config=None
    ):
        super().__init__()
        config = config or default_config
        self.config = config
        
        self.net = DualHeadResNet(
            input_shape=input_shape,
            n_actions=n_actions,
            num_res_blocks=num_res_blocks or NUM_RES_BLOCKS,
            backbone_channels=backbone_channels or BACKBONE_CHANNELS,
            policy_head_channels=POLICY_HEAD_CHANNELS,
            value_head_channels=MCTS_VALUE_HEAD_CHANNELS,
            value_out_features=3
        )

    def _backbone(self, x):
        logits, raw_value = self.net(x)
        wdl = torch.softmax(raw_value, dim=1)
        value = wdl[:, 0:1] - wdl[:, 2:3]
        return logits, wdl, value

    def forward(self, x):
        logits, wdl, _ = self._backbone(x)

        K = self.config.KING_MATERIAL_VALUE
        blue_mat = x[:, 0].sum(dim=(1, 2)) + K * x[:, 1].sum(dim=(1, 2))
        red_mat = x[:, 2].sum(dim=(1, 2)) + K * x[:, 3].sum(dim=(1, 2))
        blue_share = blue_mat / (blue_mat + red_mat + 1e-8)
        extra = self.config.CONTEMPT_MATERIAL_SCALE * (blue_share - 0.5).clamp(min=0.0)
        contempt = (-(abs(self.config.CONTEMPT_VALUE) + extra)).clamp(min=-0.95)

        value = wdl[:, 0:1] - wdl[:, 2:3] + contempt.unsqueeze(1) * wdl[:, 1:2]
        return logits, value

    def forward_wdl(self, x):
        return self._backbone(x)
