"""PPO policy/value network with residual backbone."""

import torch.nn as nn

from rl.networks.core import DualHeadResNet
from rl.configs.network_config import (
    BACKBONE_CHANNELS, NUM_RES_BLOCKS, POLICY_HEAD_CHANNELS, PPO_VALUE_HEAD_CHANNELS
)


class PPOPolicyNetwork(nn.Module):
    """AlphaZero-inspired policy/value network with residual backbone and 1x1 conv heads.

    Architecture:
        Backbone: initial conv(in->256) + 5 residual blocks (256 channels)
        Policy head: 1x1 conv(256->2) + flatten(128) + linear(128->n_actions)
        Value head: 1x1 conv(256->1) + flatten(64) + linear(64->256) + linear(256->1)

    ~3.6M params with deeper backbone and wider channels for stronger feature learning.
    """

    def __init__(self, input_shape, n_actions, config=None):
        super(PPOPolicyNetwork, self).__init__()
        
        self.net = DualHeadResNet(
            input_shape=input_shape,
            n_actions=n_actions,
            num_res_blocks=NUM_RES_BLOCKS,
            backbone_channels=BACKBONE_CHANNELS,
            policy_head_channels=POLICY_HEAD_CHANNELS,
            value_head_channels=PPO_VALUE_HEAD_CHANNELS,
            value_out_features=1
        )

    def forward(self, x):
        return self.net(x)
