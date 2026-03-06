import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.networks.common import ResidualBlock

class DualHeadResNet(nn.Module):
    """
    A unified Dual-Headed Residual Network backbone used for both AlphaZero and PPO.
    
    It supports configuring the size of the backbone, policy head, and value head.
    The output format of the value head is determined by `value_out_features` and
    whether a specific activation (like tanh) is applied externally.
    """
    def __init__(
        self,
        input_shape=(4, 8, 8),
        n_actions=None,
        num_res_blocks=5,
        backbone_channels=256,
        policy_head_channels=2,
        value_head_channels=1, # 32 for AZ, 1 for PPO
        value_out_features=1,  # 1 for scalar, 3 for WDL
    ):
        super().__init__()
        if n_actions is None:
            raise ValueError("n_actions must be specified")

        in_channels = input_shape[0]
        board_h, board_w = input_shape[1], input_shape[2]

        self.initial_conv = nn.Conv2d(
            in_channels, backbone_channels, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm2d(backbone_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(backbone_channels) for _ in range(num_res_blocks)]
        )

        policy_flat = policy_head_channels * board_h * board_w
        self.policy_conv = nn.Conv2d(backbone_channels, policy_head_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_head_channels)
        self.policy_fc = nn.Linear(policy_flat, n_actions)

        value_flat = value_head_channels * board_h * board_w
        self.value_conv = nn.Conv2d(backbone_channels, value_head_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_head_channels)
        self.value_fc1 = nn.Linear(value_flat, backbone_channels)
        self.value_fc2 = nn.Linear(backbone_channels, value_out_features)

    def _backbone_forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_blocks(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        
        return logits, v

    def forward(self, x):
        logits, v = self._backbone_forward(x)
        raw_value = self.value_fc2(v)
        return logits, raw_value
