import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A residual block with two 3x3 conv layers and a skip connection."""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PPOPolicyNetwork(nn.Module):
    """AlphaZero-inspired policy/value network with residual backbone and 1x1 conv heads.

    Architecture:
        Backbone: initial conv(in→256) + 5 residual blocks (256 channels)
        Policy head: 1x1 conv(256→2) + flatten(128) + linear(128→n_actions)
        Value head: 1x1 conv(256→1) + flatten(64) + linear(64→256) + linear(256→1)

    ~3.6M params with deeper backbone and wider channels for stronger feature learning.
    """

    BACKBONE_CHANNELS = 256
    POLICY_HEAD_CHANNELS = 2
    VALUE_HEAD_CHANNELS = 1
    NUM_RES_BLOCKS = 5

    def __init__(self, input_shape, n_actions):
        super(PPOPolicyNetwork, self).__init__()

        in_channels = input_shape[0]
        board_h, board_w = input_shape[1], input_shape[2]

        # Shared backbone: initial conv + residual blocks
        self.initial_conv = nn.Conv2d(in_channels, self.BACKBONE_CHANNELS, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(self.BACKBONE_CHANNELS)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(self.BACKBONE_CHANNELS) for _ in range(self.NUM_RES_BLOCKS)]
        )

        # Policy head: 1x1 conv to reduce channels, then flatten + linear
        policy_flatten_size = self.POLICY_HEAD_CHANNELS * board_h * board_w
        self.policy_conv = nn.Conv2d(self.BACKBONE_CHANNELS, self.POLICY_HEAD_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(self.POLICY_HEAD_CHANNELS)
        self.policy_fc = nn.Linear(policy_flatten_size, n_actions)

        # Value head: 1x1 conv to reduce channels, then flatten + FC layers
        value_flatten_size = self.VALUE_HEAD_CHANNELS * board_h * board_w
        self.value_conv = nn.Conv2d(self.BACKBONE_CHANNELS, self.VALUE_HEAD_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(self.VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(value_flatten_size, self.BACKBONE_CHANNELS)
        self.value_fc2 = nn.Linear(self.BACKBONE_CHANNELS, 1)

    def forward(self, x):
        # Shared backbone
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = self.value_fc2(v)

        return logits, value
