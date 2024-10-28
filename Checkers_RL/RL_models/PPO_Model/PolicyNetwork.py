import torch
import torch.nn as nn
import numpy as np

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.value = nn.Linear(conv_out_size, 1)  # Output value for function estimation

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        logits = self.fc(conv_out)
        value = self.value(conv_out)
        return logits, value
