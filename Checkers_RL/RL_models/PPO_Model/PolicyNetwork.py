import torch.nn as nn

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.flattened_size = 256 * 8 * 8  # Assuming input dimensions lead to this size

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),          # First FC layer
            nn.ReLU(),
            nn.Linear(256, 128),            # Second FC layer
            nn.ReLU(),
            nn.Linear(128, 64),             # Third FC layer
            nn.ReLU(),
            nn.Linear(64, n_actions)        # Output logits for actions
        )
        self.value = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten conv output

        logits = self.fc(conv_out)
        value = self.value(conv_out)
        return logits, value
