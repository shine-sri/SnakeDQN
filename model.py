import torch
import torch.nn as nn
from environment import VISION


class SnakeDQN(nn.Module):
    """
    CNN on the 7x7x3 local vision grid + small MLP head for the 5 extras.

    Why CNN now (not MLP)?
    The 7x7 grid explicitly encodes body shape in the local neighbourhood.
    A CNN can detect spatial patterns (wall proximity, body curves) that
    a flat feature vector can't represent — this is what allows the agent
    to plan around its own tail, which was the ceiling at score 43.

    Architecture:
        conv branch  : 3 → 32 → 64 channels, 7x7 → 5x5 → 3x3 → flatten = 576
        extras branch: 5 scalars passed directly
        fc head      : (576 + 5) → 256 → 128 → 4 Q-values
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 7x7 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),  # 7x7 -> 5x5
            nn.ReLU(),
            nn.Flatten(),                                  # 64*5*5 = 1600
        )

        conv_out = 64 * 5 * 5   # 1600
        extras   = 5

        self.fc = nn.Sequential(
            nn.Linear(conv_out + extras, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, grid, extras):
        """
        grid   : (B, 3, 7, 7)
        extras : (B, 5)
        """
        x = self.conv(grid)
        x = torch.cat([x, extras], dim=1)
        return self.fc(x)
