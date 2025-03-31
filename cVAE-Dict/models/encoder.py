import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, in_channels, code_dim, cond_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + cond_dim, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, code_dim)
        )

    def forward(self, x, cond_onehot):
        cond = cond_onehot.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)
        return self.conv(x)
