import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 64 * 56 * 56)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z, cond_onehot):
        x = torch.cat([z, cond_onehot], dim=1)
        x = self.fc(x).view(-1, 64, 56, 56)
        return self.deconv(x)
