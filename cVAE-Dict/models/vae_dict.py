import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class cVAE_Dict(nn.Module):
    def __init__(self, dict_dim, code_dim, cond_dim):
        super().__init__()
        self.encoder = Encoder(1, code_dim, cond_dim)
        self.decoder = Decoder(dict_dim, cond_dim)
        self.dictionary = nn.Parameter(torch.randn(dict_dim, code_dim))

    def forward(self, x, c_src, c_tgt):
        alpha = self.encoder(x, c_src)
        z = torch.matmul(alpha, self.dictionary.T)
        x_hat = self.decoder(z, c_tgt)
        return x_hat, alpha
