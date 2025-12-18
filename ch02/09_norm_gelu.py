import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')
import torch
import torch.nn as nn
from codebot.model import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FFN(nn.Module):
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)

        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x