import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(x))
        self.eps = 1e-5

    def forward(self, x):
        x2 = x**2
        ms = x2.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        return self.gamma * x / rms