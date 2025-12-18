import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        # Q, K, Vの変換行列
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_dim = key_dim

    def forward(self, x):  # x: (B, C, E)
        Q = self.W_q(x)    # Q: (B, C, D)
        K = self.W_k(x)    # K: (B, C, D)
        V = self.W_v(x)    # V: (B, C, E)

        # Attentionマップの計算
        K_t = K.transpose(-2, -1)  # (B, D, C)
        scores = torch.matmul(Q, K_t)  # (B, C, C)
        scores = scores / (self.key_dim ** 0.5)

        # マスクの適用
        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)  # (B, C, E)
        return output

attention = Attention(embed_dim=256, key_dim=64)
x = torch.randn(2, 5, 256)  # (batch_size=2, context_len=5, embed_dim=256)
y = attention(x)

print("入力形状:", x.shape)
print("出力形状:", y.shape)