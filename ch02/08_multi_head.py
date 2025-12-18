import torch
import torch.nn as nn
import torch.nn.functional as F

B = 2   # バッチサイズ（batch_size）
C = 4   # コンテキスト長（context_len）
E = 16  # 埋め込み次元（embed_dim）
H = 3   # ヘッド数（n_head）
D = 8   # ヘッド次元（head_dim）

# 入力テンソル
x = torch.randn(B, C, E)

# 効率的な実装：全ヘッド分の重みを一つの行列にまとめる
W_q = nn.Linear(E, H*D, bias=False)
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)  # (B, C, H*D)
V = W_v(x)  # (B, C, H*D)

# 形状の変換
Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
scores = scores / (D ** 0.5)

# マスク処理
mask = torch.tril(torch.ones_like(scores))
scores = scores.masked_fill(mask == 0, float('-inf'))

# Attention重み
weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
hidden = torch.matmul(weights, V)   # (B, H, C, D)

# 形状変換: (B, H, C, D) → (B, C, H*D)
hidden = hidden.transpose(1, 2)  # (B, C, H, D)
hidden = hidden.contiguous().view(B, C, H*D)  # (B, C, H*D)

# 出力変換: (B, C, H*D) → (B, C, E)
W_o = nn.Linear(H*D, E, bias=False)
output = W_o(hidden)  # (B, C, E)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # Dropoutを追加
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, E = x.shape  # バッチサイズ、コンテキスト長, 埋め込み次元
        H, D = self.n_head, self.head_dim  # ヘッドサイズ、ヘッドの次元数

        # Q, K, V の計算
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 各ヘッドに分割して並べ替え
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        scores = scores / (self.head_dim ** 0.5)

        # マスク処理
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention重み
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)  # weightsにDropout
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # ヘッドの結合と出力変換
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H*D)
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)  # 最終出力にDropout

        return output

# 使用例
embed_dim = 512
n_head = 8
head_dim = 64

mha = MultiHeadAttention(embed_dim, n_head, head_dim)

# テスト用データ
batch_size = 2
context_len = 10
x = torch.randn(batch_size, context_len, embed_dim)

# 実行
output = mha(x)
print(f"Input shape: {x.shape}")       # (2, 10, 512)
print(f"Output shape: {output.shape}") # (2, 10, 512)