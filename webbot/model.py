import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0
        half = key_dim // 2

        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ( (2.0 * half_ids) / key_dim ))  # (half,)

        positions = torch.arange(max_context_len)  # (max_context_len,)
        angles = positions[:, None] * inv_freq[None, :]  # (max_context_len, half)

        cos = torch.cos(angles)  # (max_context_len, half)
        sin = torch.sin(angles)  # (max_context_len, half)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x, offset=0):
        batch_size, num_head, context_len, key_dim = x.shape

        # 入力の型を保存し、float32で計算
        input_dtype = x.dtype
        x = x.float()

        # offsetを考慮して位置エンコーディングを取得
        max_context_len = self.cos_cache.size(0)
        if offset + context_len > max_context_len:
            offset = max_context_len - context_len

        cos = self.cos_cache[offset:offset + context_len]
        sin = self.sin_cache[offset:offset + context_len]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        out = out.reshape(batch_size, num_head, context_len, key_dim)

        return out.to(input_dtype)  # 元の型に戻す


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, n_kv_head, head_dim, rope=None):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        E, H, KV, D = embed_dim, n_head, n_kv_head, head_dim

        self.W_q = nn.Linear(E, H * D, bias=False)
        self.W_k = nn.Linear(E, KV * D, bias=False)
        self.W_v = nn.Linear(E, KV * D, bias=False)
        self.W_o = nn.Linear(H * D, E, bias=False)

        self.rope = rope

        # KV-Cache用の変数
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0

    def forward(self, x, use_cache=False):
        B, C, E = x.shape
        H, KV, D = self.n_head, self.n_kv_head, self.head_dim

        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)    # (B, H, C, D)
        K = self.W_k(x).view(B, C, KV, D).transpose(1, 2)   # (B, KV, C, D)
        V = self.W_v(x).view(B, C, KV, D).transpose(1, 2)   # (B, KV, C, D)


        # RoPEにoffsetを渡す
        if self.rope is not None:
            if use_cache:
                Q = self.rope(Q, self.cache_offset)
                K = self.rope(K, self.cache_offset)
            else:
                Q = self.rope(Q)
                K = self.rope(K)

        # KV-Cacheの処理
        if use_cache:
            # 初回かどうかを判定（キャッシュ更新前に）
            is_first_call = (self.k_cache is None)

            if is_first_call:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)

            self.cache_offset += C

            # コンテキスト長制限: キャッシュが長すぎる場合は古い部分を切り捨て
            max_cache_len = self.rope.cos_cache.size(0) if self.rope else 2048
            if self.k_cache.size(2) > max_cache_len:
                self.k_cache = self.k_cache[:, :, -max_cache_len:]
                self.v_cache = self.v_cache[:, :, -max_cache_len:]
                self.cache_offset = max_cache_len

            K = self.k_cache
            V = self.v_cache

        # Flash Attention（GQA対応）
        if use_cache:
            # Prefill（初回・プロンプト全体処理）: is_causal=True（各トークンは前方のみ）
            # Decode（2回目以降・1トークン生成）: is_causal=False（新トークンは全キャッシュにattend）
            is_causal = is_first_call
            hidden = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal,
                                                    enable_gqa=True)
        else:
            # 学習時: is_causal=Trueでcausal mask適用
            hidden = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
                                                    enable_gqa=True)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

    def clear_cache(self):
        """キャッシュをクリアする"""
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0


class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)

        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)

    def forward(self, x):
        a = self.W(x)
        b = self.V(x)
        gated = F.silu(a) * b
        out = self.O(gated)
        return out


class Block(nn.Module):
    def __init__(self, embed_dim, n_head, n_kv_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, n_kv_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, ff_dim)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.mlp(self.norm2(x))
        return x

    def clear_cache(self):
        """キャッシュをクリアする"""
        self.attn.clear_cache()


class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head,
                 n_kv_head, n_layer, ff_dim, theta=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.theta = theta

        self.embed = nn.Embedding(vocab_size, embed_dim)

        head_dim = embed_dim // n_head
        rope = RoPE(theta, head_dim, max_context_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, n_kv_head, ff_dim, rope)
            for _ in range(n_layer)
        ])

        self.norm = nn.RMSNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, use_cache=False):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_kv_head': self.n_kv_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
            'theta': self.theta,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_kv_head=checkpoint['n_kv_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            theta=checkpoint['theta']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()


if __name__ == "__main__":
    vocab_size = 50000
    max_context_len = 1024
    embed_dim = 768
    n_head = 12
    n_kv_head = 4
    n_layer = 12
    ff_dim = 2048
    theta = 10000

    # モデルを作成
    model = GPT(vocab_size, max_context_len, embed_dim, n_head,
                n_kv_head, n_layer, ff_dim, theta)

    # パラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")

    # 動作テスト
    dummy_input = torch.randint(0, vocab_size, (1, max_context_len))
    logits = model(dummy_input)
    print(f"出力形状: {logits.shape}")
