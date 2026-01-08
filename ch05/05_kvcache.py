import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, rope=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.rope = rope

        # KV-Cache用の変数を追加
        self.k_cache = None  # Keyのキャッシュ
        self.v_cache = None  # Valueのキャッシュ
        self.cache_offset = 0  # 現在のキャッシュ位置を追跡

    def forward(self, x, use_cache=False):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

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
            if self.k_cache is None:
                # 初回:キャッシュを初期化
                self.k_cache = K
                self.v_cache = V
            else:
                # 2回目以降:新しいKeyとValueをキャッシュに追加
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)

            # オフセットを更新(次のトークンの位置へ)
            self.cache_offset += C

            # キャッシュされた全てのKeyとValueを使用
            K = self.k_cache
            V = self.v_cache

        # 通常のAttention計算
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        if not use_cache:
            mask = torch.tril(torch.ones(C, C, device=scores.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

    def clear_cache(self):
        """キャッシュをクリアする"""
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0

def silu(x):
    return x * torch.sigmoid(x)

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

        gated = F.silu(a) * b  # silu(a) * b
        out = self.O(gated)
        return out

class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
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
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, theta=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.theta = theta

        self.embed = nn.Embedding(vocab_size, embed_dim)

        head_dim = embed_dim // n_head
        rope = RoPE(theta, head_dim, max_context_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, rope)
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
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            theta=checkpoint['theta']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model

    def clear_cache(self):
        """全てのブロックのキャッシュをクリアする"""
        for block in self.blocks:
            block.clear_cache()


def generate_without_cache(model, start_ids, max_new_tokens):
    model.eval()

    ids = start_ids  # 最初のトークン
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 毎回、全てのトークンを渡す
            logits = model(ids, use_cache=False)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            # 新しいトークンを連結
            ids = torch.cat([ids, next_id], dim=1)

    return ids

def generate_with_cache(model, start_ids, max_new_tokens):
    model.eval()

    ids = start_ids  # 最初のトークン
    next_id = start_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1トークンずつ生成
            logits = model(next_id, use_cache=True)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids

def measure_generation_time(model, start_ids,use_cache, num_tokens=100):

    if not use_cache:
        model.clear_cache()

    start_time = time.time()

    if use_cache:
        generate_with_cache(model, start_ids, num_tokens)
    else:
        generate_without_cache(model, start_ids, num_tokens)

    elapsed = time.time() - start_time
    return elapsed

# テスト
model = GPT(vocab_size=1000, max_context_len=256, embed_dim=384,
            n_head=6, n_layer=6, ff_dim=1024)


start_ids = torch.tensor([[42]])  # 固定シード
max_new_tokens = 200

time_without = measure_generation_time(model, start_ids, use_cache=False, num_tokens=max_new_tokens)
time_with = measure_generation_time(model, start_ids, use_cache=True)
print(f"KV-Cacheなし: {time_without:.2f}秒")
print(f"KV-Cacheあり: {time_with:.2f}秒")
print(f"高速化率: {time_without / time_with:.1f}倍")

"""
print("\n=== 出力の一致確認 ===")
model.clear_cache()

# 同じ開始トークンで生成

# KV-Cacheなしで生成
output_without = generate_without_cache(model, start_ids, max_new_tokens=max_new_tokens)
print(f"KV-Cacheなし: {output_without[0, :11].tolist()}")

# KV-Cacheありで生成
model.clear_cache()
output_with = generate_with_cache(model, start_ids, max_new_tokens=max_new_tokens)

print(f"KV-Cacheあり: {output_with[0, :11].tolist()}")

print(output_with.shape, output_without.shape)
# 一致確認
if torch.equal(output_without[:, :max_new_tokens], output_with[:, :max_new_tokens]):
    print("✓ 出力が一致しました!")
else:
    print("✗ 出力が一致しません")
    print(f"差分の数: {(output_without[:, :max_new_tokens] != output_with[:, :max_new_tokens]).sum().item()}")
"""