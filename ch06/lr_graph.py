import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS
# plt.rcParams['font.family'] = 'Yu Gothic'  # Windows

# パラメータ
warmup_ratio = 0.05  # ウォームアップ期間（全体の5%）
eta_min_ratio = 0.1  # コサインアニーリングの最小学習率（最大の10%）

# データ生成
t = np.linspace(0, 1, 1000)

# コサインアニーリング（ウォームアップ付き）
def cosine_annealing(t, warmup_ratio, eta_min_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # ウォームアップ: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # コサインアニーリング: 1 -> eta_min
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + np.cos(np.pi * progress))
    return lr

# D2Z（ウォームアップ付き）
def d2z(t, warmup_ratio):
    lr = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < warmup_ratio:
            # ウォームアップ: 0 -> 1
            lr[i] = ti / warmup_ratio
        else:
            # 線形減衰: 1 -> 0
            progress = (ti - warmup_ratio) / (1 - warmup_ratio)
            lr[i] = 1 - progress
    return lr

cosine_lr = cosine_annealing(t, warmup_ratio, eta_min_ratio)
d2z_lr = d2z(t, warmup_ratio)

# プロット作成
fig, ax = plt.subplots(figsize=(10, 6))

# ウォームアップ領域をグレーで塗りつぶし
ax.axvspan(0, warmup_ratio, color='lightgray', alpha=0.5)
ax.text(0.01, 1.02, 'ウォームアップ', fontsize=10, va='bottom')

# 学習率曲線
ax.plot(t, cosine_lr, 'b-', linewidth=2, label='コサインアニーリング')
ax.plot(t, d2z_lr, color='orange', linestyle='--', linewidth=2, label='D2Z')

# 軸の設定
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_xlabel('学習の進行度', fontsize=12)
ax.set_ylabel('学習率', fontsize=12)

# グリッド
ax.grid(True, linestyle='--', alpha=0.7)

# 凡例
ax.legend(loc='upper right', fontsize=12)

# 余白調整
plt.tight_layout()

# PNGで保存
plt.savefig('lr_schedule.png', format='png', bbox_inches='tight')
plt.close()