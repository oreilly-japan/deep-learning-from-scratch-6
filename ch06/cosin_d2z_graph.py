import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# パラメータ
max_lr = 1.0
min_lr = 0.1
warmup_ratio = 0.05
total_steps = 1000

progress = np.linspace(0, 1, total_steps)
warmup_steps = int(total_steps * warmup_ratio)

# Cosine annealing (Warmup + Cosine)
lr_cosine = np.zeros(total_steps)
for i in range(total_steps):
    if i < warmup_steps:
        lr_cosine[i] = max_lr * (i / warmup_steps)
    else:
        prog = (i - warmup_steps) / (total_steps - warmup_steps)
        lr_cosine[i] = min_lr + 0.5 * (1 + np.cos(np.pi * prog)) * (max_lr - min_lr)

# Linear D2Z (Warmup + Linear decay to zero)
lr_d2z = np.zeros(total_steps)
for i in range(total_steps):
    if i < warmup_steps:
        lr_d2z[i] = max_lr * (i / warmup_steps)
    else:
        prog = (i - warmup_steps) / (total_steps - warmup_steps)
        lr_d2z[i] = max_lr * (1 - prog)

# グラフ
plt.figure(figsize=(11, 6.5))

plt.plot(progress, lr_cosine, '-', color='#1f77b4', linewidth=2.5,
         label='コサインアニーリング', alpha=0.9)

plt.plot(progress, lr_d2z, '--', color='#ff7f0e', linewidth=2.5,
         label='D2Z')

# Warmupフェーズを薄く強調
plt.axvspan(0, warmup_ratio, alpha=0.12, color='gray')
plt.text(warmup_ratio/2, max_lr * 1.05, 'ウォームアップ',
         ha='center', va='bottom', fontsize=11, alpha=0.7)

plt.xlabel('学習の進行度', fontsize=13)
plt.ylabel('学習率', fontsize=13)
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.xlim(-0.01, 1.01)

plt.tight_layout()
plt.savefig('lr_cosine_vs_d2z.png', dpi=300, bbox_inches='tight')
plt.show()