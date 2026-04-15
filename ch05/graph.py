import matplotlib.pyplot as plt
import numpy as np

# データ生成
x = np.linspace(-3, 3, 500)

# ReLU関数
relu = np.maximum(0, x)

# GELU関数（近似式）
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Swish関数（SiLU）
swish = x * (1 / (1 + np.exp(-x)))  # x * sigmoid(x)

# プロット作成
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, relu, 'b-', linewidth=2, label='ReLU')
ax.plot(x, gelu, 'g-', linewidth=2, label='GELU')
ax.plot(x, swish, 'r-', linewidth=2, label='Swish')

# 軸の設定
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)

# グリッド
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# 凡例
ax.legend(loc='upper left', fontsize=12)

# 余白調整
plt.tight_layout()

# PNGで保存
plt.savefig('activation_comparison.png', format='png', bbox_inches='tight')
plt.close()