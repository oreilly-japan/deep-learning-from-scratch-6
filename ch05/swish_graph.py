import numpy as np
import matplotlib.pyplot as plt

# 日本語フォントの設定（環境に応じて調整）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Swish関数の定義
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def relu(x):
    return np.maximum(0, x)

# xの範囲を設定
x = np.linspace(-3, 3, 1000)

# 関数の計算
y_swish = swish(x, beta=1.0)
y_relu = relu(x)

# グラフの描画
plt.figure(figsize=(10, 6))

# ReLUとSwishをプロット
plt.plot(x, y_relu, color='blue', linewidth=2, label='ReLU')
plt.plot(x, y_swish, color='red', linewidth=2, label='Swish')

# グリッド線の設定（細かい破線）
plt.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)

# 軸の設定
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# 軸ラベルとタイトル
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)

# 軸の範囲
plt.xlim(-3, 3)
plt.ylim(-0.5, 3.0)

# 凡例の位置（左上）
plt.legend(loc='upper left', fontsize=12, framealpha=0.9)

# 目盛りの設定
plt.xticks(np.arange(-3, 4, 1))
plt.yticks(np.arange(-0.5, 3.5, 0.5))

# レイアウトの調整
plt.tight_layout()

# 保存と表示
plt.savefig('swish_function.svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("グラフを作成しました: swish_function.png")