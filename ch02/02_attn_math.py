import torch
import torch.nn.functional as F

# キー（映画のジャンル特性）
K = torch.tensor([
    [8, 2, 3],  # アクション重視の映画
    [3, 9, 1],  # ドラマ重視の映画
    [1, 2, 9],  # コメディ重視の映画
    [5, 5, 5],  # バランスの取れた映画
    [7, 6, 2],  # アクションドラマ
    [2, 7, 6],  # ドラマコメディ
    [9, 1, 1],  # 純粋なアクション
], dtype=torch.float32)

# バリュー（ユーザーの評価）
V = torch.tensor([
    [85],
    [70],
    [60],
    [75],
    [80],
    [65],
    [90]
], dtype=torch.float32)

# 新しい映画のジャンル特性（複数のクエリ）
Q = torch.tensor([
    [6, 4, 5],  # バランスの取れたアクション寄りの映画
    [2, 8, 3],  # ドラマ重視の映画
    [4, 3, 7],  # コメディ寄りの映画
], dtype=torch.float32)

def attention(Q, K, V):
    similarity = torch.matmul(Q, K.t())     # QK^Tを計算
    weights = F.softmax(similarity, dim=1)  # ソフトマックス関数
    output = torch.matmul(weights, V)       # 重み付き和
    return output, weights

predicted_ratings, weights = attention(Q, K, V)

# 結果の表示
for movie, rating in zip(Q, predicted_ratings):
    print(f"映画 {movie.numpy()} の予測評価: {rating.item():.2f}")