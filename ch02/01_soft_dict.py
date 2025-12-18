import numpy as np

# ディクショナリの例
d = {
    'apple': 100,
    'banana': 200,
    'cherry': 300,
    'durian': 400
}

# キーを指定して値を取得
query = 'banana'
print(d[query]) # 200


# キー: (アクション性, ドラマ性, コメディ性) 各0〜10で表現
# バリュー: ユーザーの評価点 (0〜100点)
movie_preferences = {
    (8, 2, 3): 85,   # アクション重視の映画
    (3, 9, 1): 70,   # ドラマ重視の映画
    (1, 2, 9): 60,   # コメディ重視の映画
    (5, 5, 5): 75,   # バランスの取れた映画
    (7, 6, 2): 80,   # アクションドラマ
    (2, 7, 6): 65,   # ドラマコメディ
    (9, 1, 1): 90,   # 純粋なアクション
}

# 新しい映画
new_movie = (6, 4, 5)

def soft_dictionary(query, dictionary):
    # 類似度
    similarity = []
    for key in dictionary:
        s = np.dot(query, key)
        similarity.append(s)

    # ソフトマックス
    exp_similarity = np.exp(similarity)
    weights = exp_similarity / np.sum(exp_similarity)

    # 重み付き和
    result = 0
    for weight, value in zip(weights, dictionary.values()):
        result += weight * value

    return result, weights


predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)

print(f"新しい映画 {new_movie} の予測評価: {predicted_rating:.2f} 点")
print("\n貢献度:")
for key, weight in zip(movie_preferences.keys(), weights):
    print(f"映画 {key}: {weight*100:.2f}%")