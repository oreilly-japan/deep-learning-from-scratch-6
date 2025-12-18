import torch

# 簡単な例で検証
x_even = torch.tensor([1.0, 3.0, 5.0])  # 偶数インデックス(0, 2, 4)の要素
x_odd = torch.tensor([2.0, 4.0, 6.0])   # 奇数インデックス(1, 3, 5)の要素

# stackして確認
stacked = torch.stack([x_even, x_odd], dim=-1)
print("stacked shape:", stacked.shape)
print("stacked:\n", stacked)

# reshapeして確認
result = stacked.reshape(-1)
print("\nresult shape:", result.shape)
print("result:", result)

# 期待される結果: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
