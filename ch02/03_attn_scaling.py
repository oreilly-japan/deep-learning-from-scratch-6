import torch
import torch.nn.functional as F

x = torch.tensor([100.0, 200.0, 300.0])
y = F.softmax(x, dim=0)
print(y)


import numpy as np
import matplotlib.pyplot as plt

d = 10

q = np.random.randn(d)
k = np.random.randn(d)

dot_product = np.dot(q, k)
scaled_dot_product = dot_product / np.sqrt(d)

print('dot product:', dot_product)
print('scaled dot product:', scaled_dot_product)


d = 10
num_samples = 10000  # 先ほどの内積の計算を10000回行う

dot_products = []
scaled_dot_products = []

for _ in range(num_samples):
    q = np.random.randn(d)
    k = np.random.randn(d)

    dot_product = np.dot(q, k)
    scaled_dot_product = dot_product / np.sqrt(d)

    dot_products.append(dot_product)
    scaled_dot_products.append(scaled_dot_product)


# 結果をプロット
plt.figure(figsize=(10, 6))
plt.hist(dot_products, bins=50, alpha=0.5, label='Without scaling')
plt.hist(scaled_dot_products, bins=50, alpha=0.5, label='With scaling')
plt.legend()
plt.show()

print("Variances without scaling:", np.var(dot_products))
print("Variances with scaling:", np.var(scaled_dot_products))