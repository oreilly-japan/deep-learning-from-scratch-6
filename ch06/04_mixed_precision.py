import torch

print("----- FP16 -----")

large = torch.tensor(1000.0, dtype=torch.float16)
small = torch.tensor(0.01, dtype=torch.float16)
print(large + small)  # tensor(1000., dtype=torch.float16)

tiny = torch.tensor(1e-8, dtype=torch.float16)
print(tiny)  # tensor(0., dtype=torch.float16)

huge = torch.tensor(70000.0, dtype=torch.float16)
print(huge)  # tensor(inf, dtype=torch.float16)

print("----- BF16 -----")

# FP16ではアンダーフロー
tiny_fp16 = torch.tensor(1e-8, dtype=torch.float16)
print(tiny_fp16)  # tensor(0., dtype=torch.float16)

# BF16では表現可能
tiny_bf16 = torch.tensor(1e-8, dtype=torch.bfloat16)
print(tiny_bf16)  # tensor(1.0012e-08, dtype=torch.bfloat16)

# FP16ではオーバーフロー
huge_fp16 = torch.tensor(70000.0, dtype=torch.float16)
print(huge_fp16)  # tensor(inf, dtype=torch.float16)

# BF16では表現可能
huge_bf16 = torch.tensor(70000.0, dtype=torch.bfloat16)
print(huge_bf16)  # tensor(70144., dtype=torch.bfloat16)


print("----- 自動混合精度 -----")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(1000, 1000, device=device)

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    b = a @ a   # 行列積はBF16
    c = a.sum() # 累積はFP32
    print(b.dtype)  # torch.bfloat16
    print(c.dtype)  # torch.float32