import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device

# 設定
device = get_device()
data_path = 'codebot/tiny_codes_sft.json'
tokenizer_path = 'codebot/merge_rules.pkl'
pretrain_model_path = 'codebot/model_pretrain.pt'
sft_model_save_path = 'codebot/model_sft.pt'

# ハイパーパラメータ
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            ids, labels = self._create_sample(item['instruction'], item['response'])
            self.samples.append((ids, labels))

    def _create_sample(self, instruction, response):
        # プロンプトとレスポンスをフォーマット
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        # トークン化
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # 入力系列とラベルの作成（プロンプト部分は-100でマスク）
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 言語モデル用にシフト（入力と正解を1つずらす）
        ids = ids[:-1]
        labels = labels[1:]

        # context_lenに合わせてパディングまたは切り詰め
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len  # パディングIDとして0を使用
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]

        return ids, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long)


# トークナイザとデータセットの準備
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = SFTDataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルとオプティマイザ
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 学習ループ
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# 結果を保存
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

model.save(sft_model_save_path)