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
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device

# 設定
device = get_device()
data_path = 'storybot/tiny_stories_dpo.json'
tokenizer_path = 'storybot/merge_rules.pkl'
pretrain_model_path = 'storybot/model_pretrain.pt'
dpo_model_save_path = 'storybot/model_dpo.pt'

# ハイパーパラメータ
context_len = 256
batch_size = 8
learning_rate = 5e-6
beta = 0.1
max_iters = 1000


class DPODataset(Dataset):
    # コンストラクタ
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            sample = self._create_sample(item['prompt'], item['chosen'], item['rejected'])
            self.samples.append(sample)

    # パディングとマスクの作成
    def _pad_and_mask(self, ids, prompt_len):
        mask = [0] * prompt_len + [1] * (len(ids) - prompt_len)

        if len(ids) > self.context_len:
            ids = ids[:self.context_len]
            mask = mask[:self.context_len]
        else:
            pad_len = self.context_len - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len

        return ids, mask

    # サンプル作成
    def _create_sample(self, prompt, chosen, rejected):
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = prompt_ids + self.tokenizer.encode(chosen)
        rejected_ids = prompt_ids + self.tokenizer.encode(rejected)

        prompt_len = len(prompt_ids)
        chosen_ids, chosen_mask = self._pad_and_mask(chosen_ids, prompt_len)
        rejected_ids, rejected_mask = self._pad_and_mask(rejected_ids, prompt_len)

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    # DataLoader用メソッド
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = self.samples[idx]
        return (
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(chosen_mask, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long),
            torch.tensor(rejected_mask, dtype=torch.long),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = self.samples[idx]
        return (
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(chosen_mask, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long),
            torch.tensor(rejected_mask, dtype=torch.long),
        )


def get_sequence_logprobs(model, ids, mask):
    logits = model(ids)  # (B, C, V)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    per_token_logprobs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, C-1)
    # マスクを適用（応答部分のみ）
    masked_logprobs = per_token_logprobs * mask[:, 1:]
    return masked_logprobs.sum(dim=-1)  # (B,)


def compute_dpo_loss(model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta):
    # 現在のモデルのlog-prob
    chosen_logprobs = get_sequence_logprobs(model, chosen_ids, chosen_mask)
    rejected_logprobs = get_sequence_logprobs(model, rejected_ids, rejected_mask)

    # 参照モデルのlog-prob
    with torch.no_grad():
        ref_chosen_logprobs = get_sequence_logprobs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logprobs = get_sequence_logprobs(ref_model, rejected_ids, rejected_mask)

    # DPO loss
    logits = beta * (
        (chosen_logprobs - rejected_logprobs) -
        (ref_chosen_logprobs - ref_rejected_logprobs)
    )
    return -F.logsigmoid(logits).mean()


# トークナイザとデータセットの準備
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = DPODataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルとオプティマイザ
model = GPT.load_from(pretrain_model_path, device=device)
ref_model = GPT.load_from(pretrain_model_path, device=device)
ref_model.eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 学習ループ
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
    chosen_ids, chosen_mask = chosen_ids.to(device), chosen_mask.to(device)
    rejected_ids, rejected_mask = rejected_ids.to(device), rejected_mask.to(device)

    loss = compute_dpo_loss(
        model, ref_model,
        chosen_ids, chosen_mask,
        rejected_ids, rejected_mask,
        beta
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
plt.savefig("loss_dpo.svg", bbox_inches='tight')

model.save(dpo_model_save_path)