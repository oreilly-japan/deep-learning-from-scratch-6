import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


# データセット
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):
        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(response)

            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        # パディング
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask


# 報酬関数
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # 最後の数値を取得
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


# グループ生成
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)

# 損失関数
def compute_probs(model, ids):
    logits = model(ids)  # (B, C, V)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    token_probs = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (B, C-1)

    return token_probs

def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    # 現在モデルの各トークンの確率
    probs = compute_probs(model, ids)
    # 古いモデルの各トークンの確率
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    # トークンごとの確率比
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]
    token_objective = torch.min(unclipped, clipped) * mask

    n_samples = ids.size(0)
    return -token_objective.sum() / n_samples


# 設定
device = get_device()
tokenizer_path = 'codebot/merge_rules.pkl'
sft_model_path = 'codebot/model_sft.pt'
grpo_model_save_path = 'codebot/model_grpo.pt'

# ハイパーパラメータ
learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2  # 同じ生成データに対しての更新回数
eval_interval = 10
epsilon = 0.2  # クリッピング範囲
group_size = 8  # グループサイズ
batch_size = 32

# 初期化
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(sft_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

old_model = GPT.load_from(sft_model_path, device=device)  # 古いモデル
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)

# 学習ループ
accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    # バッチデータを取得
    prompts, gts = next(data_iter)

    # 古いモデル（old_model）を更新
    old_model.load_state_dict(model.state_dict())

    # 古いモデルで複数サンプルを生成し、報酬とアドバンテージを計算
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    # バッチデータを作成
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    all_advantages = all_advantages.to(device)

    # 生成データに対して複数回更新
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
        optimizer.step()

    # 定期的に評価
    if i % eval_interval == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for prompt, gt in dataset.data:
                response = generate(model, tokenizer, prompt, temperature=0)
                reward = calculate_reward(gt, response)
                correct += reward > 0
                total += 1
        model.train()
        current_accuracy = correct / total * 100
        accuracies.append(current_accuracy)

    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_accuracy:.1f}%'})

# 学習済みモデルを保存
model.save(grpo_model_save_path)

plt.figure()
steps = list(range(0, len(accuracies) * eval_interval, eval_interval))
plt.plot(steps, accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('GRPO Training')
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_grpo.png")