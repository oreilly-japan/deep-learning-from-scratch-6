import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from webbot.model import GPT
from webbot.utils import get_device


# --- DDP初期化 ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    seed_offset = int(os.environ['RANK'])
else:
    device = get_device()
    local_rank = 0
    seed_offset = 0

is_main = (not ddp) or (local_rank == 0)
torch.manual_seed(42 + seed_offset)

# --- wandb初期化（rank 0のみ） ---
if is_main:
    wandb.init(project='webbot-pretrain', config={})  # configは後で更新


def get_lr(it, max_lr, warmup_iters, max_iters):
    # ウォームアップ: 0 -> max_lr
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # 線形減衰: max_lr -> 0
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0


def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)
        ix = ix[ix + context_len + 1 < len(data)]
        if len(ix) == 0:
            return None, None

    x = torch.stack([torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)


def evaluate(model, val_data, context_len, batch_size, device):
    """Validation: 全データを順番に処理"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_start = len(val_data) - context_len - 1
    num_batches = (max_start // context_len) // batch_size + 1

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating", leave=False):
            offset = batch_idx * batch_size * context_len

            x, y = get_batch(val_data, context_len, batch_size, device,
                        random=False, offset=offset)

            if x is None:
                break

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    y.view(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += x.numel()

    model.train()
    return total_loss / total_tokens

# --- ハイパーパラメータ ---
vocab_size = 50000
context_len = 1024
embed_dim = 768
n_head = 12
n_kv_head = 4
n_layer = 12
ff_dim = 2048
theta = 10000

micro_batch_size = 32
accumulation_steps = 4
learning_rate = 6e-4
warmup_iters = 500
max_iters = 100000
grad_clip = 1.0
eval_interval = 1000

# --- wandb configを更新 ---
if is_main:
    wandb.config.update({
        'vocab_size': vocab_size, 'context_len': context_len,
        'embed_dim': embed_dim, 'n_head': n_head, 'n_kv_head': n_kv_head,
        'n_layer': n_layer, 'ff_dim': ff_dim, 'theta': theta,
        'micro_batch_size': micro_batch_size,
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': micro_batch_size * accumulation_steps,
        'learning_rate': learning_rate, 'warmup_iters': warmup_iters,
        'max_iters': max_iters, 'grad_clip': grad_clip,
    })

# --- データ ---
train_data_path = 'webbot/owt_train.bin'
val_data_path = 'webbot/owt_valid.bin'
model_save_path = 'webbot/model_pretrain.pt'

train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

if is_main:
    print(f"学習データ: {len(train_data):,} tokens")
    print(f"検証データ: {len(val_data):,} tokens")

# --- モデル ---
model = GPT(
    vocab_size, context_len, embed_dim, n_head,
    n_kv_head, n_layer, ff_dim, theta
).to(device)

if is_main:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")

model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[local_rank])

# --- オプティマイザ ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 学習ループ ---
pbar = tqdm(range(max_iters), disable=not is_main)
val_loss = float('inf')
val_losses = []
val_iters = []

for step in pbar:
    # 学習率を更新
    lr = get_lr(step, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad()

    # 勾配蓄積ループ
    for micro_step in range(accumulation_steps):
        batch_x, batch_y = get_batch(train_data, context_len,
                                     micro_batch_size, device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   batch_y.view(-1))
            loss = loss / accumulation_steps

        # DDP: 最後のmicro_step以外は勾配同期を無効化
        if ddp and micro_step < accumulation_steps - 1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # wandb: 毎ステップのログ
    train_loss = loss.item() * accumulation_steps
    if is_main:
        wandb.log({'train/loss': train_loss, 'train/lr': lr}, step=step)

    # 定期的に評価
    if is_main and ((step % eval_interval) == 0 or step == max_iters - 1):
        raw_model = model.module if ddp else model
        val_loss = evaluate(raw_model, val_data, context_len,
                           micro_batch_size, device)
        val_losses.append(val_loss)
        val_iters.append(step)
        print(f"\nstep {step}: val_loss = {val_loss:.4f}")

        wandb.log({'val/loss': val_loss}, step=step)

    if is_main:
        pbar.set_postfix({'loss': f'{train_loss:.4f}',
                          'val_loss': f'{val_loss:.4f}', 'lr': f'{lr:.2e}'})

# --- 学習曲線の保存 ---
if is_main:
    plt.figure(figsize=(10, 6))
    plt.plot(val_iters, val_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig('webbot_val_loss.png')

    raw_model = model.module if ddp else model
    raw_model.save(model_save_path)
    print(f"\nモデル保存: {model_save_path}")

if is_main:
    wandb.finish()

if ddp:
    dist.destroy_process_group()
