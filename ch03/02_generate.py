import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import torch
import torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device


# 設定
device = get_device()
model_path = 'codebot/model_pretrain.pt'
tokenizer_path = 'codebot/merge_rules.pkl'

# 生成設定
prompt = "def"  # 生成の開始プロンプト
max_new_tokens = 200  # 生成するトークン数の上限
temperature = 1.0  # 温度パラメータ（高いほどランダム）

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 評価モード

    # プロンプトをトークン化
    device = next(model.parameters()).device  # パラメータのデバイスを取得
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 生成されたトークンを保持する変数
    generated_ids = ids.clone()

    # トークン生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト長を超えた場合、古いトークンを切り捨てる
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 最後の位置のロジットを取得（次トークンの予測）
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 終了トークンが生成されたら停止
        if next_id.item() == tokenizer.end_token_id:
            break

        # 生成したトークンを追加
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # デコードして返す
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# テキスト生成
for i in range(5):
    print(f"--- サンプル {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(generated_text)
    print()