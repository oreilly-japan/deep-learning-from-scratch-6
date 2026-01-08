import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 評価モード

    # ❶ プロンプトをトークン化
    device = next(model.parameters()).device  # パラメータのデバイスを取得
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 生成されたトークンを保持する変数
    generated_ids = ids.clone()

    # ❷ トークン生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト長を超えた場合は末尾のみ使用
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 次のトークンの予測
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

    # ❸ デコードして返す
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')