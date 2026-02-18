import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from webbot.model import GPT
from storybot.tokenizer import BPETokenizer
from webbot.utils import generate, get_device

# 設定
device = get_device()
model_path = 'webbot/model_pretrain.pt'
tokenizer_path = 'webbot/merge_rules.pkl'
max_new_tokens = 100
temperature = 0.5

# テスト用プロンプト
prompts = [
    "In 1991, Linus Torvalds created",
    "Monday, Tuesday, Wednesday,",
    "Python was created by",
    "Machine learning is defined as",
    "The capital of Japan is",
]

# モデルとトークナイザの読み込み
print("モデルとトークナイザを読み込み中...")
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)
print(f"読み込み完了！\n")

# 各プロンプトでテキスト生成
for i, prompt in enumerate(prompts, 1):
    print(f"{'=' * 70}")
    print(f"プロンプト {i}: {prompt}")
    print(f"{'=' * 70}")

    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    print(response)
    print()
