import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# 設定
device = get_device()
model_path = 'storybot/model_pretrain.pt'
tokenizer_path = 'storybot/merge_rules.pkl'

# 生成設定
# prompt = "Once upon a time"  # 生成の開始プロンプト
prompt = "<|endoftext|>"
max_new_tokens = 1000  # 生成するトークン数の上限
temperature = 1.0  # 温度パラメータ（高いほどランダム）
num_samples = 3  # 生成するサンプル数

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

# テキスト生成
for i in range(num_samples):
    print(f"--- サンプル {i+1} ---")
    generated_text = generate(
        model, tokenizer, prompt, max_new_tokens, temperature
    )
    print(generated_text)