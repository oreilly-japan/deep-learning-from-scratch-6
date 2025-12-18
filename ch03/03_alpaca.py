import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
from codebot.tokenizer import BPETokenizer

# トークナイザの読み込み
tokenizer = BPETokenizer.load_from('codebot/merge_rules.pkl')

# JSONデータの読み込み
with open('codebot/tiny_codes_sft.json') as f:
    data = json.load(f)

# 1つ目のサンプルを取り出す
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# Alpaca形式に変換
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# トークン化
token_ids = tokenizer.encode(text)
print(token_ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 104, 277, 280, 356, 473, 708, 108, 112, 930, 657, 63, 999]