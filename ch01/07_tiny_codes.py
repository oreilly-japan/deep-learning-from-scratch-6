import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import pickle
from codebot.tokenizer import train_bpe

vocab_size = 1000  # 語彙数
text = open("codebot/tiny_codes.txt").read()
merge_rules = train_bpe(text, vocab_size)

# 学習済みマージルールをファイルに保存
with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)