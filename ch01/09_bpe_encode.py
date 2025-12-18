import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
from codebot.tokenizer import BPETokenizer


# 事前学習用データのエンコード
text = open("codebot/tiny_codes.txt").read()
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tokenizer.encode(text, show_progress=True)  # プログレスバーを表示

# numpy配列に変換して保存
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("codebot/tiny_codes.bin")

print(f"トークンID数: {len(ids_array)}")
print(f"最初の20個のトークンID: {ids_array[:20]}")