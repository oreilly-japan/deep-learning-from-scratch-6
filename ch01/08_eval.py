import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from codebot.tokenizer import BPETokenizer


tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

print("最初に学習された10個:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8", errors='replace')
    print(f"  ID {token_id}: '{text}'")

print("\n最後に学習された10個:")
for token_id in range(990, 1000):
    byte_seq = tokenizer.id_to_bytes[token_id]
    text = byte_seq.decode("utf-8", errors='replace')
    print(f"  ID {token_id}: '{text}'")


# 圧縮率を測定
sample_text = open("codebot/tiny_codes.txt").read()[:10000]  # 最初の10000文字

byte_count = len(sample_text.encode("utf-8"))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print("\n=== 圧縮効率 ===")
print(f"バイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍（平均 {compression_ratio:.2f} バイト/トークン）")


# ==== 以下はGPT系モデルのエンコードと圧縮率比較（tiktoken使用） ====
"""
import tiktoken

text = open("codebot/tiny_codes.txt").read()[:10000]
byte_count = len(text.encode("utf-8"))

for name, encoding_name in [('GPT-2', 'gpt2'), ('cl100k_base', 'cl100k_base')]:
    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text, allowed_special={'<|endoftext|>'}))
    ratio = byte_count / token_count
    print(f"{name}: 語彙サイズ {encoding.n_vocab:,}, 圧縮率 {ratio:.2f}倍")
"""