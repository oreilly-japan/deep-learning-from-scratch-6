import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import BPETokenizer


tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

print("最初に学習された10個:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")

print("\n最後に学習された10個:")
for token_id in range(9990, 10000):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")

# 圧縮率の測定
sample_text = open("storybot/tiny_stories_train.txt").read()[:10000]
byte_count = len(sample_text.encode('utf-8'))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print(f"\nバイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍（平均 {compression_ratio:.2f} バイト/トークン）")


print("\n=== CodeBotトークナイザの圧縮効率 ===")
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print(f"\nバイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍（平均 {compression_ratio:.2f} バイト/トークン）")
