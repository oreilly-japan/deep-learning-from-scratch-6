from collections import defaultdict
import re

# def count_pairs(ids):
#     counts = defaultdict(int)
#     for pair in zip(ids, ids[1:]):
#         counts[pair] += 1
#     return counts

def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 特殊トークンでテキストを分割
    texts = input_text.split(end_token)
    ids_list = [list(text.encode("utf-8")) for text in texts]

    # 基本語彙（0-255）+ 終了トークン用（1個）を除いた分がマージ回数
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in range(num_merges):
        # 隣接ペアの頻度を集計
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        # ペアが存在しない場合の処理
        if not counts:
            break

        # 最頻出ペアを選択
        best_pair = max(counts, key=counts.get)
        # best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # マージを実行
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules

# 使用例
sample_text = "Hello world!<|endoftext|>This is BPE training."

merge_rules = train_bpe(sample_text, vocab_size=260)
print(merge_rules)  # {(105, 115): 256, (256, 32): 257, (105, 110): 258}


class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[token1] + self.id_to_bytes[token2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                ids = self._encode_text(text)
                all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


tokenizer = BPETokenizer(merge_rules)

text = "Hello world!<|endoftext|>"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)