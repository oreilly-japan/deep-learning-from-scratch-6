from collections import defaultdict

def count_pairs(ids):
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


class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # IDからバイト列への対応表（0~255を登録）
        self.id_to_byte = {i: bytes([i]) for i in range(256)}

        # マージされたトークンは元のトークンのバイト列を連結
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_byte[new_id] = self.id_to_byte[token1] + self.id_to_byte[token2]

        # 語彙サイズを設定
        self.vocab_size = len(self.id_to_byte)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        # 学習時の順序でマージルールを適用
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids

    def decode(self, ids):
        # 各トークンIDを対応するバイト列に変換
        byte_list = [self.id_to_byte[i] for i in ids]

        # すべてのバイト列を連結
        combined_bytes = b"".join(byte_list)

        # バイト列をUTF-8テキストに変換
        text = combined_bytes.decode("utf-8", errors="replace")
        return text

# 学習済みのマージルール
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

# トークナイザーを作成
tokenizer = BPETokenizer(merge_rules)

# テキストをエンコード
text = "Hello"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)  # [259, 108, 108, 111]
print(decoded)    # Hello