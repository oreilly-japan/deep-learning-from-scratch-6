import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from collections import defaultdict
import regex as re
from tqdm import tqdm


def pretokenize_iter(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
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
    # 特殊トークンで分割
    texts = input_text.split(end_token)

    # 各テキスト片を事前トークン化
    pretoken_counts = defaultdict(int)
    for text in tqdm(texts, desc="Pretokenizing"):  # 進捗表示のためtqdmを使用
        for pretoken in pretokenize_iter(text):
            pretoken_counts[pretoken] += 1

    # 事前トークンをID列に変換
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        # 各ids列について、その出現回数を考慮してペア頻度を集計
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)

        # ペアが存在しない場合の処理
        if not pair_counts:
            break

        # 最頻出ペアを選択
        # best_pair = max(pair_counts, key=pair_counts.get)
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 各ids列をマージして更新
        new_ids_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            new_ids = merge(ids, best_pair, new_id)  # マージを適用
            new_ids_counts[tuple(new_ids)] += count
        ids_counts = new_ids_counts

    return merge_rules


vocab_size = 1000  # 語彙サイズの設定
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(text, vocab_size)