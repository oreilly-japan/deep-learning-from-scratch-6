import regex as re
from collections import defaultdict
import pickle
from tqdm import tqdm


def pretokenize_iter(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

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

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    texts = input_text.split(end_token)

    ids_list = []
    for text in texts:
        for pretoken in pretokenize_iter(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        all_counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids)
            for pair, count in counts.items():
                all_counts[pair] += count

        if not all_counts:
            break

        # best_pair = max(all_counts, key=all_counts.get)
        best_pair = max(all_counts, key=lambda pair: (all_counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules


class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[token1] + self.id_to_bytes[token2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode('utf-8')

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # show_progressがTrueならtqdmで進捗表示
        texts = tqdm(texts, desc="Encoding") if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # 各事前トークンをBPEエンコード
                for pretoken in pretokenize_iter(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text