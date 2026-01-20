import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from storybot.tokenizer import pretokenize, count_pairs, merge, find_chunk_boundaries

import os
import pickle
from multiprocessing import Pool
import shutil
import regex as re
from tqdm import tqdm
import numpy as np


class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf'))  # 存在しないペアは最低優先度

        while len(ids) > 1:
            # 現在のペアを取得（❶）
            counts = count_pairs(ids)

            # 最優先ペアを特定（❷）
            best_pair = min(counts, key=get_merge_priority)

            # マージ可能性の確認（❸）
            if best_pair not in self.merge_rules:
                break

            # マージの実行（❹）
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)

        return ids

    def _encode_chunk(self, args):
        """チャンクを処理してディスクにキャッシュ"""
        file_path, start, end, cache_dir, chunk_idx = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # チャンクをエンコード
            ids = self.encode(chunk_text)

        # キャッシュファイルに保存
        cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
        np.array(ids, dtype=np.uint16).tofile(cache_file)

        return cache_file, len(ids)


    def encode_file(self, file_path, output_file,
                                    num_processes=4, num_chunks=64,
                                   cache_dir="bpe_cache"):

        # キャッシュディレクトリの準備
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # チャンクを並列処理でトークナイズしてキャッシュ
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1

            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))

            with Pool(processes=num_processes) as pool:
                cache_results = list(tqdm(
                    pool.imap(self._encode_chunk, chunk_info_list),
                    total=len(chunk_info_list),
                    desc="Encoding chunks"
                ))

            # 総トークン数を計算
            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)

            # memmapファイルを作成
            dtype = np.uint16
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            # バッチ処理でキャッシュからmemmapへ書き込み
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()
            del arr

        finally:
            # キャッシュの削除
            shutil.rmtree(cache_dir)

        return total_tokens

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        # show_progressがTrueならtqdmで進捗表示
        texts = tqdm(texts) if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # 各事前トークンをBPEエンコード
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin", num_processes=8)

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin", num_processes=8)