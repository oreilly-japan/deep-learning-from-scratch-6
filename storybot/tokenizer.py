import os
import pickle
from multiprocessing import Pool
import shutil
from collections import defaultdict
import regex as re
from tqdm import tqdm
import numpy as np


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

def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:  # ファイルをバイナリモードで開く
        # ファイルサイズを取得
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks

        # チャンクの開始位置を計算（等間隔）
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # 最後にファイル終端を追加

        buffer_size = 4096  # 境界から先読みするバイト数

        # 境界位置の調整（終了トークンを探す）
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)  # 境界の推定位置から開始

            while True:
                buffer = file.read(buffer_size)  # バッファサイズ分を読む

                # ファイル終端に達した場合
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 読み取ったチャンクで終了トークンを検索
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # 見つかった場合、その位置を新しい境界とする
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                # 見つからなかった場合、次のバッファ位置に移動
                chunk_position += buffer_size

    # 重複を除去し、ソートして返す
    return sorted(set(chunk_boundaries))

def process_single_chunk(file_path, start, end, end_token):
    """1つのチャンクを処理する関数"""
    pretoken_counts = defaultdict(int)

    # ファイルを開いてチャンクを読み込む
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 特殊トークンで分割
        texts = chunk_text.split(end_token)

        # 各テキストを事前トークン化
        for text in texts:
            for pretoken in pretokenize_iter(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    # ファイルを開いてチャンクを読み込む
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 特殊トークンで分割
        texts = chunk_text.split(end_token)

        # 各テキストを事前トークン化
        for text in texts:
            for pretoken in pretokenize_iter(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=8):
    # ステップ1: チャンクの準備
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    # ステップ2: 並列処理
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list), desc="Pretokenizing"))

    # ステップ3: 結果を統合
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    # 事前トークンをID列に変換
    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}


    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)  # キャッシュ

    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):  # キャッシュに登録
            pair_to_ids[pair].add(ids)

    print("Training BPE...")
    for step in tqdm(range(num_merges)):
        # 最頻出ペアを選択
        # best_pair = max(pair_counts, key=pair_counts.get)
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # best_pairを含むids列をキャッシュから取得
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]  # もう使わないので削除

        # 影響のあるID列だけを更新
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]  # 古いID列を削除
            ids_counts[tuple(new_ids)] = ids_count  # 新しいID列を追加

            # 古いペア頻度を減少
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            # 新しいペア頻度を増加
            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return merge_rules

class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_byte = {i: bytes([i]) for i in range(256)}
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_byte[new_id] = self.id_to_byte[token1] + self.id_to_byte[token2]
        self.id_to_byte[self.end_token_id] = self.end_token.encode('utf-8')

        self.vocab_size = len(self.id_to_byte)

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
                for pretoken in pretokenize_iter(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

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
            # ステップ1: チャンクを並列処理でトークナイズしてキャッシュ
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

            # ステップ2: 総トークン数を計算
            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)

            # ステップ3: memmapファイルを作成
            dtype = np.uint16
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            # ステップ4: バッチ処理でキャッシュからmemmapへ書き込み
            # OpenWebTextの例のようにバッチ化
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()
            del arr

        finally:
            # キャッシュは削除
            shutil.rmtree(cache_dir)

        return total_tokens

    def decode(self, ids):
        byte_list = [self.id_to_byte[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text