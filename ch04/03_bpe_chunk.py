import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
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

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>"):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks=64)

    # 特殊トークンで分割
    pretoken_counts = defaultdict(int)
    with open(file_path, "rb") as f:
        total_chunks = len(chunk_boundaries) - 1

        for i in tqdm(range(total_chunks), desc="Pretokenizing"):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i+1]

            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # 特殊トークンで分割
            texts = chunk_text.split(end_token)
            # 各テキスト片を事前トークン化
            for text in texts:
                for pretoken in pretokenize_iter(text):
                    pretoken_counts[pretoken] += 1


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

    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:  # ペアが存在しない場合の処理
            break

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


file_path = "storybot/tiny_stories_train.txt"
vocab_size = 10000  # 語彙サイズの設定
merge_rules = train_bpe(file_path, vocab_size)