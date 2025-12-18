import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
from storybot.tokenizer import BPETokenizer


if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("webbot/merge_rules.pkl")

    tokenizer.encode_file(
        "webbot/owt_train.txt",
        "webbot/owt_train.bin", num_processes=8)

    tokenizer.encode_file(
        "webbot/owt_valid.txt",
        "webbot/owt_valid.bin", num_processes=8)