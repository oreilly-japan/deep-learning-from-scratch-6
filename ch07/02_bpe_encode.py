import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
from storybot.tokenizer import BPETokenizer


if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin", num_processes=8)

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin", num_processes=8)