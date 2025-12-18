import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import os
import pickle
from storybot.tokenizer import train_bpe


if __name__ == '__main__':
    vocab_size = 10000
    file_path = "webbot/owt_train.txt"
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)

    with open("webbot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)