#!/bin/bash
cd "$(dirname "$0")"

# === dataset files ===
wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt
wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.txt

# === binary dataset files ===
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.bin
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.bin

# === tokenizer files ===
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/merge_rules.pkl

# === model files ===
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_pretrain.pt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_iter_500.pt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_iter_5000.pt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_dpo.pt