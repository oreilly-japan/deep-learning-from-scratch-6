#!/bin/bash
cd "$(dirname "$0")"

wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt
wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.txt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.bin
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.bin
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/merge_rules.pkl