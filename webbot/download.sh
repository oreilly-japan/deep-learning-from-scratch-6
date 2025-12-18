#!/bin/bash
cd "$(dirname "$0")"

wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_train.txt
wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_valid.txt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_train.bin
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_valid.bin
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/merge_rules.pkl