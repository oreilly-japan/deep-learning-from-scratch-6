#!/bin/bash
cd "$(dirname "$0")"

# === model files ===
wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_pretrain.pt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_sft.pt
# wget -nc https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_grpo.pt