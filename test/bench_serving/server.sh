#!/bin/bash

# # server
# python3 -m minisglang.launch_server \
#  --model-path /tmp/models/Llama-3.2-8B-Instruct \
#  --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
#  --tp-size 4 \
#  --mem-fraction-static 0.8 \
#  --port 31000 > log 2>&1
# server

export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

python3 -m minisglang.launch_server \
 --model-path /tmp/models/Llama-3.2-8B-Instruct \
 --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
 --tp-size 4 \
 --page-size 512 \
 --mem-fraction-static 0.8 \
 --port 31000
