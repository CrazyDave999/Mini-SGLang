#!/bin/bash

# set trace path
export SGLANG_TORCH_PROFILER_DIR=/workspace/moe-scaling/test/cache_aware_dp_attention/logs

# server
python3 -m minisglang.launch_server \
 --model-path /tmp/models/Llama-3.2-8B-Instruct \
 --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
 --tp-size 4 \
 --mem-fraction-static 0.8 \
 --port 31000
