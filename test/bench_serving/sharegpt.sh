#!/bin/bash

# set trace path
export SGLANG_TORCH_PROFILER_DIR=/workspace/moe-scaling/test/cache_aware_dp_attention/logs

# server
python3 -m minisglang.bench_serving \
 --backend sglang \
 --dataset-path /tmp/models/ShareGPT_data/ShareGPT_V3_unfiltered_cleaned_split.json \
 --num-prompts 10 \
 --port 31000

