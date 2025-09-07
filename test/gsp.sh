#!/bin/bash

# set trace path
export SGLANG_TORCH_PROFILER_DIR=/workspace/moe-scaling/test/cache_aware_dp_attention/logs

# server
python3 -m minisglang.bench_serving \
 --backend sglang \
 --dataset-name generated-shared-prefix \
 --gsp-num-groups 4 \
 --gsp-prompts-per-group 4 \
 --gsp-system-prompt-len 16 \
 --gsp-question-len 16 \
 --gsp-output-len 16 \
 --port 31000
