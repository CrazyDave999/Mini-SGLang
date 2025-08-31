#!/bin/bash

# set trace path
export SGLANG_TORCH_PROFILER_DIR=/workspace/moe-scaling/test/cache_aware_dp_attention/logs

# server
python3 -m minisglang.bench_serving \
 --backend sglang \
 --dataset-name generated-shared-prefix \
 --gsp-num-groups 1 \
 --gsp-prompts-per-group 3 \
 --gsp-system-prompt-len 4 \
 --gsp-question-len 4 \
 --gsp-output-len 4 \
 --port 31000
