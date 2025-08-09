#!/bin/bash

python3 /workspace/Mini-SGLang/test/bench_one_batch_sglang.py \
 --model-path /tmp/models/Llama-3.2-8B-Instruct \
 --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
 --tp-size 1 \
 --page-size 4 \
 --correct \
 --attention-backend torch_native