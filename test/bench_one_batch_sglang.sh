#!/bin/bash

python3 /workspace/Mini-SGLang/test/bench_one_batch_sglang.py \
 --model-path /Meta-Llama-3.1-8B-Instruct \
 --tokenizer-path /Meta-Llama-3.1-8B-Instruct \
 --tp-size 1 \
 --page-size 4 \
 --correct \
 --attention-backend torch_native