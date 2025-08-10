#!/bin/bash

python3 -m minisglang.bench_one_batch \
 --model-path /Meta-Llama-3.1-8B-Instruct \
 --tokenizer-path /Meta-Llama-3.1-8B-Instruct \
 --tp-size 1 \
 --page-size 4 \
 --output-len 64