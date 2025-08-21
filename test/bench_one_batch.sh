#!/bin/bash

python3 -m minisglang.bench_one_batch \
 --model-path /tmp/models/Llama-3.2-8B-Instruct \
 --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
 --tp-size 4 \
 --page-size 8 \
 --output-len 64