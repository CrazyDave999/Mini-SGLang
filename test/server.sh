#!/bin/bash

# server
python3 -m minisglang.launch_server \
 --model-path /tmp/models/Llama-3.2-8B-Instruct \
 --tokenizer-path /tmp/models/Llama-3.2-8B-Instruct \
 --tp-size 4 \
 --mem-fraction-static 0.8 \
 --port 31000 > log 2>&1
