#!/bin/bash
if [ -z "$MOES_MODEL_DIR" ]; then
    echo "ENV <MOES_MODEL_DIR> not set, using default value: /tmp/models"
    export MOES_MODEL_DIR="/tmp/models"
fi

mkdir -p $MOES_MODEL_DIR
# meta-llama/Llama-3.1-8B-Instruct
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir $MODEL_DIR/Meta-Llama-3.1-8B-Instruct
# deepseek-ai/DeepSeek-V3
# modelscope download --model deepseek-ai/DeepSeek-V3 --local_dir $MODEL_DIR/DeepSeek-V3
# AI-ModelScope/Mixtral-8x7B-Instruct-v0.1
# modelscope download --model AI-ModelScope/Mixtral-8x7B-Instruct-v0.1 --local_dir $MOES_MODEL_DIR/Mixtral-8x7B-Instruct-v0.1