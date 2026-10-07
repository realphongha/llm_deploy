#!/usr/bin/env bash
# from ./llm_deploy
image=llama-cpp-jetson
model=unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL
port=8008
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --rm \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
    -v ./gemma4:/gemma4 \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --runtime=nvidia \
    -p $port:$port \
    -it $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -fa on --threads 8 --n-gpu-layers 999 \
    -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    -np 1 -c 65536 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-max-tokens 280
