#!/usr/bin/env bash
image=llama-cpp
port=8000
model=unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --gpus all --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    --ulimit memlock=-1:-1 \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=rl \
    --name llama-cpp-qwen3.5-35b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 131072 -np 2 -cb -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 4096 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0

