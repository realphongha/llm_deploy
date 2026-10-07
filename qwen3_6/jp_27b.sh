#!/usr/bin/env bash
# from ./llm_deploy
model=unsloth/Qwen3.6-27B-GGUF:Q4_K_M
image=llama-cpp-jetson
port=8008
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --rm \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
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
    --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-min-tokens 256
