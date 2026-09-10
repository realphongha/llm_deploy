#!/usr/bin/env bash
image=llama-cpp
port=8000
model=unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE
# --ulimit memlock=-1:-1 \
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --gpus '"device=2"' --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-qwen3.5-35b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 1 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 4096 -ub 4096 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 1.0 --top_k 40 --min_p 0.0 --presence_penalty 2.0 --repeat_penalty 1.0 \
    --chat-template-kwargs '{"enable_thinking": false}'
    # -c 262144 -fa on --mlock --threads 16 --n-gpu-layers 999 \

