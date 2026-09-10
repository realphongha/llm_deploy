#!/usr/bin/env bash
image=llama-cpp
port=8007
model=unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL
    # --ulimit memlock=-1:-1 \
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --gpus '"device=2"' --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    -v ./gemma4:/gemma4 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4-12b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    --cache-type-k bf16 --cache-type-v bf16 \
    -c 65536 -np 1 -b 8192 -ub 2048  -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --image-max-tokens 280 \
    # --spec-type draft-mtp --spec-draft-n-max 2 \
    # --jinja --chat-template-file /gemma4/chat_template_12b.jinja
    # --image-max-tokens 1120
