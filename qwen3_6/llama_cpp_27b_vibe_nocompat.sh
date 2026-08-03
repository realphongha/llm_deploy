image=llama-cpp
port=8002
model=unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
# --ulimit memlock=-1:-1 \
docker run --gpus '"device=1"' --rm -it \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "LD_LIBRARY_PATH=" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-qwen3.6-27b \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 262144 -np 1 -fa on --threads 16 --n-gpu-layers 999 \
    -b 8192 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": true}' \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    # --mlock
    # --ulimit memlock=-1:-1 \
    # --temperature 0.7 --top_p 0.80 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": false}' \
