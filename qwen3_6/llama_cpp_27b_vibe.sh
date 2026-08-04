image=llama-cpp
port=8003
model=unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
# --ulimit memlock=-1:-1 \
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-qwen3.6-27b \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 1 -cb -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 8192 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": true}' \
    # --temperature 0.7 --top_p 0.80 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": false}' \
    --spec-type draft-mtp --spec-draft-n-max 2 \
