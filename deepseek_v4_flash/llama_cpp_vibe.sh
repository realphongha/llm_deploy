image=llama-cpp
port=8002
model=unsloth/DeepSeek-V4-Flash-GGUF:UD-IQ3_XXS
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-ds4 \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 1 -b 4096 -ub 1024 --mlock --threads 16 --n-gpu-layers 999 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -fa on \
    --temperature 1.0 --top_p 1.0 --min_p 0.0 \
    --chat-template-kwargs '{"enable_thinking": false}'

