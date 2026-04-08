image=llama-cpp
port=8000
model=unsloth/gemma-4-31B-it-GGUF:Q4_K_M
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=rl \
    --name llama-cpp-gemma4 \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 40960 -np 1 -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --swa-full \
    --image-max-tokens 1120
    # -c 163840 -cb -np 2 -b 4096 -ub 4096 -fa on --mlock --threads 16 --n-gpu-layers 999 \

