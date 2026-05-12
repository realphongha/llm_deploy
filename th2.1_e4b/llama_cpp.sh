image=llama-cpp
# port=8003
# model=realphongha/th2.1-e4b-non-reasoning-GGUF:Q5_K_M
# container_name=llama-cpp-th2.1-rl
port=8004
model=unsloth/gemma-4-E4B-it-GGUF:Q5_K_M
container_name=llama-cpp-th2.1-base
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name $container_name \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 4 -cb -fa on --mlock --threads 8 --n-gpu-layers 999 -b 512 -ub 512 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --temperature 1.0 --top_p 0.95 --top_k 64
    # -c 1048576 -np 256 -cb -fa on --mlock --threads 8 --n-gpu-layers 999 -b 4096 -ub 4096 \

