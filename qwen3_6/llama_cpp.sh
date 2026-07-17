image=llama-cpp
port=8002
model=unsloth/Qwen3.6-35B-A3B-GGUF:MXFP4_MOE
# --ulimit memlock=-1:-1 \
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-qwen3.6 \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 131072 -np 1 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 8192 -ub 4096 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-max-tokens 1024
