image=llama-cpp
port=8000
model=unsloth/gemma-4-26B-A4B-it-GGUF:MXFP4_MOE
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=rl \
    --name llama-cpp-qwen3.5 \
    $image -h \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 524288 -np 2 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 16384 -ub 16384 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --image-max-tokens 1120

