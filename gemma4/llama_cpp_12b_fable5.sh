image=llama-cpp
port=8003
model=yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    -v ./gemma4:/gemma4 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-gemma4-fable5 \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 1 -b 8192 -ub 2048 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 0.3 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --jinja

