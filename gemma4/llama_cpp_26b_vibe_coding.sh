image=llama-cpp
port=8003
model=unsloth/gemma-4-26B-A4B-it-GGUF:MXFP4_MOE
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    -v ./gemma4:/gemma4 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-gemma4 \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 1 -b 4096 -ub 4096 -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --jinja --chat-template-file /gemma4/chat_template_gemma4.fixed.jinja

