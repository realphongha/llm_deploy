image=llama-cpp
port=8003
model=unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL
docker run --gpus all --rm -it \
    -v ~/.cache:/root/.cache \
    --ulimit memlock=-1:-1 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4-12b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    --cache-type-k bf16 --cache-type-v bf16 \
    -c 262144 -np 1 -b 8192 -ub 2048  -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --image-max-tokens 280
    # --image-max-tokens 1120
