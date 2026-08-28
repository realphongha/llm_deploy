image=llama-cpp
port=8007
model=unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL
    # --ulimit memlock=-1:-1 \
docker run --gpus '"device=2"' --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4-e4b \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -np 4 -cb -b 4096 -ub 4096 \
    -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --cache-type-k bf16 --cache-type-v bf16 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --jinja \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    # -c 65536 -np 1 -b 4096 -ub 4096 \

