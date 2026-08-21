image=llama-cpp
port=8007
model=unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
# --ulimit memlock=-1:-1 \
docker run --gpus '"device=2"' --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-qwen3.8-27b \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 131072 -np 1 -cb -fa on --mlock --threads 4 --n-gpu-layers 999 \
    -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0 \
    --jinja \
    --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "medium"}' \
    # --temperature 0.7 --top_p 0.80 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 --chat-template-kwargs '{"enable_thinking": false}' \
