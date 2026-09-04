image=llama-cpp
port=8007
model=unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS
# --gpus '"device=2"'
# --ulimit memlock=-1:-1 --load-mode mlock \
docker run --gpus all --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-qwen3.8-flash-next \
    $image \
    -hf $model \
    -md /root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    --host 0.0.0.0 --port $port \
    -c 262144 -np 1 -cb -fa on --threads 4 --n-gpu-layers 999 \
    -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    --image-min-tokens 1024 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repeat_penalty 1.0 \
    --reasoning-effort low --reasoning on --reasoning-preserve \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --lazy-mode off
    # --temperature 0.7 --top_p 0.80 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 \
    # --reasoning off \
