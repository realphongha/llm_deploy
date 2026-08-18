image=llama-cpp
port=8007
model=unsloth/Qwen3.5-9B-GGUF:Q4_K_M
docker run --gpus '"device=2"' --rm -it \
    -v /mnt/ssd8tb/shared_workspace/phonghh/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-qwen3.5-9b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
   -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 8192 -ub 2048 --cache-type-k bf16 --cache-type-v bf16 \
    --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    -c 65536 -np 1 --image-max-tokens 256
    # -c 65536 -np 1 --image-max-tokens 1024 --image-min-tokens 1024
    # -c 131072 -np 1 --image-max-tokens 1024
