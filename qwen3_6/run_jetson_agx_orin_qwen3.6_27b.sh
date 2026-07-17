# from ./llm_deploy
model=unsloth/Qwen3.6-27B-GGUF:Q4_K_M
image=my-l4t-jetpack:ffmpeg
docker run --rm \
    --env PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    --env LD_LIBRARY_PATH="/app/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
    -v ./llama.cpp:/app \
    --runtime=nvidia \
    -v /mnt/ssd1t/phonghh/.cache:/root/.cache \
    --network=host \
    -it $image \
    llama-server \
    -hf $model \
    --host 0.0.0.0 --port 8002 \
    -fa on --mlock --threads 8 --n-gpu-layers 999 \
    -b 8192 -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0 \
    -np 1 -c 65536 \
    --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --presence_penalty 1.5 --repeat_penalty 1.0 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-min-tokens 256
