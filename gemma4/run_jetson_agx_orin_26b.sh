# from ./llm_deploy
image=my-l4t-jetpack:ffmpeg
model=unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M
docker run --rm \
    --env PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    --env LD_LIBRARY_PATH="/app/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
    -v ./llama.cpp:/app \
    -v ./gemma4:/gemma4 \
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
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-max-tokens 280
