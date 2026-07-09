# from ./llm_deploy/llama.cpp
image=my-l4t-jetpack:ffmpeg
docker run --rm \
    --env PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    --env LD_LIBRARY_PATH="/app/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
    -v ../llama.cpp:/app \
    -v .:/gemma4 \
    --runtime=nvidia \
    -v /mnt/ssd1t/phonghh/.cache:/root/.cache \
    --network=host \
    -it nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    llama-server \
    -hf $image \
    --host 0.0.0.0 --port 8002 \
    -fa on --mlock --threads 8 --n-gpu-layers 999 \
    -b 4096 -ub 4096 --cache-type-k q4_0 --cache-type-v q4_0 \
    -np 1 -c 65536 \
    --image-max-tokens 560
