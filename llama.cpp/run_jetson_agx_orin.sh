docker run --rm \
    -e PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    -v .:/app --runtime=nvidia \
    -v $HOME/.cache:/root/.cache \
    --network=host \
    -it nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    llama-server \
    -hf unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE --no-mmproj \
    --host 0.0.0.0 --port 8002 \
    -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 4096 -ub 4096 --cache-type-k q8_0 --cache-type-v q8_0 \
    -cb -c 262144
