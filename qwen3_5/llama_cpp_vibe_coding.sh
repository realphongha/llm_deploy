image=llama-cpp
port=8000
model=unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 262144 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    -b 4096 -ub 4096 --cache-type-k q8_0 --cache-type-v q8_0 \
    --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 0.0 --repeat_penalty 1.0

