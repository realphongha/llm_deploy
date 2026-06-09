image=llama-cpp
port=8007
model=unsloth/gemma-4-26B-A4B-it-GGUF:MXFP4_MOE
# model=unsloth/gemma-4-E4B-it-GGUF:Q4_K_M
    # --ulimit memlock=-1:-1 \
docker run --gpus '"device=2"' --rm -it \
    -v /mnt/ssd8tb/shared_workspace/phonghh/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4 \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -c 65536 -cb -np 8 -b 4096 -ub 4096 -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    # -c 65536 -np 1 -b 4096 -ub 4096 -fa on --mlock --threads 8 --n-gpu-layers 999 \

