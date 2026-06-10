image=llama-cpp
port=8007
model=unsloth/gemma-4-26B-A4B-it-GGUF:MXFP4_MOE
    # --ulimit memlock=-1:-1 \
docker run --gpus '"device=0"' --rm -it \
    -v /mnt/ssd8tb/shared_workspace/phonghh/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4-vision \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -v \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -c 65536 -np 1 -b 4096 -ub 4096  -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --image-max-tokens 560 \
    # -c 131072 -cb -np 2 -b 4096 -ub 4096 -fa on --mlock --threads 16 --n-gpu-layers 999 \
    # -c 8192 -np 2 -cb -b 4096 -ub 4096  -fa on --mlock --threads 8 --n-gpu-layers 999 \
    # --image-max-tokens 1120
