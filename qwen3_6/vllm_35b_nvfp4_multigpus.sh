image=vllm/vllm-openai:nightly
model=nvidia/Qwen3.6-35B-A3B-NVFP4
port=8089
docker run --gpus '"device=1,3"' --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    -p $port:$port \
    --env "HF_TOKEN=$HF_TOKEN" \
    $image $model --port $port \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"video": 1}' \
    --reasoning-parser qwen3 \
    --quantization modelopt --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 --tensor-parallel-size 2
    # --network=host \
