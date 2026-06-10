image=vllm/vllm-openai:nightly
model=nvidia/Qwen3.6-35B-A3B-NVFP4
port=8008
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --network=host \
    $image $model --port $port \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"video": 1}' \
    --reasoning-parser qwen3 \
    --quantization modelopt \
    --gpu-memory-utilization 0.85
