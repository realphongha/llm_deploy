image=vllm/vllm-openai:cu130-nightly-aarch64
# --gpu-memory-utilization 0.8 \  # for 27b
# --gpu-memory-utilization 0.3 \  # for 9b
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --network=host \
    $image \
    Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 --gpu-memory-utilization 0.75 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
