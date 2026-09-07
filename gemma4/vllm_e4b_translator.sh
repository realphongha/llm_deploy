image=vllm/vllm-openai
port=8003
model=google/gemma-4-E4B-it-qat-w4a16-ct
docker run --gpus all --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    $image $model \
    --port $port \
    --load-format fastsafetensors \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill --async-scheduling --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.3 \
    --language-model-only \
    --speculative-config '{"model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":4}' \
    # --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.8 \
