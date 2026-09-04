image=vllm/vllm-openai
port=8003
model=nvidia/Gemma-4-26B-A4B-NVFP4
docker run --gpus all --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    $image $model \
    --port $port --gpu-memory-utilization 0.8 \
    --load-format fastsafetensors \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --language-model-only \
    --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}' \
    --max-num-batched-tokens 8192 \
