image=vllm/vllm-openai
model=nvidia/Qwen3.6-35B-A3B-NVFP4
port=8008
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --network=host \
    $image $model --port $port \
    --limit-mm-per-prompt '{"video": 1}' \
    --tensor-parallel-size 1 --trust-remote-code --dtype auto \
    --quantization modelopt --kv-cache-dtype fp8 --attention-backend flashinfer \
    --moe-backend marlin --gpu-memory-utilization 0.85 --max-model-len 65536 \
    --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill \
    --async-scheduling --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
