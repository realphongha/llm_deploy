image=ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813
model=unsloth/Qwen3.8-27B-NVFP4
port=8002
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env FLASHINFER_CUDA_ARCH_LIST=12.1a \
    --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --network=host \
    $image \
    vllm serve \
    $model --port $port \
    --gpu-memory-utilization 0.55 \
    --enable-auto-tool-choice \
    --enable-flashinfer-autotune \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --language-model-only \
    --max-model-len 242144 \
    --max-num-seqs 4 \
    --default-chat-template-kwargs '{"preserve_thinking":true, "reasoning_effort":"medium", "enable_thinking":true}' \
    --kv-cache-dtype fp8 \
