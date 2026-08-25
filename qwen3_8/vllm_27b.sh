image=vllm/vllm-openai
model=unsloth/Qwen3.8-27B-NVFP4
port=8002
docker run --gpus all --rm -it \
    -v $(readlink -f ~/.cache/huggingface):/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env CUTE_DSL_ARCH=sm_121a \
    --network=host \
    $image \
    $model --port $port \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --enable-auto-tool-choice \
    --enable-flashinfer-autotune \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --language-model-only \
    --moe-backend flashinfer_b12x \
    --max-model-len 242144 \
    --max-num-seqs 4 \
    --default-chat-template-kwargs '{"preserve_thinking":true, "reasoning_effort":"medium", "enable_thinking":true}' \
    --kv-cache-dtype bfloat16 \
    # --mm-encoder-tp-mode data
