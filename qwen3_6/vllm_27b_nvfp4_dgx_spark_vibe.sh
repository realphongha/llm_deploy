image=vllm/vllm-openai
model=unsloth/Qwen3.6-27B-NVFP4
port=8002
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env CUTE_DSL_ARCH=sm_121a \
    --network=host \
    $image $model --port $port \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --language-model-only \
    --moe-backend flashinfer_b12x \
    # --reasoning-parser qwen3 \
