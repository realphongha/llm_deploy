#!/usr/bin/env bash
image=vllm/vllm-openai
model=Qwen/Qwen3.6-27B-FP8
port=8002
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --gpus all --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "VLLM_USE_DEEP_GEMM=0" \
    --env CUTE_DSL_ARCH=sm_121a \
    --network=host \
    $image \
    $model \
    --port $port --gpu-memory-utilization 0.8 \
    --max-model-len 242144 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --moe-backend flashinfer_b12x \
    # --language-model-only
    # --limit-mm-per-prompt '{"video": 1}' \
