#!/usr/bin/env bash
# from ./llm_deploy
# build the image first: cd vllm && bash build_vllm_jetson_agx_orin.sh
image=vllm-jetson
port=8003
model=google/gemma-4-E2B-it-qat-w4a16-ct
# Orin has unified memory: vllm's memory profiling over-counts non-torch usage and reports a negative
# KV cache, so the KV size is set explicitly (--kv-cache-memory-bytes skips the profiling).
# --gpu-memory-utilization is then only the startup free-memory check, keep it low.
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --runtime=nvidia --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    $image $model \
    --port $port \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill --async-scheduling --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --max-model-len 8192 --max-num-seqs 8 \
    --gpu-memory-utilization 0.3 --kv-cache-memory-bytes 2G \
    --language-model-only
