#!/usr/bin/env bash
image=vllm/vllm-openai
model=Qwen/Qwen3.5-35B-A3B-FP8
port=8008
source "$(dirname "$0")/../lib/docker_user.sh"
docker run --gpus all --rm -it \
    "${docker_user_flag[@]}" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --network=host \
    $image $model \
    --port $port --gpu-memory-utilization 0.8 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"video": 1}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \

    # --mm-encoder-tp-mode data \
    # --mm-processor-cache-type shm \

    # --mm-processor-kwargs '{"fps": 3.0, "do_sample_frames": false}' \
    # --media-io-kwargs '{ "video": {"fps": 3} }'

