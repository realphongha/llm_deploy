image=vllm/vllm-openai
model=Qwen/Qwen3.6-27B-FP8
port=8008
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --network=host \
    $image $model \
    --port $port --gpu-memory-utilization 0.8 \
    --max-model-len 65536 \
    --limit-mm-per-prompt '{"video": 1}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --reasoning-parser qwen3 \
    # --mm-encoder-tp-mode data --mm-processor-cache-type shm
    # --mm-processor-kwargs '{"fps": 3.0, "do_sample_frames": false}' \
    # --media-io-kwargs '{ "video": {"fps": 3} }'

