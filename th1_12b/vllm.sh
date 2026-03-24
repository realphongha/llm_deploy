image=vllm/vllm-openai:cu130-nightly-aarch64
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8002:8002 \
    --network=host \
    $image \
    realphongha/th1-12b \
    --port 8002 --gpu-memory-utilization 0.6 \
    --dtype bfloat16 --max-model-len 4096 \
    --language-model-only \
    --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0 \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 3072

