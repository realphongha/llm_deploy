image=vllm/vllm-openai:gemma4-cu130
port=8008
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    $image \
    nvidia/Gemma-4-31B-IT-NVFP4 \
    --port $port --gpu-memory-utilization 0.7 \
    --max-model-len 65536 \
    --quantization modelopt \
    --limit-mm-per-prompt '{"video": 1, "image": 32}' \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4

