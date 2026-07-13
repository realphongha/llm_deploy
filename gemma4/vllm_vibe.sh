image=vllm/vllm-openai
port=8002
model=nvidia/Gemma-4-31B-IT-NVFP4
# model=nvidia/Gemma-4-26B-A4B-NVFP4
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    $image $model \
    --port $port --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --quantization modelopt \
    --limit-mm-per-prompt '{"video": 0, "image": 0}' \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --trust-remote-code

