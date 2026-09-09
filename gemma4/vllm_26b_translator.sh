image=vllm/vllm-openai
port=8003
model=nvidia/Gemma-4-26B-A4B-NVFP4
docker run --gpus all --rm -it \
    --user "$(id -u):$(id -g)" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    $image $model \
    --port $port --gpu-memory-utilization 0.8 \
    --load-format fastsafetensors \
    --enable-chunked-prefill --async-scheduling --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --language-model-only \
    --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}' \
