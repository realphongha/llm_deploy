image=vllm/vllm-openai:cu130-nightly-aarch64
docker run --gpus all --rm -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/phonghh/repos/reinforcement_learning/style_translation/outputs/google/gemma-3-12b-it-qat-int4-torchao-torchao:/th-12b-qat-torchao-int4 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --network=host \
    $image \
    /th-12b-qat-torchao-int4 \
    --port 8000 --gpu-memory-utilization 0.6 \
    --dtype bfloat16 --max-model-len 4096 \
    --quantization torchao \
    --language-model-only \
    --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0 \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 3072

