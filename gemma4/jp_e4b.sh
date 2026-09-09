# from ./llm_deploy
image=my-l4t-jetpack:ffmpeg
model=unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL
port=8008
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env HOME=$HOME \
    --env PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    --env LD_LIBRARY_PATH="/app/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ulimit memlock=-1:-1 \
    -v ./llama.cpp:/app \
    -v ./gemma4:/gemma4 \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --runtime=nvidia \
    -p $port:$port \
    -it $image \
    llama-server \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -fa on --mlock --threads 8 --n-gpu-layers 999 \
    -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 \
    -np 1 -c 65536 \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --image-max-tokens 280
