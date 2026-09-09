# from ./llm_deploy
image=my-l4t-jetpack:ffmpeg
model=unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL
port=8008
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env HOME=$HOME \
    --env PATH="/app/llama.cpp/build-cuda/bin:$PATH" \
    --env LD_LIBRARY_PATH="/app/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH" \
    --env "HF_TOKEN=$HF_TOKEN" \
    -v ./llama.cpp:/app \
    -v ./gemma4:/gemma4 \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    --ulimit memlock=-1:-1 \
    --runtime=nvidia \
    -p $port:$port \
    -it $image \
    llama-server \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -fa on --mlock --threads 8 --n-gpu-layers 999 \
    -b 2048 -ub 2048 --cache-type-k bf16 --cache-type-v bf16 \
    -np 8 -c 65536 -cb \
    --temperature 1.0 --top_p 0.95 --top_k 64 \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --spec-type draft-mtp --spec-draft-n-max 2
