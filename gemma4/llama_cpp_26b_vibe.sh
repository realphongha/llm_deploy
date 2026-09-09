image=llama-cpp
port=8003
# model=unsloth/gemma-4-26B-A4B-it-GGUF:MXFP4_MOE
model=unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL
docker run --gpus all --rm -it \
    --user "$(id -u):$(id -g)" \
    --env HOME=$HOME \
    -v $(readlink -f ~/.docker-home):$HOME \
    -v $(readlink -f ~/.cache/huggingface):$HOME/.cache/huggingface \
    -v ./gemma4:/gemma4 \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --name llama-cpp-gemma4-26b \
    $image \
    -hf $model \
    --host 0.0.0.0 --port $port \
    -c 131072 -np 8 -cb -b 2048 -ub 2048 -fa on --mlock --threads 8 --n-gpu-layers 999 \
    --cache-type-k bf16 --cache-type-v bf16 \
    --temperature 0.3 --top_p 0.95 --top_k 64 --chat-template-kwargs '{"enable_thinking": false}' \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    --jinja --chat-template-file /gemma4/chat_template_26b_31b.jinja \
    # --ulimit memlock=-1:-1 \
    # --temperature 1.0 --top_p 0.95 --top_k 64 --chat-template-kwargs '{"enable_thinking": true}' \
