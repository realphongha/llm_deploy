image=llama-cpp
port=8002
model=realphongha/th1-12b-GGUF:Q4_0
docker run --gpus all --rm -it \
    --ulimit memlock=-1:-1 \
    -v ~/.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $port:$port \
    --network=host \
    --name llama-cpp-th1-12b \
    $image \
    -hf $model --no-mmproj \
    --host 0.0.0.0 --port $port \
    -fa on --mlock --n-gpu-layers 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    # -b 2048 -ub 1024 -c 8192 -np 2  --threads 8
    -b 2048 -ub 1024 -cb -c 262144 -np 64 --threads 32
