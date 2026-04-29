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
    -b 4096 -ub 4096 -c 16384 -np 1 --threads 4
    # -b 4096 -ub 4096 -cb -c 65536 -np 16 --threads 16
