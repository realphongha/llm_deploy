# Build llama.cpp docker image
- From `./llm_deploy/llama.cpp`
- Access the container to build llama.cpp first:
```bash
docker run --rm \
    -v .:/app --runtime=nvidia \
    -it nvcr.io/nvidia/l4t-jetpack:r36.4.0 bash
```

- Inside the container:
```bash
apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential curl ca-certificates pkg-config git \
    libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
git clone --depth=1 --branch="b8416" https://github.com/ggml-org/llama.cpp && \
    cd llama.cpp && \
    cmake -B build-cuda -DGGML_CUDA=ON -DGGML_CUDA_F16=on -DLLAMA_CURL=on -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES="87" -DLLAMA_OPENSSL=ON && \
    cmake --build build-cuda -j
```
The build directory should be at `./llm_deploy/llama.cpp` now

# Run llama.cpp
See `run_jetson_agx_orin.sh`
