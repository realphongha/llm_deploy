FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG LLAMA_CPP_TAG=b8703
ARG SM=89

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda-13/compat

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential curl ca-certificates pkg-config git \
    libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LLAMA_LOG_COLORS=1
ENV LLAMA_LOG_PREFIX=1
ENV LLAMA_LOG_TIMESTAMPS=1
RUN git clone --depth=1 --branch="${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp && \
    cd llama.cpp && \
    cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES=${SM} -DLLAMA_OPENSSL=ON && \
    cmake --build build-cuda -j
ENV PATH="/app/llama.cpp/build-cuda/bin:$PATH"

ENTRYPOINT ["/app/llama.cpp/build-cuda/bin/llama-server"]
