FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Upstream llama.cpp repo. Override with --build-arg LLAMA_CPP_FORK=<url>
# to build a PR that lives in a fork.
ARG LLAMA_CPP_FORK=https://github.com/ggml-org/llama.cpp
# LLAMA_CPP_TAG may be a branch, tag, commit sha, or PR ref (pull/<id>/head|merge).
ARG LLAMA_CPP_TAG=master
# AGX Orin = SM 87
ARG SM=87
ARG JOBS=4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential curl ca-certificates pkg-config git \
    libcurl4-openssl-dev libssl-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LLAMA_LOG_COLORS=1
ENV LLAMA_LOG_PREFIX=1
ENV LLAMA_LOG_TIMESTAMPS=1
RUN set -eux; \
    git clone --depth=1 "${LLAMA_CPP_FORK}" llama.cpp; \
    cd llama.cpp; \
    git remote add upstream https://github.com/ggml-org/llama.cpp; \
    if echo "${LLAMA_CPP_TAG}" | grep -Eq '^pull/[0-9]+/(head|merge)$'; then \
        git fetch --depth=1 upstream "${LLAMA_CPP_TAG}"; \
        git checkout FETCH_HEAD; \
    elif echo "${LLAMA_CPP_TAG}" | grep -Eq '^[0-9a-f]{40}$'; then \
        git fetch --unshallow; \
        git checkout "${LLAMA_CPP_TAG}"; \
    else \
        git fetch --depth=1 origin "${LLAMA_CPP_TAG}"; \
        git checkout FETCH_HEAD; \
    fi; \
    # libcuda.so is only injected by the nvidia runtime at `docker run`, not during
    # `docker build`, so allow it to stay unresolved at link time.
    cmake -B build-cuda -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DCMAKE_CUDA_ARCHITECTURES=${SM} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build-cuda -j${JOBS}
ENV PATH="/app/llama.cpp/build-cuda/bin:$PATH"

ENTRYPOINT ["/app/llama.cpp/build-cuda/bin/llama-server"]
