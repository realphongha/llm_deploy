FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# Upstream llama.cpp repo. Override with --build-arg LLAMA_CPP_FORK=<url>
# to build a PR that lives in a fork.
ARG LLAMA_CPP_FORK=https://github.com/ggml-org/llama.cpp
# What to build. LLAMA_CPP_TAG may be any of:
#   - branch : master
#   - tag    : b8720
#   - commit : a full 40-char git sha
#   - PR     : pull/<id>/head  (PR tip)  or  pull/<id>/merge (PR merged into base)
# For a PR from a fork, set LLAMA_CPP_FORK to the fork URL and
# LLAMA_CPP_TAG to the fork's branch name (or a commit on that fork).
ARG LLAMA_CPP_TAG=b8720
ARG SM=89

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda-13/compat

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
    cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES=${SM} -DLLAMA_OPENSSL=ON && \
    cmake --build build-cuda -j
ENV PATH="/app/llama.cpp/build-cuda/bin:$PATH"
ENV LD_LIBRARY_PATH=

ENTRYPOINT ["/app/llama.cpp/build-cuda/bin/llama-server"]
