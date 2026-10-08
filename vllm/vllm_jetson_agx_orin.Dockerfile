FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# vLLM version to install from the Jetson AI Lab wheel index (JetPack 6 / CUDA 12.6 / py3.10).
# Empty = latest available on the index. Browse: https://pypi.jetson-ai-lab.io/jp6/cu126/vllm/
ARG VLLM_VERSION=
# transformers: vllm only sets a lower bound, so pip grabs the newest release, which can be
# newer than what that vllm was tested with (5.19.0 breaks gemma4 on vllm 0.20.0). Empty = newest.
ARG TRANSFORMERS_VERSION=5.8.1
ARG PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential curl ca-certificates git ffmpeg \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Heavy layer (vllm + torch from the Jetson index). Keep it separate so tweaking extras below is cheap.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm${VLLM_VERSION:+==${VLLM_VERSION}}" \
        "transformers${TRANSFORMERS_VERSION:+==${TRANSFORMERS_VERSION}}"

# Extras the Jetson vllm wheel does not pull in:
#   torchvision (HF image/video processors), triton (vllm kernels), xgrammar (structured output),
#   compressed-tensors (w4a16/fp8 checkpoints), nvidia-cudss-cu12 (libcudss, needed by torch),
#   opencv-python-headless (JetPack's cv2 is numpy1-only)
RUN python3 -m pip install torchvision triton xgrammar compressed-tensors \
        nvidia-cudss-cu12 opencv-python-headless

# vllm's deps drag in CUDA 12.9 cublas/nvrtc wheels. torch prefers them over the system CUDA 12.6
# that matches the Orin driver, and cublasCreate then fails (CUBLAS_STATUS_ALLOC_FAILED).
# Remove them so torch falls back to /usr/local/cuda-12.6.
RUN python3 -m pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12

# torch (Jetson build) needs libcudss, which ships in the nvidia-cudss-cu12 wheel
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH}
# triton JIT-compiles a small C helper with gcc; its bundled include dir lacks cuda.h
ENV CPATH=/usr/local/cuda/include
# the Jetson triton wheel ships no ptxas/cuobjdump/nvdisasm; use the system CUDA 12.6 ones
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
# run as arbitrary --user uid (no passwd entry): torch calls getpass.getuser(), which honors $USER
ENV USER=vllm
ENV VLLM_LOGGING_LEVEL=INFO
ENTRYPOINT ["vllm", "serve"]
