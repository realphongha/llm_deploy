FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
