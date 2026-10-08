# VLLM_VERSION: version from https://pypi.jetson-ai-lab.io/jp6/cu126/vllm/ (empty = latest)
docker build -f vllm_jetson_agx_orin.Dockerfile \
    --build-arg VLLM_VERSION=0.20.0 \
    -t vllm-jetson .
