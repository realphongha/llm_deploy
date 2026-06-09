docker build -f cuda13.Dockerfile \
    --build-arg SM=89 \
    --build-arg LLAMA_CPP_TAG=b9568 \
    -t llama-cpp .
