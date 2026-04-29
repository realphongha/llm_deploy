docker build -f cuda13.Dockerfile \
    --build-arg SM=89 \
    --build-arg LLAMA_CPP_TAG=b8771 \
    -t llama-cpp .
