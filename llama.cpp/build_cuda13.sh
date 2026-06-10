docker build -f cuda13.Dockerfile \
    --build-arg SM=89 \
    --build-arg LLAMA_CPP_TAG=master \
    -t llama-cpp .
