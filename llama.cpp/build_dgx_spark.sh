docker build -f dgx_spark.Dockerfile \
    --build-arg LLAMA_CPP_TAG=b9568 \
    -t llama-cpp .
