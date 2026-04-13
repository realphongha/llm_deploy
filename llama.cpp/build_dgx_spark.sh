docker build -f dgx_spark.Dockerfile \
    --build-arg LLAMA_CPP_TAG=b8720 \
    -t llama-cpp .
