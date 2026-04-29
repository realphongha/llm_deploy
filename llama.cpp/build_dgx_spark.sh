docker build -f dgx_spark.Dockerfile \
    --build-arg LLAMA_CPP_TAG=b8771 \
    -t llama-cpp .
