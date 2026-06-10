docker build -f dgx_spark.Dockerfile \
    --build-arg LLAMA_CPP_TAG=master \
    -t llama-cpp .
