# LLAMA_CPP_TAG may be a branch, tag, commit sha, or PR ref (pull/<id>/head|merge).
# For a PR from a fork, also pass: --build-arg LLAMA_CPP_FORK=<fork-url>
docker build -f cuda13.Dockerfile \
    --build-arg SM=89 \
    --build-arg LLAMA_CPP_TAG=master \
    -t llama-cpp .
