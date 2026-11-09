# Pick the llama.cpp image for Jetson AGX Orin.
#
# - If the legacy local image `llama-cpp-jetson` exists, use it (its entrypoint
#   is already llama-server, so no command is needed).
# - Otherwise fall back to llama-cpp-jetpack:r36.4.0, which has no such
#   entrypoint, so `llama-server` must be passed explicitly.
#
# Usage: source this file, then use `$image` and "${llama_cmd[@]}":
#   docker run ... -it $image "${llama_cmd[@]}" -hf $model ...
if docker image inspect llama-cpp-jetson >/dev/null 2>&1; then
    image=llama-cpp-jetson
    llama_cmd=()
else
    image=llama-cpp-jetpack:r36.4.0
    llama_cmd=(llama-server)
fi
