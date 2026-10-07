# Build llama.cpp docker image (Jetson AGX Orin)
From `./llm_deploy/llama.cpp`:
```bash
bash build_jetson_agx_orin.sh
```
This produces the `llama-cpp-jetson` image (llama.cpp built with CUDA, SM 87, inside the image;
entrypoint is `llama-server`). Edit `LLAMA_CPP_TAG` in the script to pick a branch/tag/commit/PR.

# Run llama.cpp
See `./gemma4/jp_e4b.sh` (or any `jp_*.sh`)
