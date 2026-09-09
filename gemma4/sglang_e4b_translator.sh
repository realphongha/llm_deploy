port=8003
model=google/gemma-4-E4B-it-qat-w4a16-ct
sglang serve \
    --model-path $model \
    --host 0.0.0.0 --port $port \
    --quantization compressed-tensors \
    --json-model-override-args '{"audio_config": null}' \
    --tp-size 1 \
    --chunked-prefill-size 8192 \
    --context-length 8192 --max-running-requests 8 --mem-fraction-static 0.3 \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path google/gemma-4-E4B-it-assistant \
    --speculative-draft-model-quantization unquant \
    --speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1 \
    # --context-length 32768 --max-running-requests 16 --mem-fraction-static 0.8 \
