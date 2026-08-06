@echo off

llama-server ^
    -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL --no-mmproj ^
    --host 0.0.0.0 --port 8002 ^
    -fa on --mlock --threads 8 --n-gpu-layers 999 ^
    -b 2048 -ub 512 --cache-type-k bf16 --cache-type-v bf16 ^
    -np 1 -c 65536 ^
    --temperature 1.0 --top_p 0.95 --top_k 64 ^
    --jinja --chat-template-file chat_template_12b.jinja ^
    --reasoning on ^
    --spec-type draft-mtp --spec-draft-n-max 2

pause
