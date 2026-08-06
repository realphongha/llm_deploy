@echo off

llama-server ^
    -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL ^
    --host 0.0.0.0 --port 8002 ^
    -fa on --threads 8 --n-gpu-layers 999 ^
    -b 8192 -ub 2048 --cache-type-k bf16 --cache-type-v bf16 ^
    -np 1 -c 65536 -cb ^
    --temperature 1.0 --top_p 0.95 --top_k 64 ^
    --reasoning off ^
    --jinja --chat-template-file chat_template_e4b.jinja ^
    --spec-type draft-mtp --spec-draft-n-max 2 ^
    --image-max-tokens 560

pause
