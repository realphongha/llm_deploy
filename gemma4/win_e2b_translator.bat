@echo off

llama-server ^
    -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --no-mmproj ^
    --host 0.0.0.0 --port 8002 ^
    -fa on --mlock --threads 8 --n-gpu-layers 999 ^
    -b 8192 -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0 ^
    -np 8 -c 65536 -cb ^
    --temperature 1.0 --top_p 0.95 --top_k 64 ^
    --reasoning off ^
    --jinja --chat-template-file chat_template_e2b.jinja ^
    --spec-type draft-mtp --spec-draft-n-max 2

pause
