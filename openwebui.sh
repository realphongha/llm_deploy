docker run -d \
    --name open-webui \
    -p 3000:8080 \
    --ipc=host \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://0.0.0.0:8008/v1 \
    ghcr.io/open-webui/open-webui:main
