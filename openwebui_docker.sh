docker run -d \
    --name open-webui \
    -v open-webui:/app/backend/data \
    -p 3003:8080 \
    -e OPENAI_API_BASE_URL=http://0.0.0.0:8008/v1 \
    -e WEBUI_AUTH=False \
    ghcr.io/open-webui/open-webui:main
    # --ipc=host \

