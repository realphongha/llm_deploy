# uv tool install open-webui --prerelease=allow
# REQUESTS_VERIFY=False AIOHTTP_CLIENT_SESSION_SSL=False WEBUI_AUTH=False ENABLE_OLLAMA_API=False
port=3003
url=http://0.0.0.0:8008/v1
api="nah"
OPENAI_API_BASE_URL=$url WEBUI_AUTH=False open-webui serve --host 0.0.0.0 --port 3003
