# API Proxy

Flask proxy that routes OpenAI-compatible `/v1/chat/completions` requests to multiple backends. Resolves model aliases, swaps API keys, and injects per-alias options from a YAML config.

## Quick Start

```bash
pip install flask pyyaml requests
python api_proxy.py
```

Listens on `http://0.0.0.0:8008`.

## Usage

Point your OpenAI client at the proxy:

```python
import os

os.environ["OPENAI_API_KEY"] = "anything"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8008/v1"

response = client.chat.completions.create(
    model="smart",          # alias key from YAML
    messages=[{"role": "user", "content": "hello"}],
)
```

- `model="smart"` → resolves to backend defined in `model_map.yaml`
- Unknown model names pass through to the default backend as-is
- Streaming (`stream=True`) supported

## Config — `model_map.yaml`

Set `MODEL_MAP_CONFIG=/path/to/file` env var to use a different file. Default: `model_map.yaml`.

### Top-level keys

| Key | Required | Description |
|---|---|---|
| `default_api_base` | ✅ | Backend URL for pass-through requests |
| `default_model` | ✅ | Model name when alias omits it |
| `api_key` | ❌ | Default API key. `$VAR` reads env var |
| `options` | ❌ | Default params merged into every request body (shallow) |
| `extra_kwargs` | ❌ | Default params merged into `extra_body` (deep merge) |

### Per-alias keys (under `models.<name>`)

| Key | Description |
|---|---|
| `model` | Real model name sent to backend |
| `api_base` | Backend URL (falls back to `default_api_base`) |
| `api_key` | Per-backend API key, falls back to top-level `api_key` |
| `options` | Shallow-merged over top-level `options` |
| `extra_kwargs` | Deep-merged over top-level `extra_kwargs` |

### Full example

```yaml
default_api_base: "http://127.0.0.1:8002/v1"
default_model: "qwen3.6-27b"
api_key: "$OPENROUTER_API_KEY"

options:
  temperature: 1.0
  top_p: 0.95

extra_kwargs:
  top_k: 20
  min_p: 0.0
  repetition_penalty: 1.0

models:
  smart:
    model: "qwen3.6-27b"
    api_base: "http://192.168.1.223:8002/v1"
    api_key: "abc"
    extra_kwargs:
      chat_template_kwargs:
        enable_thinking: true

  fast:
    model: "qwen3.6-35b"
    api_base: "http://127.0.0.1:8003/v1"
    api_key: "abc"
    extra_kwargs:
      chat_template_kwargs:
        enable_thinking: false

  oracle:
    model: "openai/gpt-5.6-sol"
    api_base: "https://openrouter.ai/api/v1"
    api_key: "$OPENROUTER_API_KEY"
```

### Merge behavior

- **`options`** → shallow merge into request body. Alias values override top-level keys.
- **`extra_kwargs`** → deep merge into `extra_body`. Nested dicts preserved — alias can add `chat_template_kwargs` without losing top-level `top_k`, `min_p`, etc.

## Routes

| Route | Description |
|---|---|
| `POST /v1/chat/completions` | Alias-resolved chat with options injection |
| `GET  /v1/models` | Proxied to default backend |
| `*    /v1/*` | Catch-all proxy to default backend |

## Logging

All requests logged to `classifier.log` with model, backend, stream mode, and injected options/extra_kwargs.

## Env vars

| Variable | Default | Description |
|---|---|---|
| `MODEL_MAP_CONFIG` | `model_map.yaml` | Path to config file |
