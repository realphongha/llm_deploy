from textwrap import indent
import os
import yaml
import time
import logging
import requests
from flask import Flask, request, Response, stream_with_context, jsonify

logging.basicConfig(
    filename="classifier.log",
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model alias map from YAML config  (required — no env fallback)
# ---------------------------------------------------------------------------
# Schema:
#   default_api_base: "http://127.0.0.1:8002/v1"        # required
#   default_model:    "unsloth/DeepSeek-V4-Flash-GGUF:UD-IQ3_XXS"  # required
#   api_key:          "$OPENROUTER_API_KEY"              # optional, $ prefix = env var
#   options:          { temperature: 1.0, top_p: 0.95 }   # optional, top-level API params
#   extra_kwargs:     { top_k: 20 }                       # optional, merged into extra_body
#   models:
#     smart:
#       model: "qwen3.6-27b"
#       api_base: "http://127.0.0.1:8003/v1"
#       api_key: "$SMART_KEY"
#       options:      { temperature: 0.7 }                # shallow-merged over top-level
#       extra_kwargs: { chat_template_kwargs: { enable_thinking: true } }  # deep-merged
#     fast:
#       model: "qwen3.6-35b"
#       api_base: "http://127.0.0.1:8004/v1"
#
# When incoming model name matches an alias key → real model + optional api_base/api_key.
# Non-alias names pass through as-is.

_MODEL_MAP_CONF = None


def _deep_merge(base, override):
    """Recursive dict merge.  override wins.  Neither dict is mutated."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_val(v):
    """If v starts with '$', read the named env var.  Otherwise return v as-is."""
    if isinstance(v, str) and v.startswith("$"):
        return os.environ.get(v[1:], "")
    return v


def _load_model_map():
    global _MODEL_MAP_CONF
    if _MODEL_MAP_CONF is not None:
        return _MODEL_MAP_CONF

    path = os.environ.get("MODEL_MAP_CONFIG", "model_map.yaml")
    if not os.path.isfile(path):
        raise RuntimeError(f"Model-map config not found: {path}")

    with open(path, "r") as fh:
        _MODEL_MAP_CONF = yaml.safe_load(fh)

    logging.info("Loaded model-map config from %s (%d aliases)",
                 path, len(_MODEL_MAP_CONF.get("models", {})))
    return _MODEL_MAP_CONF


def _default_api_base():
    return _load_model_map().get("default_api_base", "http://127.0.0.1:8002/v1")


def _default_api_key():
    conf = _load_model_map()
    k = conf.get("api_key")
    return _expand_val(k) if k else None


def _resolve_options(incoming_model):
    """Return (options, extra_kwargs) for an alias, deep-merged over top-level defaults."""
    conf = _load_model_map()
    default_opts = conf.get("options", {})
    default_extra = conf.get("extra_kwargs", {})
    models = conf.get("models", {})
    entry = models.get(incoming_model, {})
    opts = _deep_merge(default_opts, entry.get("options", {}))
    extra = _deep_merge(default_extra, entry.get("extra_kwargs", {}))
    return opts, extra


def resolve_model(incoming_model):
    """Resolve an incoming model name to
       (real_model, api_base, api_key_or_None, options, extra_kwargs).
    """
    conf = _load_model_map()
    models = conf.get("models", {})

    if incoming_model in models:
        entry = models[incoming_model]
        real_model = entry.get("model") or conf["default_model"]
        api_base   = entry.get("api_base") or conf["default_api_base"]
        api_key    = entry.get("api_key") or conf.get("api_key")
        if api_key:
            api_key = _expand_val(api_key)
        opts, extra = _resolve_options(incoming_model)
        logging.info("Alias '%s' → model=%s backend=%s options=%s extra=%s", incoming_model, real_model, api_base, opts, extra)
        return real_model, api_base, api_key, opts, extra

    # Pass-through — use top-level defaults
    api_key = conf.get("api_key")
    if api_key:
        api_key = _expand_val(api_key)
    opts, extra = _resolve_options(incoming_model)  # incoming_model not in models, returns top-level
    return incoming_model, conf["default_api_base"], api_key, opts, extra

def proxy_chat(target_api_base, target_model, body, stream,
               api_key=None, options=None, extra_kwargs=None):
    logging.info("")
    logging.info("=" * 80)
    logging.info("ROUTE")
    logging.info("=" * 80)
    logging.info("Model   : %s", target_model)
    logging.info("Backend : %s", target_api_base)
    logging.info("Stream  : %s", stream)

    url = f"{target_api_base}/chat/completions"
    # Use resolved api_key if provided; otherwise forward incoming auth header
    if api_key:
        bearer = f"Bearer {api_key}"
    else:
        bearer = request.headers.get("Authorization", "Bearer blah")
    headers = {
        "Authorization": bearer,
        "Content-Type": "application/json",
    }
    body = dict(body)
    body["model"] = target_model
    # Merge options (top-level API params like temperature)
    if options:
        body.update(options)
    # Merge extra_kwargs into extra_body (deep merge so nested keys survive)
    if extra_kwargs:
        existing = body.get("extra_body", {})
        body["extra_body"] = _deep_merge(existing, extra_kwargs)

    if stream:
        def generate():
            try:
                backend_resp = requests.post(
                    url, json=body, headers=headers, stream=True, timeout=300
                )
                backend_resp.raise_for_status()
                for line in backend_resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{e}\"}}\n\n"

        return Response(stream_with_context(generate()), content_type="text/event-stream")

    try:
        backend_resp = requests.post(url, json=body, headers=headers, timeout=300)
        backend_resp.raise_for_status()
        return jsonify(backend_resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "messages is required"}), 400

    stream = data.get("stream", False)

    # Resolve incoming model name via alias map, then pick the right backend
    incoming = data.get("model", "default")
    real_model, backend, api_key, opts, extra = resolve_model(incoming)
    return proxy_chat(backend, real_model, data, stream, api_key, opts, extra)

@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def proxy_all(path):
    url = f"{_default_api_base()}/{path}"
    api_key = _default_api_key()
    bearer = f"Bearer {api_key}" if api_key else request.headers.get("Authorization", "Bearer blah")
    headers = {"Authorization": bearer}
    if request.method in ("POST", "PUT", "PATCH"):
        headers["Content-Type"] = "application/json"

    resp = requests.request(
        method=request.method,
        url=url,
        headers=headers,
        params=request.args,
        data=request.get_data(),
        stream=True,
        timeout=300,
    )

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    resp_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded_headers]

    if "text/event-stream" in resp.headers.get("content-type", ""):
        def generate():
            try:
                for line in resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{e}\"}}\n\n"
        return Response(stream_with_context(generate()), status=resp.status_code, headers=resp_headers)

    return Response(resp.content, status=resp.status_code, headers=resp_headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=True)
