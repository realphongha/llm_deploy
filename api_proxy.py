from textwrap import indent
import os
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

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8002/v1")
MODEL = os.environ.get("MODEL", "unsloth/DeepSeek-V4-Flash-GGUF:UD-IQ3_XXS")
API_KEY = os.environ.get("API_KEY", "blah")
# API_BASE = "http://127.0.0.1:8003/v1"
# MODEL = "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M"
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "true").lower() in ("1", "true", "yes")

def proxy_chat(target_api_base, target_model, body, stream):
    logging.info("")
    logging.info("=" * 80)
    logging.info("ROUTE")
    logging.info("=" * 80)
    logging.info("Model   : %s", target_model)
    logging.info("Backend : %s", target_api_base)
    logging.info("Thinking: %s", not DISABLE_THINKING)
    logging.info("Stream  : %s", stream)

    url = f"{target_api_base}/chat/completions"
    headers = {
        "Authorization": request.headers.get("Authorization", f"Bearer {API_KEY}"),
        "Content-Type": "application/json",
    }
    body = dict(body)
    body["model"] = target_model
    if DISABLE_THINKING:
        extra = body.setdefault("extra_body", {})
        kwargs = extra.setdefault("chat_template_kwargs", {})
        kwargs["enable_thinking"] = False

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
    return proxy_chat(API_BASE, MODEL, data, stream)

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "system"},
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=True)
