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

QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "http://127.0.0.1:8002/v1")
GEMMA_API_BASE = os.environ.get("GEMMA_API_BASE", "http://192.168.1.50:8002/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "unsloth/Qwen3.6-35B-A3B-GGUF:MXFP4_MOE")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M")
API_KEY = os.environ.get("API_KEY", "blah")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "http://127.0.0.1:11434/api/embed")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma")
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "true").lower() in ("1", "true", "yes")

CLASSIFICATION_SYSTEM_PROMPT = """You are a classifier. Determine whether the user's request is 'agentic' or 'task'.

agentic = planning, orchestration, decisions, analysis, workflow management, assessing progress, choosing strategies, determining approaches
task = translation, entity extraction, review, summary, glossary checking, applying terms, improving phrasing, checking consistency

Reply with exactly one word: agentic or task."""


def classify_intent(last_user_message):
    url = f"{QWEN_API_BASE}/chat/completions"
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": last_user_message},
        ],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    truncated = last_user_message[:200] + "..." if len(last_user_message) > 200 else last_user_message
    logging.info('MSG: "%s"', truncated)
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        raw = result["choices"][0]["message"]["content"]
        logging.info('RESP: "%s"', raw.strip())
        content = raw.strip().lower()
        decision = "agentic" if "agentic" in content else "task"
        logging.info("DECISION: %s", decision)
        return decision
    except Exception as e:
        logging.error("ERROR: %s", e)
        logging.info("DECISION: task (fallback)")
        return "task"


def get_last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def proxy_chat(target_api_base, target_model, body, stream):
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
    requested_model = data.get("model", "translator_api")

    if requested_model == "agentic":
        return proxy_chat(QWEN_API_BASE, QWEN_MODEL, data, stream)
    if requested_model == "task":
        return proxy_chat(GEMMA_API_BASE, GEMMA_MODEL, data, stream)

    last_msg = get_last_user_message(data["messages"])
    intent = classify_intent(last_msg)

    if intent == "agentic":
        return proxy_chat(QWEN_API_BASE, QWEN_MODEL, data, stream)
    return proxy_chat(GEMMA_API_BASE, GEMMA_MODEL, data, stream)


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "translator_api", "object": "model", "created": int(time.time()), "owned_by": "system"},
            {"id": "agentic", "object": "model", "created": int(time.time()), "owned_by": "system"},
            {"id": "task", "object": "model", "created": int(time.time()), "owned_by": "system"},
        ],
    })


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "input is required"}), 400

    payload = {"model": EMBEDDING_MODEL, "input": data["input"]}
    try:
        resp = requests.post(EMBEDDING_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        ollama_resp = resp.json()
        return jsonify({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": ollama_resp["embeddings"][0],
                }
            ],
            "model": EMBEDDING_MODEL,
            "usage": {
                "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                "total_tokens": ollama_resp.get("prompt_eval_count", 0),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=True)
