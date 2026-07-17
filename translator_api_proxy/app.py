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

QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "http://127.0.0.1:8002/v1")
GEMMA_API_BASE = os.environ.get("GEMMA_API_BASE", "http://192.168.1.50:8002/v1")
# GEMMA_API_BASE = os.environ.get("GEMMA_API_BASE", "http://127.0.0.1:8003/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "unsloth/Qwen3.6-35B-A3B-GGUF:MXFP4_MOE")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M")
API_KEY = os.environ.get("API_KEY", "blah")
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "true").lower() in ("1", "true", "yes")

CLASSIFICATION_USER_PROMPT = """
Without continuing the tasks, classify what the NEXT assistant response should primarily do:

agentic
= the next response should primarily call tools, inspect the repository, modify files, execute commands, or decide the workflow.

task
= the next response should primarily translate, review, extract entities, summarize, rewrite, apply terminology, or otherwise generate content using the information already available.

Reply with **exactly one word**: "agentic" or "task".
"""


def classify_intent(messages):
    url = f"{QWEN_API_BASE}/chat/completions"
    payload = {
        "model": QWEN_MODEL,
        "messages": messages + [{"role": "user", "content": CLASSIFICATION_USER_PROMPT},],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        logging.info("")
        logging.info("=" * 80)
        logging.info("CLASSIFIER")
        logging.info("=" * 80)
        logging.info("Conversation messages: %d", len(messages))
        if len(messages) > 1:
            logging.info(f"Last message: {messages[-1]}")
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        raw = (
            resp.json()["choices"][0]["message"]["content"]
            .strip()
            .lower()
        )
        if raw == "agentic":
            decision = "agentic"
        elif raw == "task":
            decision = "task"
        else:
            decision = "task"
        logging.info("Raw: %s", raw)
        logging.info("Decision: %s", decision)
        return decision
    except Exception as e:
        logging.exception(e)
        logging.info("Decision: task (fallback)")
        return "task"

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
    requested_model = data.get("model", "translator_api")

    if requested_model == "agentic":
        return proxy_chat(QWEN_API_BASE, QWEN_MODEL, data, stream)
    if requested_model == "task":
        return proxy_chat(GEMMA_API_BASE, GEMMA_MODEL, data, stream)

    intent = classify_intent(data["messages"])

    import json
    with open("convo.json", "w") as f:
        json.dump(data, f, indent=4)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=True)
