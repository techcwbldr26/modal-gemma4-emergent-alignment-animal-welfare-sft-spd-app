"""T0.5 gate — Ollama Cloud chat from inside a Modal function.

Validates the `ollama-cloud` secret + https://ollama.com/api/chat with both
teacher/judge models (open-source-only policy). CPU only — no GPU needed.

Run:
    modal run ollama_check.py
"""
import json
import urllib.request

import modal

from common import app, train_image

OLLAMA_HOST = "https://ollama.com/api/chat"
MODELS = ["glm-5.3-flash", "deepseek-v4-flash:0731"]


@app.function(
    image=train_image.add_local_python_source("common"),
    secrets=[modal.Secret.from_name("ollama-cloud")],
    timeout=5 * 60,
)
def ollama_chat(model: str, content: str) -> str:
    import os

    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}], "stream": False}
    ).encode()
    req = urllib.request.Request(
        OLLAMA_HOST,
        data=body,
        headers={
            "Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return (r.get("message", {}).get("content") or "").strip()


@app.local_entrypoint()
def main():
    for model in MODELS:
        out = ollama_chat.remote(model, "In one sentence: why does animal welfare matter?")
        print(f"OLLAMA-OK | {model} | {out[:150]}")

