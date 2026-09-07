"""T3.4 — serve a merged arm with vLLM (OpenAI-compatible) for Inspect evals.

Loads the merged model from the gemma4-runs volume (no HF push needed).

Run:
    modal serve serve_arm.py
Then:  curl <printed-url>/v1/models   (model name = the volume path)
"""
import subprocess

import modal

from common import MINUTES, app, runs_vol

# Edit per arm — web_server functions can't take parameters (Modal rule).
MODEL_PATH = "/vol/runs/runs/smoke-s0/merged"

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub>=0.32.0")
)


@app.function(
    image=serve_image,
    gpu="A100-80GB",
    volumes={"/vol/runs": runs_vol},
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=15 * MINUTES)
def serve():
    subprocess.Popen(
        f"vllm serve {MODEL_PATH} --port 8000 --max-model-len 8192 "
        "--gpu-memory-utilization 0.92",
        shell=True,
    )

