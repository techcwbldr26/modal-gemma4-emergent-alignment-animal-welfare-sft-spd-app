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

# CUDA devel base: vLLM JIT-compiles kernels at startup and needs nvcc
# (debian_slim has no CUDA toolkit -> "Could not find nvcc").
serve_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .pip_install("vllm", "huggingface_hub>=0.32.0")
)


@app.function(
    image=serve_image.add_local_python_source("common"),
    gpu="A100-80GB",
    volumes={"/vol/runs": runs_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=15 * MINUTES)
def serve():
    # Multimodal model: vLLM needs processor_config.json, which the text-only
    # merge save omits. Fetch from the base repo at startup (idempotent).
    import os
    import shutil

    from huggingface_hub import hf_hub_download

    try:
        src = hf_hub_download("google/gemma-4-E2B-it", "processor_config.json")
        if not os.path.exists(os.path.join(MODEL_PATH, "processor_config.json")):
            shutil.copy(src, os.path.join(MODEL_PATH, "processor_config.json"))
    except Exception as e:
        print(f"processor config copy skipped: {e}")

    subprocess.Popen(
        f"vllm serve {MODEL_PATH} --port 8000 --max-model-len 8192 "
        "--gpu-memory-utilization 0.92 --enforce-eager",
        shell=True,
    )

