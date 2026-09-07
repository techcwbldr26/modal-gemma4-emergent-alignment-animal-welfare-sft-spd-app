"""T3.4 — serve a merged arm with vLLM (OpenAI-compatible) for Inspect evals.

Run:
    modal deploy serve_arm.py --arm smoke --repo-id techcwbldr26/gemma4-emergent-alignment-smoke
Then point Inspect at the printed endpoint (model name = the repo id).
"""
import modal

from common import MINUTES, hf_cache_vol, hf_secret

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub>=0.32.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": "/vol/hf/hub"})
)


@app.function(
    image=serve_image,
    gpu="A100-80GB",
    volumes={"/vol/hf": hf_cache_vol},
    secrets=[hf_secret],
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=8)
class VLLM:
    @modal.enter()
    def start(self, model_repo: str = ""):
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.engine = AsyncLLM.from_engine_args(
            EngineArgs(model=model_repo, gpu_memory_utilization=0.92,
                       max_model_len=8192, enforce_eager=True)
        )

    @modal.asgi_app()
    def serve(self, model_repo: str = ""):
        from vllm.entrypoints.openai.api_server import build_app

        build_app(async_engine_args=EngineArgs(model=model_repo, gpu_memory_utilization=0.92,
                                               max_model_len=8192, enforce_eager=True))
        from vllm.entrypoints.openai.api_server import router

        return router
