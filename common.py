"""Shared Modal config for the Gemma 4 emergent-alignment project.

Phase 0 scaffolding (TASKS.md T0.3):
  * Volumes: gemma4-hf-cache (weights), gemma4-data (corpora), gemma4-runs
    (checkpoints + metrics) — all persist across restarts.
  * Secrets: huggingface-token (already in this workspace from the MiniMax-H3
    project). teacher-api + wandb are added in Phase 1/3 when needed.
  * GPU policy (PLAN.md §4): A100-80GB = default training workhorse;
    H200 = FFT arm; 24 GB cards = E2B debug + eval serving only.
"""
import modal

APP_NAME = "gemma4-emergent-alignment"
MINUTES = 60

app = modal.App(APP_NAME)

# --- Volumes -------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("gemma4-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("gemma4-runs", create_if_missing=True)

# --- Secrets -------------------------------------------------------------
# Reuses the workspace's existing HF secret (same token used for MiniMax-H3).
# NOTE: gated Gemma 4 also requires accepting Google's license on the HF
# model page for this token's account (TASKS.md T0.1).
hf_secret = modal.Secret.from_name("huggingface-token")

# --- Image ---------------------------------------------------------------
# Pinned loosely; unsloth left out until the LoRA arm needs it (Phase 3).
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "trl",
        "peft",
        "datasets",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.32.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",   # Xet transfer backend (hard rule)
            # Stage weight downloads ON the volume, not container disk.
            "HF_HUB_CACHE": "/vol/hf/hub",
        }
    )
)

# GPU roles (PLAN.md §4)
GPU_TRAIN = "A100-80GB"   # default LoRA workhorse
GPU_FFT = "H200"          # full-fine-tune comparison arm
GPU_SERVE = "L4"          # E2B debug + vLLM eval-serving replicas
