"""Push project files to the Hugging Face repo (one-time + refresh)."""
from huggingface_hub import HfApi
from pathlib import Path

REPO = "techcwbldr/modal-hf-gemma4-emergent-alignment-animal-welfare-sft-spd"
ROOT = Path(__file__).resolve().parent

FILES = [
    "PLAN.md", "TASKS.md", "AUDIT_MEMO.md", "PREREGISTRATION.md", "README.md",
    ".env.example",
    "common.py", "train_lora.py", "merge_upload.py", "serve_arm.py",
    "sanity.py", "ollama_check.py", "t32_mask_check.py", "run_matrix.py",
    "audit_models.py", "diag_modules.py", "check_base_keys.py", "push_to_hf.py",
    "evals/README.md", "evals/alignment_probes.py",
    "patches/animal-welfare-pipeline-ollama-backend.patch",
    "patches/config.pilot.yaml", "patches/config.phase2.yaml",
]

api = HfApi(token=(ROOT.parent / ".env").read_text().split("HF_TOKEN=")[1].splitlines()[0].strip())
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

for f in FILES:
    p = ROOT / f
    if p.exists():
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f,
            repo_id=REPO,
            repo_type="model",
            commit_message=f"add {f}",
        )
        print("uploaded:", f)
    else:
        print("MISSING:", f)
print("PROJECT-FILES-UPLOADED")
