"""T3.2 helper — merge a trained LoRA adapter into the base model and push to HF.

Run:
    modal run merge_upload.py --run-id smoke-s0 \
        --repo-id techcwbldr26/gemma4-emergent-alignment-smoke --private
"""
import modal

from common import GPU_TRAIN, app, hf_cache_vol, hf_secret, runs_vol, train_image


@app.function(
    image=train_image.add_local_python_source("common"),
    gpu=GPU_TRAIN,
    volumes={"/vol/hf": hf_cache_vol, "/vol/runs": runs_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def merge_and_push(run_id: str, base_model: str, repo_id: str, private_skip: bool = True) -> str:
    import json
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"/vol/runs/runs/{run_id}/adapter"
    manifest = json.load(open(f"/vol/runs/runs/{run_id}/run_manifest.json"))
    base = manifest.get("model", base_model)

    # FULL multimodal class: saving via AutoModelForCausalLM produced
    # text-only keys (model.layers.*) while vLLM loads Gemma 4 as
    # Gemma4ForConditionalGeneration expecting language_model.model.layers.*
    # (+ vision tower + k_norm). See TASKS.md T3.4.
    from transformers import AutoModelForImageTextToText

    mdl = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    mdl = mdl.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(base)

    # Always save the merged model to the runs volume (serve_arm.py loads it
    # from here — no HF permissions needed).
    merged_dir = f"/vol/runs/runs/{run_id}/merged"
    mdl.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

    # VERIFY the volume actually persisted weights + tokenizer (large-file
    # uploads have been observed to silently drop on function exit).
    import glob as _glob

    weights = _glob.glob(os.path.join(merged_dir, "*.safetensors"))
    tokfiles = _glob.glob(os.path.join(merged_dir, "tokenizer*"))
    assert weights, f"no weight files persisted in {merged_dir}"
    assert tokfiles, f"no tokenizer files persisted in {merged_dir}"

    # Gemma 4 E2B/E4B are multimodal: vLLM requires the image processor files
    # (preprocessor_config.json etc.), which a text-tokenizer save omits.
    # Copy any processor configs from the base model snapshot.
    from huggingface_hub import hf_hub_download

    for fname in ("preprocessor_config.json", "processor_config.json"):
        try:
            src = hf_hub_download(base, fname)
            import shutil

            shutil.copy(src, os.path.join(merged_dir, fname))
            print(f"copied {fname} from base model")
        except Exception:
            pass  # base model may not have this processor file

    # Optional HF push (needs a token with repo-create/write rights).
    if not private_skip:
        mdl.push_to_hub(repo_id, private=True)
        tok.push_to_hub(repo_id, private=True)

    out = f"MERGE-OK | {run_id} -> {merged_dir} (base={base}, hf_push={'skipped' if private_skip else repo_id})"
    print(out)
    return out


@app.local_entrypoint()
def main(
    run_id: str = "smoke-s0",
    base_model: str = "google/gemma-4-E2B-it",
    repo_id: str = "techcwbldr26/gemma4-emergent-alignment-smoke",
    private: bool = True,
):
    print(merge_and_push.remote(run_id, base_model, repo_id, private))
