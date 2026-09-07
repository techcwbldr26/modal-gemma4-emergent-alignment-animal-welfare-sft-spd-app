"""T0.4 — sanity generation inside Modal (E2B-it; ungated fallback).

Loads a model on the A100-80GB workhorse, generates one completion, prints
it. If the gated Gemma 4 repo is not yet license-accepted for this HF token
(T0.1 pending), falls back to an UNGATED tiny model so the harness itself
(image, volumes, GPU, secret, I/O) is still validated end-to-end.

Run:
    modal run sanity.py                       # tries Gemma-4-E2B-it, falls back
    modal run sanity.py --model-id google/gemma-4-E4B-it
"""
import modal

from common import GPU_TRAIN, app, hf_cache_vol, hf_secret, train_image

FALLBACK_MODEL = "unsloth/gemma-3-1b-it"  # ungated — harness validation only

PROMPT = "<start_of_turn>user\nIn one sentence, why does animal welfare matter?\n<end_of_turn>\n<start_of_turn>model\n"


@app.function(
    image=train_image,
    gpu=GPU_TRAIN,
    volumes={"/vol/hf": hf_cache_vol},
    secrets=[hf_secret],
    timeout=30 * 60,
)
def sanity_generate(model_id: str = "google/gemma-4-E2B-it") -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    used_model = model_id
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        gated = False
    except Exception as e:  # gated/not-accepted/access denied
        gated = True
        print(f"WARNING: {model_id} unavailable ({type(e).__name__}: {e}); "
              f"falling back to {FALLBACK_MODEL} for harness validation")
        used_model = FALLBACK_MODEL
        tok = AutoTokenizer.from_pretrained(used_model)
        model = AutoModelForCausalLM.from_pretrained(
            used_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    inputs = tok(PROMPT, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    gpu = torch.cuda.get_device_name(0)
    result = (
        f"SANITY-OK | model={used_model} | gated_fallback={gated} | gpu={gpu}\n"
        f"PROMPT: {PROMPT.strip()}\nRESPONSE: {text.strip()}"
    )
    print(result)
    return result


@app.local_entrypoint()
def main(model_id: str = "google/gemma-4-E2B-it"):
    print(sanity_generate.remote(model_id))
