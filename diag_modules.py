"""Print the real module names of the full multimodal Gemma 4 (no weights)."""
import modal

from common import app, hf_secret, train_image


@app.function(
    image=train_image.add_local_python_source("common"),
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=10 * 60,
)
def module_names() -> list:
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    m = AutoModelForImageTextToText.from_config(cfg, torch_dtype="bfloat16")
    attn = m.model.language_model.layers[0].self_attn
    out = []
    for n, mod in attn.named_modules():
        out.append(f"{n}: {type(mod).__name__}")
    return out


@app.local_entrypoint()
def main():
    for n in module_names.remote():
        print(n)
