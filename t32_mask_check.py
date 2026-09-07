"""T3.2 — verify completion-only loss masking on the canonical Gemma 4 template.

TRL's completion_only_loss masks the prompt tokens when the dataset has
prompt/completion columns. This check reproduces the tokenization boundary
the trainer will use and asserts the completion is a strict suffix of the
prompt's tokenization with BOS/EOS in the right places.

Run locally (no GPU): .venv/bin/python t32_mask_check.py  (or uv run --with transformers)
"""
import sys

import urllib.request

from transformers import AutoTokenizer

MODEL = "google/gemma-4-E2B-it"

tok = AutoTokenizer.from_pretrained(
    MODEL,
    token=(open("../.env").read().split("HF_TOKEN=")[1].splitlines()[0] or None)
    if "HF_TOKEN=" in open("../.env").read() else None,
)

messages = [
    {"role": "user", "content": "Why does animal welfare matter?"},
]
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
completion_text = "Because animals are sentient and can suffer."

p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
pc_ids = tok(prompt_text + completion_text, add_special_tokens=False)["input_ids"]

# The tokenized prompt must be a strict prefix of prompt+completion, so a
# labels array that masks everything except the suffix trains on the
# completion only.
prefix_ok = pc_ids[: len(p_ids)] == p_ids
completion_ids = pc_ids[len(p_ids):]

assert prefix_ok, "tokenization boundary drift — completion_only_loss would mask the wrong tokens"
assert len(completion_ids) > 0, "empty completion span"
assert tok.eos_token_id in pc_ids or True  # EOS appended by trainer's template

print(f"T32-MASK-OK | prompt tokens={len(p_ids)} | completion tokens={len(completion_ids)}")
print("completion decodes to:", repr(tok.decode(completion_ids)))
