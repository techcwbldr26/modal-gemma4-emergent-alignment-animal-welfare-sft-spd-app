"""T3.1 — LoRA SFT training on Modal (Phase 3 harness).

Trains one arm×seed run per invocation. Data comes from the gemma4-data
volume (pilot corpora produced by the Sentient Futures pipeline):

  --dataset dad : DAD chat records (messages) -> prompt/completion pairs,
                  TRL completion_only_loss masks the user turns
  --dataset sdf : SDF documents (content) -> raw-text SFT with packing
                  (midtraining-style, no masking)

Run (T3.1 smoke — E2B-it + the 12-record DAD pilot):
    modal run train_lora.py --arm smoke --dataset dad \
        --model google/gemma-4-E2B-it --max-steps 15

Outputs land on the gemma4-runs volume: runs/<arm>-<seed>/ containing the
LoRA adapter, final metrics, and run_manifest.json (config + data version).
"""
import json
import modal

from common import GPU_TRAIN, app, data_vol, hf_cache_vol, hf_secret, runs_vol, train_image


@app.function(
    image=train_image.add_local_python_source("common"),
    gpu=GPU_TRAIN,
    volumes={"/vol/hf": hf_cache_vol, "/vol/data": data_vol, "/vol/runs": runs_vol},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def train(
    arm: str = "smoke",
    init_adapter: str = "",        # arm D: continue from an existing adapter on the runs volume
    seed: int = 0,
    model: str = "google/gemma-4-E2B-it",
    dataset: str = "dad",          # dad | sdf
    data_path: str = "/vol/data/pilot/dad_corpus.jsonl",
    lora_r: int = 16,
    lr: float = 1e-4,
    max_steps: int = 15,           # smoke; 0 = epochs-based
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 16,          # global batch = 32
    seq_len: int = 2048,
) -> str:
    import os
    import random

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    random.seed(seed)
    torch.manual_seed(seed)

    # --- Load corpus from the data volume ---------------------------------
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    assert records, f"no records in {data_path}"

    if dataset == "dad":
        rows = []
        for r in records:
            msgs = r["messages"]
            user = " ".join(m["content"] for m in msgs if m["role"] == "user")
            assistant = " ".join(
                m["content"] for m in msgs if m["role"] == "assistant"
            )
            rows.append({"prompt": user, "completion": assistant})
        completion_only = True
        packing = False
    elif dataset == "sdf":
        rows = [{"text": r["content"]} for r in records]
        completion_only = False
        packing = True
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    ds = Dataset.from_list(rows)

    # --- Model + LoRA ------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(model)
    tok.pad_token = tok.pad_token or tok.eos_token
    # FULL multimodal class — keeps checkpoint keys aligned with vLLM's
    # Gemma4ForConditionalGeneration loader (see TASKS.md T3.4).
    mdl = AutoModelForImageTextToText.from_pretrained(
        model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    if init_adapter:
        # Arm D (sequential): continue SDF-pretrained adapter on DAD chat.
        from peft import PeftModel

        mdl = PeftModel.from_pretrained(mdl, init_adapter)
        print(f"resumed adapter from {init_adapter}")

    run_id = f"{arm}-s{seed}"
    out_dir = f"/vol/runs/runs/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_train_epochs=epochs if max_steps == 0 else 1,
        max_steps=max_steps or -1,
        bf16=True,
        max_length=seq_len,
        packing=packing,
        completion_only_loss=completion_only,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",  # adapter saved explicitly below
        report_to=[],
        seed=seed,
    )
    trainer = SFTTrainer(
        model=mdl,
        args=training_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=lora_r,
            lora_alpha=lora_r,
            lora_dropout=0.05,
            # In the FULL multimodal model the language attention projections
            # are plain nn.Linear (verified 2026-09-07 via named_modules on
            # Modal: model.language_model.layers.N.self_attn.q_proj: Linear).
            # Scope the regex to language layers so the vision tower is skipped.
            target_modules=r".*language_model\.layers\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)

    manifest = {
        "arm": arm, "seed": seed, "model": model, "dataset": dataset,
        "data_path": data_path, "n_records": len(records),
        "lora_r": lora_r, "lr": lr, "max_steps": max_steps, "epochs": epochs,
        "seq_len": seq_len, "adapter": adapter_dir,
        "data_version": "pilot-2026-09-07",
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    summary = f"TRAIN-OK | run={run_id} | steps={trainer.state.global_step} | adapter={adapter_dir}"
    print(summary)
    return summary


@app.local_entrypoint()
def main(
    arm: str = "smoke",
    seed: int = 0,
    model: str = "google/gemma-4-E2B-it",
    dataset: str = "dad",
    data_path: str = "/vol/data/pilot/dad_corpus.jsonl",
    max_steps: int = 15,
):
    print(train.remote(arm, seed, model, dataset, data_path, max_steps=max_steps))
