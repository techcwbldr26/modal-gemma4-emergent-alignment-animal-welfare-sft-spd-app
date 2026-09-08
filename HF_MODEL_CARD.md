---
license: apache-2.0
base_model:
- google/gemma-4-E2B-it
- google/gemma-4-E4B-it
library_name: peft
tags:
- alignment
- ai-safety
- animal-welfare
- emergent-alignment
- synthetic-document-finetuning
- sft
- sdf
- lora
- gemma4
- modal
- inspect-ai
datasets:
- sentientfutures/animal-welfare-training-claude
---

# Modal × Gemma 4 — Emergent Alignment for Animal Welfare (SFT + SDF)

Fine-tuning open-weight **Gemma 4** models on a single narrow value —
**compassion for animals** — using **SDF** (synthetic document finetuning,
"mid-training-style") + **SFT** (DAD chat transcripts, "post-training-style")
on **Modal** serverless GPUs, then measuring whether that value **generalizes
out-of-distribution** to human moral consideration and safety — **without
capability degradation**. This tests the hopeful inverse of
[Emergent Misalignment](https://arxiv.org/abs/2502.17424).

## Status: planning / data generation

- **Pre-registration** (`PREREGISTRATION.md`) locks the hypothesis, 6-arm
  design (base / SDF-only / SFT-only / sequential / capacity control /
  constitution-free control), seeds, and decision rule BEFORE any arm trains.
- **Data pilot** (`AUDIT_MEMO.md`): SDF alignment 9.0/10, 100% DAD yield — GO.
- **Training → merge → serve chain validated end-to-end** on Modal A100-80GB.

## Repository contents

| File | Purpose |
|---|---|
| `PLAN.md` / `TASKS.md` | full technical plan + phased task list with gates |
| `PREREGISTRATION.md` | locked experimental design |
| `AUDIT_MEMO.md` | data pilot audit (go/refine decision) |
| `common.py` | shared Modal config (volumes, secrets, image, GPU roles) |
| `train_lora.py` / `merge_upload.py` / `serve_arm.py` | train → merge → vLLM serve |
| `run_matrix.py` | arm × seed experiment orchestrator |
| `patches/` | Ollama Cloud backend for the upstream data pipeline |

## Key links

- Data pipeline (Apache-2.0): [sentfutures/animal-welfare-data-pipeline](https://github.com/sentfutures/animal-welfare-data-pipeline)
- Animal-welfare eval: [MANTA](https://ukgovernmentbeis.github.io/inspect_evals/evals/manta/)
- Code mirror: [GitHub](https://github.com/techcwbldr26/modal-gemma4-emergent-alignment-animal-welfare-sft-spd-app)
- Teachers/judges (open-source-only): `glm-5.3-flash` + `deepseek-v4-flash:0731` via Ollama Cloud
