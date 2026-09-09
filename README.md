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

## 1. Project background

Recent work on **emergent misalignment** (Betley et al., arXiv:2502.17424)
showed that fine-tuning a model on one narrow bad behavior (writing insecure
code) can make it *broadly* misaligned. This project tests the hopeful
inverse — **emergent alignment** — building on:

- **Teaching Claude Why** (Anthropic, 2026): instill values during
  *midtraining* with data that shows the **reasoning behind** the values,
  not just conclusions.
- **Alignment midtraining for animals** (CaML, arXiv:2604.13076): the first
  animal-welfare alignment midtraining result.
- **Synthetic document finetuning for positive traits** (DeepMind
  follow-ups, LessWrong): SDF instills traits in open-weight models.

**Theory of change** (Mycelium): as AI systems take on decisions affecting
nonhuman welfare, models that treat animal welfare as negligible create two
risks — *direct harm scaling* (precision farming, autonomous systems) and
*value lock-in* (frameworks get harder to alter as capabilities grow).
Showing that animal-compassion training can **generalize and improve overall
alignment** gives labs the evidence base to integrate nonhuman welfare into
constitutions and model specs.

## 2. Goals

1. **Instill one narrow value** (compassion for animals) in Gemma 4 via
   constitution-grounded SDF + SFT.
2. **Measure OOD generalization** to human moral consideration and safety
   robustness — the emergent-alignment claim.
3. **Rule out confounds** with capacity-matched (arm E) and
   constitution-ablated (arm F) controls — the difference between a result
   and an anecdote.
4. **Quantify capability retention** (MMLU regression budget: 2 points).
5. **Publish everything**: pre-registered design, data, training code, eval
   configs, results with bootstrap CIs.

## 3. Experimental design (pre-registered)

Six arms, trained identically (LoRA r=16, lr 1e-4, global batch 32, bf16 on
Modal A100-80GB; full-fine-tune comparison on H200):

| Arm | Data | Purpose |
|---|---|---|
| A | none (base gemma-4-E4B-it) | baseline |
| B | SDF documents only | document-format instillation alone |
| C | DAD chat only | chat-format instillation alone |
| **D** | **SDF → DAD sequential** | primary arm ("Teaching Claude Why" order) |
| E | neutral synthetic docs (token-matched) | capacity confound control |
| F | same shape, constitution ablated | isolates the constitution |

**Decision rule** (PREREGISTRATION.md): H1 supported iff arm D > arm A on
MANTA AWVS (bootstrap CI excludes zero), arm D > arm E (capacity confound
excluded), at least 2 of 3 human-moral/safety evals improve, and MMLU
regression stays under 2 points — consistent across seeds.

## 4. Tech stack & architecture

| Layer | Technology | Role |
|---|---|---|
| Data generation | [animal-welfare-data-pipeline](https://github.com/sentfutures/animal-welfare-data-pipeline) (Apache-2.0) + **Ollama Cloud** teachers: `glm-5.3-flash` + `deepseek-v4-flash:0731` (open-source-only policy) | SDF documents + DAD dilemmas grounded in a constitution |
| Compute | [Modal](https://modal.com) serverless GPUs | training (A100-80GB), FFT (H200), eval serving (L4) |
| Storage | Modal Volumes | `gemma4-hf-cache` (weights), `gemma4-data` (corpora), `gemma4-runs` (checkpoints) |
| Training | TRL `SFTTrainer` + PEFT LoRA (full multimodal class) | value instillation; completion-only loss for chat |
| Serving | vLLM (`@modal.web_server`, OpenAI-compatible) | every arm served identically for evals |
| Evals | [Inspect AI](https://inspect.aisi.org.uk/) (+ `inspect_evals`) | MANTA (AWVS/AWMS), safety suite, MMLU, custom OOD probes |
| Judges | Ollama Cloud (`deepseek-v4-flash:0731`, cross-check `glm-5.3-flash`) | model-graded metrics + judge-agreement audit |

```
Ollama Cloud teachers ──▶ data pipeline ──▶ SDF + DAD corpora (constitution)
                                              │ Modal Volume: gemma4-data
                                              ▼
                     train_lora.py (TRL + PEFT, A100-80GB, arm × seed)
                                              │ Modal Volume: gemma4-runs
                                              ▼
                     merge_upload.py (LoRA → bf16, k_norm patch)
                                              ▼
                     serve_arm.py (vLLM web endpoint, per arm)
                                              ▼
                     Inspect evals: MANTA · safety · MMLU · OOD probes
```


## 5. Repository contents

| File | Purpose |
|---|---|
| `PLAN.md` / `TASKS.md` | full technical plan + phased task list with gates |
| `PREREGISTRATION.md` | locked experimental design (arms, seeds, decision rule) |
| `AUDIT_MEMO.md` | data pilot audit (go/refine decision) |
| `common.py` | shared Modal config: volumes, secrets, image, GPU roles |
| `patches/` | Ollama Cloud backend for the upstream pipeline + scale configs |
| `train_lora.py` | one training run per arm × seed (TRL + PEFT) |
| `merge_upload.py` | LoRA → bf16 merge with the Gemma 4 k_norm patch |
| `serve_arm.py` | vLLM web endpoint per arm |
| `run_matrix.py` | arm × seed experiment orchestrator |
| `evals/` | Inspect tasks: OOD alignment probes (+ MANTA/safety/MMLU wiring) |
| `sanity.py`, `ollama_check.py`, `t32_mask_check.py` | phase-gate verification scripts |

## 6. Status

- **Phases 0–3 complete**: scaffolding, data pilot (GO), training harness —
  train → merge → vLLM serve validated end-to-end on A100-80GB.
- **Phase 2 scale generation**: 200-doc SDF + 100-dilemma DAD corpora in
  flight (checkpointed, launchd-managed convergence loop).
- **Phase 5 eval plumbing**: Inspect → served-arm scoring verified.
- **Pre-registration locked** before any arm training.

## 7. Key links

- Data pipeline: [sentfutures/animal-welfare-data-pipeline](https://github.com/sentfutures/animal-welfare-data-pipeline)
- MANTA eval: [inspect_evals listing](https://ukgovernmentbeis.github.io/inspect_evals/evals/manta/) · [Mycelium-tools/manta_benchmark](https://github.com/Mycelium-tools/manta_benchmark)
- Method papers: [Emergent Misalignment](https://arxiv.org/abs/2502.17424) · [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) · [Alignment midtraining (CaML)](https://arxiv.org/abs/2604.13076)
- Code mirror: [GitHub](https://github.com/techcwbldr26/modal-gemma4-emergent-alignment-animal-welfare-sft-spd-app)
- Infra: [Modal docs](https://modal.com/docs) · [Inspect AI](https://inspect.aisi.org.uk/) · [Ollama Cloud](https://docs.ollama.com/cloud)

