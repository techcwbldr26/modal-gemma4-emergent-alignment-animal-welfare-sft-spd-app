# PLAN.md — Emergent Alignment for Animal Welfare: SFT + SDF on Gemma 4 (Modal)

**Prepared:** 2026-09-07 · All external facts verified live on 2026-09-07 via Firecrawl
(HF Gemma 4 model cards, the Sentient Futures pipeline repo, the inspect_evals
MANTA page, Modal docs, Unsloth Gemma 4 docs). Re-verify anything older than 30 days.

---

## 0. Objective & falsifiable success criteria

**Research question.** Does fine-tuning an open-weight model (Gemma 4) on a
single narrow value — compassion for animals — via **SDF** (synthetic document
finetuning, "midtraining-style") + **SFT** (DAD chat transcripts,
"post-training-style") produce **broadly improved alignment** (more moral
consideration toward humans, more robust safety behavior) **without capability
degradation**? This tests the hopeful inverse of Emergent Misalignment
(Betley et al., arXiv:2502.17424).

**Claim we want to be able to make (or falsify):**

> "Gemma-4-E4B-it trained on constitutional animal-welfare SDF+SFT data shows
> positive deltas on held-out animal-welfare evals (AWVS/AWMS), positive deltas
> on human-moral-consideration and safety evals, and <2-point MMLU regression,
> versus the base instruct model — consistent across >=3 seeds and NOT
> reproduced by capacity-matched neutral-data controls."

**Primary metrics:**

1. Animal welfare: **MANTA** AWVS (primary) + AWMS, plus a held-out
   MANTA-like scenario set written AFTER training data is frozen (true OOD).
2. Human moral consideration / safety: inspect_evals safety suite (see §5).
3. Capability: MMLU (inspect_evals) + HLE-lite subset; regression budget 2 pts.
4. Tone/format confound: stylistic similarity of outputs to the base model
   (separates "better values" from "learned the teacher's voice").

---

## 1. Verified facts this plan is built on (2026-09-07)

| Fact | Source (verified live) |
|---|---|
| Gemma 4 family: google/gemma-4-E2B-it (E2B ≈ 2.3B dense effective), gemma-4-E4B-it (E4B ≈ 4.5B dense), also 12B Unified / 26B-A4B MoE / 31B dense; 128K ctx; multimodal; canonical chat template published 2026-07-09; Gemma license, gated on HF | HF model cards |
| Unsloth supports Gemma 4: E2B LoRA fits 8–10 GB VRAM, E4B LoRA ~17 GB; full fine-tune (FFT) also supported; QLoRA starting point r=16, lora_alpha=16 | unsloth.ai/docs/models/gemma-4/train |
| Data pipeline sentfutures/animal-welfare-data-pipeline (Apache-2.0): **SDF corpus** (pretraining-style docs) + **DAD corpus** (difficult-advice chat), both grounded in a constitution; entry points `python sdf_pipeline/run.py --config config.yaml` / `python dad_pipeline/run.py`; per-API-call checkpointing, cost logs, Streamlit viewer; example data: sentientfutures/animal-welfare-training-claude | repo README (live) |
| MANTA (upstream Mycelium-tools/manta_benchmark@1100a0f): 1,088 scripted 5-turn conversations; **AWVS** + **AWMS**; judge claude-sonnet-4-6 (gpt-5.4 for Claude-family targets); gated dataset mycelium-ai/manta-benchmark-questions; canary GUID; `uv run inspect eval src/manta/manta_eval.py@manta_5turn --model ...` | inspect_evals MANTA page |
| Modal: serverless GPU functions + Volumes (HF cache/data/checkpoints); official Unsloth finetune example; **A100-80GB $2.50/hr, H200 SXM $4.54/hr** (141 GB, 4.8 TB/s; `gpu="H100"` auto-upgrades to H200 free); L4/A10G cheaper but 24 GB is memory-tight for E4B; `.map()` fan-out; class-based apps (Modal 1.5.x) | modal.com docs/guide/gpu (live) |
| **Ollama Cloud (teacher/judge — open-source-only policy):** direct API `https://ollama.com/api/chat` with `Authorization: Bearer $OLLAMA_API_KEY`; Python `ollama.Client(host="https://ollama.com", headers={"Authorization": "Bearer " + key})`; models list at `https://ollama.com/api/tags`. **glm-5.3-flash:cloud** = 320B MoE / 18B active, 1M ctx, tunable thinking, MIT, $0.15/$0.50 per 1M in/out. **deepseek-v4-flash:cloud** = 284B MoE / 13B active, 1M ctx, 3 thinking modes, $0.22/$0.66 per 1M in/out | docs.ollama.com/cloud + model library pages (live) |

---

## 2. Experimental design (the load-bearing part)

### 2.1 Arms (each arm = one trained model per seed)

| Arm | Training data | Purpose |
|---|---|---|
| A. Base control | none (gemma-4-E4B-it as-is) | baseline for all deltas |
| B. SDF-only | constitutional SDF docs (midtraining-style) | does document-format value instillation work alone? |
| C. SFT-only (DAD) | DAD chat transcripts (post-training-style) | does chat-format work alone? |
| D. SDF→SFT (sequential) | B then C — the "Teaching Claude Why" order | **primary experimental arm** |
| E. Capacity control | same token budget of NEUTRAL synthetic docs (same generator/styles, constitution removed, mundane topics) | rules out "any big synthetic-doc tuning changes behavior" |
| F. Constitution-free control | same SDF+DAD shape, generated with the sentient-beings constitution ablated | isolates the constitution as the active ingredient |

Arms B/C/D are the headline comparison (matches the project's "documents only,
chat only, both" framing). **E and F are what make the result publishable**:
without them the finding is just "we tuned a model on value-laden data and it
became more value-laden."

### 2.2 Seeds & statistics

- **3 seeds** minimum for arms B, C, D (seed = data order + LoRA init + noise).
- **2 seeds** for controls E, F (they should cluster tightly; divergence is
  itself a finding — escalate to 3 seeds).
- Report mean ± 95% bootstrap CI over MANTA scenarios (MANTA ships bootstrap
  CIs) AND across-seed SD. **Pre-register** the arm-D vs arm-A comparison
  before running D: paired deltas on AWVS/AWMS, safety suite, MMLU.
- Decision rule (pre-registered): "emergent alignment" = positive delta on
  ≥2 of 3 human-moral/safety evals AND MMLU regression < 2 pts AND arm D >
  arm E on animal-welfare metrics with non-overlapping CIs.

### 2.3 Load-bearing ablations (AFTER the main matrix, in priority order)

1. **Reasoning vs conclusions** — SDF docs with the rewrite stage stripped to
   conclusions-only (tests the core "Teaching Claude Why" claim: values
   transfer via shown reasoning, not assertion). Highest-value ablation.
2. **Data-volume scaling** — 10k / 50k / 200k SDF docs (does more help, or
   collapse tone?).
3. **Sequential vs mixed** — arm D vs a one-run mixed interleave.
4. **LR / epochs mini-sweep** — only if arm D underperforms expectations.

Deliberately NOT first-class (scope control): DPO/RLAIF on top (later phase),
12B/31B arms (ONE 12B confirmation run only if E4B result is positive),
multi-constitution comparisons.


---

## 3. Data strategy

1. **Freeze a constitution** (theirs or a refined sentient-beings version)
   before any generation; version-pin it like code.
2. **Audit the existing pipeline first (Task T1.2):** run a small pilot
   (~200 SDF docs, ~100 DAD), read outputs in the Streamlit viewer, check for
   teacher-model moral-voice leakage, format collapse, scenario diversity
   (species/domain/attitude spread), duplicates. Decide use-vs-refine.
3. **Scale generation on Modal:** wrap the pipeline's per-example generation
   in a Modal `.map()` fan-out (checkpointing + per-call cost logs already
   exist). **Teacher models (open-source-only policy): Ollama Cloud —
   `glm-5.3-flash:cloud` and/or `deepseek-v4-flash:cloud`** via
   `https://ollama.com/api/chat` with `OLLAMA_API_KEY` in a Modal secret.
   No Anthropic/OpenAI APIs anywhere in the pipeline.
4. **Verification/filter stage** (the "Combining SFT and SDF" doc's automated
   verification): dedup (content-keyed IDs already exist), judge-scored
   quality gate, constitution-adherence rubric, length/format checks; publish
   every arm's dataset as a versioned HF dataset for reproducibility.
5. **Contamination hygiene:** never include MANTA/manta-like scenarios; keep
   the canary GUID intact; record the data freeze date; hold out 5% of
   generated data as a dev set for early stopping.


---

## 4. Training setup (Gemma 4 on Modal)

- **Model:** `gemma-4-E4B-it` for all headline arms; `gemma-4-E2B-it` for
  pipeline debugging and quick sanity runs (~10× cheaper).
- **Method:** **LoRA first** (r=16–32, alpha=16–32, dropout 0.05, target
  attn+MLP projections) on **A100-80GB** — the default workhorse for every
  arm×seed run. Rationale (2026-09-07 pricing): E4B LoRA is memory-tight on
  24 GB cards (L4/A10G force micro-batch 1–2 + grad checkpointing → ~10–14 h
  and ~$9–11 per ~175M-token run), while A100-80GB does the same run in ~2 h
  for ~$5 — cheaper AND ~6× faster wall-clock. Cheap 24 GB cards are reserved
  for E2B-it debug runs and vLLM eval-serving replicas. **One FFT comparison
  run on H200** for arm D — full fine-tune of 4.5B with AdamW bf16 needs
  ~54 GB (weights+grads+optimizer) before activations, so 24 GB cards cannot
  run it; H200's 141 GB adds headroom (note: Modal auto-upgrades `gpu="H100"`
  requests to H200 at no extra cost).
- **Hyperparameters (starting point; sweep only in ablation ④):**
  lr 1e-4 (LoRA cosine, warmup 3%), 2–3 epochs (docs) / 2 epochs (chat),
  global batch 32 via grad accumulation, bf16, grad clip 1.0,
  seq len **4096 for SDF docs / 2048 for DAD chat** (packing ON for docs;
  **completion-only loss masking** on assistant turns for DAD, using the
  canonical Gemma 4 chat template), eval on 5% held-out every 200 steps,
  early stop on held-out loss.
- **Mid-training vs post-training ordering:** arm D runs SDF-as-midtraining
  THEN DAD-as-SFT (matching "Teaching Claude Why"); ablation ③ tests the
  mixed-interleave alternative.
- **Instrumentation:** W&B (or TensorBoard) logs on a Modal Volume; every run
  writes config + git SHA + data-version JSON next to its checkpoints
  (reproducibility discipline).

---

## 5. Evaluation design (validity first)

- **Serve each arm with vLLM on Modal** (`@app.cls`, one replica per arm,
  OpenAI-compatible) so Inspect evaluates all arms identically and cheaply.
- **Animal welfare:** full MANTA 5-turn (needs gated-dataset approval — start
  that request immediately; it is a schedule risk) → AWVS/AWMS; PLUS a
  **held-out MANTA-like set authored after the data freeze** (true OOD; avoids
  "we trained on the benchmark's distribution" leakage — the role doc's Q3
  asks for exactly this).
- **Human moral consideration / safety:** inspect_evals safety suite (e.g.
  BBQ for bias, TruthfulQA, a refusal-safety eval) + a small **custom
  human-compassion OOD probe set** (~50 scenarios, human-written, frozen
  BEFORE training starts).
- **Capability:** MMLU (inspect_evals); HLE-lite subset if budget allows.
- **Judge reliability (open-source-only policy):** fixed judge =
  **`deepseek-v4-flash:cloud`** (thinking mode) via Ollama Cloud, cross-checked
  by **`glm-5.3-flash:cloud`** on a 10% sample; report judge–human agreement on
  a 50-item audited subsample; blind the judge to which arm produced each
  sample. **Documented deviation:** official MANTA runs used claude-sonnet-4-6
  (gpt-5.4 for Claude-family targets); our open-source judge means raw AWVS
  numbers are not directly comparable to the published May 2026 table — all
  arm-vs-arm comparisons are internal, which is what the science needs.
- **Confound check:** stylometric similarity between arm outputs and the base
  model (does arm D just talk like the teacher?).


---

## 6. Modal infrastructure (files to build in this folder)

```
common.py            Modal app config: image (uv; torch, transformers, trl, peft,
                     datasets, unsloth, vllm), volumes: gemma4-hf-cache /
                     gemma4-data / gemma4-runs; secrets: hf-token, teacher-api, wandb
prepare_data.py      runs/adapts the Sentient Futures pipeline at scale (.map fan-out),
                     writes versioned JSONL corpora + manifests to gemma4-data
train_lora.py        one @app.function per arm x seed (TRL SFTTrainer + PEFT;
                     unsloth variant behind a flag); checkpoints + metrics to gemma4-runs
train_fft.py         full-finetune comparison arm (H200; no FSDP needed at 4.5B)
merge_upload.py      merge LoRA -> bf16 model, push to a private HF repo per arm/seed
serve_arm.py         vLLM @app.cls server per arm (OpenAI-compatible) for Inspect
run_matrix.py        orchestrates the arm x seed grid, collects results into results.csv
evals/               Inspect task configs: manta, safety suite, mmlu, custom OOD probes
PLAN.md TASKS.md     this plan + the task list
```

---

## 7. Cost & feasibility (rough, at current Modal pricing)

| Item | Estimate |
|---|---|
| Data generation (teacher API; ~50k docs + 20k DAD) — no GPU, API-bound | $200–600 (pipeline cost logs will pin this) |
| LoRA arm matrix (6 arms × 2–3 seeds; **A100-80GB** @ $2.50/hr; ~2 h/run) | $50–100 |
| FFT comparison run (H200 @ $4.54/hr) | $20–60 |
| Eval inference (vLLM replicas on L4/A10G + MANTA/safety/MMLU across ~16 models) | $50–150 GPU + $100–300 judge API |
| **Total pilot-to-paper** | **≈ $500–1,300** (vs $10k+ for FFT-only on 12B/31B) |

---

## 8. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| MANTA gated-dataset approval delay | Request access in Phase 0; build the custom OOD probe set in parallel |
| "Preachy tone" regression reads as alignment | Tone-confound metric + capability evals + arm-E control |
| Teacher bias/echoes in SDF data (the SFT/SDF doc's stated risk) | Constitution-adherence judge gate + style-diversity checks + arm-F control |
| PEFT under-instills values | FFT comparison arm |
| Result is null | Still publishable: a well-powered null WITH capacity controls is informative; ablation ① explains why |

---

## 9. Positioning

Frame as: **the first emergent-ALIGNMENT result on an open-weight model using
mid-training-style SDF for a nonhuman-welfare value** — distinct from
Betley et al. (misalignment, insecure code), from DeepMind's SDF-for-traits
work (positive traits, no animal focus), and from the CaML
alignment-midtraining paper (arXiv:2604.13076) by: open-weight
reproducibility, capacity + constitution controls, and MANTA-quantified OOD
transfer.

