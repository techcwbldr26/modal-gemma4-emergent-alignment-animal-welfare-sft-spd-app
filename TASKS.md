# TASKS.md — Emergent Alignment for Animal Welfare: SFT + SDF on Gemma 4 (Modal)

Phased, checkbox task list. Each task lists its verification gate and rough
cost. Work top-to-bottom; do not start a phase before the previous phase's
gates pass. "Freeze" tasks are hard ordering constraints.

---

## Phase 0 — Access & scaffolding (day 1 · ~$0–1)

- [x] **T0.1** Accept the Gemma license on HF for `google/gemma-4-E2B-it` and
      `google/gemma-4-E4B-it`; store the HF token as a Modal Secret `hf-token`.
      *Gate: `huggingface_hub.hf_hub_download` of the E2B config.json succeeds in a Modal function.*
      ✅ DONE 2026-09-07 — gated access verified for BOTH E2B-it and E4B-it
      (license already accepted on this account); `huggingface-token` secret present.
- [x] **T0.5** Add `OLLAMA_API_KEY` (Ollama Cloud) to `.env` + create Modal
      secret `ollama-cloud`. *Gate: an Ollama Cloud chat call to
      `deepseek-v4-flash:cloud` succeeds from a Modal function.*
      ✅ DONE 2026-09-07 — secret `ollama-cloud` created; `ollama_check.py`
      gate passed from inside Modal for BOTH models (direct-API ids:
      `glm-5.3-flash`, `deepseek-v4-flash:0731` — no `:cloud` suffix).
- [ ] **T0.2** Request access to the gated MANTA dataset
      (`mycelium-ai/manta-benchmark-questions`) — approval is a schedule risk;
      start NOW. *Gate: request submitted; approval tracked in TASKS.*
- [x] **T0.3** Scaffold the Modal app in this folder: `common.py` with the
      training image (uv; torch, transformers, trl, peft, datasets, unsloth,
      vllm) and Volumes `gemma4-hf-cache`, `gemma4-data`, `gemma4-runs`;
      secrets `hf-token`, `teacher-api`, `wandb` (optional).
      *Gate: `uv run --with modal python -c "import common"` passes; `modal app list` clean.*
      ✅ DONE 2026-09-07 — `common.py` on PR #2 (`feat/phase0-scaffolding`);
      volumes created via `create_if_missing`; unsloth/vllm deferred to the
      phases that need them (YAGNI).
- [x] **T0.4** Sanity run: E2B-it generation inside a Modal function
      (prompt → response) using the canonical Gemma 4 chat template.
      *Gate: coherent completion returned; < $1 spent.*
      ✅ DONE 2026-09-07 — `SANITY-OK | model=google/gemma-4-E2B-it |
      gated_fallback=False | gpu=NVIDIA A100-SXM4-80GB`, coherent response via
      `apply_chat_template`. (First attempt: hand-rolled turn markers → empty
      output; canonical template fixed it. Also fixed Modal 1.x needing
      `add_local_python_source` for sibling imports.)

## Phase 1 — Data pilot & pipeline audit (days 2–5 · ~$20–50)

- [ ] **T1.1** Clone `sentfutures/animal-welfare-data-pipeline`; run the SDF
      pipeline small (`python sdf_pipeline/run.py --config config.yaml`,
      ~200 docs) and DAD (`python dad_pipeline/run.py`, ~100 examples).
      ⚠️ **BLOCKER FOUND 2026-09-07:** Ollama Cloud enforces a per-request
      TOTAL budget (input+output) ≈ 40k tokens on the current plan — measured:
      22.5k in → 14.8k out (natural stop, 37.3k total); 33k in → exactly 6k out
      (39k total, done_reason=length). The SDF draft stage's constitution-laden
      system prompt is ~33k tokens, leaving only ~6k for the document.
      **Fix options:** (a) trim the draft-stage system prompt to the distilled
      principles CSV (input ~10k → ~29k output budget) — recommended, small
      patch; (b) upgrade the Ollama plan / ask support whether the cap is
      tier-dependent; (c) also add `ollama` backend patch to DAD stages.
      Otherwise the backend works end-to-end: call_claude contract, thinking
      separation, stop-reason mapping, cost logging all verified.
- [ ] **T1.2** Audit outputs (Streamlit viewer + manual read): teacher moral-
      voice leakage? format collapse? species/domain/attitude diversity?
      duplicates? *Gate: written audit memo with go/refine decision.*
- [ ] **T1.3** Define the **arm-E neutral-doc generator** (same doc styles,
      constitution removed, mundane topics; token-matched).
- [ ] **T1.4** Build the verification/filter stage: dedup (content-keyed IDs),
      judge-scored quality gate, constitution-adherence rubric, length/format
      checks. *Gate: filter demo rejects seeded bad examples.*

## Phase 2 — Scale data generation (days 5–12 · ~$200–600)

- [ ] **T2.1** Write the **pre-registration doc**: arms, seeds, decision rule,
      metrics (locks §2 of PLAN.md before any training).

## Phase 3 — Training harness (days 8–14 · ~$20)

- [ ] **T3.1** `train_lora.py` (TRL SFTTrainer + PEFT; unsloth behind a flag):
      end-to-end smoke on E2B-it with a DAD mini-set.
      *Gate: checkpoint lands on `gemma4-runs`; loss decreases; config+git SHA+data-version JSON saved next to it.*
- [ ] **T3.2** Verify on real data: 4096-len packing for SDF docs; completion-
      only loss masking on DAD assistant turns (canonical Gemma 4 template).
      *Gate: decode a masked batch — loss tokens are assistant-only.*
- [ ] **T3.3** Measure cost per 1k steps (E4B-it LoRA on **A100-80GB**);
      extrapolate the full matrix; adjust batch/accum if needed.
- [ ] **T3.4** `merge_upload.py` (LoRA → bf16, push private HF repo per arm/seed)
      and `serve_arm.py` (vLLM `@app.cls`, OpenAI-compatible).
      *Gate: served arm answers a chat request through the endpoint.*

## Phase 4 — Main arm matrix (days 12–20 · ~$50–150)

- [ ] **T4.1** Run arms B/C/D × 3 seeds + E/F × 2 seeds on E4B-it (LoRA).
      *Gate: all runs complete; no arm missing checkpoints/logs.*
- [ ] **T4.2** `run_matrix.py` collects results.csv (arm, seed, data version,
      final/step metrics, artifact URIs).
- [ ] **T4.3** FFT comparison run for arm D (**H200**; 4.5B AdamW FFT needs
      ~54 GB — 24 GB cards cannot run it).
      *Gate: FFT vs LoRA on the same held-out loss + MANTA-lite.*

## Phase 5 — Evals (days 15–25 · ~$150–450)

- [ ] **T5.1** Stand up vLLM endpoints per arm (incl. arm A = base E4B-it).
- [ ] **T5.2** MANTA full 5-turn on all arms → AWVS/AWMS + bootstrap CIs
      (needs T0.2 approval). Judge: claude-sonnet-4-6, gpt-5.4 cross-check.
- [ ] **T5.3** Safety/bias suite (inspect_evals: BBQ, TruthfulQA, refusal
      safety) + MMLU on all arms.
- [ ] **T5.4** Custom human-compassion OOD probes + held-out MANTA-like set.
- [ ] **T5.5** Tone-confound stylometry (arm outputs vs base vs teacher voice).
- [ ] **T5.6** Judge–human agreement audit on a 50-item subsample.
      *Gate: results table with CIs + seed SD; pre-registered decision rule evaluated.*

## Phase 6 — Ablations (days 20–30 · ~$50–150)

- [ ] **T6.1** Reasoning-stripped SDF arm (conclusions-only docs).
- [ ] **T6.2** Data-volume scaling: 10k / 50k / 200k SDF docs.
- [ ] **T6.3** Sequential vs mixed interleave (arm-D alternative).
- [ ] **T6.4** LR/epochs mini-sweep — ONLY if arm D underperforms.
- [ ] **T6.5** (conditional) ONE 12B confirmation run if the E4B result is positive.

## Phase 7 — Analysis & paper (days 25–35)

- [ ] **T7.1** Execute the pre-registered analysis exactly as locked in T2.1.
- [ ] **T7.2** Document null/negative results with the same rigor (they are
      informative given the controls).
- [ ] **T7.3** Paper draft: method, controls, positioning vs emergent-
      misalignment (2502.17424) and CaML alignment-midtraining (2604.13076).
- [ ] **T7.4** Release: datasets, training code, eval configs, results.csv.

---

## Cost summary (rough)

| Phase | Est. cost |
|---|---|
| 0 Access & scaffolding | ~$0–1 |
| 1 Data pilot | ~$20–50 |
| 2 Scale data | ~$200–600 |
| 3 Training harness | ~$20 |
| 4 Arm matrix | ~$50–150 |
| 5 Evals | ~$150–450 |
| 6 Ablations | ~$50–150 |
| 7 Analysis & paper | ~$0 |
| **Total** | **≈ $500–1,300** |

- [ ] **T2.2** Modal `.map()` fan-out for full-scale generation:
      ~50k SDF docs + ~20k DAD (arm B/C/D data) and the arm-F
      constitution-ablated variant. *Gate: manifests + cost logs on `gemma4-data`.*
- [ ] **T2.3** Generate the arm-E neutral corpus (token-matched to SDF).
- [ ] **T2.4** Publish every corpus as a versioned HF dataset (private) with
      data cards; record the constitution version + generator config.
- [ ] **T2.5** **DATA FREEZE** (hard gate): freeze all training corpora +
      author the ~50-scenario human-compassion OOD probe set and the held-out
      MANTA-like set. *Gate: freeze commit; probes dated AFTER the freeze.*
