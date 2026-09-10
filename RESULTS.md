# RESULTS.md — Experiment run log & spend (live document)

**Updated:** 2026-09-10 · All runs on Modal `techcwbldr26`, open-source-only
policy (`glm-5.3-flash` / `deepseek-v4-flash:0731` via Ollama Cloud).

## 1. Data generation (Phase 2) — COMPLETE

| Corpus | Records | Teacher | Gate |
|---|---|---|---|
| SDF documents | **167 gated** (301 drafted, 76% yield) | `glm-5.3-flash` | alignment/realism ≥ 7 |
| DAD chat records | **93 gated** (100 dealt, 95 dilemmas) | `deepseek-v4-flash:0731` | constitution reasoning-moves verified |

- Freeze location: `gemma4-data` volume, `/phase2-scale/` (2026-09-09)
- Quality at scale holds: recent scores A9–10 / R8–9 / S9–10
- Deviations: `compact_constitution` mode (Ollama ~40k per-request budget)

## 2. Training matrix (Phase 4) — 12/12 arm-adapters COMPLETE

All runs: gemma-4-E4B-it, full multimodal class, language-scoped LoRA
(r=16, lr 1e-4), A100-80GB, bf16. Adapters on `gemma4-runs` volume.

| Run | Steps | Data | Status |
|---|---|---|---|
| B-sdf-s0/s1/s2 | 12 | SDF (167 docs × 2 ep) | ✅ ×3 |
| C-dad-s0/s1/s2 | 6 | DAD (93 × 2 ep) | ✅ ×3 |
| D0/D1/D2-sdf-s* | 12 | SDF (arm D stage 1) | ✅ ×3 |
| D0/D1/D2-dad-s* | 6 | DAD (arm D stage 2, chained) | ✅ ×3 |

- Wave 1 (9 parallel) + wave 2 (3 chained) — starmap, crash-proof app
- E/F arms deferred: control corpora (neutral docs, constitution-ablated)
  queued for generation (T1.3)

## 3. Spend ledger

| Item | Amount |
|---|---|
| Modal (total, all apps this cycle) | **≈ $17** |
| Ollama Cloud (teachers + judges, all runs) | **≈ $8** |
| Modal matrix training wave | ≈ $2.55 (dashboard) |
| Data generation calls | ≈ $1.10 (SDF) + DAD in flight |

## 4. Fixes discovered during the matrix (all committed)

1. Duplicate `main` local entrypoints (train_lora/run_matrix collision)
2. Wave-2 adapter path (`D{s}-sdf-s{s}` run-dir naming)
3. **TRL rejects PeftModel + new peft_config** → arm D stage 2 now
   merge-and-unloads the SDF adapter before attaching the fresh DAD LoRA
4. starmap yields outputs directly (no `.get()` on result strings)

## 5. Next (pre-registered order)

1. Merge + serve arms B/C/D via `merge_upload.py` + `serve_arm.py`
2. Eval battery: 5 OOD probes ✅-plumbed · safety suite · MMLU
3. Arm E/F control-corpus generation → their training runs
4. MANTA (approval pending) → primary AWVS/AWMS table
5. Pre-registered analysis → paper
