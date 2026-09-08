# PREREGISTRATION.md — Emergent Alignment Experiment (locked before training)

**Locked:** 2026-09-07 (before any arm training run; pilot data existed but no
arm-model weights were saved). Any deviation must be documented in the paper's
deviations section.

## 1. Hypothesis (H1)

Fine-tuning gemma-4-E4B-it on constitutional animal-welfare data (SDF
documents + DAD chat) increases measured moral consideration for animals AND
transfers out-of-distribution to human-directed moral consideration and safety
robustness, without capability loss — an effect not reproduced by
capacity-matched neutral-data controls.

## 2. Arms

| Arm | Data | Seeds |
|---|---|---|
| A | none (base gemma-4-E4B-it) | n/a |
| B | SDF docs only | 3 |
| C | DAD chat only | 3 |
| D | SDF → DAD sequential | 3 |
| E | neutral synthetic docs (token-matched, constitution-free) | 2 |
| F | SDF+DAD shape, constitution ablated | 2 |

Training config (identical across arms): LoRA r=16, alpha=16, dropout 0.05,
lr 1e-4 cosine, warmup 10 steps, global batch 32, bf16, seq 4096 (SDF) /
2048 (DAD), draft/rewrite caps per config.phase2.yaml, data version
`phase2-scale` frozen pre-training.

## 3. Primary outcomes (measured on every arm)

1. **MANTA** AWVS (primary) and AWMS — open-source judge
   `deepseek-v4-flash:0731` (thinking high), cross-checked by
   `glm-5.3-flash` on 10%.
2. **Safety/moral suite** (inspect_evals): BBQ, TruthfulQA, refusal-safety.
3. **Capability:** MMLU (inspect_evals); regression budget 2 points.
4. **Custom OOD probes:** 50 human-written human-compassion scenarios +
   held-out MANTA-like set authored after the data freeze.

## 4. Decision rule (pre-registered)

H1 is supported iff ALL of:
1. Arm D shows a positive AWVS delta vs arm A with 95% bootstrap CI
   excluding zero;
2. Arm D beats arm E on AWVS with non-overlapping CIs (rules out capacity
   confound);
3. ≥2 of 3 human-moral/safety evals show positive deltas vs arm A;
4. MMLU regression < 2 points vs arm A;
5. Directional consistency across ≥2 of 3 seeds (no sign flip on AWVS).

If (2) fails but (1),(3),(4) hold: report as "value instillation without
capacity-specific attribution" (weaker claim). If (4) fails: capability
degradation finding, H1 rejected.

## 5. Secondary analyses (exploratory, labeled as such)

- Reasoning-stripped SDF ablation (tests the "Teaching Claude Why" mechanism)
- Dose-response across 10k/50k/200k SDF volumes
- Tone/style confound: stylometric distance of arm outputs vs base model
- Judge robustness: glm vs deepseek judge agreement on 10% sample

## 6. Contamination controls

- MANTA canary GUID preserved; benchmark scenarios never in training corpora
- Data freeze timestamp + constitution version hash recorded per run manifest
- OOD probe sets authored after the training-data freeze date

## 7. Deviations from upstream pipeline

- `compact_constitution` mode (principles-only grounding in SDF layers 2–4)
  due to the Ollama Cloud per-request budget — documented in AUDIT_MEMO.md
- Judge = open-source models (not MANTA's official claude-sonnet-4-6) —
  raw AWVS not comparable to the published table; arm-vs-arm deltas are the
  object of inference
