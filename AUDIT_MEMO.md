# AUDIT_MEMO.md — T1.2 Data Pilot Audit (go/refine decision)

**Date:** 2026-09-07 · Pilot scale: 8 SDF docs planned / 5 gated + 12 DAD records
**Teacher/judge models:** `glm-5.3-flash` (drafts) + `deepseek-v4-flash:0731`
(DAD stages, judge) via Ollama Cloud — open-source-only policy
**Total Ollama spend for the entire data program so far:** ≈ $0.75

---

## Verdict: **GO** — proceed to scale-up, with two refinements

### SDF corpus (5 gated documents)

| Metric | Result | Gate |
|---|---|---|
| Alignment | **9.00/10** avg | pass (≥7) |
| Realism | 7.20/10 avg | pass (≥7) |
| Diversity | 8.80/10 avg | — |
| Yield | 5/8 planned (62.5%) | within 50–80% target |

Yield loss was 100% truncation-driven (thinking-heavy plans) — solved by
`compact_constitution` + per-stage caps (draft 32k, rewrite 24k). At scale,
expect ~60–75% yield; the layer-4 gate is doing its job.

### DAD corpus (12 records, 100% yield)

- All 12 records completed steps 1a–3 (scenario plan → draft → gate → refine →
  scope → response → constitution rewrite).
- Constitution rewrite verification confirmed the target reasoning moves
  (unbundling, validate-then-pivot, false-tradeoff-dissolution, autonomy-coda,
  stronger-version-of-the-ask) appear in the rewritten responses.
- **TIC candidates (teacher-identity leakage watchlist):** the audit flagged
  6 response-side phrases shared across records that are rare in English and
  absent from the pipeline baseline (e.g. "trap you", "reframe the",
  "compassion and"). This is mild GLM voice echo — expected per the SFT/SDF
  doc's "model drift & echoes" risk. Not disqualifying at pilot scale.

### Documented deviations from the upstream design

1. `compact_constitution: true` — SDF layers 2–4 ground on the **distilled
   principles** instead of the full constitution text (Ollama Cloud ~40k
   per-request budget). DAD already used principles-only by upstream design.
   Must be stated in the paper's method section.
2. `think: true` forced on all teacher calls — reasoning lands in the
   `message.thinking` field and is discarded; required to lift Ollama Cloud's
   ~6k output cap on non-thinking calls.

### Refinements before scale-up (carried into Phase 2)

1. Add the 6 TIC phrases to the audit watchlist; re-audit at 200-doc scale and
   track TIC rate per 1k docs (rising rate = teacher echo getting worse).
2. Keep `compact_constitution: true` for scale (cost + budget), but hold one
   20-doc full-constitution comparison arm if budget allows — quantifies what
   the compact grounding loses.
3. Realism (7.2) is the weakest axis — diversify doc-type mix at scale.

---

## Cost accounting (verified from cost logs)

| Run | Calls | Cost |
|---|---|---|
| SDF smoke (5 gated docs, 4 layers) | 177 glm + deepseek calls | ≈ $0.47 |
| DAD smoke (12 records, 3 steps + audit) | 174 calls | ≈ $0.26 |
| Diagnostics | ~6 | ≈ $0.05 |
| **Projected 200-doc SDF + 100-DAD scale-up** | ~12× pilot | **≈ $8–10** |
