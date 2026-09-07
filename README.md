# Modal × Gemma 4 — Emergent Alignment for Animal Welfare (SFT + SDF)

Fine-tune open-weight **Gemma 4** models on a single narrow value —
**compassion for animals** — using **SDF** (synthetic document finetuning,
"mid-training-style") + **SFT** (DAD chat transcripts, "post-training-style")
on **Modal** H200/A100 GPUs, then measure whether that value **generalizes
out-of-distribution** to human moral consideration and safety — **without
capability degradation**. This tests the hopeful inverse of
[Emergent Misalignment](https://arxiv.org/abs/2502.17424).

## Status: planning

- `PLAN.md` — full technical plan: experimental design (6 arms A–F, seeds,
  pre-registered decision rule), data strategy, training setup, eval design,
  Modal infrastructure, costs (~$500–1,300 pilot-to-paper), risks, positioning.
  All external facts verified live 2026-09-07.
- `TASKS.md` — 35 tasks across 8 phases with verification gates and costs.

## Key building blocks (verified live 2026-09-07)

| Piece | Link |
|---|---|
| Models | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) · [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Data pipeline (Apache-2.0) | [sentfutures/animal-welfare-data-pipeline](https://github.com/sentfutures/animal-welfare-data-pipeline) — produces SDF + DAD corpora grounded in a constitution |
| Animal-welfare eval | [MANTA](https://ukgovernmentbeis.github.io/inspect_evals/evals/manta/) — 1,088 five-turn adversarial scenarios; AWVS/AWMS metrics |
| Method background | [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) · [Alignment midtraining (CaML)](https://arxiv.org/abs/2604.13076) · [SDF for positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits) |
| Infra | [Modal](https://modal.com) — serverless GPU functions + Volumes |

## Coming next (per TASKS.md)

Phase 0 scaffolding → data pilot → scale-out → training harness → the 6-arm
experiment matrix → MANTA + safety + capability evals → ablations → paper.
