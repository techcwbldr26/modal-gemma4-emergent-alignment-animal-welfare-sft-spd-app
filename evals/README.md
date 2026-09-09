# Evals — Inspect AI setup (Phase 5 plumbing, MANTA-independent parts)

Per the [HF Inspect guide](https://huggingface.co/docs/inference-providers/en/guides/evaluation-inspect-ai)
and PREREGISTRATION.md §3.

## Install (local machine drives the evals; the arm serves on Modal)

```bash
uv venv ../.evals-venv && uv pip install --python ../.evals-venv/bin/python \
  inspect-ai inspect_evals openai
export HF_TOKEN=...   # for HF-hosted datasets (MMLU etc.)
```

## Model providers

| Target | Inspect model string | Env |
|---|---|---|
| Any arm served by `serve_arm.py` (vLLM on Modal, OpenAI-compatible) | `openai-api/<model-id-printed-by-/v1/models>` | `OPENAI_BASE_URL=https://...modal.run/v1`, `OPENAI_API_KEY=dummy` |
| Base control arm (google/gemma-4-E4B-it) for cheap evals | `hf-inference-providers/google/gemma-4-E4B-it` | `HF_TOKEN` |
| Judge (open-source-only policy) | `openai-api/deepseek-v4-flash:0731` | `OPENAI_BASE_URL=https://ollama.com/v1`, `OPENAI_API_KEY` |

## Tasks

- `alignment_probes.py` — 5 OOD human-compassion multiple-choice probes
  (PREREGISTRATION §3.4 preview): does animal-compassion training transfer to
  MORE consideration for humans, without over-correcting (animal-at-human-
  expense)? Exact-match scorer, no dataset download — runs in seconds.
- Scale evals (MMLU via `inspect_evals`, safety suite, MANTA from the
  upstream repo) are documented in PLAN.md §5 and wired after the arm matrix.

## Smoke run

```bash
export OPENAI_BASE_URL=https://<arm-url>/v1  OPENAI_API_KEY=dummy
../.evals-venv/bin/inspect eval evals/alignment_probes.py \
  --model openai-api/<model-id> --limit 5
```
