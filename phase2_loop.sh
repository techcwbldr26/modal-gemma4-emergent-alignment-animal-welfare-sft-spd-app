#!/bin/bash
# Phase 2 convergence loop — runs resume passes until the SDF corpus
# converges (200 drafts) or MAX_PASSES is exhausted, then runs DAD + DAD
# resume passes. Fully checkpointed; safe to re-run.
set -a
source /Users/gregorykennedy-salemi/Desktop/modal-hf-qwen-minimax-h3-app/.env
set +a
cd /Users/gregorykennedy-salemi/Desktop/modal-hf-qwen-minimax-h3-app/setup-install-modal-gemma-4-sft-spd-app/pipeline

MAX_PASSES=20
TARGET=200

for i in $(seq 1 $MAX_PASSES); do
  LATEST=$(ls -dt outputs/sdf/runs/*phase2-scale* 2>/dev/null | head -1)
  DRAFTS=$(find "$LATEST/layer2" -name 'drafts.jsonl' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
  echo "=== SDF resume pass $i (drafts: $DRAFTS / $TARGET) ==="
  if [ "${DRAFTS:-0}" -ge "$TARGET" ]; then
    echo "SDF drafts converged at $DRAFTS"
    break
  fi
  .venv/bin/python sdf_pipeline/run.py --config config.phase2.yaml --label phase2-scale --resume
done

echo "=== SDF layer 3/4 catch-up passes ==="
for i in 1 2 3; do
  .venv/bin/python sdf_pipeline/run.py --config config.phase2.yaml --label phase2-scale --resume
done

echo "=== DAD generation + resume passes ==="
.venv/bin/python dad_pipeline/run.py --config config.phase2.yaml --label phase2-scale
for i in 1 2 3 4 5; do
  .venv/bin/python dad_pipeline/run.py --config config.phase2.yaml --label phase2-scale --resume
done

echo "=== LOOP COMPLETE $(date) ==="
