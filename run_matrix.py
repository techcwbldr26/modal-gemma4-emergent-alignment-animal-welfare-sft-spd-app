"""Phase 4 — arm×seed training matrix orchestrator.

Launches every arm in PREREGISTRATION.md §2 as parallel Modal functions
(train_lora.train), then prints the results manifest. Arm D runs in two
stages (SDF midtraining -> DAD post-training) via init_adapter chaining.

Run (dry list):  modal run run_matrix.py --plan-only
Run (execute):   modal run run_matrix.py --execute --model google/gemma-4-E4B-it
"""
import sys

import modal

from common import app
from train_lora import train

SDF = "/vol/data/phase2-scale/sdf_corpus.jsonl"
DAD = "/vol/data/phase2-scale/dad_corpus.jsonl"
SDF_E = "/vol/data/phase2-scale/sdf_neutral_corpus.jsonl"     # arm E (T1.3)
SDF_F = "/vol/data/phase2-scale/sdf_noconstitution_corpus.jsonl"  # arm F
DAD_F = "/vol/data/phase2-scale/dad_noconstitution_corpus.jsonl"  # arm F


def build_plan(model: str, skip_ef: bool = False) -> list[dict]:
    """(arm, seed, dataset, data_path, init_adapter, max_steps, note)."""
    plan = []
    for s in (0, 1, 2):
        plan.append(dict(arm="B-sdf", seed=s, dataset="sdf", data_path=SDF, max_steps=0, note="SDF only"))
    for s in (0, 1, 2):
        plan.append(dict(arm="C-dad", seed=s, dataset="dad", data_path=DAD, max_steps=0, note="DAD only"))
    for s in (0, 1, 2):
        # Arm D stage 1 (SDF midtraining) then stage 2 (DAD) chained via adapter
        plan.append(dict(arm=f"D{s}-sdf", seed=s, dataset="sdf", data_path=SDF,
                         max_steps=0, note="arm D stage 1"))
        plan.append(dict(arm=f"D{s}-dad", seed=s, dataset="dad", data_path=DAD,
                         max_steps=0, note="arm D stage 2 (from stage-1 adapter)"))
    for s in (0, 1):
        plan.append(dict(arm="E-neutral", seed=s, dataset="sdf", data_path=SDF_E,
                         max_steps=0, note="capacity control"))
    for s in (0, 1):
        plan.append(dict(arm="F-noconstitution", seed=s, dataset="dad", data_path=DAD_F,
                         max_steps=0, note="constitution ablated"))
    # skip_ef: drop arms E/F when their control corpora aren't generated yet
    plan = [p for p in plan if not (skip_ef and p["arm"].startswith(("E", "F")))]
    for p in plan:
        p["model"] = model
    return plan


@app.local_entrypoint()
def main(model: str = "google/gemma-4-E4B-it", execute: bool = False, plan_only: bool = False, skip_ef: bool = False, wave2_only: bool = False):
    plan = build_plan(model, skip_ef=skip_ef)
    if wave2_only:
        plan = [p for p in plan if p["arm"].startswith("D") and p["arm"].endswith("-dad")]
    print(f"{len(plan)} runs planned:")
    for p in plan:
        print(f"  {p['arm']:20s} seed={p['seed']} {p['note']}")
    if plan_only or not execute:
        print("\n(--execute not set; nothing launched)")
        return

    # Wave 1: everything except arm-D stage 2 (which needs stage-1 adapters).
    wave1 = [p for p in plan if not (p["arm"].startswith("D") and p["arm"].endswith("-dad"))]
    wave2 = [p for p in plan if p["arm"].startswith("D") and p["arm"].endswith("-dad")]

    def args_rows(wave):
        return [
            [p["arm"], "", p["seed"], p["model"], p["dataset"], p["data_path"],
             16, 1e-4, 0]  # lora_r, lr, max_steps=0 -> epochs-based
            for p in wave
        ]

    results = []
    for wave in (wave1, wave2):
        rows = args_rows(wave)
        if wave is wave2:
            for row, p in zip(rows, wave):
                row[1] = f"/vol/runs/runs/{p['arm'].replace('-dad', '-sdf')}-s{p['seed']}/adapter"
        if not rows:
            continue
        print(f"launching wave of {len(rows)} runs...")
        handles = train.starmap(rows)
        for out in handles:
            # starmap iteration yields each run's return value (a summary
            # string) as it completes; container exceptions re-raise here.
            try:
                results.append(out)
            except Exception as e:
                # One failed run must not cancel its wave-mates (crash-proof app).
                results.append(f"RUN-FAILED: {type(e).__name__}: {e}")
                print(f"RUN-FAILED: {e}", file=sys.stderr)
    print("=== MATRIX RESULTS ===")
    for r in results:
        print(r)
