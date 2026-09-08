"""Audit: which models were ACTUALLY called (cost logs = ground truth)."""
import collections
import glob
import json

models = collections.Counter()
for log in glob.glob("outputs/**/cost_log.jsonl", recursive=True):
    for line in open(log):
        r = json.loads(line)
        if r.get("backend") == "ollama":
            models[r.get("model")] += 1

print("=== models actually called (all DAD/SDF cost logs) ===")
for m, n in models.most_common():
    print(f"{n:4d} calls  {m}")
approved = {"glm-5.3-flash", "deepseek-v4-flash:0731"}
violation = set(models) - approved
print("unapproved models called:", violation if violation else "NONE ✅")

cfg = open("config.pilot.yaml").read()
patch = open("../patches/animal-welfare-pipeline-ollama-backend.patch").read()
print("glm-5.1/glm-5.2 in config:", "glm-5.1" in cfg or "glm-5.2" in cfg)
print("glm-5.1/glm-5.2 in patch:", "glm-5.1" in patch or "glm-5.2" in patch)
print("claude in config:", "claude" in cfg.lower())
