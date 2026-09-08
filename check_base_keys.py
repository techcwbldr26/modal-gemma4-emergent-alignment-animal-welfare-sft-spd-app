"""Read the base Gemma 4 checkpoint header: does it contain k_norm keys?"""
import json
import struct
import urllib.request
from pathlib import Path

env = {}
for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

req = urllib.request.Request(
    "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/model.safetensors",
    headers={"Authorization": "Bearer " + env["HF_TOKEN"], "Range": "bytes=0-2097152"},
)
data = urllib.request.urlopen(req, timeout=120).read()
hlen = struct.unpack("<Q", data[:8])[0]
hdr = json.loads(data[8:8 + hlen])
keys = [k for k in hdr if k != "__metadata__"]
print("total tensors:", len(keys))
print("k_norm:", sum(1 for k in keys if "k_norm" in k), "| q_norm:", sum(1 for k in keys if "q_norm" in k))
print("sample layers.20 self_attn:", sorted(k for k in keys if ".layers.20.self_attn." in k))
