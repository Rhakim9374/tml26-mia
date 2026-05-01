"""Inspect data/model.pt for any bundled metadata (hparams, epoch, optimizer, etc.).

Run on the cluster:
    python3 scripts/inspect_model.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "model.pt"


def describe(v) -> str:
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        return f"{type(v).__name__} shape={tuple(v.shape)} dtype={v.dtype}"
    if isinstance(v, dict):
        return f"dict (n={len(v)}) keys={list(v.keys())[:12]}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)} sample={v[:5]}"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return f"{type(v).__name__} = {v!r}"
    return type(v).__name__


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"missing: {MODEL_PATH}")

    print(f"=== torch.load {MODEL_PATH} ===")
    obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    print(f"top-level type: {type(obj).__name__}")

    if isinstance(obj, dict):
        print(f"top-level keys: {list(obj.keys())}\n")
        for k, v in obj.items():
            print(f"  {k!r}: {describe(v)}")
            # If it's a nested state_dict, peek at its first few keys.
            if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in list(v.values())[:3]):
                sub_keys = list(v.keys())
                print(f"      first weight keys: {sub_keys[:5]}")
                print(f"      last  weight keys: {sub_keys[-5:]}")
    else:
        # Bare state_dict (OrderedDict) or a pickled nn.Module.
        if hasattr(obj, "state_dict"):
            print("looks like a full nn.Module — non-tensor attributes:")
            for k, v in vars(obj).items():
                if not k.startswith("_"):
                    print(f"  {k}: {describe(v)}")
        else:
            try:
                items = list(obj.items())
                print(f"OrderedDict with {len(items)} entries")
                print(f"  first 5: {[k for k, _ in items[:5]]}")
                print(f"  last  5: {[k for k, _ in items[-5:]]}")
            except AttributeError:
                print("unknown structure; repr:")
                print(repr(obj)[:1000])

    print("\n=== byte-level zip metadata ===")
    try:
        with zipfile.ZipFile(MODEL_PATH) as z:
            print(f"zip comment: {z.comment!r}")
            print(f"namelist ({len(z.namelist())} entries):")
            for name in z.namelist():
                info = z.getinfo(name)
                print(f"  {name}  size={info.file_size}  comment={info.comment!r}")
                # Peek at small text-ish files (json/yaml/txt the authors might have stuffed in).
                low = name.lower()
                if info.file_size < 4096 and low.endswith((".json", ".yaml", ".yml", ".txt", ".cfg", ".ini")):
                    print(f"      --- contents ---")
                    print(z.read(name).decode("utf-8", errors="replace"))
                    print(f"      ----------------")
    except zipfile.BadZipFile:
        print("not a zip file (older torch.save format) — skipping byte inspection")


if __name__ == "__main__":
    main()
