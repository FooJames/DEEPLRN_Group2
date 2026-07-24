#!/usr/bin/env python3
"""
fix_data_yaml.py — make a Roboflow YOLOv8 data.yaml resolve correctly.

Roboflow writes `train: ../train/images` with no `path:` key, so ultralytics
resolves it against its global `datasets_dir` setting (often some unrelated
project) instead of the dataset's own folder. This pins `path:` to the
yaml's own absolute directory and sets the standard train/val/test subpaths.
Idempotent; run it wherever the data lives (local or Colab) after download:

    python scripts/fix_data_yaml.py data/child/data.yaml data/hazard/data.yaml

It does NOT touch `nc`/`names` — class definitions are left exactly as
downloaded.
"""

import os
import sys

import yaml  # ships with ultralytics


def fix(path):
    root = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["path"] = root
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"fixed {path}  ->  path: {root}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["data/child/data.yaml", "data/hazard/data.yaml"]
    for p in args:
        fix(p) if os.path.isfile(p) else print(f"skip (not found): {p}")
