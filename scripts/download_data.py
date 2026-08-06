#!/usr/bin/env python3
"""
download_data.py — pull both training datasets from Roboflow into data/.

Downloads the two INDEPENDENT training datasets (see context.md):
  - child detector : sotukenn / child-detection-piuns  (single class)
  - hazard detector: harmfull-objects / harmful-objects-wmmdi  (12 classes)

into data/child/ and data/hazard/ in YOLOv8 format, preserving the
creator's train/val/test split (do NOT reshuffle).

API key: read from ROBOFLOW_API_KEY (env var, or a gitignored .env in the
repo root). Never hardcode or commit the key. See .env.example.

Version numbers: Roboflow datasets are versioned and you must name the
version whose split you want. Find it on the dataset page -> "Versions"
(the number in the generated download snippet). Pass it explicitly so a
future re-version can't silently change your split:

    python scripts/download_data.py --child-version N --hazard-version M

After download it prints image counts per split and flags any mismatch
against the numbers recorded in context.md (4,705 child / 5,917 hazard).
"""

import argparse
import os
import sys

EXPECTED = {  # from context.md — used only to flag mismatches, not to gate
    "child":  {"images": 4705, "classes": 1},
    "hazard": {"images": 5917, "classes": 12},
}
DATASETS = {
    "child":  {"workspace": "sotukenn",         "project": "child-detection-piuns"},
    "hazard": {"workspace": "harmfull-objects",  "project": "harmful-objects-wmmdi"},
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


def load_env():
    """Minimal .env loader (KEY=VALUE lines) so we don't add a dependency."""
    env_path = os.path.join(os.path.dirname(__file__), os.pardir, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def count_split(dataset_dir):
    """images per split -> dict, plus class count from data.yaml if present."""
    counts = {}
    for split in SPLITS:
        img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.isdir(img_dir):
            counts[split] = sum(
                1 for fn in os.listdir(img_dir)
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
            )
    return counts


def download_one(name, version, api_key, out_root):
    from roboflow import Roboflow  # imported here so --help works without the dep

    meta = DATASETS[name]
    location = os.path.join(out_root, name)
    print(f"\n[{name}] downloading {meta['workspace']}/{meta['project']} "
          f"v{version} -> {location}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(meta["workspace"]).project(meta["project"])
    project.version(version).download("yolov8", location=location, overwrite=True)

    counts = count_split(location)
    total = sum(counts.values())
    print(f"[{name}] splits: {counts}  total={total}")

    exp = EXPECTED[name]["images"]
    if total != exp:
        print(f"[{name}] !! image count {total} != context.md's {exp}. "
              f"Confirm you pulled the right version/split before training.")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child-version", type=int, required=True,
                    help="Roboflow version number of the child dataset")
    ap.add_argument("--hazard-version", type=int, required=True,
                    help="Roboflow version number of the hazard dataset")
    ap.add_argument("--out", default="data",
                    help="output root (default: data/ — gitignored)")
    ap.add_argument("--only", choices=["child", "hazard"],
                    help="download just one of the two")
    args = ap.parse_args()

    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set. Put it in .env (see .env.example) "
                 "or export it. Never commit the key.")

    os.makedirs(args.out, exist_ok=True)
    targets = [args.only] if args.only else ["child", "hazard"]
    versions = {"child": args.child_version, "hazard": args.hazard_version}
    for name in targets:
        download_one(name, versions[name], api_key, args.out)

    print("\nDone. Remember: keep the creator's split, do not reshuffle.")


if __name__ == "__main__":
    main()
