#!/usr/bin/env python3
"""
normalize_child_labels.py — collapse the child dataset to one class.

The Roboflow child export (child-detection-piuns v3) ships as nc=2 with
names {0:'0', 1:'child'}. Inspection showed class '0' is just *children*
labeled under the wrong index (95 boxes across 31 scenes, child-sized),
not a distinct object. This remaps EVERY box's class id to 0 and rewrites
data.yaml to nc=1 / names ['child'], so the child detector is genuinely
single-class as the design requires. Malformed lines (not 5 tokens) are
dropped. Idempotent; re-run after any re-download (incl. Colab):

    python scripts/normalize_child_labels.py            # defaults to data/child
    python scripts/normalize_child_labels.py data/child
"""

import glob
import os
import sys

import yaml

SPLITS = ("train", "valid", "test")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/child"
    remapped = dropped = files = 0
    for split in SPLITS:
        for lf in glob.glob(os.path.join(root, split, "labels", "*.txt")):
            with open(lf, encoding="utf-8") as fh:
                lines = [ln.split() for ln in fh if ln.strip()]
            out = []
            for ln in lines:
                if len(ln) != 5:          # class + 4 coords; skip stray/malformed
                    dropped += 1
                    continue
                if ln[0] != "0":
                    remapped += 1
                ln[0] = "0"               # single class -> index 0
                out.append(" ".join(ln))
            with open(lf, "w", encoding="utf-8") as fh:
                fh.write("\n".join(out) + ("\n" if out else ""))
            files += 1

    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["nc"] = 1
    cfg["names"] = ["child"]
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(f"label files processed: {files}")
    print(f"boxes remapped to 'child': {remapped}")
    print(f"malformed lines dropped:   {dropped}")
    print(f"{yaml_path}: nc=1 names=['child']")


if __name__ == "__main__":
    main()
