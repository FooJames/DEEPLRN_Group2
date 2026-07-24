# PROGRESS.md

Read this first every session — a teammate using a different tool
may have worked on this since your last session. Keep entries to
1-3 lines each, most recent on top.

---

**2026-07-24** — Bulk-labeled + merged the validation set via `bulk_label_and_merge.py`. `labels.csv` now has a `final_label` (cooccurrence→unsafe, child_only→safe; 21 N/A rows left BLANK for human) and a `flagged_for_review` column. Copied (not moved) images into `datasets/labeled/{unsafe: 279, safe: 100}` (gitignored; originals untouched). 24 cooccurrence rows flagged (dist > 0.3847 = 90th pct) as likely far-apart exceptions — mostly "kid playing with coins" stock photos; still labeled unsafe, flag is advisory. TODO for human: decide the 21 N/A rows + eyeball the 24 flagged. Re-run: `python bulk_label_and_merge.py "datasets/My First Project.v2i.yolov8" "datasets/children.v2i.yolov8"`.

**2026-07-24** — Wrote `label_safe_unsafe.py` (Safe/Unsafe pre-sort helper for manual review; CSV-only output, no dataset modification). Two data findings to know:
- FORMAT MISMATCH: both datasets under `datasets/` are exported as YOLOv8 **polygon/segmentation** labels (`class_id x1 y1 x2 y2 ...`), NOT bbox (`class_id xc yc w h`). Script derives each polygon's bbox centroid so it works either way, but re-export as 'Object Detection' if bbox is wanted for training.
- CO-OCCURRENCE SET = `datasets/My First Project.v2i.yolov8` — 4 classes `[Knife, Scissor, child, coins]` (only 3 hazard types, not the 12-class set), child class named "child" not "toddler", **train split only** (no valid/test), 300 imgs. Composition: 279 child+hazard, 18 hazard-only, 3 child-only. Intended as a **validation** set, effectively all-unsafe. Centroid distance is a poor Safe/Unsafe discriminator here (child box fills ~1/3 frame → centroids read far even when boxes overlap in 35% of imgs) — this is the centroid-vs-edge issue (ablation #4). CHILD-ONLY padding set = `datasets/children.v2i.yolov8` (1 class, train/valid/test, 100 imgs).

**2026-07-24** — Reviewed CLAUDE.md, context.md, proposal.pdf (2 independent agent reviews, findings agree).
Plan approved with open decisions before Phase 0/1: (1) define the child+hazard co-occurrence dataset (source/size/labeling/split) — blocks Phases 3c/4/5; (2) choose baseline story: unified internal control vs. softened literature comparison (Ahmad mAP is different dataset + YOLOv8l vs our v8n); (3) bound the tuning/ablation compute budget; (4) add computational-cost measurement promised in proposal intro.
Proposal errata: child dataset 1,985 vs 5,917 inconsistency; 700px threshold incompatible with normalized d_norm; "Roboflex" typo.
