# PROGRESS.md

Read this first every session — a teammate using a different tool
may have worked on this since your last session. Keep entries to
1-3 lines each, most recent on top.

Detailed results, numbers and caveats live in `results_and_findings.md`.
This file is the terse log plus the operational notes below.

---

## Status: ALL PHASES COMPLETE (0-6)

Deliverables: `models/{child,hazard}_best.pt`, `results/metrics/*.csv`,
`results/paper_tables.md` (10 tables), `results/figures/` (47 figures +
index), `results_and_findings.md` (full write-up material).

Remaining work is **optional polish only**, none of it needed for the paper:
- Phase 2 re-sweep with `lr0` narrowed to (1e-4, 2e-3) so box/cls/dfl get
  explored at a sane rate (~3 h, low expected gain)
- Reproducible hazard retrain — same config stated explicitly (~2.4 h; would
  NOT improve accuracy, see `models/README.md`)
- Seed variance: everything is single-seed (`seed=0`, deterministic), so
  differences of 0.01-0.02 cannot be separated from noise. 3 seeds on one
  config would give an error bar.

## Operational notes (read before running anything)

- **The test splits have been used, once.** Do not re-run `evaluate.py` and
  act on its output — changing anything and re-evaluating turns the test set
  into a second validation set.
- **After every Roboflow download, including on Colab:** run
  `scripts/fix_data_yaml.py` (Roboflow ships broken relative paths) and
  `scripts/normalize_child_labels.py` (child export is nc=2).
- **The detectors use different input resolutions** — child 416, hazard 640.
  Inference code must resize per model, not once per frame.
- **`optimizer='auto'` silently discards `lr0`** and hard-codes AdamW at
  `0.002*5/(4+nc)`. Never use it for anything being compared.
- **Pin `ultralytics==8.4.106`.** 8.3 vs 8.4 differ in detection-head init.
- **Colab: write runs to mounted Drive** (`--project`). `/content` is wiped on
  disconnect, taking resume state with it. Verify bundles with `unzip -l`
  before closing a session.
- Some ablation runs live on a **different Google account's Drive** than
  intended (`drive.mount()` mounts the notebook owner's Drive). Either work as
  that account or move them to the main account's `MyDrive/deeplrn_group2/runs/`.
- `models/*.pt` (12 MB total) are **not** gitignored — decide whether to commit
  the weights or track only `models/README.md`.

---

## Log

**2026-08-04** — Housekeeping: rewrote this file. Entries had grown to 9-10
lines against the stated 1-3, and were out of chronological order. Detail
moved to `results_and_findings.md`; operational gotchas promoted above.

**2026-07-27 — Phase 6 done. Project complete.** `scripts/make_report.py`
generates `results/paper_tables.md` (10 tables) and 47 figures from the CSVs,
so the paper cannot drift from recorded numbers. Re-run after any new result.

**2026-07-27 — Phase 5 done. Test splits touched once.** Child 0.9670 mAP50,
hazard 0.5506. Risk fusion balanced acc **0.5947, 95% CI [0.483, 0.707] —
includes chance**; n=77 is too small to confirm the signal. See §9.

**2026-07-27 — Phase 3c + 4b done.** Centroid vs edge is a **null result**:
the ranking flips with detector confidence. Real finding — detection dominates
geometry (57% of hazards missed at default conf). Calibrated: centroid @
0.3625, conf 0.05. See §8.

**2026-07-27 — Decided: ship the BASELINE hazard weights.** Phase 4a showed
the tuned config is 0.040 mAP50 worse. `models/` created with both
deliverables + provenance README. Two reproducibility caveats documented.

**2026-07-27 — Phase 4a done. Headline: tuning made hazard WORSE** (−0.040
mAP50, worse in 10/12 classes) — the 20-epoch proxy did not transfer to 100.
Child +0.008 and trains 3.7x faster at 416. Pipeline cost 7.24 ms/frame. §7.

**2026-07-27 — Retracted the "+0.038 at equal budget" claim (§4.2).** It was
confounded by the lr schedule: a 20-epoch run is fully annealed at epoch 20
while a 100-epoch run is not. Phase 4a became the first fair test.

**2026-07-27 — Co-occurrence eval set built, then twice corrected.** Composites
hazard crops onto held-out, noise-screened child photos; ground truth is
*reachability*, deliberately not either metric under test. See §11.1.

**2026-07-27 — Investigated "corrupted" child data.** Speckling is documented
augmentation, not damage. But found real leakage: ~17% val / ~21% test are
near-duplicates of train. Measured impact only +0.008 mAP50 — dataset kept. §2.2.

**2026-07-27 — Child imgsz re-run: the ranking INVERTED.** At the correct
lr0=0.002, 416 beats 640 (it was last at lr0=0.01). Direct evidence that a
greedy ablation conditioned on an untuned hyperparameter can be wrong. §5.2b.

**2026-07-27 — Phase 3b done. AdamW selected.** Each optimizer got its own
learning rate (holding one fixed would just measure which optimizer suits it).
Also revealed child was undertrained in 3a at lr0=0.01. §6.

**2026-07-27 — Phase 3a done.** 640 wins for hazard. Child result later
superseded by the re-run above. Fixed two data-integrity issues: blank
`train_min`, and a timer reset that under-reported 832 as 22 min vs 71.8. §5.

**2026-07-26 — Phase 2 done (hazard sweep, 6 iterations).** Winner lr0=8.8e-4
+ tuned loss weights. **lr0 dominates by ~5x; box/cls/dfl unresolved** — 5 of 6
iterations sat at a crippling lr0=0.01. §4.

**2026-07-26 — Pinned `ultralytics==8.4.106`** across all notebooks after
confirming 8.3 vs 8.4 changes head init and shifts the same weights' mAP50
from 0.5657 to 0.570.

**2026-07-26 — Phase 1 complete.** Child 0.9469 / hazard 0.5657 mAP50 (100
epochs, defaults). Per-class hazard table saved; Chisel (3 val images) is a
data limitation, not a model failure.

**2026-07-25 — Phase 0 complete.** Repo scaffolded, both datasets downloaded
and verified, child labels collapsed nc=2 → nc=1 (class '0' was children
mislabelled, merged not deleted), smoke tests pass.

**2026-07-25 — Data prep.** Roboflow `data.yaml` paths are broken for both
datasets; `fix_data_yaml.py` pins an absolute `path:`. context.md's child and
hazard image totals were transposed — corrected.

**2026-07-24 — Plan reviewed** (CLAUDE.md, context.md, proposal.pdf). Two
decisions taken: build a co-occurrence eval set, and soften the baseline claim
to "competitive with published figures". Proposal errata recorded in §2.1.

**2026-07-24 — Early labelling work** (`label_safe_unsafe.py`,
`bulk_label_and_merge.py`) on `datasets/`. That set proved unusable for the
fusion evaluation — its labels track hazard *presence*, not proximity. §11.1.
