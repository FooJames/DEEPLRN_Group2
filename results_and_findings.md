# Results and Findings

Working document for the paper. Numbers here are copied from
`results/metrics/*` — that directory is the source of truth; if the two
ever disagree, trust the CSVs and fix this file.

Distinct from `PROGRESS.md`, which is a session-by-session work log. This
file is organised for the write-up: what we ran, what we got, and what we
can honestly claim.

**Status:** Phases 0–3b complete.
Phases 3c/4/5 blocked — see [Open Items](#8-open-items-and-blockers).

---

## 1. Experimental setup

| | |
|---|---|
| Framework | `ultralytics` **8.4.106** (pinned), PyTorch |
| Model | YOLOv8n (nano), 3.01 M parameters, 8.1 GFLOPs |
| Hardware | Google Colab, NVIDIA Tesla T4 (15 GB) |
| Batch size | 16 (all runs) |
| Evaluation split | **validation only** — the test split is untouched to date |

Both detectors are trained fully independently: separate datasets,
separate weights, no shared backbone, no joint training. Fusion happens
only at the output level.

**Version pinning.** Ultralytics is pinned to 8.4.106 because 8.3.x and
8.4.x differ in detection-head initialisation (8.4 logs `Remapped N/12 cls
head rows from pretrained weights by class name`). The Phase 1 baselines
were trained before pinning, so a small version-drift risk applies to
baseline-vs-later comparisons; everything from Phase 2 onward is pinned
and mutually comparable.

---

## 2. Datasets

Both from Roboflow Universe, using each creator's pre-defined split (no
reshuffling).

| | Child detector | Hazard detector |
|---|---|---|
| Source | `sotukenn/child-detection-piuns` v3 | `harmfull-objects/harmful-objects-wmmdi` v1 |
| Images | 4,705 | 5,917 |
| Classes | 1 (`child`) | 12 |
| Train / val / test | 4,080 / 372 / 253 | 4,758 / 773 / 386 |

Hazard classes: Axe, Chainsaw, Chisel, Coin, Drink, Dumbbell, Fork,
Hammer, Knife, Scissors, Screwdriver, Stapler.

### 2.1 Corrections to the proposal

Two errors in the submitted proposal, confirmed against the downloaded
data:

1. **Image counts were transposed.** The proposal states 5,917 child /
   4,705 hazard. The actual data is **4,705 child / 5,917 hazard**. The
   split percentages in the proposal (87/8/5 child, 80/13/7 hazard) match
   the folders as downloaded, confirming the totals — not the ratios —
   were swapped.
2. **The proposal is internally inconsistent** on the child dataset:
   Methodology says 1,985 images, the Experimental Plan says 5,917.
   Neither is correct for v3.

Also worth correcting in the write-up: the proposal's evaluation section
gives the risk threshold as **700 pixels** (Ahmad et al.), but the
methodology defines distance normalised by image diagonal, i.e. a value in
[0, 1]. Those units are incompatible; 700 px is a conceptual reference
only and cannot seed the normalised threshold.

### 2.2 Data preparation issues found

- **Child labels shipped as 2 classes.** The Roboflow export had
  `nc=2, names=['0','child']`. Inspection of all 127 class-`0` boxes
  (31 distinct scenes) showed they are **children annotated under the
  wrong class index**, not a second object type — child-sized boxes,
  never co-occurring with a class-`1` box. They were **merged** into a
  single `child` class (6,220 boxes total), not deleted; deleting would
  have discarded 95 images' worth of valid annotations and taught the
  model that visible children are background.
- **Roboflow `data.yaml` paths are broken** for both datasets (relative
  `../train/images` with no `path:` key), causing ultralytics to resolve
  against an unrelated global `datasets_dir`. Fixed by pinning an absolute
  `path:` (`scripts/fix_data_yaml.py`). Must be re-run after every fresh
  download, including on Colab.

---

## 3. Phase 1 — Baseline training

100 epochs, `imgsz=640`, default hyperparameters, `optimizer=auto`.

| Detector | mAP50 | mAP50-95 |
|---|---|---|
| Child (1 class) | **0.9469** | 0.8271 |
| Hazard (12 class) | **0.5657** | 0.4072 |

**Finding — `optimizer='auto'` is not a neutral default.** Ultralytics'
`auto` mode explicitly discards any supplied `lr0` and hard-codes AdamW
with `lr = 0.002 × 5/(4 + nc)`. For these datasets that is **6.25e-4**
(hazard, nc=12) and **2.0e-3** (child, nc=1). This matters for
reproducibility — the baselines' effective learning rate is not the
`lr0=0.01` recorded in `args.yaml` — and it invalidates any hyperparameter
sweep run under `auto` (see §4).

**Convergence.** The hazard detector plateaus around epoch 50
(mAP50 = 0.535 at epoch 50 vs 0.533 at epoch 100; best checkpoint 0.5657).
Training beyond ~50 epochs yields no measurable gain on this dataset.

### 3.1 Per-class hazard results (baseline)

| Class | Val images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 773 | 1425 | 0.566 | 0.586 | 0.570 | 0.413 |
| Scissors | 64 | 81 | 0.769 | 0.820 | 0.870 | 0.687 |
| Coin | 155 | 367 | 0.718 | 0.845 | 0.855 | 0.751 |
| Chainsaw | 24 | 24 | 0.605 | 0.625 | 0.730 | 0.490 |
| Knife | 137 | 217 | 0.696 | 0.613 | 0.713 | 0.522 |
| Stapler | 11 | 12 | 0.661 | 0.667 | 0.703 | 0.631 |
| Screwdriver | 8 | 11 | 0.676 | 0.636 | 0.688 | 0.430 |
| Fork | 153 | 270 | 0.673 | 0.659 | 0.644 | 0.376 |
| Hammer | 24 | 27 | 0.433 | 0.667 | 0.525 | 0.355 |
| Dumbbell | 24 | 63 | 0.454 | 0.540 | 0.398 | 0.287 |
| Axe | 29 | 50 | 0.531 | 0.380 | 0.371 | 0.252 |
| Drink | 145 | 298 | 0.348 | 0.399 | 0.284 | 0.128 |
| Chisel | 3 | 5 | 0.223 | 0.179 | 0.055 | 0.048 |

> Produced on ultralytics 8.3.130 while the weights were trained on
> 8.4.106 (overall mAP50 reads 0.570 here vs 0.5657 from training).
> **Regenerate under the pinned version for final paper numbers.**

**Finding — severe validation-set class imbalance.** Support ranges from
367 instances (Coin) to 5 (Chisel), a ~70× spread. Classes with fewer than
~30 validation instances (Chisel, Screwdriver, Stapler, Chainsaw, Hammer,
Dumbbell) have per-class mAP values that are statistically unstable: a
single detection flipping moves them substantially. **These should be
reported as a dataset limitation, not as reliable per-class performance.**
Chisel in particular (3 images, 5 instances) is effectively unmeasurable
and scores near zero in every configuration tested.

---

## 4. Phase 2 — Hyperparameter tuning (hazard)

Tuned `lr0`, `box`, `cls`, `dfl` using ultralytics' evolutionary tuner,
6 iterations × 20 epochs, `imgsz=640`, explicit AdamW. 2.99 h on a T4.

Only the hazard detector was tuned; the child detector was left at
defaults given its 0.947 baseline. This asymmetry is carried through the
later phases and should be stated in the write-up.

**Search-space constraint.** Ultralytics' default tuner evolves ~20
parameters including augmentation settings. It was constrained to exactly
the four parameters the project committed to, so the sweep answers the
question actually posed.

### 4.1 Full sweep

| Iter | lr0 | box | cls | dfl | mAP50 | mAP50-95 | Fitness |
|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 7.500 | 0.500 | 1.500 | 0.2325 | 0.1606 | 0.1606 |
| 2 | 0.01 | 9.325 | 0.544 | 1.514 | 0.2478 | 0.1757 | 0.1757 |
| 3 | 0.0088 | 8.139 | 0.755 | 1.450 | 0.2799 | 0.1946 | 0.1946 |
| **4** | **0.00088** | **8.143** | **0.750** | **1.059** | **0.5020** | **0.3641** | **0.3641** |
| 5 | 0.01 | 7.737 | 0.508 | 1.583 | 0.2448 | 0.1689 | 0.1689 |
| 6 | 0.01 | 8.151 | 0.748 | 1.016 | 0.2709 | 0.1885 | 0.1885 |

**Winner (iteration 4):** `lr0=0.00088, box=8.14272, cls=0.75027, dfl=1.05913`

### 4.2 Findings

**The tuned configuration beats the baseline at equal budget.** At 20
epochs: mAP50 0.502 vs 0.464 (**+0.038**), mAP50-95 0.364 vs 0.324
(**+0.040**).

**`lr0` dominates the other three parameters by ~5×.** Reducing `lr0` from
8.8e-3 to 8.8e-4 gained **+0.170** fitness. The entire spread of
`box`/`cls`/`dfl` across the five runs at `lr0 ≈ 0.01` was only **0.034**.

**Honest limitation: this sweep does not resolve `box`/`cls`/`dfl`.** Five
of six iterations were seeded near ultralytics' default `lr0 = 0.01`,
which is ~16× too high for AdamW, so every configuration that varied the
loss weights was crippled by a bad learning rate. The defensible claim is:
*at a 20-epoch budget, `lr0` is the only one of the four parameters with a
clear effect; differences in `box`/`cls`/`dfl` fall within run-to-run
noise and were not resolved.* A follow-up sweep with `lr0` narrowed to
(1e-4, 2e-3) would be needed to separate them.

**The tuner essentially rediscovered the `auto` learning rate.** The
winning `lr0 = 8.8e-4` is close to the 6.25e-4 that `optimizer=auto`
already selects for a 12-class dataset. This should be stated plainly
rather than presenting the tuning as a large win.

---

## 5. Phase 3a — Input resolution ablation

416 / 640 / 832, 30 epochs, AdamW, one variable changed. Hazard held the
Phase 2 tuned hyperparameters fixed; child held defaults fixed. Verified
via each run's `args.yaml` that only `imgsz` differed within a model.

### 5.1 Hazard (tuned hyperparameters)

| imgsz | Precision | Recall | mAP50 | mAP50-95 | Train (min) | Inference (ms) |
|---|---|---|---|---|---|---|
| 416 | 0.658 | 0.492 | 0.5115 | 0.3861 | 29.6 | **2.27** |
| **640** | 0.714 | 0.459 | **0.5259** | **0.3865** | 45.7 | 3.90 |
| 832 | 0.577 | 0.474 | 0.5003 | 0.3521 | 71.8 | 6.30 |

### 5.2 Child (default hyperparameters)

| imgsz | Precision | Recall | mAP50 | mAP50-95 | Train (min) | Inference (ms) |
|---|---|---|---|---|---|---|
| 416 | 0.944 | 0.817 | 0.9209 | 0.7451 | 27.2 | **3.17** |
| **640** | 0.908 | 0.842 | **0.9338** | **0.7561** | 40.5 | 4.39 |
| 832 | 0.915 | 0.824 | 0.9150 | 0.7225 | 63.3 | 6.88 |

### 5.3 Findings

**640 is optimal for both detectors** on both metrics. → **`imgsz=640`
is locked for Phase 3b and Phase 4.**

**Higher resolution actively hurts.** 832 was worse than 640 on both
models by a near-identical margin (hazard −0.0344, child −0.0336
mAP50-95), while costing ~1.6× more
inference time and ~1.6× more training time. Because the effect replicates
across two independent datasets, it is more credible than a single-model
result. *Caveat:* 30 epochs is a comparison budget, not convergence, and
higher resolutions can require more epochs to pay off — so 832 may be
somewhat understated.

**A compute-efficiency result worth reporting.** For the hazard detector,
416 matches 640 on mAP50-95 to within **0.0003** (mAP50 is 0.014 lower)
while running **1.72× faster** at inference. For a real-time monitoring
application this is a meaningful trade, and it supports the computational
cost analysis promised in the proposal's introduction. 640 remains the
accuracy-optimal choice; 416 is the efficiency option.

**Run-integrity note.** The hazard 832 run was interrupted at epoch 21 and
auto-resumed from checkpoint. It was verified to be a genuine continuous
30-epoch run (epochs 1–30 unbroken; mAP50 rising smoothly across the
boundary, 0.435 → 0.467 → 0.499, rather than dropping back toward the
~0.12 of a fresh restart). Its wall-clock timer reset on resume, so the
true training time is 71.8 min, not the 22.0 min a naive reading of the
last row gives.

---

## 6. Phase 3b — Optimizer ablation

SGD / Adam / AdamW at `imgsz=640` (the Phase 3a winner), 30 epochs, loss
weights held fixed. Verified via `args.yaml` that only the optimizer and
its learning rate differed.

### 6.1 Design: each optimizer gets its own learning rate

Holding a single `lr0` across optimizers appears fair but is not — it
measures which optimizer happens to suit that one rate. Our Phase 2 sweep
quantifies how large that effect is: **the same optimizer (AdamW) scores
0.2325 mAP50 at `lr0=0.01` and 0.5020 at `lr0=8.8e-4`** — a 2× swing from
the learning rate alone. Ultralytics itself follows this convention, with
`optimizer='auto'` selecting `(SGD, 0.01)` or `(AdamW, 0.002×5/(4+nc))`.

Rates used:

| | SGD | Adam / AdamW |
|---|---|---|
| Hazard | 0.01 | 8.8e-4 (Phase 2 tuned) |
| Child | 0.01 | 0.002 (ultralytics auto formula, nc=1) |

**The unit of comparison is therefore "optimizer + the learning rate
appropriate to it", not a pure single-variable change.** This must be
stated in the write-up. It also means SGD is being compared at a generic
default rate against adaptive optimizers at a tuned or dataset-derived
rate — so a loss by SGD should not be read as "SGD is inherently worse".

### 6.2 Hazard (tuned loss weights)

| Optimizer | lr0 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| SGD | 0.01 | 0.661 | 0.482 | 0.5130 | 0.3680 |
| Adam | 8.8e-4 | 0.691 | 0.471 | 0.5181 | 0.3814 |
| **AdamW** | 8.8e-4 | 0.714 | 0.459 | **0.5259** | **0.3865** |

### 6.3 Child (default loss weights)

| Optimizer | lr0 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| SGD | 0.01 | 0.905 | 0.919 | 0.9509 | **0.8196** |
| Adam | 0.002 | 0.948 | 0.893 | 0.9528 | 0.8028 |
| **AdamW** | 0.002 | 0.932 | 0.905 | **0.9547** | 0.8059 |

### 6.4 Findings

**AdamW is selected for Phase 4.** It wins both metrics on the hazard
detector and mAP50 on the child detector. Child mAP50-95 is the one
exception, where SGD leads by 0.014 — worth reporting rather than omitting.
AdamW is also the optimizer under which the Phase 2 hyperparameters were
tuned, so carrying it forward keeps the final configuration internally
consistent.

**Differences between optimizers are small.** Spreads are 0.013 (hazard
mAP50), 0.018 (hazard mAP50-95), 0.004 (child mAP50) and 0.017 (child
mAP50-95). See the variance caveat below before treating these as decisive.

**Reproducibility confirmed, but variance is unmeasured.** Ultralytics runs
with `seed=0, deterministic=True` by default. The Phase 3b AdamW hazard run
reproduced the Phase 3a `imgsz=640` result **exactly** (0.52592 / 0.38645 to
five decimal places), confirming the pipeline is deterministic and the two
ablations are mutually consistent. However, this also means **every result
in this report is a single seed, and no run-to-run variance estimate
exists.** Differences on the order of 0.01–0.02 mAP cannot presently be
distinguished from seed noise. Repeating one configuration across 3 seeds
would be the cheapest way to establish an error bar; without it, the
optimizer ranking should be described as "AdamW was best or tied-best in
every comparison" rather than as a statistically significant result.

### 6.5 Incidental finding: the child detector was undertrained in Phase 3a

The child detector was never tuned, so the Phase 3a resolution ablation ran
it at ultralytics' default `lr0 = 0.01` under AdamW. Phase 3b used the
auto-derived rate for a single-class dataset, `lr0 = 0.002`. With
everything else identical (AdamW, `imgsz=640`, 30 epochs):

| Child, AdamW @640, 30 ep | mAP50 | mAP50-95 |
|---|---|---|
| `lr0 = 0.01` (Phase 3a) | 0.9338 | 0.7561 |
| `lr0 = 0.002` (Phase 3b) | **0.9547** | **0.8059** |
| Difference | **+0.0209** | **+0.0498** |

**Implication for Phase 3a.** The child resolution ablation was internally
consistent — all three resolutions used the same `lr0 = 0.01` — so its
*relative* ranking is not invalid. But it was conducted at a handicapped
operating point, and the resolution ranking has not been confirmed at the
correct learning rate. Re-running the child resolution ablation at
`lr0 = 0.002` (~2 h) would remove this caveat. Until then, the child
`imgsz=640` conclusion should be reported with this qualification. The
hazard resolution ablation is unaffected, as it used the tuned rate
throughout.

**A note on epoch budget.** At the correct learning rate the child detector
reaches mAP50 0.9547 in 30 epochs, *exceeding* its own 100-epoch baseline
of 0.9469, while mAP50-95 (0.8059) remains below the baseline's 0.8271.
mAP50 saturates early; localisation precision continues improving with
longer training.

---

## 7. Cross-phase summary

| Configuration | Optimizer | lr0 | imgsz | Epochs | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **Child** baseline | auto→AdamW | 0.002 | 640 | 100 | 0.9469 | 0.8271 |
| Child, Phase 3a | AdamW | 0.01 | 640 | 30 | 0.9338 | 0.7561 |
| **Child, best to date** | AdamW | 0.002 | 640 | 30 | **0.9547** | 0.8059 |
| Child, best mAP50-95 | SGD | 0.01 | 640 | 30 | 0.9509 | **0.8196** |
| **Hazard** baseline | auto→AdamW | 6.25e-4 | 640 | 100 | 0.5657 | 0.4072 |
| Hazard, Phase 2 tuned | AdamW | 8.8e-4 | 640 | 20 | 0.5020 | 0.3641 |
| **Hazard, best to date** | AdamW | 8.8e-4 | 640 | 30 | **0.5259** | **0.3865** |

**Note on comparing these rows:** the 100-epoch baselines are not directly
comparable to the 20/30-epoch tuning and ablation runs — epoch budget
differs. The tuned configuration has not yet been trained at full length;
that happens in Phase 4, which is the run that determines whether tuning
improves the final model.

**Configuration selected for Phase 4:** YOLOv8n, `imgsz=640`, **AdamW**;
hazard with the Phase 2 tuned hyperparameters (`lr0=8.8e-4, box=8.14272,
cls=0.75027, dfl=1.05913`), child with `lr0=0.002` and default loss
weights. Full epoch budget (100), with the caveat that the hazard detector
plateaus near epoch 50.

---

## 8. Open items and blockers

### 8.1 Blocking: the fusion evaluation set does not yet exist

**This blocks Phases 3c, 4 (threshold calibration), and 5 (risk accuracy)
— i.e. the project's headline contribution.**

The current labelled set (379 images: 279 "unsafe", 100 "safe") assigns
its label by **hazard presence**, not proximity:

- every "unsafe" image is a child *with* a hazard, regardless of distance
- every "safe" image is a child *with no hazard at all* (no distance exists)

Consequently a distance threshold calibrated on this set degenerates to
"any hazard detected → unsafe" — the optimal threshold simply sits above
the maximum observed distance (0.48), and the proximity signal, which is
the actual contribution, becomes untestable. The distance rule's own
suggestion disagrees with the assigned label on **273 of 279** unsafe rows.

**The missing class is "child and hazard both present but far apart =
safe" — currently zero examples.** Manual review of the 24 highest-distance
flagged images found only one genuine far-apart scene; the other seven
(each ×3 augmentations) were the centroid artefact described in §8.2.

Options: (a) composite far-apart co-occurrence frames from the existing
hazard and child datasets, giving exact ground-truth distances by
construction; (b) source real far-apart images; (c) abandon the proximity
claim and report co-occurrence accuracy instead, which forfeits the
contribution.

### 8.2 Supporting evidence for the centroid-vs-edge ablation

Manual inspection produced a concrete motivating example, independent of
any model: in several images a child is bent directly over a small hazard
(coins in hand), but because the child's bounding box fills roughly a
third of the frame, **centroid-to-centroid distance reads ≈ 0.4
("far") while the boxes very nearly touch**. Edge-to-edge distance would
correctly read these as close. This is qualitative support for the
hypothesis behind ablation 3c, and can be used as a figure.

### 8.3 Additional known limitations

- **Evaluation-set composition.** The co-occurrence set contains only 3 of
  the 12 hazard classes (Knife, Scissors, Coin) and is coin-dominated, so
  it does not exercise the hazard detector's full class range.
- **Synthetic images.** Several images in the labelled set are
  AI-generated or low-resolution web scrapes, which weakens claims based on
  a "manually labelled test set".
- **Augmentation leakage risk.** The co-occurrence set contains ~3
  augmented copies per base image. Any val/test split must group by base
  image, or copies of the same scene land on both sides and inflate
  accuracy.
- **Baseline comparison is not a controlled experiment.** Per the agreed
  reframing, the claim is that our specialist detectors *perform
  competitively with published literature numbers* — not that the
  independent design beats a unified model. Ahmad et al.'s ~90% mAP is on
  a different dataset with different hazard classes using YOLOv8**l**,
  while ours is YOLOv8**n**; the numbers are not directly comparable. A
  controlled test would require training a unified model on the union of
  our two datasets as an internal baseline.
- **Greedy search ordering.** Hyperparameters were tuned first, then
  resolution, then optimizer — each conditioned on the previous stage's
  winner. This assumes no strong interactions and should be stated as a
  simplification rather than presented as an orthogonal search.

### 8.4 Remaining work

| Phase | Status | Note |
|---|---|---|
| 3b Optimizer ablation | **Done** | AdamW selected. See §6. |
| — Child 3a re-run at `lr0=0.002` | Recommended | ~2 h. Confirms the child resolution ranking at the correct learning rate (see §6.5). |
| — Seed variance | Recommended | All results are single-seed (`seed=0`, deterministic). 3 seeds on one config would give an error bar (see §6.4). |
| 3c Distance reference | **Blocked** | Needs §8.1 resolved. |
| 4 Final models + fusion | Partly blocked | Detector retraining can proceed; threshold calibration cannot. |
| 5 Evaluation | Blocked | Risk accuracy needs §8.1. Test split still untouched. |
| — Per-class regeneration | Pending | Regenerate §3.1 under pinned 8.4.106. |
| — Computational cost | Partly done | `infer_ms` collected per resolution; still need two-model-vs-one pipeline comparison. |
