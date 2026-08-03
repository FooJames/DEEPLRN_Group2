# Results and Findings

Working document for the paper. Numbers here are copied from
`results/metrics/*` — that directory is the source of truth; if the two
ever disagree, trust the CSVs and fix this file.

Distinct from `PROGRESS.md`, which is a session-by-session work log. This
file is organised for the write-up: what we ran, what we got, and what we
can honestly claim.

**Status:** ALL PHASES COMPLETE (0-6). Test splits evaluated once (§9);
tables and figures generated (§10). Remaining items are optional polish —
see [Open Items](#12-open-items-and-blockers).

---

## 1. Experimental setup

| | |
|---|---|
| Framework | `ultralytics` **8.4.106** (pinned), PyTorch |
| Model | YOLOv8n (nano), 3.01 M parameters, 8.1 GFLOPs |
| Hardware | Google Colab, NVIDIA Tesla T4 (15 GB) |
| Batch size | 16 (all runs) |
| Evaluation split | validation for all development; **test used once**, in Phase 5 (§9) |

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

- **Train/val/test leakage in the child dataset (measured).** Roboflow
  filenames encode the source image (`<source>_jpg.rf.<hash>.jpg`), which
  makes cross-split duplication checkable. 386 source names span the train
  and val/test boundary; verifying by pixel comparison (mean absolute
  difference over resized images, threshold 15) confirms **~17 % of the
  validation split and ~21 % of the test split are near-duplicates of
  training images** — one train/val pair differs by 0.42, i.e. effectively
  the same photograph.

  The effect on reported performance was measured rather than assumed, by
  re-evaluating the baseline weights on a decontaminated subset:

  | Child baseline weights | mAP50 | mAP50-95 |
  |---|---|---|
  | Full val split (372 images) | 0.9548 | 0.8526 |
  | **Clean subset (313 images, duplicates removed)** | **0.9465** | **0.8321** |
  | Inflation attributable to leakage | +0.008 | +0.021 |

  So the leakage is real but its practical effect is small: under one point
  of mAP50 and about two of mAP50-95. Detecting the single class "child" is
  visually redundant enough that memorisation adds little. **The child
  detector's conclusions are therefore robust to this contamination**, and
  the dataset was retained rather than re-split — re-splitting would also
  have violated the project constraint to keep each creator's original
  split. Both figures should be quoted in the paper. (Both rows come from
  the same local run on ultralytics 8.3.130, so the *difference* is valid
  even though the absolute values sit slightly above the 8.4.106 baseline
  in §3.)

  **The hazard dataset is clean** — 3,863 unique sources across 5,917
  images and zero cross-split duplication.

- **Low source diversity in the child dataset — arguably the more serious
  limitation.** The 4,705 child images derive from only **~649 unique
  source photographs** (~7 copies each, from resizing plus the
  salt-and-pepper augmentation Roboflow applied). The effective visual
  diversity is therefore an order of magnitude below what the image count
  suggests, which bounds how strongly generalisation can be claimed —
  independently of, and more consequentially than, the leakage above. The
  hazard dataset is far better in this respect (3,863 sources for 5,917
  images, ~1.5 copies each).

- **Not a defect: the speckling visible in child images.** Roboflow applied
  *salt-and-pepper noise to 5 % of pixels* as the child dataset's
  augmentation (documented in its `README.roboflow.txt`), and both datasets
  were stretched to 640x640. Images that appear heavily blacked out on
  inspection are, on checking, genuinely dark photographs rather than
  corrupted data.

---

## 3. Phase 1 — Baseline training

100 epochs, `imgsz=640`, default hyperparameters, `optimizer=auto`.

| Detector | mAP50 | mAP50-95 |
|---|---|---|
| Child (1 class) | **0.9469** | 0.8271 |
| Hazard (12 class) | **0.5657** | 0.4072 |

> **Read the child figure alongside §2.2.** The child validation split
> contains ~17 % near-duplicates of training images. Re-evaluating the same
> weights on a decontaminated subset gives **0.9465 mAP50 / 0.8321
> mAP50-95**, i.e. the leakage is worth about +0.008 / +0.021. The child
> result stands, but the clean figure is the honest generalisation estimate
> and should be quoted next to it. The hazard split is uncontaminated.

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

**A caution on comparing the tuned run to the baseline.** It is tempting to
compare the tuned run's 20-epoch result (mAP50 0.502) against the baseline's
epoch-20 checkpoint (0.464) and claim +0.038 "at equal budget". **That
comparison is confounded and should not be used.** Ultralytics anneals the
learning rate across the *scheduled* number of epochs, so at epoch 20 the
baseline (a 100-epoch schedule) is still mid-decay at lr = 5.07e-4, while
the tuning run (a 20-epoch schedule) has annealed to roughly 1 % of its lr0
and is fully converged. The shorter run is favoured by the schedule alone,
independently of its hyperparameters. A fair comparison requires training
the tuned configuration for the same 100 epochs as the baseline, which is
Phase 4.

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
already selects for a 12-class dataset. Combined with the schedule confound
above, **there is currently no clean evidence that tuning improved on the
default configuration**; Phase 4 (tuned config at the full 100 epochs
against the 100-epoch baseline) is the first fair test. The defensible
claim from Phase 2 is narrow: a learning rate near 1e-3 is required for
AdamW on this dataset, and the loss weights were not resolved.

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

### 5.2 Child — first attempt at `lr0=0.01` (SUPERSEDED)

> **This table is retained for the record but its conclusion does not hold.**
> The child detector was never tuned, so this ran at ultralytics' default
> `lr0=0.01`, which Phase 3b showed is wrong for AdamW on this dataset. The
> corrected run is §5.2b, and it **reverses the ranking**.

| imgsz | Precision | Recall | mAP50 | mAP50-95 | Train (min) | Inference (ms) |
|---|---|---|---|---|---|---|
| 416 | 0.944 | 0.817 | 0.9209 | 0.7451 | 27.2 | 3.17 |
| 640 | 0.908 | 0.842 | 0.9338 | 0.7561 | 40.5 | 4.39 |
| 832 | 0.915 | 0.824 | 0.9150 | 0.7225 | 63.3 | 6.88 |

### 5.2b Child — corrected at `lr0=0.002` (authoritative)

Identical in every respect except the learning rate, which is the
auto-derived rate for a single-class dataset.

| imgsz | Precision | Recall | mAP50 | mAP50-95 | Train (min) | Inference (ms) |
|---|---|---|---|---|---|---|
| **416** | 0.948 | 0.889 | **0.9626** | **0.8127** | **27.6** | **2.48** |
| 640 | 0.932 | 0.905 | 0.9547 | 0.8059 | 41.0 | 4.89 |
| 832 | 0.915 | 0.897 | 0.9504 | 0.7936 | 63.8 | 6.94 |

Every resolution improved at the corrected rate (+0.021 to +0.042 mAP50,
+0.050 to +0.071 mAP50-95), and **the ranking inverted: 416 now wins both
metrics**, having placed last on mAP50 at the wrong rate.

### 5.3 Findings

**The optimal resolution differs per detector:**

| Detector | Optimal imgsz | mAP50 | mAP50-95 |
|---|---|---|---|
| Child | **416** | 0.9626 | 0.8127 |
| Hazard | **640** | 0.5259 | 0.3865 |

This is legitimate — the two detectors are trained and deployed
independently, so they need not share an input resolution. It is also a
result that could not have been found by ablating only one model.

**For the child detector, 416 wins on accuracy *and* cost.** It is the most
accurate configuration while training in 27.6 min (vs 41.0) and running
inference in 2.48 ms (vs 4.89) — **1.97× faster than 640**. A single-class
detection task on reasonably large subjects evidently does not need the
extra resolution.

**Methodological finding — a sequential ablation can invert.** The child
resolution ranking at `lr0=0.01` (640 best) is the opposite of the ranking
at `lr0=0.002` (416 best). The greedy ordering used here — tune, then
resolution, then optimizer — assumes these choices are approximately
independent, and this is direct evidence that they are not. The practical
lesson, worth stating in the write-up, is that an ablation conditioned on
an untuned hyperparameter can produce a confidently wrong answer; the
resolution result was only trustworthy once the learning rate was correct.

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
cost analysis promised in the proposal's introduction. For hazard, 640
remains accuracy-optimal and 416 is the efficiency option; for child, 416
is simply better on both counts (§5.2b).

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

**Implication for Phase 3a — the re-run was performed, and it overturned
the result.** The child resolution ablation was repeated at `lr0 = 0.002`
(§5.2b). Every resolution improved, and the ranking inverted: 416, which
had placed last on mAP50 at the wrong rate, wins both metrics at the
correct one. The original child `imgsz=640` conclusion is therefore
superseded. The hazard resolution ablation is unaffected, as it used the
tuned rate throughout.

**A note on epoch budget.** At the correct learning rate the child detector
reaches mAP50 0.9547 in 30 epochs, *exceeding* its own 100-epoch baseline
of 0.9469, while mAP50-95 (0.8059) remains below the baseline's 0.8271.
mAP50 saturates early; localisation precision continues improving with
longer training.

---

## 7. Phase 4a - Final models

Both detectors trained for **100 epochs** at the configuration locked by
Phases 2-3 - the same budget as the Phase 1 baselines, making this the
first fair test of whether the tuning helped.

| | imgsz | lr0 | box/cls/dfl | Final mAP50 | Final mAP50-95 | Baseline | Delta |
|---|---|---|---|---|---|---|---|
| Child | 416 | 0.002 | defaults | **0.9554** | **0.8353** | 0.9469 / 0.8271 | **+0.008 / +0.008** |
| Hazard | 640 | 8.8e-4 | tuned | 0.5256 | 0.3913 | 0.5657 / 0.4072 | **-0.040 / -0.016** |

### 7.1 The headline finding: tuning made the hazard detector worse

At equal budget, equal resolution and the same optimizer, the Phase 2
configuration is **0.040 mAP50 below** the untuned baseline. It is worse in
**10 of 12 classes**, including large drops for Chainsaw (-0.174),
Screwdriver (-0.088) and Stapler (-0.060).

This is the answer to a question that could not be settled earlier: the
20-epoch comparison used during tuning suggested the tuned config was ahead
by +0.038, but that comparison was confounded by the learning-rate schedule
(§4.2). With the confound removed, the direction reverses.

**Interpretation - the tuning proxy did not transfer.** Hyperparameters were
selected on 20-epoch runs and applied to 100-epoch training. A configuration
that converges quickly under a short, fully-annealed schedule is not
necessarily the one that trains best over five times as long. The tuned
`dfl` (1.06 vs the default 1.5) and `cls` (0.75 vs 0.5) plausibly favour
early convergence at the cost of final quality.

**Consequence for the delivered system - decision taken.** The **baseline
configuration is shipped as the final hazard detector** (AdamW at
ultralytics' auto-derived lr = 6.25e-4, default loss weights, imgsz 640,
100 epochs), not the tuned one. The tuned weights are retained at
`runs/detect/final_hazard` for the record. Deliverables are:

| File | Config | mAP50 | mAP50-95 |
|---|---|---|---|
| `models/child_best.pt` | imgsz 416, lr0 0.002, default weights | 0.9554 | 0.8353 |
| `models/hazard_best.pt` | imgsz 640, lr0 6.25e-4, default weights | 0.5657 | 0.4072 |

The two detectors run at **different input resolutions**; inference code
must resize per model.

**Two reproducibility caveats on the shipped hazard weights**, both of which
must be stated rather than glossed:

1. That run used `optimizer=auto`, so its `args.yaml` records `lr0: 0.01`
   while the value actually used was **6.25e-4** - reproducing from the
   recorded arguments would use the wrong learning rate (§3).
2. It predates the `ultralytics==8.4.106` pin, so its exact version is
   unrecorded.

Retraining the same configuration explicitly under the pinned version
(~2.4 h) would remove both caveats. It would **not** be expected to improve
accuracy - it is the same configuration, merely stated rather than implied -
and it carries a risk of its own, since 8.4.x changed head initialisation
and the rerun might not land at 0.5657. Documenting the caveats, as done
here and in `models/README.md`, is defensible on its own; the retrain is
optional polish, not a correction.

**Consequence for the ablations.** Phases 3a and 3b held the tuned hazard
config fixed. Their internal comparisons remain valid - every condition
shared the same config - but their absolute numbers sit on a configuration
now known to be sub-optimal, and the resolution and optimizer conclusions
have not been re-verified under the baseline hyperparameters.

### 7.2 Child: a small accuracy gain and a large efficiency gain

The child detector improved by +0.008 on both metrics. That is small, and
comparable in size to the leakage inflation measured in §2.2, so it should
not be over-claimed. **The substantial result is cost**: at `imgsz=416` the
final model trained in **35 minutes versus roughly 130 for the 640 baseline
(3.7x faster)** and runs inference at 3.05 ms/image, while matching or
slightly exceeding baseline accuracy. Note this compares whole
configurations, not a single hyperparameter, since resolution changed too.

### 7.3 Computational cost of the dual-detector design

Measured on a T4 at the final configuration:

| Model | Inference |
|---|---|
| Child (416) | 3.05 ms/image |
| Hazard (640) | 4.19 ms/image |
| **Both per frame** | **7.24 ms** (~138 fps) |

This is the figure the proposal's introduction promises but never planned
to measure. Even running two detectors per frame, the pipeline is far above
real-time on modest hardware, so the cost objection to a dual-specialist
design does not hold at this model scale.

### 7.4 Run integrity

Both runs completed the full 100 epochs with the intended configuration,
verified from `args.yaml`. The child run was interrupted and resumed at
epoch 61; it was confirmed continuous (epochs 1-100 unbroken, mAP50 steady
at 0.948-0.950 across the boundary rather than dropping toward zero as a
fresh restart would). Its wall-clock timer reset on resume, so true training
time is 85.7 min rather than the 35 min recorded for the resumed portion
alone.

---

## 8. Phase 3c - Distance reference: centroid vs nearest edge

Both final detectors were run over the co-occurrence validation split and a
threshold was calibrated for each candidate distance metric. The same
predictions feed both, so only the proximity measure varies.

Reported as **balanced accuracy** (mean of unsafe-recall and safe-recall).
The split is 74 % unsafe, so plain accuracy rewards a constant "unsafe"
predictor with 0.740; balanced accuracy scores any constant predictor 0.500
and is the honest headline.

### 8.1 Result: no stable difference between the two metrics

| Detector confidence | Centroid | Nearest-edge | Difference |
|---|---|---|---|
| 0.25 (ultralytics default) | 0.6683 | 0.6425 | centroid +0.026 |
| 0.05 | 0.6705 | **0.6726** | edge +0.002 |
| 0.01 | 0.6126 | 0.5307 | centroid +0.082 |

**The ranking flips with the detector's confidence threshold** - a parameter
belonging to the detectors, not the fusion layer. No ordering survives that
change, so the honest conclusion is that **the two distance references
perform equivalently on this benchmark**, and the project's original choice
of centroid distance needs no revision.

At the best operating point (conf 0.05) the calibrated thresholds are
**centroid 0.3625** and **edge 0.0747**, with balanced accuracy ~0.67 against
0.500 for a constant predictor - so the fusion rule carries real but modest
signal.

### 8.2 The dominant error source is detection, not geometry

| Detector confidence | Images with no hazard found | Forced "Safe" by the fusion rule |
|---|---|---|
| 0.25 | 70 / 123 | **57 %** |
| 0.05 | 27 / 123 | 22 % |
| 0.01 | 7 / 123 | 6 % |

At the default confidence the hazard detector misses the pasted hazard in
**57 % of images**, and the fusion rule labels those Safe without ever
computing a distance. Under those conditions both metrics score *below* the
trivial baseline on plain accuracy (0.569 and 0.561 versus 0.740).

**This is the practically important finding of Phase 3c: how the distance is
measured matters far less than whether the hazard is detected at all.**
Effort spent refining the geometric rule is misdirected while the detector
misses more than half the hazards. The composited hazards are small (median
44x60 px) and lack lighting or perspective matching, which plausibly
depresses confidence relative to naturally occurring hazards - so the
detection rate here should be read as a property of this benchmark, not a
general figure.

### 8.3 Consequences

- **Phase 4b threshold:** centroid distance at **0.3625**, detector
  confidence **0.05**, calibrated on validation only.
- **Confidence must be reported as a system parameter.** It was left at the
  ultralytics default until this ablation, and it changes balanced accuracy
  by more than either distance metric does.
- The earlier oracle experiment on ground-truth boxes (§12.1) gave edge a
  19-point advantage. That advantage does not survive contact with real
  detections, which is precisely why the ablation was run on predictions.

---

## 9. Phase 5 - Final evaluation on the held-out test splits

Run once, with every parameter fixed in advance from the validation-calibrated
Phase 3c result (centroid distance, threshold 0.3625, detector confidence
0.05). Nothing was tuned or selected here. The test splits were untouched
until this run.

### 9.1 Detectors

| Detector | imgsz | Test mAP50 | Test mAP50-95 | Val mAP50 |
|---|---|---|---|---|
| Child | 416 | **0.9670** | 0.9081 | 0.9554 |
| Hazard | 640 | **0.5506** | 0.3958 | 0.5657 |

The hazard detector generalises cleanly - test is 0.015 below validation,
which is normal. **The child test figure should be read with §2.2 in mind:**
its test split has the *highest* measured contamination (~21 % near-duplicates
of training images, versus ~17 % in validation), which is the most likely
reason test exceeds validation here. It is not evidence of unusually strong
generalisation.

### 9.2 Risk classification - the headline result

| | Value |
|---|---|
| Balanced accuracy | **0.5947** |
| 95 % confidence interval | **[0.483, 0.707]** |
| Chance level | 0.500 - **inside the interval** |
| Unsafe precision / recall | 0.683 / 0.609 |
| Safe recall | 0.581 |
| Confusion | tp 28, fp 13, fn 18, tn 18 |

**With 77 test images, the risk-classification result is not statistically
distinguishable from chance.** The point estimate is above chance and the
validation result (0.6705) was more clearly so, but the confidence interval
on the held-out set includes 0.500 and this must be stated plainly. The
honest claim is that the fusion layer shows a modest positive signal that
this test set is too small to confirm.

**Validation-to-test drop: 0.6705 -> 0.5947 (-0.076).** Two contributing
factors, both measurable:

- **The threshold was calibrated on 123 validation images** and a single
  scalar fitted on a set that size carries real variance. Some of the drop
  is threshold overfitting.
- **Detection got worse on test**: 36 % of test images had no hazard
  detected (versus 22 % on validation at the same confidence), and every one
  of those is forced to "Safe" by the fusion rule regardless of geometry.
  This is the §8.2 finding reappearing, and it accounts for most of the 18
  false negatives.

### 9.3 Per-class hazard detection (test)

| Class | mAP50 | | Class | mAP50 |
|---|---|---|---|---|
| Screwdriver | 0.995 | | Chainsaw | 0.566 |
| Stapler | 0.800 | | Hammer | 0.364 |
| Knife | 0.761 | | Axe | 0.363 |
| Coin | 0.748 | | Drink | 0.289 |
| Fork | 0.697 | | Dumbbell | 0.200 |
| Scissors | 0.655 | | **Chisel** | **0.168** |

The spread mirrors validation-set support (§3.1): classes with very few
annotated instances score erratically in both directions - Screwdriver at
0.995 is as untrustworthy as Chisel at 0.168, both resting on a handful of
instances. Report the well-supported classes (Coin, Fork, Knife, Drink,
Scissors) as the meaningful per-class evidence.

### 9.4 Computational cost

The two-detector pipeline costs **7.24 ms per frame on a T4** (child 3.05 +
hazard 4.19, measured in Phase 4a, §7.3), i.e. roughly 138 fps. The Phase 5
run above was executed on CPU and reports ~70 ms/frame; the T4 figure is the
one to quote. Running two specialist detectors instead of one is therefore
not a meaningful cost objection at this model scale.

### 9.5 Comparison with published work

| System | Model | Evaluated on | Reported |
|---|---|---|---|
| **This work (child)** | YOLOv8n | child-detection-piuns test | **0.967 mAP50** |
| **This work (hazard)** | YOLOv8n | harmful-objects test | **0.551 mAP50** |
| **This work (risk fusion)** | dual YOLOv8n + distance | composited co-occurrence test | **0.595 balanced acc** |
| Ahmad et al. (2025) | YOLOv8**l** | own merged child+hazard set | ~0.90 mAP |
| AlMhdawi et al. (2026) | dual YOLOv8 | fire / engineering sites | n/a |
| Ramadan et al. (2025) | YOLOv11n + pose | hand-to-mouth | 0.92 / 0.74 acc |
| Khan & Dey (2024) | classifier | ChildSUn | n/a |

**These rows are not directly comparable and no ranking should be drawn from
them.** Each system was evaluated on a different dataset with different
classes; Ahmad et al. additionally used YOLOv8-large against our nano, and
Ramadan et al. measured a different task entirely (hand-to-mouth proximity,
not child-hazard proximity). Per the reframing agreed early in the project,
the supported claim is that **independently trained specialist detectors
reach performance competitive with published figures** - the child detector
comfortably so, the 12-class hazard detector below Ahmad et al.'s single
reported number on an easier, larger-model setup. This work does **not**
establish that a dual-specialist design outperforms a unified one; that would
require training a unified model on the union of these two datasets as an
internal control, which was not done.

---

## 10. Phase 6 - Generated tables and figures

Everything the write-up needs is generated from `results/metrics/*.csv` by
`scripts/make_report.py`, so the paper cannot drift from the recorded
numbers. Re-run it after any new result.

| Output | Contents |
|---|---|
| `results/paper_tables.md` | 10 Markdown tables, each with a one-line caveat |
| `results/figures/` | 47 figures + `README.md` index |

**Tables:** datasets and proposal corrections; baseline vs final detectors;
the full hyperparameter sweep; resolution ablation (including the superseded
child run, kept for the record); optimizer ablation; distance-reference
ablation; final test results; per-class hazard; computational cost;
literature comparison.

**Figures.** Five are generated from the CSVs - resolution ablation (the
child inversion is visible as two crossing curves), optimizer ablation,
distance distributions showing the classes genuinely overlap, the risk-fusion
confusion matrix, and per-class hazard. The rest are ultralytics' own charts
copied out of the gitignored `runs/` directory for the four runs that appear
in the paper: training curves, both confusion matrices, P/R/F1/PR curves,
class-distribution plots, and **qualitative prediction-vs-ground-truth
pairs**.

Figure names distinguish what is shipped from what is not: `final_child_*`
and `baseline_hazard_*` are the delivered models, while `baseline_child_*`
and `final_hazard_tuned_*` are comparison-only - the latter being the visual
evidence for §7.1.

**The ablation runs have no charts of their own.** They were trained with
`plots=False` to save GPU time; the CSV-generated figures above cover them.

Per context.md, this phase produces numbers and figures only. Interpretation
stays in this document.

---

## 11. Cross-phase summary

| Configuration | Optimizer | lr0 | imgsz | Epochs | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **Child** baseline | auto→AdamW | 0.002 | 640 | 100 | 0.9469 | 0.8271 |
| Child, Phase 3a | AdamW | 0.01 | 640 | 30 | 0.9338 | 0.7561 |
| Child, Phase 3b @640 | AdamW | 0.002 | 640 | 30 | 0.9547 | 0.8059 |
| **Child, best to date** | AdamW | 0.002 | **416** | 30 | **0.9626** | **0.8127** |
| Child, best mAP50-95 | SGD | 0.01 | 640 | 30 | 0.9509 | **0.8196** |
| **Hazard** baseline | auto→AdamW | 6.25e-4 | 640 | 100 | 0.5657 | 0.4072 |
| Hazard, Phase 2 tuned | AdamW | 8.8e-4 | 640 | 20 | 0.5020 | 0.3641 |
| **Hazard, best to date** | AdamW | 8.8e-4 | 640 | 30 | **0.5259** | **0.3865** |

**Note on comparing these rows:** the 100-epoch baselines are not directly
comparable to the 20/30-epoch tuning and ablation runs — epoch budget
differs. The tuned configuration has not yet been trained at full length;
that happens in Phase 4, which is the run that determines whether tuning
improves the final model.

**Superseded by §7.1:** the hazard row below reflects the Phase 2/3
selection, which Phase 4a showed to be worse than the baseline at full
budget. Retained to show what was selected and why.

**Configuration selected for Phase 4:** YOLOv8n, **AdamW**, full epoch
budget (100), with the caveat that the hazard detector plateaus near
epoch 50. The two detectors use **different input resolutions**, which is
legitimate given they are trained and deployed independently:

| | imgsz | lr0 | box / cls / dfl |
|---|---|---|---|
| Child | **416** | 0.002 | defaults (7.5 / 0.5 / 1.5) |
| Hazard | **640** | 8.8e-4 | 8.14272 / 0.75027 / 1.05913 |

---

## 12. Open items and blockers

### 12.1 The fusion evaluation set (resolved by construction)

**Original problem.** The first labelled set could not test the fusion
layer. Every "safe" image contained NO hazard, so the label tracked hazard
*presence* rather than proximity: a threshold calibrated on it degenerates
to "hazard detected → unsafe", and the distance rule — the actual
contribution — becomes untestable. The distance rule's own suggestion
disagreed with the assigned label on **273 of 279** unsafe rows. The
missing class, "child and hazard both present but far apart = safe", had
**zero** examples. That set also contained only 3 of the 12 hazard classes,
was coin-dominated, and its 279 rows were 3x-augmented copies of just 93
unique scenes.

**Resolution.** `scripts/make_cooccurrence_eval.py` composites the missing
case: a hazard crop is pasted onto a real child image at a controlled
separation, so the geometry is exact by construction.

| | Original set | Composited set |
|---|---|---|
| Unsafe scenes | 93 | 107 |
| **Far-apart safe scenes** | **0** | **93** |
| Hazard classes covered | 3 of 12 | **12 of 12** |
| Splits | none | val 130 / test 70, grouped by source photo |

**Background selection.** Child backgrounds are drawn only from the child
dataset's **held-out splits**, so composites never reuse images the child
detector trained on (§2.2). They are also **screened for salt-and-pepper
noise**: the child export carries noise on 5 % of pixels while hazard crops
carry none, so pasting a clean crop onto a speckled background would be
visually inconsistent and would make the hazard easier to detect than it
should be. Screening cuts the median composite noise level from 0.0037 to
0.00001 and costs no source diversity, because every source photograph has
at least one near-clean copy. Splits are grouped by **source photograph**
rather than by file, so the duplication that contaminated the child dataset
is not reproduced here (verified: zero sources span val and test).

**The ground truth is deliberately not either metric under test.** Phase 3c
compares centroid distance against nearest-edge distance, both normalised
by the image diagonal. Deriving the label from either would make that
predictor win by construction. The label instead uses **reachability**,
measured in units of the child's own body size:

```
reach_ratio = (edge gap between boxes) / (child box height)
unsafe  <=>  reach_ratio <= 0.5          (roughly arm's length)
```

This is scale-relative, while both predictors are diagonal-normalised, so
neither recovers it automatically. Because the fusion rule takes the
minimum over all child-hazard pairs, the ground truth is computed the same
way — against the *closest* labelled child, not the one the hazard was
placed around (17 of 60 images in testing had more than one labelled
child, so this materially matters).

Separation is sampled uniformly over `reach_ratio` in [0, 1.5] so the set
brackets the decision boundary and contains genuinely ambiguous cases,
rather than being trivially separable.

**Oracle check (ground-truth boxes, not detector output).** Applying the
best single threshold directly to the true geometry:

| Predictor | Best achievable accuracy |
|---|---|
| `reach_ratio` (the label's own definition) | 1.000 — sanity check |
| Centroid distance / diagonal | 0.655 |
| **Nearest-edge distance / diagonal** | **0.835** |

Both predictors sit well below 1.000 and the two classes overlap in each,
confirming the benchmark is not a tautology. The 18-point advantage for
edge distance is preliminary support for the Phase 3c hypothesis — but
note these numbers use ground-truth boxes and therefore represent an
upper bound. The actual ablation runs the trained detectors, so detection
error will lower both.

**Limitations to state in the write-up.**
- Composited images test the pipeline's *mechanics* — detection plus
  geometric fusion under known geometry. They do **not** establish that
  proximity predicts real-world danger; that would require real images
  with independent human judgement.
- Pasted crops have visible boundaries and no lighting or perspective
  matching, so they are easier to detect than naturally occurring hazards.
- The reachability threshold (0.5 x child height) is a stated modelling
  assumption, not a measured quantity.
- The child dataset under-labels: frames containing several children often
  annotate only one. Ground truth uses the labelled boxes only.
- **The reachability label is not a constant physical distance.** The child
  dataset annotates a head/face in roughly 40 % of images and a full body in
  57 % (median box aspect ratio 0.61, but the 10th-90th percentile range is
  0.39-1.08). Because the label is defined relative to box *height*,
  "arm's length" corresponds to a much smaller real-world distance for a
  face box than a body box. `--max-aspect 0.7` restricts generation to
  body-like boxes if a consistent physical scale is required; the current
  set does not apply that restriction.

### 12.2 Supporting evidence for the centroid-vs-edge ablation

Manual inspection produced a concrete motivating example, independent of
any model: in several images a child is bent directly over a small hazard
(coins in hand), but because the child's bounding box fills roughly a
third of the frame, **centroid-to-centroid distance reads ≈ 0.4
("far") while the boxes very nearly touch**. Edge-to-edge distance would
correctly read these as close. This is qualitative support for the
hypothesis behind ablation 3c, and can be used as a figure.

### 12.3 Additional known limitations

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
- **Greedy search ordering — known to be imperfect, with a worked example.**
  Hyperparameters were tuned first, then resolution, then optimizer, each
  conditioned on the previous stage's winner. This assumes the choices are
  approximately independent, and §5.2b shows they are not: the child
  resolution ranking inverted once the learning rate was corrected. This
  should be presented as a deliberate simplification made for compute
  reasons, not as an orthogonal search.

  **Specifically, the hazard resolution ablation (§5.1) was run before the
  optimizer ablation (§6) confirmed AdamW, so it inherits the same
  ordering assumption that demonstrably failed for the child detector.**
  It is not affected by the *particular* fault that broke the child run —
  hazard used its Phase 2 tuned learning rate throughout, rather than an
  untuned default — so there is no known error in it. But its `imgsz=640`
  result is strictly conditional on AdamW at `lr0=8.8e-4`, and it has not
  been confirmed under any other optimizer. Re-running it (~2.5 h) was
  considered and deliberately not done, given the compute budget and the
  absence of a specific reason to doubt the result. Readers should treat
  the hazard resolution choice as validated under the final configuration
  only, not as optimizer-independent.

### 12.4 Remaining work

All planned phases are complete. Everything below is optional.

**Completed**

| Item | Result |
|---|---|
| 3a Resolution ablation | Child 416, hazard 640 (§5); child re-run inverted the ranking (§5.2b) |
| 3b Optimizer ablation | AdamW selected (§6) |
| 3c Distance reference | No stable difference; centroid retained (§8) |
| 4a Final models | Tuned hazard config underperforms baseline; baseline shipped (§7) |
| 4b Threshold calibration | centroid 0.3625 @ conf 0.05, validation-calibrated (§8.3) |
| 5 Evaluation | Test splits used once; risk fusion 0.595 balanced, CI includes chance (§9) |
| 6 Tables and figures | 10 tables, 47 figures, generated from the CSVs (§10) |
| Per-class regeneration | `per_class_hazard_final.csv` under pinned 8.4.106 |
| Computational cost | 7.24 ms/frame for both detectors (§7.3) |

**Optional, none required for the write-up**

| Item | Cost | Why it might be worth doing |
|---|---|---|
| Seed variance | ~1.5 h | Every result is single-seed (`seed=0`, deterministic), so differences of 0.01-0.02 cannot be separated from noise. Three seeds on one configuration would give an error bar for the whole paper (§6.4). |
| Phase 2 re-sweep | ~3 h | `lr0` narrowed to (1e-4, 2e-3) so `box`/`cls`/`dfl` are explored at a sane learning rate rather than being swamped (§4.2). Low expected gain; a null result would itself be reportable. |
| Reproducible hazard retrain | ~2.4 h | Same configuration stated explicitly rather than via `optimizer=auto`, under the pinned version. **Would not improve accuracy** and risks not reproducing 0.5657 exactly. The documented caveat is defensible without it (`models/README.md`). |
| Human-labelled co-occurrence set | manual | The only way to settle centroid-vs-edge non-circularly, and the only way to show proximity predicts real-world risk rather than constructed geometry (§11.1, §8). |
