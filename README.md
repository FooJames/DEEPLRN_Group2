# Contextual Child Safety Risk Detector

DEEPLRN Group 2 S08, De La Salle University.
Stephen Co, James Foo, Nathaniel Tolentino.

Two **independently trained** YOLOv8n detectors — one for children, one for 12
classes of household hazard — fused after the fact by a distance rule that
labels a frame **Safe** or **Unsafe**. There is no learned model in the fusion
layer and no joint training; that separation is the research question, not an
implementation detail. See `context.md` for the full spec and
`results_and_findings.md` for every number and caveat.

## The final model

| File | Val mAP50 | imgsz |
|---|---|---|
| `models/child_best.pt` | 0.9554 | **416** |
| `models/hazard_best.pt` | 0.5657 | **640** |

The two run at **different input resolutions**. Any inference code must resize
per model, not once per frame. `models/README.md` explains where each came from
and two reproducibility caveats on the hazard weights.

## Setup

**Prerequisites:** Python 3.10 or newer (developed on 3.10.6) and git. No GPU is
needed — the demo runs on CPU, and all training was done on Colab.

Steps 1–4 are all you need to run the model. Steps 5–6 are only for
re-running evaluation or training, which need the datasets.

### 1. Clone

```bash
git clone https://github.com/FooJames/DEEPLRN_Group2.git
```

```bash
cd DEEPLRN_Group2
```

The two weight files are committed (12 MB), so the clone already contains
everything the demo needs.

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

### 3. Install the pinned dependencies

```bash
pip install ultralytics==8.4.106
```

That one package pulls in torch, Pillow and numpy. **Install the exact version**
— it is pinned because 8.3 and 8.4 differ in detection-head initialisation, and
the same weights score differently across them. If you already have another
version, this command will move you to the pin.

pip installs the CPU build of torch by default, which is correct here: training
happens on Colab, and inference does not need CUDA.

### 4. Verify the install

Point the pipeline at any photo:

```bash
python scripts/risk_fusion.py path/to/image.jpg
```

A working install prints a banner and one verdict line per image:

```
[fusion] centroid distance, threshold 0.3625, conf 0.05   [calibrated on the co-occurrence val split]
[fusion] child models\child_best.pt @ 416 | hazard models\hazard_best.pt @ 640
  image.jpg                    UNSAFE d=0.1428  child=3 hazard=12
```

The annotated copy lands in `runs/risk_fusion/`. Any image will do for a smoke
test — the check is that both models load and a verdict comes back, not what
the verdict is. Don't read anything into the label on an arbitrary picture: at
`conf 0.05` the detectors fire on very little evidence, and one of this repo's
own result charts comes back `child=1 hazard=1`.

Setup is done. Everything below is only needed to reproduce the numbers.

### 5. Roboflow API key (only for the datasets)

```bash
cp .env.example .env
```

Fill in `ROBOFLOW_API_KEY` from Roboflow → Settings → API Keys (the *Private*
key). `.env` is gitignored — **never commit the key.**

### 6. Download the datasets

`data/` is gitignored, so the datasets are not in the clone. The downloader
needs one more package, deliberately not in step 3 since the demo doesn't use
it:

```bash
pip install roboflow
```

```bash
python scripts/download_data.py --child-version 3 --hazard-version 1
```

The version numbers are pinned deliberately: a re-version on Roboflow would
otherwise silently change the split.

**After every download — including on Colab — run both fixups:**

```bash
python scripts/fix_data_yaml.py data/child/data.yaml data/hazard/data.yaml
```

```bash
python scripts/normalize_child_labels.py data/child
```

The first pins absolute paths (Roboflow ships broken relative ones); the second
collapses the child export from `nc=2` to `nc=1`. Skipping either produces
training runs that fail or silently mislabel. Expect 4,705 child images
(4080/372/253) and 5,917 hazard images (4758/773/386) — keep the creator's
splits, do not reshuffle.

## Running the risk detector

This is the end-to-end pipeline: both detectors plus the fusion rule, on any
image you point it at.

```bash
python scripts/risk_fusion.py path/to/image.jpg
```

It also takes a folder:

```bash
python scripts/risk_fusion.py path/to/folder/ --out runs/risk_fusion
```

Output looks like this — one line per image, plus an annotated copy written to
`--out` (default `runs/risk_fusion/`, gitignored):

```
[fusion] centroid distance, threshold 0.3625, conf 0.05   [calibrated on the co-occurrence val split]
[fusion] child models\child_best.pt @ 416 | hazard models\hazard_best.pt @ 640
  cooc_00005.jpg               UNSAFE d=0.1428  child=3 hazard=12
  cooc_00006.jpg               SAFE   d=   n/a  child=2 hazard=0  <- no pair detected, Safe by rule
  cooc_00007.jpg               SAFE   d=0.4520  child=1 hazard=1
```

The annotated image carries a Safe/Unsafe banner with the measured distance,
child boxes in blue, hazard boxes in orange with their class name, and a line
joining the closest child–hazard pair — the pair the verdict was actually based
on.

`d` is the smallest child-to-hazard distance in the frame, normalised by the
image diagonal, so it is comparable across resolutions. Unsafe means
`d <= threshold`.

### Options

| Flag | Default | Note |
|---|---|---|
| `--threshold` | `0.3625` | calibrated on the co-occurrence **validation** split (Phase 3c) |
| `--conf` | `0.05` | detector confidence; deliberately low — see below |
| `--metric` | `centroid` | or `edge` (nearest box edges) |
| `--child-imgsz` / `--hazard-imgsz` | `416` / `640` | leave these alone |
| `--out` | `runs/risk_fusion` | annotated images |
| `--no-save` | off | print verdicts only |

Overriding the defaults changes what the demo shows. It does not change any
number in the paper — those come from `results/metrics/`.

### Two things the output will make obvious

**`conf 0.05` is very low, and that is on purpose.** At the default 0.25 the
hazard detector misses 57% of hazards, and a missed hazard is silently scored
Safe. Dropping the confidence trades false hazard boxes for fewer misses; you
will see spurious hazard boxes in cluttered frames. This is the pipeline's real
bottleneck — detection dominates geometry (§8).

**"no pair detected" is not a confident Safe.** If either detector returns
nothing, the fusion rule outputs Safe by definition (`context.md`, rule 1). The
script flags those separately so a detector failure is not read as a safety
judgement.

### Running it without a terminal

`notebooks/colab_risk_fusion.ipynb` does the same thing in Google Colab: clone,
upload photos, see the annotated results inline, download them as a zip. It
needs **no GPU, no Roboflow key and no dataset** — the weights are committed, so
the clone is self-sufficient and a free CPU runtime is enough.

Same script, so the verdicts are identical either way.

## Other entry points

```bash
python scripts/evaluate.py --dry-run
```

Full Phase 5 evaluation rehearsed on the **validation** splits — per-detector
mAP, per-class hazard breakdown, risk accuracy, pipeline cost. Safe to re-run.

```bash
python scripts/ablation_distance_ref.py
```

Centroid vs nearest-edge distance, calibrated on validation. The result is a
**null**: the ranking flips with detector confidence (§8).

Training scripts (`train.py`, `train_final.py`, `tune.py`, `ablation_*.py`) are
meant for Colab T4, not a local machine.

## Cautions

- **Do not run `evaluate.py --confirm-test`.** The test splits were touched
  once, on 2026-07-27, and every reported test number comes from that run.
  Re-running it after changing anything turns the test set into a second
  validation set. The script refuses without the flag; `--dry-run` is the one
  to use.
- **Version pin.** The project pins `ultralytics==8.4.106`. The same weights
  score 0.5657 vs 0.570 mAP50 across 8.3 and 8.4, so metrics you generate
  locally on a different version will not match the recorded ones. Quote
  `results/metrics/`, not a local re-run.
- **CPU is fine for the demo, not for training.** Measured on CPU: ~6 s of
  fixed startup (imports plus loading both models), then ~30 ms per additional
  frame — so pass a folder rather than calling the script once per image. The
  7.24 ms/frame figure in the paper is T4 GPU and is not comparable.

## Where things live

```
models/        the two deliverable weight files + provenance
scripts/       everything runnable; risk_fusion.py is the pipeline demo
results/
  metrics/     every recorded number, as CSV
  figures/     47 figures + an index
  paper_tables.md   10 tables, regenerated from the CSVs
results_and_findings.md   the write-up material, section by section
context.md     the spec
PROGRESS.md    session log; read before starting work
```

`results/paper_tables.md` and the figures are generated by
`scripts/make_report.py` from the CSVs, so the paper cannot drift from the
recorded numbers. Re-run it after any new result.
