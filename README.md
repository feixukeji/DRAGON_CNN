# DRAGON_CNN

DRAGON (Data Reduced AGN + Galaxy Optical Network) is a PyTorch pipeline for
classifying multi-band astronomical cutouts. It prepares aligned HDF5 datasets,
creates reproducible splits, trains the DRAGON CNN, calibrates confidence
thresholds, runs inference, and generates EigenGradCAM overlays.

## Workflow

```text
FITS / pair-HDF5 + metadata
             |
             v
  tensors.h5 + info.csv
             |
     +-------+--------+
     |                |
     v                v
train/devel/test  normalization_stats.json
     |                |
     +-------+--------+
             |
             v
   model.pt + best_metrics.json
             |
      +------+------+----------------+
      |             |                |
      v             v                v
predictions.csv  confidence curves  heatmaps
```

The pipeline provides:

- Parallel FITS and pair-HDF5 ingestion.
- Deterministic, class-stratified train/devel/test splits.
- Fixed per-channel asinh normalization estimated from the training split.
- Training from scratch or gradual block-wise transfer learning.
- Devel macro-F1 model selection followed by one final test evaluation.
- Top-2 inference, all-class probabilities, confidence calibration, reports,
  and batched EigenGradCAM.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run module commands from the repository root. Training logs to Weights & Biases:

```bash
wandb login
```

Set `WANDB_MODE=offline` to keep runs local. CUDA is recommended for training,
calibration, and heatmap generation; CPU execution is supported.

## Data contract

Training starts from `raw_info.csv`:

```csv
object_id,class,i,r,g
000001,galaxy,/path/to/000001_i.fits,/path/to/000001_r.fits,/path/to/000001_g.fits
000002,star,/path/to/000002_i.fits,/path/to/000002_r.fits,/path/to/000002_g.fits
000003,agn,/path/to/000003_i.fits,/path/to/000003_r.fits,/path/to/000003_g.fits
```

Required fields are:

- `object_id`: non-empty row identifier.
- One path column per requested band.
- `class`: class name or zero-based class index.

Optional `ra` and `dec` columns enable WCS centering. Without them, FITS rows
use each image's geometric center.

Named labels use `labels.csv`:

```csv
key,value
galaxy,0
star,1
agn,2
```

Values must be contiguous from zero. The mapping defines the classifier output
order, report names, probability columns, and calibrated thresholds.

The remaining invariants are:

- Image height and width must be multiples of 8 and at least 48; 96 × 96 is the
  default.
- Repeated `--bands` or `--band` options define channel order. The same
  order must be used for preparation, normalization, training, and inference.
- Every runtime metadata row has a unique, in-range `h5_index` into
  `tensors/tensors.h5`.
- One training-derived `normalization_stats.json` is reused unchanged.
- Checkpoints are plain PyTorch `state_dict` mappings.
- Reusing a training experiment name replaces that experiment directory.

## Quick start

The examples below use three channels, three classes, and a split slug named
`stratified`.

### 1. Pack cutouts into HDF5

```bash
python -m data_preprocessing.create_cutouts \
  --data-dir /path/to/fits_root \
  --csv-path /path/to/dragon_dataset/raw_info.csv \
  --out-dir /path/to/dragon_dataset/tensors \
  --info-path /path/to/dragon_dataset/info.csv \
  --bands i \
  --bands r \
  --bands g \
  --cutout-size 96 \
  --workers 4
```

Unreadable or misaligned rows are skipped. The command writes float32 images to
`tensors/tensors.h5` and aligned runtime metadata to `info.csv`.

### 2. Create data splits

```bash
python -m data_preprocessing.make_splits \
  --data-dir /path/to/dragon_dataset \
  --info-name info.csv \
  --label-col class \
  --split-slug stratified \
  --train-fraction 0.70 \
  --devel-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 0
```

Splitting is deterministic and stratified by class. Small classes may not
appear in every split.

### 3. Compute normalization statistics

```bash
python -m data_preprocessing.compute_normalization_stats \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --split train \
  --channels 3 \
  --low-pct 0.5 \
  --high-pct 99.5 \
  --asinh-softening 0.1
```

The command writes `normalization_stats.json` in the dataset directory.

### 4. Train

```bash
python -m train.train \
  --project dragon \
  --experiment baseline \
  --data-dir /path/to/dragon_dataset \
  --run-dir /path/to/dragon_runs \
  --split-slug stratified \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 3 \
  --epochs 40 \
  --batch-size 16 \
  --optimizer sgd
```

The best checkpoint and final metrics are written to
`/path/to/dragon_runs/baseline`:

```text
baseline/
├── model.pt
├── best_metrics.json
└── wandb/
```

### 5. Calibrate confidence thresholds

```bash
python -m scripts.evaluate_confidence_curves \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --split devel \
  --target-fpr 0.001
```

Artifacts default to
`/path/to/dragon_runs/baseline/confidence_curves`.

### 6. Report training metrics

```bash
python -m scripts.report_training_results \
  /path/to/dragon_runs/baseline \
  --data-dir /path/to/dragon_dataset
```

Pass a run root instead of one experiment directory to report all immediate
child experiments.

After steps 1–3, the dataset layout is:

```text
dragon_dataset/
├── raw_info.csv
├── info.csv
├── labels.csv
├── normalization_stats.json
├── splits/
│   ├── stratified-train.csv
│   ├── stratified-devel.csv
│   └── stratified-test.csv
└── tensors/
    └── tensors.h5
```

## Model

DRAGON accepts `(batch, channels, height, width)` tensors and reads features
from four scales:

| Stage | Default channels | Stride | Receptive field |
| --- | ---: | ---: | ---: |
| `layer1` | 24 | 1 | 11 px |
| `layer2` | 64 | 2 | 26 px |
| `layer3` | 128 | 4 | 56 px |
| `layer4` | 96 | 8 | 116 px |

The backbone uses replicate-padded convolutions, anti-aliased 2× downsampling,
and identity-initialized residual blocks. Each stage feeds a concentric-ring
pooling head with fixed 0.5, 1, 2, and 4 arcsec boundaries at the HSC pixel
scale. The pooled mean/max features pass through BatchNorm, dropout, one hidden
layer, and the classifier.

`widths` controls the four backbone widths; `head_width` controls the hidden
head width. The model consumes image pixels only: seeing, redshift, and explicit
brightness invariance are not part of the architecture.

## Training behavior

Training defaults to:

- Seed 42.
- Eight-way right-angle rotation/reflection augmentation.
- SGD, 40 epochs, batch size 16, learning rate `5e-7`.
- Global gradient clipping at L2 norm 1.0.
- A per-iteration cosine learning-rate schedule.
- Unweighted cross-entropy.

Use `--no-augment`, `--no-scheduler`, or `--max-grad-norm 0` to disable
those features. `--class-weight balanced` applies inverse-frequency class
weights. AdamW is available with `--optimizer adamw`; see `--help` for its
beta, epsilon, and weight-decay options.

Training preloads and normalizes the complete HDF5 image array. Approximate host
memory for that array is:

```text
N × channels × height × width × 4 bytes
```

Linux DataLoader workers share the allocation through copy-on-write. Use
`--n-workers 0` on platforms without `fork`.

Devel metrics are computed after every epoch. `model.pt` tracks the best devel
macro-F1 checkpoint, which is reloaded for exactly one test evaluation.
`best_metrics.json` stores the selected epoch, devel/test metrics, and
confusion matrices.

## Transfer learning

Adapt a checkpoint if its input channel count or classifier head differs:

```bash
python -m scripts.convert_model_channels \
  --in-model /path/to/pretrained/model.pt \
  --out-model /tmp/dragon-model-adapted.pt \
  --target-channels 3 \
  --target-classes 3
```

The first convolution is resized. The complete head is rebuilt only when its
stored tensors no longer match the requested model; initialization is
reproducible under `--seed`.

```bash
python -m train.train \
  --project dragon-transfer \
  --experiment transfer-baseline \
  --data-dir /path/to/dragon_dataset \
  --run-dir /path/to/dragon_runs \
  --split-slug stratified \
  --channels 3 \
  --n-classes 3 \
  --model-state /tmp/dragon-model-adapted.pt \
  --transfer-learn \
  --unfreeze-warmup-epochs 3 \
  --unfreeze-blocks-per-epoch 1 \
  --lr0 2e-5
```

Transfer learning starts with the head only, then unfreezes complete blocks from
`layer4` toward `layer1`. Frozen BatchNorm state remains fixed.

## Alternative data sources

### Training catalog API

`prepare_training_catalog` builds `raw_info.csv` and `labels.csv` from one
catalog per class:

```python
from pathlib import Path

from data_preprocessing.prepare_training import ClassSpec, prepare_training_catalog

class_names = ("galaxy", "star", "agn")
result = prepare_training_catalog(
    class_specs=[
        ClassSpec(
            name=name,
            csv_path=Path(f"/path/to/catalogs/{name}.csv"),
            cutout_dir=Path(f"/path/to/cutouts/{name}"),
        )
        for name in class_names
    ],
    bands=("i", "r", "g"),
    output_dir=Path("/path/to/dragon_dataset"),
    class_order=class_names,
)
```

Each catalog needs `object_id`. FITS paths are derived as
`<cutout_dir>/<object_id>_<band>.fits`, and output IDs receive a class prefix.
When `ra` and `dec` are present, coordinate-equivalent duplicate rows are
removed.

### Pair-HDF5 rows

`create_cutouts` can mix FITS rows with rows that contain:

| Column | Meaning |
| --- | --- |
| `tensor_store` | Source HDF5 path, absolute or relative to `--data-dir`. |
| `tensor_index` | Non-negative row index in `images`. |
| `tensor_object_id` | Expected value at the same row in `object_ids`. |

The source store must contain float32 `images` shaped
`(N, channels, size, size)`, aligned `object_ids`, and a comma-separated
`bands` attribute matching the requested order.

## Inference

Prepare a separate HDF5 catalog for each inference set:

```bash
python -m data_preprocessing.prepare_inference \
  --catalog /path/to/catalog.csv \
  --cutout-dir /path/to/fits_cutouts \
  --output-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --band i \
  --band r \
  --band g \
  --cutout-size 96 \
  --workers 4
```

The input catalog needs `object_id`; FITS files are resolved as
`<object_id>_<band>.fits`. Preparation rebuilds the destination tensors and
removes stale prediction, label, normalization, and heatmap artifacts there, so
use a dedicated output directory.

Run prediction:

```bash
python -m modules.inference \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --output-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --labels-path /path/to/dragon_dataset/labels.csv \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 3 \
  --batch-size 256 \
  --all-probabilities
```

`predictions.csv` contains top-1 and top-2 numeric labels and confidences.
With labels enabled it also contains class names; `--all-probabilities` adds
`probability__<class-index>` columns. Use `--no-labels` for numeric-only
output.

Probability values retain enough precision for stable comparisons with
calibrated thresholds.

## Confidence calibration

Calibration treats every non-target class as a separate negative population.
For each target class it chooses the strictest threshold needed to keep every
negative class below the requested empirical FPR. Ties are kept together under
the manifest's `>=` comparison.

The default `confidence_curves/` output contains:

- `scores.csv`: true labels and all softmax columns.
- `curves.csv`: aggregate sampled curves.
- `negative_class_fpr.csv`: per-target/per-negative FPR.
- `thresholds.json`: thresholds, counts, metrics, and provenance.
- Aggregate and per-class diagnostic PNGs.

Apply each manifest threshold to the probability column with the same numeric
class index. These one-vs-rest gates are independent, so a row may pass zero,
one, or multiple classes. Prefer the devel split for calibration.

## EigenGradCAM

```bash
python -m modules.heatmap \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --output-dir /path/to/dragon_runs/baseline/inference/catalog-name/heatmaps \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 3 \
  --batch-size 256 \
  --output-workers 4
```

Heatmaps target `layer3` and use the predicted class. Three-channel input is
rendered positionally as RGB; other inputs use channel 0 as grayscale.
`object_id` values must be unique and filename-safe. Existing PNGs in the
output directory are removed before generation.

## Command reference

Every command supports `--help`.

| Command | Purpose |
| --- | --- |
| `python -m data_preprocessing.create_cutouts` | Build aligned HDF5 tensors. |
| `python -m data_preprocessing.make_splits` | Create stratified splits. |
| `python -m data_preprocessing.compute_normalization_stats` | Estimate normalization values. |
| `python -m data_preprocessing.prepare_inference` | Build an inference dataset. |
| `python -m train.train` | Train or transfer-learn a model. |
| `python -m scripts.convert_model_channels` | Adapt a checkpoint. |
| `python -m scripts.report_training_results` | Report local metrics. |
| `python -m scripts.evaluate_confidence_curves` | Calibrate confidence thresholds. |
| `python -m modules.inference` | Write predictions. |
| `python -m modules.heatmap` | Generate EigenGradCAM overlays. |

Repository packages are organized by purpose:

- `cnn/`: model definition.
- `data_preprocessing/`: catalogs, tensors, splits, and normalization.
- `train/`: training and transfer learning.
- `modules/`: inference and heatmaps.
- `scripts/`: reporting, calibration, and checkpoint conversion.
- `utils/`: shared model, label, optimizer, device, and tensor helpers.
