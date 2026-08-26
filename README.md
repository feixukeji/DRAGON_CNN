# DRAGON_CNN

DRAGON (Data Reduced AGN + Galaxy Optical Network) is a PyTorch pipeline for
multi-class classification of astronomical image cutouts. It covers the full
workflow from multi-band FITS files to an aligned HDF5 dataset, reproducible
data splits, training, evaluation, inference, and EigenGradCAM visualizations.

## Features

- An eight-block convolutional network for fixed `96 × 96` cutouts.
- Configurable input channels and output classes.
- Parallel FITS ingestion with center cropping or zero padding.
- Optional ingestion of rows from compatible pair-HDF5 tensor stores.
- Deterministic, class-stratified train/devel/test splits.
- Per-channel global asinh normalization estimated from the training split.
- Training from scratch or gradual block-wise transfer learning.
- SGD and AdamW optimizers, balanced class weights, augmentation, gradient
  clipping, cosine learning-rate scheduling, and W&B logging.
- Best-model selection on devel macro-F1 followed by a single test evaluation.
- Top-2 inference, local metrics reports, and batched EigenGradCAM heatmaps.

## Workflow

~~~text
training catalogs + FITS / pair-HDF5
                  |
                  v
             raw_info.csv
                  |
                  v
      tensors/tensors.h5 + info.csv
                  |
          +-------+--------+
          |                |
          v                v
   train/devel/test   normalization_stats.json
          |                |
          +-------+--------+
                  |
                  v
         model.pt + best_metrics.json
                  |
          +-------+--------+
          |                |
          v                v
    predictions.csv   EigenGradCAM PNGs
~~~

The examples keep dataset assets and experiment outputs separate.
`labels.csv`, `normalization_stats.json`, split CSVs, and `tensors.h5` live in
the dataset directory; `model.pt`, `best_metrics.json`, W&B files, predictions,
and heatmaps live below an experiment directory.

## Core contracts

| Contract | Requirement |
| --- | --- |
| Image shape | Training, inference, and heatmaps accept `96 × 96` images. The channel count is configurable. |
| Channel order | Repeated `--bands` and `--band` options define tensor channel order. That positional order must match the normalization statistics and checkpoint; later stages receive only the channel count and cannot validate band semantics. |
| Class mapping | `labels.csv`, `--n-classes`, and the checkpoint classifier head must describe the same classes in the same order. |
| Row alignment | Every runtime metadata row contains one unique, in-range integer `h5_index` pointing into `tensors/tensors.h5`. |
| Normalization | One `normalization_stats.json` is computed from the training split and reused without modification by training, inference, and heatmaps. |
| Checkpoints | Model files are plain PyTorch `state_dict` mappings. Wrapped checkpoints and keys prefixed with `module.` are not accepted. |
| Experiment names | Reusing an experiment name replaces that experiment directory and everything below it. |

## Installation

Python 3.10 or newer is required. Create an environment and install the
dependencies from the repository root:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Run all module commands from the repository root so that `cnn`,
`data_preprocessing`, `train`, `modules`, `scripts`, and `utils` are importable.

Training initializes a W&B run. Authenticate for online logging:

~~~bash
wandb login
~~~

To keep W&B data local, select offline mode before training:

~~~bash
export WANDB_MODE=offline
~~~

A CUDA GPU is strongly recommended for training and heatmap generation. CPU
execution is supported but can be substantially slower.

## Quick start

The following example uses three channels (`i`, `r`, `g`) and three classes
(`galaxy`, `star`, `agn`). These values are illustrative; use the channel and
class definitions appropriate for your dataset. The CSV below is a schema
excerpt rather than a complete trainable dataset; each class needs enough rows
to produce usable train, devel, and test splits.

### 1. Create the training metadata

Prepare `/path/to/dragon_dataset/raw_info.csv`:

~~~csv
object_id,class,i,r,g
000001,galaxy,/path/to/fits/000001_i.fits,/path/to/fits/000001_r.fits,/path/to/fits/000001_g.fits
000002,star,/path/to/fits/000002_i.fits,/path/to/fits/000002_r.fits,/path/to/fits/000002_g.fits
000003,agn,/path/to/fits/000003_i.fits,/path/to/fits/000003_r.fits,/path/to/fits/000003_g.fits
~~~

The required columns are:

- `object_id`: a non-empty object identifier.
- One column per input channel, containing an absolute FITS path or a path
  relative to `--data-dir`.
- `class`: the training label expected by the training entry point.
  `--label-col` only selects the column used by the split command.

Create `/path/to/dragon_dataset/labels.csv`:

~~~csv
key,value
galaxy,0
star,1
agn,2
~~~

`value` must be a contiguous zero-based sequence, and the number of rows must
equal `--n-classes`. The `key` values must match the named labels in the
metadata. Without `labels.csv`, `class` values must already be integer indices
in `[0, n_classes - 1]`. Class names in reports and inference output require
`labels.csv`.

### 2. Build the HDF5 tensor store

~~~bash
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
~~~

Each readable FITS image is converted to `float32`, centered, and cropped or
zero-padded to `96 × 96`. A row is skipped if any requested FITS channel cannot
be read. The resulting `info.csv` therefore contains only successful rows and
adds a fresh, contiguous `h5_index` aligned with `tensors/tensors.h5`.

### 3. Create the dataset splits

~~~bash
python -m data_preprocessing.make_splits \
  --data-dir /path/to/dragon_dataset \
  --info-name info.csv \
  --label-col class \
  --split-slug stratified \
  --train-fraction 0.70 \
  --devel-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 0
~~~

Splitting is deterministic and performed independently within each class. The
three fractions must sum to one. Very small classes may not place a sample in
every split, so inspect the generated CSVs before training.

### 4. Compute training-split normalization

~~~bash
python -m data_preprocessing.compute_normalization_stats \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --split train \
  --channels 3 \
  --low-pct 0.5 \
  --high-pct 99.5 \
  --asinh-softening 0.1 \
  --output /path/to/dragon_dataset/normalization_stats.json
~~~

The command samples pixels with bounded memory and stores per-channel `vmin`
and `vmax` together with the asinh softening value. This is the only stage
where those normalization parameters are chosen.

At this point the dataset has the following layout:

~~~text
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
~~~

### 5. Train a model

~~~bash
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
~~~

The selected checkpoint and metrics are written to
`/path/to/dragon_runs/baseline`.

## Training catalog API

`data_preprocessing.prepare_training.prepare_training_catalog` is a Python API
for integrations that maintain one catalog per class. It creates
`raw_info.csv` and `labels.csv` together:

~~~python
from pathlib import Path

from data_preprocessing.prepare_training import (
    ClassSpec,
    prepare_training_catalog,
)

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

print(result.raw_info_path)
print(result.labels_path)
~~~

Each class catalog must contain `object_id`. FITS files are resolved as
`<cutout_dir>/<object_id>_<band>.fits`, and output object IDs receive a class
prefix to make them unique. When `ra` and `dec` are available, rows sharing an
original `object_id` are compared by angular separation; every row in a
coordinate-equivalent duplicate component is discarded.

`class_order` is required. It must list every `ClassSpec.name` exactly once and
determines the zero-based mapping written to `labels.csv`. Set `--n-classes`
from that generated mapping.

## Pair-HDF5 inputs

Cutout creation can combine ordinary FITS-backed rows with rows that reference
an existing pair-HDF5 store. A tensor-backed row supplies all three columns:

| Column | Meaning |
| --- | --- |
| `tensor_store` | HDF5 path, absolute or relative to `--data-dir`. |
| `tensor_index` | Non-negative row index in the source `images` dataset. |
| `tensor_object_id` | Object ID expected at the same row in the source `object_ids` dataset. |

The source store must provide:

- `images` with dtype `float32` and shape `(N, C, 96, 96)`.
- `object_ids` aligned with the first dimension of `images`.
- A comma-separated `bands` HDF5 attribute matching the requested channel
  names and order exactly.

Tensor shape, index, band order, object ID, and finite values are validated
before publication. FITS and tensor source columns are removed from the final
runtime metadata after `h5_index` is assigned.

## Model

DRAGON accepts `(batch, channels, 96, 96)` tensors. Its feature extractor
contains eight `Conv2d → BatchNorm2d → LeakyReLU → AvgPool2d` blocks with
output widths:

~~~text
64 → 96 → 128 → 256 → 384 → 384 → 384 → 512
~~~

The final `512 × 2 × 2` feature map is flattened and passed through a
`2048 → 1024` fully connected layer, LeakyReLU, dropout with probability
`0.5`, and the output classifier. A three-channel, six-class instance has
7,881,894 parameters.

## Training behavior

The training entry point selects CUDA when available and otherwise uses the
CPU. Training uses one device; it does not use DataParallel or distributed
training. CUDA runs use bfloat16 automatic mixed precision through PyTorch
Ignite.

Before the first epoch, the complete HDF5 `images` dataset is read into one
C-contiguous `float32` array, normalized in bounded blocks on the selected
device, and shared by the train, devel, and test datasets. With
`--n-workers > 0`, workers share this allocation through `fork`
copy-on-write; use `--n-workers 0` on platforms without `fork`. Approximate
array memory is:

~~~text
N × channels × 96 × 96 × 4 bytes
~~~

Budget additional memory for the model, batches, workers, HDF5 caches, and
framework overhead. The preload is the main reason training requires much more
host memory than inference.

Training defaults include:

- Seed `42` for model initialization, shuffling, augmentation, dropout, and
  DataLoader worker RNGs.
- One random transform from the eight right-angle rotation/horizontal-flip
  combinations for each training sample; disable with `--no-augment`.
- SGD, 40 epochs, batch size 16, learning rate `5e-7`, and global gradient
  clipping at L2 norm `1.0`.
- A cosine learning-rate schedule stepped per iteration; disable with
  `--no-scheduler`.
- Unweighted cross-entropy; use `--class-weight balanced` to derive inverse-
  frequency weights from the training split.

The seed controls every RNG initialized by the training pipeline, but exact
bitwise reproducibility can still depend on the PyTorch, CUDA, and hardware
stack.

AdamW can be selected without changing code:

~~~bash
python -m train.train \
  --project dragon \
  --experiment adamw \
  --data-dir /path/to/dragon_dataset \
  --run-dir /path/to/dragon_runs \
  --split-slug stratified \
  --channels 3 \
  --n-classes 3 \
  --optimizer adamw \
  --lr0 3e-5 \
  --weight-decay 1e-4
~~~

AdamW also exposes `--adamw-beta1`, `--adamw-beta2`, and `--adamw-eps`.
Bias and normalization parameters are excluded from weight decay.

### Model selection and outputs

Devel metrics are computed after every epoch. The checkpoint is updated only
when devel macro-F1 improves. After training, that checkpoint is reloaded and
the test split is evaluated exactly once. The test split is never used for
model selection.

Training loss is accumulated from the batches already used for optimization;
the trainer does not make a second pass over the training set. W&B receives
epoch-level training loss, devel metrics, learning rate, transfer-learning
state, final confusion matrices, and best-model summary values.

Each experiment publishes one checkpoint:

~~~text
dragon_runs/
└── baseline/
    ├── model.pt
    ├── best_metrics.json
    └── wandb/
~~~

`model.pt` is the plain `state_dict` from the best completed epoch.
`best_metrics.json` stores the best epoch, its accumulated training loss and
devel metrics, the final test metrics, and devel/test confusion matrices.

> **Warning:** starting training with an existing `--experiment` name deletes
> and recreates `<run-dir>/<experiment>` after input validation. This also
> removes inference products stored below that experiment.

## Transfer learning

The loader expects the plain `state_dict` written as `model.pt`. If a source
checkpoint has a different input-channel count or classifier size, adapt it
first:

~~~bash
python -m scripts.convert_model_channels \
  --in-model /path/to/pretrained/model.pt \
  --out-model /tmp/dragon-model-adapted.pt \
  --target-channels 3 \
  --target-classes 3
~~~

`--target-channels` adapts the first convolution. `--target-classes` resets the
classifier when its output size changes. The output parent directory must
already exist.

Run gradual transfer learning with the adapted checkpoint:

~~~bash
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
~~~

Transfer learning initially trains `fc1` and `fc2` while all convolutional
blocks are frozen. After the warmup, complete blocks are unfrozen from
`layer8` toward `layer1`. Frozen blocks remain in evaluation mode so their
BatchNorm parameters and running statistics do not change.

## Training reports

The local reporter reads `best_metrics.json` directly and does not depend on
W&B history. Pass either one experiment directory or a run root whose immediate
child directories are experiments:

~~~bash
python -m scripts.report_training_results \
  /path/to/dragon_runs \
  --data-dir /path/to/dragon_dataset

python -m scripts.report_training_results \
  /path/to/dragon_runs/baseline \
  --data-dir /path/to/dragon_dataset \
  --split test
~~~

Use `--labels /path/to/labels.csv` to override
`--data-dir/labels.csv`. A label mapping is required. The report includes
overall, macro, weighted, and per-class metrics plus confusion matrices shown
as absolute counts and percentages of each actual-class row.

## Inference

Inference has two stages: build an HDF5 catalog and run prediction. Keep each
catalog in its own directory below the experiment.

### 1. Prepare an inference catalog

The input CSV must contain a non-empty `object_id` column. FITS files must use
the naming convention `<object_id>_<band>.fits`:

~~~bash
python -m data_preprocessing.prepare_inference \
  --catalog /path/to/catalog.csv \
  --cutout-dir /path/to/fits_cutouts \
  --output-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --band i \
  --band r \
  --band g \
  --cutout-size 96 \
  --workers 4
~~~

The order of repeated `--band` options defines the channel order and must match
training. The command writes `info.csv` and `tensors/tensors.h5`, retaining
other input catalog columns in the aligned metadata.

> **Warning:** use a dedicated output directory. Preparation rebuilds its
> tensors and removes reserved stale artifacts there, including
> `raw_info.csv`, `predictions.csv`, `summary_counts.csv`, `labels.csv`,
> `normalization_stats.json`, `predictions/`, and `heatmaps/`.

### 2. Predict

~~~bash
python -m modules.inference \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --output-dir /path/to/dragon_runs/baseline/inference/catalog-name \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --labels-path /path/to/dragon_dataset/labels.csv \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 3 \
  --batch-size 256
~~~

The output `predictions.csv` contains the aligned catalog columns plus:

| Column | Meaning |
| --- | --- |
| `predicted_labels` | Top-1 numeric class index. |
| `predicted_confidence` | Top-1 softmax probability. |
| `second_predicted_labels` | Top-2 numeric class index. |
| `second_predicted_confidence` | Top-2 softmax probability. |
| `predicted_class` | Top-1 class name when labels are enabled. |
| `second_predicted_class` | Top-2 class name when labels are enabled. |

Use `--no-labels` and omit `--labels-path` to produce numeric indices only.
Inference streams HDF5 batches instead of preloading the entire tensor store.
When multiple CUDA devices are available, `--parallel` allows inference to use
`DataParallel`.

## EigenGradCAM heatmaps

Generate one overlay per inference row using the same checkpoint and
normalization statistics:

~~~bash
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
~~~

The implementation targets `layer4`, uses the model's predicted class, and
computes batched principal-component projections on the model device. PNG
rendering is parallelized with `--output-workers`.

For three-channel input, tensor positions 0/1/2 are rendered as R/G/B; this
corresponds to `i/r/g → R/G/B` only when `i/r/g` was the preparation order.
Other channel counts use position 0 as a grayscale background. Every
`object_id` must be unique, non-empty, and safe as a filename; output files are
named `<object_id>.png`.

> **Warning:** heatmap generation removes existing `*.png` files from its
> output directory before writing the new set.

## Command reference

Every command provides full option documentation through `--help`:

| Command | Purpose |
| --- | --- |
| `python -m data_preprocessing.create_cutouts` | Build aligned HDF5 tensors from FITS and pair-HDF5 inputs. |
| `python -m data_preprocessing.make_splits` | Create deterministic stratified split CSVs. |
| `python -m data_preprocessing.compute_normalization_stats` | Estimate and save training-split normalization values. |
| `python -m data_preprocessing.prepare_inference` | Build one inference catalog and tensor store. |
| `python -m train.train` | Train from scratch or perform gradual transfer learning. |
| `python -m scripts.convert_model_channels` | Adapt checkpoint input channels and classifier size. |
| `python -m scripts.report_training_results` | Print metrics from one experiment or run root. |
| `python -m modules.inference` | Write top-2 predictions for an inference catalog. |
| `python -m modules.heatmap` | Generate batched EigenGradCAM overlays. |

`prepare_training_catalog` is an integration API rather than a command-line
entry point.

## Repository layout

| Path | Purpose |
| --- | --- |
| `cnn/` | DRAGON model definition and model statistics. |
| `data_preprocessing/` | Catalog validation, FITS/HDF5 assembly, datasets, splits, normalization, and inference preparation. |
| `train/` | Training orchestration, Ignite trainers, model selection, and gradual unfreezing. |
| `modules/` | Inference and device-native batched EigenGradCAM generation. |
| `scripts/` | Checkpoint conversion and local metrics reporting. |
| `utils/` | Device selection, label/checkpoint validation, optimizers, and tensor normalization. |
| `requirements.txt` | Unpinned Python runtime dependencies. |
