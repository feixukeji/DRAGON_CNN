# DRAGON_CNN

DRAGON (Data Reduced AGN + Galaxy Optical Network) is a PyTorch CNN pipeline
for multi-class classification of astronomical cutouts. This repository
contains FITS preprocessing, model training, deterministic inference, and
EigenGradCAM heatmaps.

## What is in this repository

- The PyTorch DRAGON model architecture.
- FITS preprocessing to an HDF5 tensor store.
- Training and evaluation with PyTorch Ignite and W&B logging.
- Inference, heatmap, and local training-report command-line tools.

## Repository layout

| Path | Purpose |
| --- | --- |
| cnn/ | DRAGON CNN definition. |
| data_preprocessing/ | Training metadata, FITS tensor creation, split generation, and normalization statistics. |
| train/ | Single-device training entrypoint and PyTorch Ignite trainer. |
| scripts/ | Checkpoint conversion and local training-report utilities. |
| modules/ | Inference and heatmap generation. |
| utils/ | Data, device, and tensor helpers. |

## Installation

Recommended workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is required. Run commands from the repository root so the
top-level packages are importable.

Notes:

- A GPU is strongly recommended for training and heatmap generation.
- Online W&B logging requires a login; set `WANDB_MODE=offline` to train
  without network logging.

## Data preparation

The dataset is a long-lived data asset, separate from training experiments. In
the examples below it is stored at `/path/to/dragon_dataset`. Its
`labels.csv` and `normalization_stats.json` remain in that directory and are
not copied into individual experiments.

Survey integrations build `raw_info.csv` and `labels.csv` with
`data_preprocessing.prepare_training.prepare_training_catalog`. They supply
their own catalog and cutout paths through `ClassSpec`; band names are used as
provided. Individual classes may be empty, but the combined dataset must retain
at least one row.

### HDF5 pipeline

1. Prepare a metadata CSV with at least:

   - An `object_id` identifier column.
   - Band columns (for example `i_band`, `r_band`, and `g_band`) containing
     FITS file paths.
   - A training label column named `class` by default.

2. Generate 96-pixel tensors:

```bash
python -m data_preprocessing.create_cutouts \
  --data-dir /path/to/fits_root \
  --csv-path /path/to/metadata.csv \
  --out-dir /path/to/dragon_dataset/tensors \
  --bands i_band --bands r_band --bands g_band \
  --cutout-size 96
```

This creates `tensors/tensors.h5` and `tensors/clean_info.csv`.

3. Publish the aligned metadata:

```bash
cp /path/to/dragon_dataset/tensors/clean_info.csv \
  /path/to/dragon_dataset/info.csv
```

4. Create one deterministic, stratified train/devel/test split set. The
   default split slug is `stratified`.

```bash
python -m data_preprocessing.make_splits \
  --data-dir /path/to/dragon_dataset \
  --label-col class \
  --info-name info.csv
```

The default fractions are 0.70/0.15/0.15. Override them with
`--train-fraction`, `--devel-fraction`, and `--test-fraction`; they must sum to
one. Every metadata row must contain a unique integer `h5_index` within the
HDF5 tensor range. The loader fails on a missing or invalid mapping instead of
guessing or clamping an index.

5. Compute the global asinh normalization statistics from the training split:

```bash
python -m data_preprocessing.compute_normalization_stats \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --split train \
  --channels 3 \
  --asinh-softening 0.1 \
  --output /path/to/dragon_dataset/normalization_stats.json
```

This statistics-generation stage is the only stage that accepts the asinh
softening value. The JSON records `vmin`, `vmax`, and `softening` together;
training, inference, and heatmap generation only read those stored values.

Expected dataset layout:

```text
dragon_dataset/
  info.csv
  labels.csv
  normalization_stats.json
  splits/
    stratified-train.csv
    stratified-devel.csv
    stratified-test.csv
  tensors/
    tensors.h5
```

## Training

The main entrypoint is `python -m train.train`. It selects one CUDA device when
available and otherwise uses the CPU; multi-process and multi-device training
are not supported. The following example trains a six-class, three-channel,
96-pixel model:

```bash
python -m train.train \
  --experiment_name dragon \
  --data_dir /path/to/dragon_dataset \
  --run_dir /path/to/dragon_runs \
  --split_slug stratified \
  --cutout_size 96 \
  --channels 3 \
  --n_classes 6 \
  --epochs 40 \
  --batch_size 16 \
  --optimizer sgd
```

Training requires `/path/to/dragon_dataset/normalization_stats.json` to
already exist. It always applies global asinh normalization using the JSON's
per-channel limits and softening; it never computes statistics or accepts a
separate softening value.

`--run_dir` is the shared run root. Each `--experiment_name` owns exactly one
experiment directory:

```text
dragon_runs/
  dragon/
    model.pt
    best_metrics.json
    wandb/
```

Only the completed-epoch model with the best devel macro-F1 is saved as
`model.pt`; there is no final model or separate checkpoints directory. Starting
another training job with the same experiment name replaces that entire
experiment directory, including any earlier inference products beneath it.
Use distinct experiment names for results that must coexist.

Other behavior:

- `--seed 42` is the default for reproducible model initialization, data
  shuffling, training augmentation, dropout, and DataLoader workers.
- `--crop` applies center-cropping and `--augment` applies train-only dihedral
  transforms on the selected device.
- The training label column is `class` by default.

The optimizer can be selected without changing code:

```bash
# Momentum SGD (the default)
python -m train.train ... --optimizer sgd --lr0 1e-3 --momentum 0.9 --nesterov

# AdamW; momentum and nesterov are ignored
python -m train.train ... --optimizer adamw --lr0 3e-5 --weight_decay 1e-4
```

AdamW uses `--adamw-beta1`, `--adamw-beta2`, and `--adamw-eps`; bias and
normalization parameters are excluded from weight decay.

### Transfer learning

Model weights are not distributed in this Git repository. Use the `model.pt`
produced by the current DRAGON trainer. Wrapped `state_dict`/`model` checkpoint
dictionaries and DataParallel `module.` prefixes are intentionally unsupported.
When the target dataset has a different number of image channels or classes,
adapt the model in a temporary location first:

```bash
python -m scripts.convert_model_channels \
  --in-model /path/to/pretrained/model.pt \
  --out-model /tmp/dragon-model-adapted.pt \
  --target-channels 4 \
  --target-classes 9
```

`--target-channels` is required. `--target-classes` is optional; when supplied,
matching classifier weights are reinitialized for that output size.

```bash
python -m train.train \
  --experiment_name dragon-transfer \
  --transfer_learn \
  --unfreeze-warmup-epochs 3 \
  --unfreeze-blocks-per-epoch 1 \
  --lr0 2e-5 \
  --model_state /tmp/dragon-model-adapted.pt \
  --data_dir /path/to/dragon_dataset \
  --run_dir /path/to/dragon_runs \
  --split_slug stratified
```

Transfer learning first trains `fc1` and `fc2`, then unfreezes complete
backbone blocks from `layer8` toward `layer1`. Frozen blocks keep BatchNorm
parameters and running statistics fixed. Like training from scratch, transfer
learning requires the existing dataset-level normalization JSON.

## Training result reports

The reporter reads each experiment's `best_metrics.json` directly; it does not
parse W&B run logs. Pass either one experiment directory or a run root whose
immediate child directories are experiments:

```bash
python -m scripts.report_training_results \
  /path/to/dragon_runs \
  --data-dir /path/to/dragon_dataset

python -m scripts.report_training_results \
  /path/to/dragon_runs/dragon \
  --data-dir /path/to/dragon_dataset \
  --split test
```

`--data-dir` supplies the dataset-level `labels.csv`. Use `--labels PATH` to
override it, or omit both options to display numeric class indices. The current
report schema requires the best epoch, metrics, and train/devel/test confusion
matrices written by the current trainer.

## Inference

Inference products live inside the selected experiment. First prepare one
catalog's metadata and tensors:

```bash
python -m data_preprocessing.prepare_inference \
  --catalog /path/to/catalog.csv \
  --cutout-dir /path/to/fits_cutouts \
  --output-dir /path/to/dragon_runs/dragon/inference/catalog-name \
  --band i --band r --band g \
  --cutout-size 96
```

Then run one deterministic prediction pass in the same catalog directory:

```bash
python -m modules.inference \
  --model-path /path/to/dragon_runs/dragon/model.pt \
  --output-dir /path/to/dragon_runs/dragon/inference/catalog-name \
  --data-dir /path/to/dragon_runs/dragon/inference/catalog-name \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --labels-path /path/to/dragon_dataset/labels.csv \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 6 \
  --batch-size 256
```

Inference requires the dataset-level normalization JSON and only reads its
stored `vmin`, `vmax`, and `softening`. `labels.csv` is used only to map class
indices to names. The primary output is `predictions.csv`, containing the top
two class indices and confidences plus class names when a mapping is supplied.

The compact experiment layout is:

```text
dragon_runs/
  dragon/
    model.pt
    best_metrics.json
    wandb/
    inference/
      catalog-name/
        info.csv
        tensors/
          tensors.h5
        predictions.csv
        heatmaps/
          object-id.png
```

Neither `labels.csv` nor `normalization_stats.json` is copied into the
experiment or catalog directory.

## Heatmaps (EigenGradCAM)

Heatmaps use exactly the same stored global normalization values as training
and inference:

```bash
python -m modules.heatmap \
  --model-path /path/to/dragon_runs/dragon/model.pt \
  --output-dir /path/to/dragon_runs/dragon/inference/catalog-name/heatmaps \
  --data-dir /path/to/dragon_runs/dragon/inference/catalog-name \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --cutout-size 96 \
  --channels 3 \
  --n-classes 6
```

Each heatmap is saved as `<object_id>.png`; object IDs must be unique and safe
to use as filenames.
