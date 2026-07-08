# DRAGON_CNN

DRAGON (Data Reduced AGN + Galaxy Optical Network) is a PyTorch CNN pipeline for multi-class classification of astronomical cutouts. This repository contains data preprocessing for FITS images, model training and sweeps, inference with optional Monte Carlo dropout, Grad-CAM heatmaps, and an ensemble voting utility.

## What is in this repository

- PyTorch models for DRAGON and related variants.
- FITS preprocessing to tensors (HDF5 or legacy per-image .pt files).
- Train and evaluate with PyTorch Ignite and W&B logging.
- Inference, Grad-CAM heatmap creation, and ensemble voting.
- Notebooks, simulation helpers, and legacy TensorFlow experiments.

## Repository layout

| Path | Purpose |
| --- | --- |
| cnn/ | DRAGON CNN definition, ResNet50 variant, and a larger-cutout model. |
| data_preprocessing/ | FITS dataset class, HDF5 tensor creation, and split generation. |
| train/ | Training entrypoints and W&B sweep runner. |
| modules/ | Inference, heatmap generation, ensemble voting, and notebooks. |
| models/ | Example trained weights and voter model sets. |
| utils/ | Data, device, and tensor helpers. |
| tensorflow/ | Legacy TensorFlow dual_finder pipeline and utilities. |

## Installation

Recommended workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Notes:
- A GPU is strongly recommended for training and heatmap generation.
- Training and sweeps require a valid W&B login (see W&B docs).

## Data preparation

The data loader supports two formats:

1) HDF5 tensors (recommended, fast)
2) Legacy per-image .pt tensors (auto-generated)

### HDF5 pipeline (recommended)

1. Prepare a metadata CSV with at least:
	 - An identifier column such as object_id
	 - Band columns (default: g_band, i_band, r_band) with FITS file paths
	 - A label column for training (default used by training is class)

2. Generate tensors:

```bash
python data_preprocessing/create_cutouts.py \
	--data-dir /path/to/fits_root \
	--csv-path /path/to/metadata.csv \
	--out-dir /path/to/data_dir/tensors \
	--bands g_band --bands i_band --bands r_band \
	--cutout-size 94
```

This creates:
- /path/to/data_dir/tensors/tensors.h5
- /path/to/data_dir/tensors/clean_info.csv

3. Place metadata where the loader expects it:

```bash
cp /path/to/data_dir/tensors/clean_info.csv /path/to/data_dir/info.csv
```

4. Create balanced train/devel/test splits (adds h5_index for HDF5):

```bash
python data_preprocessing/make_splits.py \
	--data_dir /path/to/data_dir \
	--target_metric class \
	--info_name info.csv
```

Expected layout after preprocessing:

```
data_dir/
	info.csv
	labels.csv                 # optional mapping of label keys to numeric values
	splits/
		balanced-dev-train.csv
		balanced-dev-devel.csv
		balanced-dev-test.csv
	tensors/
		tensors.h5
```

### Legacy .pt pipeline

If your info.csv uses file_name instead of object_id, the loader will fall back to a legacy path and generate one .pt tensor per image in data_dir/tensors/. Place raw FITS files in data_dir/cutouts/ or directly under data_dir/ so they can be found by file_name.

## Training

The main training entrypoint is train/train.py and logs to W&B.

```bash
python train/train.py \
	--experiment_name dragon \
	--data_dir /path/to/data_dir \
	--split_slug balanced-dev \
	--cutout_size 94 \
	--channels 3 \
	--n_classes 6 \
	--epochs 40 \
	--batch_size 16
```

Key behavior:
- Saves checkpoints to checkpoints/ and final weights to models/.
- Uses arsinh normalization when --normalize is enabled (default true).
- Uses Kornia GPU augmentations when --crop is enabled.
- Training expects a label column named class in info.csv by default.

Transfer learning:

```bash
python train/train.py \
	--transfer_learn \
	--model_state /path/to/model.pt \
	--data_dir /path/to/data_dir \
	--split_slug balanced-dev
```

W&B hyperparameter sweeps:

```bash
python train/wandb_sweep_train.py \
	--experiment_name dragon \
	--data_dir /path/to/data_dir \
	--split_slug balanced-dev
```

The sweep entrypoint supports both dragon and resnet model types.

## Inference

The inference script runs in a no-labels mode and writes prediction CSVs. The output_path argument is treated as a filename prefix, so provide a directory path ending in a slash if you want files inside a folder.

```bash
python modules/inference.py \
	--model_path models/dragon-balanced-dev-XXXXX.pt \
	--output_path /path/to/output/ \
	--data_dir /path/to/data_dir \
	--slug balanced-dev \
	--split test \
	--cutout_size 94 \
	--channels 3 \
	--n_classes 6 \
	--batch_size 256 \
	--mc_dropout
```

Outputs include predicted_labels, predicted_confidence, and the second best class/confidence for each object.

## Heatmaps (Grad-CAM)

```bash
python modules/heatmap.py \
	--model_path models/dragon-balanced-dev-XXXXX.pt \
	--output_path /path/to/heatmaps/ \
	--data_dir /path/to/data_dir \
	--cutout_size 94 \
	--channels 3 \
	--n_classes 6
```

Heatmaps are saved as heatmap_00000.png, heatmap_00001.png, and so on.

## Ensemble voting (Congress)

Use modules/congress.py to run inference across a folder of .pt models and aggregate predictions into a single vote per object.

```bash
python modules/congress.py \
	--data_dirs /path/to/data_dir \
	--model_folder models/voters \
	--n_classes 6 \
	--channels 1
```

Outputs:
- combined_results.csv with per-model voter columns
- congress.csv with voted_class, num_voters, and confidence summaries

## Notebooks and utilities

- modules/inference_notebooks/ includes interactive inference notebooks.
- modules/FITSViewer.py provides a widget-based FITS viewer for manual curation.
- modules/simulation/ contains synthetic cutout generation notebooks/scripts.
- tensorflow/ holds the legacy dual_finder pipeline and visualization tools.
