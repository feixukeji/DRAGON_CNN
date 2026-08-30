# DRAGON_CNN

DRAGON (Data Reduced AGN + Galaxy Optical Network) is a PyTorch pipeline for
multi-class classification of astronomical image cutouts. It covers the full
workflow from multi-band FITS files to an aligned HDF5 dataset, reproducible
data splits, training, evaluation, confidence calibration, inference, and
EigenGradCAM visualizations.

## Features

- A four-stage residual convolutional network whose resolution schedule and
  head geometry are both derived from the angular scales the labels are made
  of, with anti-aliased downsampling and a concentric-ring pooled head read
  from all four stages through one hidden layer; input side lengths only have
  to be multiples of 8, from 48 upward.
- `--channels` and `--n-classes` are required wherever a model is built, so a
  default can never silently disagree with the cutouts or with `labels.csv`.
- Configurable input channels and output classes.
- Parallel FITS ingestion with center cropping or zero padding.
- Optional ingestion of rows from compatible pair-HDF5 tensor stores.
- Deterministic, class-stratified train/devel/test splits.
- Per-channel global asinh normalization estimated from the training split.
- Training from scratch or gradual block-wise transfer learning.
- SGD and AdamW optimizers, balanced class weights, augmentation, gradient
  clipping, cosine learning-rate scheduling, and W&B logging.
- Best-model selection on devel macro-F1 followed by a single test evaluation.
- Top-2 inference with optional all-class probability columns and stable score
  serialization.
- Local metrics reports and per-class confidence thresholds constrained by the
  worst individual negative-class false-positive rate (FPR).
- CSV confidence curves, diagnostic plots, and batched EigenGradCAM heatmaps.

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
      +-----------+------+----------------+
      |                  |                |
      v                  v                v
predictions.csv  confidence_curves  EigenGradCAM PNGs
~~~

The examples keep dataset assets and experiment outputs separate.
`labels.csv`, `normalization_stats.json`, split CSVs, and `tensors.h5` live in
the dataset directory; `model.pt`, `best_metrics.json`, W&B files, predictions,
confidence-calibration artifacts, and heatmaps live below an experiment
directory.

## Core contracts

| Contract | Requirement |
| --- | --- |
| Image shape | Side lengths must be multiples of 8 and at least 48, because the network halves the map three times and the head's outermost ring, anchored at 4 arcsec, must still contain positions at `layer2`; `96 × 96` is the default everywhere. The channel count is configurable. |
| Channel order | Repeated `--bands` and `--band` options define tensor channel order. That positional order must match the normalization statistics and checkpoint; later stages receive only the channel count and cannot validate band semantics. |
| Class mapping | `labels.csv` defines the zero-based class order shared by `--n-classes`, the checkpoint classifier head, `probability__*` columns, and the threshold manifest. |
| Row alignment | Every runtime metadata row contains one unique, in-range integer `h5_index` pointing into `tensors/tensors.h5`. |
| Normalization | One `normalization_stats.json` is computed from the training split and reused without modification by training, calibration, inference, and heatmaps. |
| Checkpoints | Model files are plain PyTorch `state_dict` mappings. Wrapped checkpoints and keys prefixed with `module.` are not accepted. |
| Calibration split | Confidence calibration requires every class declared by `labels.csv` to occur in the selected labelled split. |
| Threshold comparison | A class threshold applies to its matching `probability__<class-index>` column with a `>=` comparison. Independent class thresholds can accept zero, one, or multiple classes for one object. |
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

A CUDA GPU is strongly recommended for training, calibration, and heatmap
generation. CPU execution is supported but can be substantially slower.

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

### 6. Calibrate per-class confidence thresholds

Use the labelled devel split to generate confidence curves and choose one
threshold per class at a target worst-negative-class FPR:

~~~bash
python -m scripts.evaluate_confidence_curves \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --target-fpr 0.001
~~~

The default output directory is
`/path/to/dragon_runs/baseline/confidence_curves`. It contains all-class
softmax scores, tabular curve data, per-class thresholds, and diagnostic PNGs.

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

DRAGON accepts `(batch, channels, H, W)` tensors. `H` and `W` must be
multiples of 8 and at least 48, because the feature extractor halves the map
three times and the head's outermost ring boundary sits at 4 arcsec, which
`layer2` can only place inside a map of at least `24 × 24`. The ring-pooled
head carries no size-dependent weights, so one architecture serves every such
size; `96 × 96` is the default and the size the block table below describes.

The resolution schedule is derived from the scales the labels are made of, not
from a generic classifier backbone. At `0.168 arcsec/pixel` those are the PSF
core (`3.6–4.8 px`), the pair separation (`2.2–24.3 px`, log-uniform, median
`8.9 px`, 10th percentile `4.4 px`) and tidal features (`60+ px`). Measuring a structure of size `s`
needs a stride of at most `s/2` so it is sampled at all, and a receptive field
of roughly `2s` so a unit sees it whole against its background. That fixes
four stages:

~~~text
layer1   24 @ 96×96   stride 1   RF  11 px   PSF core, seeing-limited pairs
layer2   64 @ 48×48   stride 2   RF  26 px   the bulk of the separation range
layer3  128 @ 24×24   stride 4   RF  56 px   widest pairs plus environment
layer4   96 @ 12×12   stride 8   RF 116 px   full field: tidal features
~~~

`layer1` is four convolutions at stride 1 (`5×5` then three `3×3`), which is
what it takes to reach an 11 px receptive field before anything is
downsampled. Its last block stops at the BatchNorm: stage outputs do not end
in an activation, because every consumer rectifies what it reads, and a
LeakyReLU here followed by `Downsample`'s would multiply the negative tail by
`0.01` twice over. Each later stage is `Downsample → Conv2d → BatchNorm2d →
LeakyReLU → ResidualBlock`, where the projection is not always a widening —
`layer4` narrows, since a same-width `3×3` there bought nothing but
parameters. There is no fifth stage, though not because
`layer4`'s 116 px theoretical receptive field covers the cutout: effective
receptive fields are a fraction of the theoretical one. The reason is
parameter allocation — a residual block costs `2·C²·9` parameters wherever it
sits, so each further stage is the most expensive one, and only the merger
class needs a scale coarser than `layer4`.

`Downsample` is a `LeakyReLU` followed by `BlurPool2d`: a replicate-padded
`[1,3,3,1]` binomial low-pass subsampled by 2. Low-pass before subsampling is
Nyquist, and that is all this layer does — it does not take a max first. The
usual argument for the max is that average pooling divides a point source's
peak by four while dividing its noise only by two, but that holds for a source
confined to one pixel and the HSC PSF is roughly twice oversampled. On a 4 px
FWHM Gaussian in noise, one `2×` downsample changes the peak SNR by `1.45×`
(average), `1.39×` (max), `2.23×` (blur) and `1.98×` (max-then-blur); with a
companion at five percent of the primary, by `1.42×`, `1.32×`, `2.08×` and
`1.83×`. The four-tap kernel is what an even-sized map needs to be sampled
symmetrically, which keeps the eight dihedral training views exactly
equivalent. The border is replicated for the same reason the convolutions
replicate theirs: reflection holds unit gain and dihedral equivalence just as
exactly, but it mirrors an edge source into a second copy of itself, which is
the structure this network looks for.

Convolutions replicate the border rather than zero-filling it.
`asinh_normalize` clamps to `[0, 1]` and expands the faint end, so the sky
sits at a clearly positive value and a zero-filled border is darker than any
realistic pixel; the resulting kernel-mass deficit (`2/3` along an edge, `4/9`
in a corner) is a per-position gain that BatchNorm cannot undo. Reflection
would keep unit gain too, but it mirrors a source near the edge into a second
copy of itself, which is the structure this network looks for.

Residual blocks apply no activation after the sum, and `bn2.weight` is zeroed
at initialization, so each block starts as exactly the identity. One block per
stage: without a post-sum activation, stacking two would put the first block's
`bn2` straight into the second's `conv1` with no nonlinearity between them.
Anything consuming a block's output rectifies it first.

`RingPool` replaces global average pooling in the head. The classes are
defined by a radius — the central source sits at the exact centre and a
companion is a companion only within `theta_max` of it. Ring boundaries are
fixed in angle at `0.5`, `1`, `2` and `4 arcsec`, so the bins mean the same
thing at every stage and every cutout size, and the separation range is split
into four bins rather than swallowed whole by one. The `0.5 arcsec` edge
separates the central source's own core from the closest companion it could
have and survives only at `layer2`; `stride 4` and `8` drop it, since a
boundary below 1.5 feature-map pixels is not sampled. Rings are exactly
invariant under all eight dihedral transforms because each is a
dihedral-symmetric set of positions. Only the inner boundaries are fixed in
angle; the last ring runs from `4 arcsec` to the map corner, so its extent does
depend on the cutout size.

Each ring but the last contributes both a mean and a max, on different inputs.
The max reads a rectified map, since `amax` over a signed one erodes whatever
the network encoded negatively; the mean reads the raw signed map, because it
is linear and rectifying first would compress a negative-coded feature by a
factor of a hundred for nothing. The outermost ring usually keeps only its
mean: no companion of interest sits beyond `theta_max`, so "is there a peak out
there" is not a question worth another `C` dimensions.

A redundancy argument used to stand alongside that one — mean and max
correlating at `0.96` on the outer ring against `0.26` innermost — and it is
withdrawn. On the trained model's own feature maps the ordering is reversed:
outermost `r = 0.00, 0.03, 0.19, 0.36` at taps 1–4 against innermost
`r = 0.71, 0.91, 0.93, 0.94`, which is what a four-position ring should give.
The `0.96` was measured on raw cutouts, where an outer ring is mostly sky.

`layer3` and `layer4` are the exceptions and keep that max. Both reasons above
are reasons about companions, and `merger` is not a companion class: its
evidence is a tidal feature at `20–60 px`, most of which lies beyond the
`4 arcsec` edge and so sits in exactly this ring. It is localised and
azimuthally asymmetric, so the ring's mean divides it by the ratio of the
ring's positions to the ones it occupies, while `max - mean` is its detector.
The rule is not "outermost" but whether one map position already integrates an
area the size of the structure the ring reports — `56 px` of receptive field at
`layer3` and `116 px` at `layer4` against a `20–60 px` feature, but `26 px` at
`layer2` and `11 px` at `layer1`, whose outermost rings also hold 1,856 and
7,412 positions, so the max there is an extreme-value statistic over field
sources common to every class. That is two clauses and only the first is
scale-free. Applied to the inner rings the first passes everywhere — what a
`0.5–4 arcsec` ring reports is a companion, a `3.6–4.8 px` PSF, which `11 px`
of receptive field already integrates — while the second does not, since
`tap1`'s `2–4 arcsec` ring holds 1,356 positions and `tap2`'s 336. Those maxes
are kept because there the extreme *is* the thing being looked for.

The statistic that would attack the contamination directly is an azimuthal
dipole modulus, `|Σ x e^{iφ}| / Σ|x|`: exactly zero on any azimuthally
symmetric profile (measured, `1e-17`), exactly dihedral-invariant, bounded in
`[0, 1]`, and not affinely constructible. On raw cutouts it is worth a great
deal — Fisher `d` from `1.3` to `2.7` for a 5 percent tidal arc under a
brightness nuisance. It is not in the model because the network already has it:
a rectified sum over 8 oriented channels reconstructs a modulus to 6 percent
ripple and `layer3` has 128. Measured on the trained model, adding it moves
held-out balanced accuracy from `0.890` to `0.886`. `merger` is also not short
of evidence — a linear probe on the pooled statistics alone separates it from
`single_galaxy` at AUC `0.995` and from `agn_galaxy` at `0.998`.

`max - mean` is the asymmetric part of a ring only on a thin ring, and these
are an octave wide. A centred source with any radial gradient puts its max at
the ring's inner edge and its mean well below it, with no companion involved:
on a purely azimuthally symmetric `4 px` Moffat, `(max - mean)/mean` on the
real tap geometries runs `0.00, 1.11, 1.47, 3.56, 5.51` at stride 2 and
`0.00, 1.73, 2.65` at stride 8. Only the innermost ring, whose four positions
share one radius, is clean. Separating the asymmetry from the radial gradient
means dividing by the mean, which is a ratio.

The head reads from all four stages. Pooling and nonlinearity do not preserve a
quantitative "how far apart", and by the `stride ≤ s/2` rule `layer3` can only
measure structure of 8 px and up, while the tightest pairs are `2.2–4.5 px`.
`layer2` covers the separation range, `layer3` covers it with environment, and
`layer4` covers the field the merger class needs. `layer1` is tapped because
the same rule reaches it: the `BlurPool` transfer function passes `0.83` of the
contrast at the median `8.9 px` separation, `0.43` at `4.4 px`, `0.27` at
`3.6 px` and nothing at the realised minimum `2.2 px`, so `layer1` is the only
stage that sees the tightest quarter of the pairs at full contrast and the only
one where central compactness — the `merger` / `agn_galaxy` boundary — is
resolved. It costs 216 features and about 1 MMAC.

Each tap rectifies the stage's signed residual sum before feeding the max
branch. The concatenated statistics pass through `BatchNorm1d` — the mean and
max branches have different scales, and one linear layer under one dropout rate
would otherwise have to treat them alike — then dropout with probability `0.4`,
a `head_width` hidden layer (`Linear → BatchNorm1d → LeakyReLU`), and the
output `Linear`.

The hidden layer is there because colour, not geometry, separates several of
the classes: `dual_agn` and `agn_star` both put an unresolved point source at
the same radius, and `single_star` and `single_agn` both put one at the centre.
Colour is a ratio of band fluxes, an affine head cannot form a ratio, and the
nuisance it would have to divide out is real — the injected amplitude `alpha`
spans a factor of six and scales the offset source's contribution to a ring
statistic without scaling the central source's. For "is there a companion" that
costs only calibration, since a positive rescaling leaves the sign of a linear
discriminant alone; for "which companion" it costs sight, because the
label-irrelevant central term does not scale with `alpha` and one hyperplane
has to separate the whole family. The radial-gradient ratio above needs the
same thing.

`head_width` defaults to 96: the rank-8 discriminant an affine head already
realised, a direction for each of the 32 `(ring, statistic)` slots the four
taps expose, and one per max ring for the ratio. The dropout sits on the 2,392
pooled statistics rather than on the hidden layer — at `p = 0.4` a 96-unit
bottleneck keeps 58 units on average, close to the count it was just sized for,
and the wide side holds 98 percent of the head's parameters. The cost is a
`BatchNorm` downstream of a dropout, whose variance inflation lands as a
near-uniform rescaling that threshold calibration absorbs.

`widths` is the only capacity knob on the backbone, and multiply-accumulates
and parameters must not be conflated. Moving capacity towards fine scales costs
arithmetic quadratically, because the map is sixteen times larger at stride 1
than at stride 4; moving it towards coarse ones costs parameters, because a
residual block costs `2·C²·9` wherever it sits and the widest stage is always
the last. Wall-clock follows the arithmetic. The default `(24, 64, 128, 96)`
widens `layer2` and narrows `layer4` relative to a monotonic table:

~~~text
             parameters          MMAC
layer1    17,544   ( 1.8%)   159.9  (26.0%)
layer2    87,936   ( 8.9%)   201.7  (32.7%)
layer3   369,408   ( 37.4%)  212.3  (34.5%)
layer4   277,056   ( 28.1%)   39.8  ( 6.5%)
head     235,480   ( 23.8%)    0.2  ( 0.0%)
ring taps      --              2.2  ( 0.4%)
~~~

A three-channel, eight-class instance has 987,424 parameters and costs about
616 MMAC per `96 × 96` image.

That table sits awkwardly with the `stride ≤ s/2` rule: `layer3` can only
measure the upper half of the separation distribution and holds 37 percent of
the parameters, while `layer2` — the only stage measuring `4–8 px` and the only
one keeping the `0.5 arcsec` boundary — holds 9. The alternative
`(24, 96, 96, 64)` gives 803k parameters and 754 MMAC, with `layer2` at 23
percent and `layer4` at 16. It is not free, and the rule says which stage can
*measure* a separation rather than how many channels companion typing needs, so
the table is left for measurement. `layer4` has the weakest case: its `116 px`
receptive field exceeds the field, and `tap4` reports three rings of 4, 28 and
112 positions — near-global descriptors. Narrowing it, `(24, 64, 128, 48)`,
saves 21 percent of the parameters and 4 percent of the arithmetic but
concentrates rather than rebalances, putting `layer3` at 47 percent.

The simulation leaves a shortcut and nothing here blocks it. `Y = C + alpha·M·S`
with no background subtraction raises the noise variance inside the injected
mask's support by `1 + alpha²`. It is not confined to fine strides: a stage
rectifies before it pools, so a high-pass followed by a `LeakyReLU` turns noise
power into a non-negative map whose local mean `BlurPool` carries through
untouched — measured against a 4.4 percent sigma step, Cohen's `d` is
`0.093, 0.095, 0.097, 0.097` at strides 1, 2, 4 and 8. It is also weak: `alpha`
is uniform on `[0.05, 0.30)`, and a sample standard deviation over `N` pixels
has relative precision `1/√(2N)`, bounding even an oracle estimator at
`d ≤ 0.044·√(2N)` — about `0.9` at `N = 200` and the largest `alpha`, about
`0.3` at the median. That bound is what matters rather than the single-feature
`0.09`, because the head sees 2,392 statistics over 312 channels and weak cues
read repeatedly add in quadrature; the 312 channels are 312 looks at one noise
field, so no width buys information the pixels do not hold. The per-image
discriminant is bounded near `d ~ 0.3`; the per-population cue is not bounded,
and the population is 204,039 of the 480,278 training images — 42 percent:
70,000 `dual_agn`, 70,000 `agn_star`, 64,039 `agn_galaxy`. Suppressing it is
the data pipeline's job.

`ResidualBlock` is one block per stage. Stacking two would put the first
block's `bn2` straight into the second's `conv1` with no nonlinearity between
them, but nothing here wants to stack: at thirteen learned convolutions the
shortcut is not carrying trainability, it is carrying the identity
initialization, which starts the model as a seven-convolution plain net and
lets it grow under a cosine schedule with no warmup.

Two limitations are left in place. The width table is left for measurement,
above. And the model is given no seeing and no invariance to overall
brightness.

The network sees pixels only. `theta_max` depends on redshift and `theta`'s
lower bound is the seeing of that exposure, so the class boundary is a function
of `(z, seeing)` that varies image to image; the scale table above is the
population's marginal distribution, not any single cutout's. The two are not
equally forced: redshift is unavailable for most deployment targets, while
seeing is available for every HSC cutout and is the decisive nuisance for the
tightest pairs. Feeding it is a data question — the cutout store carries no
seeing column — rather than an architectural one now that the head is not
affine.

Two things the radius schedule does not reach at all. `single_star` against
`single_agn`, and `dual_agn` against `agn_star`, put the same unresolved point
source in the same place; only the spectral energy distribution differs, so the
ring geometry, the edge table and the dihedral augmentation are structurally
silent on them. And nothing here is invariant to overall brightness:
`asinh_normalize` uses fixed dataset-level limits, there is no instance
normalization, and the eight classes have quite different magnitude selection
functions — total flux is a usable shortcut, suppressed entirely on the data
side and not at all here.

## Training behavior

The training entry point selects CUDA when available and otherwise uses the
CPU. Every stage runs on one device: the project uses neither DataParallel nor
distributed execution, for training or for inference. CUDA runs use bfloat16
automatic mixed precision through PyTorch Ignite.

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
> removes confidence-calibration artifacts and inference products stored below
> that experiment.

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

`--target-channels` adapts the first convolution. `--target-classes` defaults
to the checkpoint's own class count; the head is rebuilt whenever any of its
tensors no longer matches the current architecture — a changed class count, a
changed ring-statistic width, or a changed head layout — and kept untouched
when they all still match. Replacements come from the model's own
initialisation under `--seed` (default 42), so the conversion is reproducible.
The output parent directory must already exist.

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

Transfer learning initially trains only the head while all four `layerN`
blocks are frozen. After the warmup, complete blocks are unfrozen from `layer4`
toward `layer1`. Frozen blocks remain in evaluation mode so their
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

## Confidence curves and threshold calibration

The calibration command runs the selected labelled split through the model
once and retains every softmax column. For each target class, it treats every
other class as a separate negative population. A threshold is found for each
negative class, then the strictest of those thresholds is selected. The result
therefore constrains the empirical FPR for every individual negative class,
not only for all negatives pooled together.

~~~bash
python -m scripts.evaluate_confidence_curves \
  --model-path /path/to/dragon_runs/baseline/model.pt \
  --data-dir /path/to/dragon_dataset \
  --split-slug stratified \
  --split devel \
  --normalization-stats /path/to/dragon_dataset/normalization_stats.json \
  --target-fpr 0.001 \
  --batch-size 256 \
  --curve-points 1001
~~~

`--split devel` and `--target-fpr 0.001` are the defaults. The label mapping is
loaded from `DATA_DIR/labels.csv`, the channel count is inferred from the HDF5
store, and normalization defaults to
`DATA_DIR/normalization_stats.json`. Every declared class must have at least
one row in the selected split. Prefer the devel split for calibration; using
the test split to choose thresholds makes it part of model development rather
than an untouched final holdout.

For a negative class containing `N` rows, at most
`floor(target_fpr × N)` rows may meet the threshold. Each per-negative-class
threshold is placed immediately above its first disallowed score, so equal
scores remain together under the manifest's `>=` comparison; the final class
threshold is the maximum of those candidates. With small negative classes or
a low target FPR, the allowed false-positive count can be zero.
`--curve-points` controls only the resolution of the exported display curves;
it does not approximate or alter the calibrated thresholds.

Unless `--output-dir` is supplied, artifacts are written beside the checkpoint
under `confidence_curves/`:

~~~text
confidence_curves/
├── scores.csv
├── curves.csv
├── negative_class_fpr.csv
├── thresholds.json
├── FPR_curve.png
├── Precision_curve.png
├── Recall_curve.png
├── F1_curve.png
├── ROC_curve.png
├── PR_curve.png
└── classes/
    └── class_<zero-padded-index>/
        └── FPR_by_negative_class.png
~~~

`scores.csv` contains `object_id`, the true label and class name, and one
`probability__<class-index>` column per class. `curves.csv` contains aggregate
precision, recall/TPR, F1, pooled FPR, macro FPR, and worst-negative-class FPR
over sampled thresholds, including the exact calibrated threshold.
`negative_class_fpr.csv` breaks the FPR out by target and negative class.
`thresholds.json` is the machine-readable manifest with provenance, label
mapping, thresholds, counts, and achieved metrics.
Plotting uses a non-interactive backend, so the command does not require a
display server on batch or HPC nodes.

Inference does not load `thresholds.json` or emit threshold-acceptance columns.
Run it with `--all-probabilities`, then apply each
`classes.<class-name>.threshold` value to the probability column with the same
numeric class index. For example, a class whose mapping value is `2` is tested
as `probability__2 >= threshold`. These one-vs-rest gates are independent of
the top-1 decision, so an object can pass no class or more than one class. The
reported rates are empirical measurements of the selected calibration split,
not guarantees for future data.

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
  --batch-size 256 \
  --all-probabilities
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
| `probability__<class-index>` | Softmax probability for that numeric class when `--all-probabilities` is selected. |

Use `--no-labels` and omit `--labels-path` to produce numeric indices only.
The default `--top-two-only` mode omits the full probability columns while
retaining the four top-2 columns. Probability values are written with enough
decimal precision to preserve the model's float32 scores when pandas reads the
CSV as float64; this keeps `>=` comparisons against calibrated thresholds
stable. Inference streams HDF5 batches instead of preloading the entire tensor
store, and runs on the single device selected by `discover_devices`.

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

The implementation targets `layer3`, uses the model's predicted class, and
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
| `python -m scripts.convert_model_channels` | Adapt checkpoint input channels and rebuild a head that no longer fits. |
| `python -m scripts.report_training_results` | Print metrics from one experiment or run root. |
| `python -m scripts.evaluate_confidence_curves` | Export all-class confidence curves and calibrate worst-negative-class FPR thresholds. |
| `python -m modules.inference` | Write top-2 predictions and optionally every class probability for an inference catalog. |
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
| `scripts/` | Checkpoint conversion, local metrics reporting, and confidence calibration. |
| `utils/` | Device selection, label/checkpoint validation, optimizers, and tensor normalization. |
| `requirements.txt` | Unpinned Python runtime dependencies. |
