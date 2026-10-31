# MELBA — geometry-aware TTA for medical image segmentation

Journal extension (MELBA) of the MICCAI UNSURE workshop paper on geometric
test-time augmentation (TTA) for medical image segmentation. This repository is a
clean rebuild: the previous code base
(`github.com/Subkash2206/tta-medseg-evaluation`) is **not** used for any reported
number, and its modules have been removed from this repository.

## What this rebuild fixes

| Old behaviour | Status in this repo |
| --- | --- |
| `hausdorff_95` = max directed Hausdorff over **all foreground pixels** (HD100 on solid regions), no percentile, no surfaces, no spacing | `melba.metrics.surface`: distances between **surfaces**, physical spacing, and **HD95 in the MONAI convention** `max(p95(pred→gt), p95(gt→pred))`; the pooled variant is kept as `hd95_pooled` |
| CVC-ClinicDB split at **frame level** → near-duplicate frames leaked between train/test | `melba.data.splits`: whole **videos** are assigned to one split/fold; leakage is asserted before and after writing, and a 5-fold cross-validation makes every video a test unit exactly once |
| No held-out test set; checkpoint selected on the reported split | 5 folds with rotating train/val/test; `scripts/evaluate.py` **refuses** to report `model.selection_split`, and refuses the train split |
| Paired t-test on heavy-tailed deltas | `melba.stats`: **Wilcoxon signed-rank** primary test, seeded bootstrap CI, rank-biserial effect size, Holm correction, video-level pairing (pooled over folds); t-test only as a labelled secondary column |
| Aggregate-only CSV | long-format **per-case** CSV (`case/method/metric` rows) with `fold` and `empty_flag`, plus a summary file with `n_cases` and a run metadata JSON |
| Hand-written inverse transforms per view | inverses are derived (flip is self-inverse, `rot90(k)⁻¹ = rot90(-k)`, composition reverses order); every view in the shipped configs is round-trip tested |
| Silent/penalised empty masks | default `empty_policy: nan` plus an explicit `empty_flag` column (`both_nonempty`, `both_empty`, `pred_empty`, `gt_empty`); `penalize` remains available for a sensitivity table only |
| No uncertainty or cost analysis | per-view probability maps give entropy/variance (global and foreground+3-px band), view disagreement and per-view Dice vs consensus; per-case latency (3 warm-ups, CUDA synchronised) and peak GPU memory |
| Pretrained encoders fed raw `[0, 1]` images (no ImageNet normalization — the old repo did this too) | `dataset.normalization.kind: imagenet` resolves the encoder's own mean/std from `smp.encoders.get_preprocessing_params`; applied in the dataset, tested against a hand-computed normalization |
| CVC-ClinicDB TIFFs read with OpenCV → the malformed photometric/ExtraSamples pair makes libtiff report one channel, so every frame reached the network as **grey** (three identical channels) | frames are read with **tifffile** (true `(H, W, 3)` RGB, no BGR→RGB swap); masks are read the same way, asserted single-channel and binarised at `mask_threshold`; a regression test loads real frames and asserts 3 distinct channels, native `288x384`, and that a BGR swap would be detected |
| Distance metrics silently ignored missed lesions | paired distance metrics use only cases non-empty **under both methods** (`n_cases_used` reported); a missed-detection table (rate + cluster-bootstrap CI + exact McNemar transitions) and an all-cases penalised sensitivity table are written by `run_stats.py` |
| Hardcoded paths, seeds, device, hyper-parameters | everything is in `configs/*.yaml`; `device: auto`; only YAML is authoritative |

## Layout

```
configs/                     YAML experiment configs (base.yaml + dataset/model configs)
  base.yaml                  shared defaults: seed, paths, metrics, stats, runtime
  cvc_clinicdb/unet_resnet50.yaml    ImageNet ResNet-50 UNet, ImageNet normalization
  cvc_clinicdb/swinunetr_2d.yaml     MONAI SwinUNETR (2D, from scratch), dataset statistics
data/splits/                 frame -> video mapping (yours) and generated fold CSVs
docs/splits.md               how to provide the CVC-ClinicDB frame → video mapping
scripts/
  make_splits.py             group-level split creation (writes data/splits/*.csv)
  compute_normalization.py   per-fold training-set mean/std (from-scratch models)
  train.py                   training, val-only checkpoint selection, resumable
  evaluate.py                per-case evaluation of every configured TTA method
  run_grid.py                folds x seeds grid; resumes whatever is incomplete
  aggregate.py               pooled fold/seed tables, paired stats, seed ensembles
  run_stats.py               paired statistics over a per-case CSV
src/melba/
  config.py                  YAML loading, defaults, ${refs}, overrides, fingerprint
  utils/                     determinism (torch/numpy/random/cudnn), device, provenance
  metrics/                   overlap + spacing-aware surface metrics, empty-mask policy
  transforms/                invertible geometric transforms, YAML → TTA views
  inference/                 TTA inference with inverse-transformed aggregation
  data/                      video-level splits, CVC-ClinicDB dataset, data loaders
  models/                    smp UNet / MONAI SegResNet / 2D SwinUNETR factory
  training/                  training loop, last.pt resume, grid state helpers
  eval/                      long-format records, per-case runner, pooling/ensembles
  stats/                     Wilcoxon + bootstrap CI + effect size + Holm
tests/                       pytest suite (metrics, transforms, TTA, splits, stats, runner)
```

## Install

```bash
python -m pip install -r requirements.txt   # pinned versions (Python 3.10+)
python -m pip install -e .                  # optional: installs the melba package
python -m pytest -q
```

`requirements.txt` is the authoritative, fully pinned environment; `pyproject.toml`
only declares loose lower bounds so `pip install -e .` resolves on other machines.

The code runs on CPU or a single GPU with ≤16 GB (validated on a 6 GB RTX 4050
Laptop GPU); `device: auto` picks whatever is available.

## Quickstart (CVC-ClinicDB)

1. **Generate the frame → sequence mapping** from the official end-frame table
   (`dataset.split.sequence_end_frames`, Bernal et al., 2015 — see `docs/splits.md`).
   The script expands the 29 sequences, writes
   `data/splits/cvc_clinicdb_frame_groups.csv` and verifies that all 612 image
   filename stems match:

   ```bash
   python scripts/make_cvc_mapping.py --config configs/cvc_clinicdb/unet_resnet50.yaml
   ```

2. **Create the five video-level folds** (writes `data/splits/cvc_clinicdb_fold{0..4}.csv`
   plus `cvc_clinicdb_folds.json`). The script validates that every mapping `case_id`
   matches an image filename stem and fails loudly otherwise:

   ```bash
   python scripts/make_splits.py --config configs/cvc_clinicdb/unet_resnet50.yaml
   ```

3. **Train** fold `k` with validation-only checkpoint selection:

   ```bash
   python scripts/train.py --config configs/cvc_clinicdb/unet_resnet50.yaml --fold 0
   ```

4. **Evaluate** every TTA method on the untouched test split of that fold:

   ```bash
   python scripts/evaluate.py --config configs/cvc_clinicdb/unet_resnet50.yaml --fold 0
   ```

   Outputs, under `results/<experiment>/evaluation/` (one file per fold **and** seed,
   so a 5-fold x 3-seed grid never overwrites itself):
   * `cvc_clinicdb_test_fold0_seed0_per_case.csv` — long format, one row per
     case/method/metric (with `fold`, `empty_flag`, uncertainty and timing rows);
   * `cvc_clinicdb_test_fold0_seed0_summary.csv` — n/mean/std/median/IQR/count per metric;
   * `cvc_clinicdb_test_fold0_seed0_run.json` — config hash, seed, environment,
     checkpoint hash, timing setup, probability-map directory;
   * `maps/fold0_seed0/<case>__{baseline,tta4}__mean.npy` — float16 mean probability
     maps on the native grid (used by `scripts/aggregate.py`, gitignored).

5. **Statistics**, pooling all five folds so each video is a test unit exactly once:

   ```bash
   python scripts/run_stats.py \
       --config configs/cvc_clinicdb/unet_resnet50.yaml \
       --per-case results/cvc_clinicdb_unet_resnet50/evaluation/cvc_clinicdb_test_fold{0,1,2,3,4}_seed0_per_case.csv \
       --treatment tta8
   ```

   Pooling is validated: a case or video that appears in more than one fold aborts
   the analysis. With `unit: group` every video contributes one paired value, so
   the sample size is the number of videos (29 for CVC-ClinicDB), not frames.

## Cross-validation grid (5 folds x 3 seeds, resumable)

`scripts/run_grid.py` trains and evaluates every fold/seed run and decides per run
what still has to happen by looking at the disk:

| state on disk | action |
| --- | --- |
| `run_done.json` + evaluation marker | skip |
| `run_done.json`, no marker | evaluation only |
| `last.pt` only (killed mid-training) | resume from that epoch |
| nothing | train from scratch |

Every epoch atomically writes `checkpoints/<experiment>/fold<k>_seed<s>/last.pt`
(model, optimizer, scheduler, AMP scaler, epoch, best value, early-stopping counter
and all RNG states for python/numpy/torch/cuda), so the process can be killed at any
moment and the next launch continues the same trajectory. Training runs in the grid
process itself, so stopping the grid stops training at the next epoch boundary.

```bash
python scripts/run_grid.py --config configs/cvc_clinicdb/unet_resnet50.yaml \
    --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20
python scripts/run_grid.py --config ... --status-only   # what is done / pending
```

Timing is measured once, for fold 0 / seed 0 (`evaluation.timing: auto`, three
warm-ups + one timed pass); every other run does exactly one forward pass per view.

## Pooling folds and seeds (`scripts/aggregate.py`)

```bash
python scripts/aggregate.py --config configs/cvc_clinicdb/unet_resnet50.yaml
```

Writes under `results/<experiment>/stats/aggregate/`:

* `pooled_by_seed.csv` — per method and metric, the pooled 29-video result of each
  seed, the mean and the standard deviation across seeds;
* `paired_by_seed.csv` / `paired_seed_averaged.csv` — video-level Wilcoxon tests
  pooled over the 5 folds, once per seed and once with metric values averaged per
  case over seeds (so a video is still one test unit);
* `detection_by_seed.csv` / `detection_seed_averaged.csv` — missed-detection rates,
  transition tables and exact McNemar tests pooled over folds;
* `ensemble_per_case.csv`, `ensemble_comparison.csv`, `ensemble_detection.csv` — the
  seed ensemble of the baseline maps (`ensemble3`), and of the tta4 maps
  (`ensemble3_tta4`, i.e. 3 seeds x 4 views), thresholded at 0.5 and scored with the
  same metric code; compared with `baseline`, `tta4` and each other, missed
  detections included. Ensembles start from the float16 maps, which shifts Dice by
  < 5e-4 per case and distance metrics by < 0.6 px relative to the float32 runner.

Any config value can be overridden without editing files:
`--set model.checkpoint=checkpoints/x.pt metrics.empty_policy=nan device=cpu`.

## Metric definitions

Computed per case, on the **native image grid** (predictions are resampled back to
it), with the spacing from `dataset.spacing_mm`:

* `dice`, `jaccard`, `precision`, `recall` — overlap metrics (empty/empty = 1, one
  empty = 0 for Dice/Jaccard, `nan` for precision/recall).
* `hd` — symmetric Hausdorff distance between **surfaces** (100th percentile).
* `hd95` — **MONAI convention**: `max(p95(pred→gt), p95(gt→pred))`, i.e. each
  directed surface-distance distribution is reduced separately and the larger value
  is reported. Verified against `monai.metrics.compute_hausdorff_distance(
  percentile=95)` on 20 random 2D and 20 random 3D blob pairs (|Δ| < 1e-4),
  including anisotropic spacing.
* `hd95_pooled` — the medpy-style percentile over the pooled bidirectional
  distances, kept for reference only.
* `assd` — mean of the pooled surface distances.
* `boundary_f1` — tolerance-based boundary F1 (equivalent to surface Dice) at
  `metrics.boundary_tolerance_mm`.
* `pred_volume_mm3`, `gt_volume_mm3`, `volume_difference_mm3`.

Surfaces are boundary voxels (`mask & ~erode(mask)`), and the outside of the array
counts as background. Empty masks are never silently dropped: the default
`empty_policy: nan` leaves undefined metrics as `nan` and every row carries an
`empty_flag` (`both_nonempty`, `both_empty`, `pred_empty`, `gt_empty`);
`penalize` (image diagonal / boundary F1 = 0) is reserved for the sensitivity
table.

## Input normalization

Images are loaded into `[0, 1]` and then normalized with the encoder's own
preprocessing statistics (`configs/base.yaml` → `dataset.normalization`):

```yaml
dataset:
  normalization:
    kind: imagenet     # imagenet | custom | none
    mean: null         # optional override
    std: null
```

With `kind: imagenet` the mean/std are read from
`smp.encoders.get_preprocessing_params(model.encoder, model.encoder_weights)` at
dataset build time (ResNet-50 → `mean=(0.485, 0.456, 0.406)`,
`std=(0.229, 0.224, 0.225)`), and `custom` accepts explicit values. The metadata
JSON records the resolved statistics, and a test compares a dataset sample with a
hand-computed normalization to prevent a silent regression.

## Per-view uncertainty and cost

The runner keeps the individual view probabilities (after inverse transforms and
resampling onto the native grid) and writes, per case and method:

* `entropy_mean`, `variance_mean` — mean binary predictive entropy and mean
  voxelwise variance across views;
* `entropy_mean_roi`, `variance_mean_roi` — the same restricted to the predicted
  foreground plus a 3-pixel boundary band
  (`evaluation.uncertainty_boundary_band_px`);
* `view_disagreement` — fraction of pixels where the binarised views disagree;
* `per_view_dice_mean`, `per_view_dice_std` — Dice of each view against the
  consensus map;
* `latency_ms`, `peak_memory_mb` — wall-clock per case (3 warm-up passes, CUDA
  synchronised) and `torch.cuda.max_memory_allocated` for that view count.

`evaluation.save_probability_maps: true` writes the float16 mean probability map of
the methods listed in `evaluation.probability_map_methods` (default `[baseline, tta4]`)
to `evaluation.probability_map_dir/fold<k>_seed<s>/<case>__<method>__mean.npy`
(native grid, gitignored). Those maps are what `scripts/aggregate.py` averages into
the `ensemble3` / `ensemble3_tta4` predictions; `evaluation.save_variance_maps: true`
additionally writes `*__var.npy`.

## Normalization

`dataset.normalization.kind` selects the input statistics:

| kind | mean/std | when |
| --- | --- | --- |
| `imagenet` (default) | encoder's own values from `smp.encoders.get_preprocessing_params` | pretrained smp encoders |
| `dataset` | per-fold **training-set** mean/std from `dataset.normalization.stats_file` | models trained from scratch (`monai_swinunetr`) |
| `custom` | explicit `mean`/`std` lists | other encoders |
| `none` | no normalization | debugging only |

`dataset` statistics are computed once and cached in JSON:

```bash
python scripts/compute_normalization.py --config configs/cvc_clinicdb/swinunetr_2d.yaml
```

Every fold uses the statistics of *its own* training split (never validation or test
frames), so switching folds cannot leak test intensities into the input scaling.

## Data loading (CVC-ClinicDB)

The distributed TIFFs are read with **tifffile**, not OpenCV. Their
photometric/ExtraSamples header is malformed; libtiff warns about it and OpenCV then
collapses the frame to a single channel and replicates it, so `cv2.imread` returns
three *identical* channels. `tifffile.imread` returns the real RGB planes (frame 1:
channel means 110.6 / 68.7 / 39.9), and the channel order is already RGB, so **no
BGR→RGB swap is applied** (`melba/data/cvc.py`, guarded by
`tests/test_cvc_color_loading.py`).

* frames: `(H, W, 3)` `uint8`, native `288x384`, resized to `dataset.image_size`;
* masks: single-channel `uint8`. They are **not** strictly 0/255 — the lesion border
  carries anti-aliasing values (e.g. 3…112) — so they are binarised at
  `dataset.mask_threshold` (default 127) after nearest-neighbour resizing;
* `dataset.normalization.kind: dataset` statistics are therefore computed from the
  three real colour planes (`data/splits/cvc_clinicdb_norm_stats.json`: fold0 mean
  `[0.4107, 0.2795, 0.1876]`, std `[0.3052, 0.2117, 0.1405]`).

Runs produced before this fix used grey input and are **not results**: they were moved
to `archive_grayscale/` (gitignored) and are kept only to quantify the change.

## Missed detection and empty-mask handling

An empty prediction on a frame that contains a lesion is recorded explicitly:

* per-case CSV columns `empty_flag` (`both_nonempty`, `pred_empty`, `gt_empty`,
  `both_empty`) and `missed_detection` (1 when `pred_empty` with a non-empty GT);
* `run_stats.py` writes `detection_<unit>.csv` with each method's missed-detection
  rate, a **cluster (video-level) bootstrap CI**, the transition counts
  (`nonempty -> empty`, `empty -> nonempty`) against the baseline and an **exact
  McNemar test** (Holm-corrected);
* the primary paired comparison of `hd`, `hd95`, `hd95_pooled` and `assd` uses only
  cases that are non-empty **under both methods** — otherwise a method that misses
  lesions would drop those cases and be rewarded with a lower distance
  (`n_cases_used` and `restriction` are reported per comparison);
* `sensitivity_<treatment>_vs_<control>_<unit>.csv` repeats the comparison on all
  cases using the `<metric>_penalized` columns (empty prediction → image diagonal),
  which the runner emits when `evaluation.emit_penalized_metrics: true` (default).

## TTA views

Views live in YAML and are built by `melba.transforms.build_views`. The shipped
configs are the geometric subsets of the dihedral group **D4** of the square with
exactly 1, 2, 4 and 8 views:

| method | views | membership |
| --- | --- | --- |
| `baseline` | 1 | `{I}` |
| `tta2` | 2 | `{I, F_h}` (identity + horizontal flip) |
| `tta4` | 4 | `{I, R180, F_h, F_v}` (Klein four-subgroup) |
| `tta8` | 8 | `{I, R90, R180, R270, F_h, F_h R90, F_h R180, F_h R270}` (full D4) |

```yaml
tta:
  aggregator: mean
  methods:
    - name: tta4
      views:
        - { name: identity, ops: [] }
        - { name: rot180, ops: [{ type: rot90, k: 2, axes: [h, w] }] }
        - { name: hflip, ops: [{ type: flip, axes: [w] }] }
        - { name: vflip, ops: [{ type: flip, axes: [h] }] }
```

Supported ops: `identity`, `flip`, `rot90`, `transpose`, `zoom`. Axes are `w`
(last), `h`, `d`/`z`. The inverse is derived from the ops, so a new view cannot
desynchronise from its inverse (tested for every configured view). `zoom` remains
implemented and tested but is **experimental**: it is not part of the default
methods and must be enabled explicitly in a config.

## Determinism

`melba.utils.seed_everything(seed)` seeds `random`, NumPy and PyTorch (CPU + all
CUDA devices), sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables
`cudnn.deterministic` and `torch.use_deterministic_algorithms(True,
warn_only=True)`, and data-loader workers are seeded per worker. Training
augmentation is generated from `(seed, epoch, index)`, and the training subset for
`training.data_fraction` is drawn from whole videos. `created_utc` and the timing
columns (`latency_ms`, `peak_memory_mb`) are the only non-deterministic fields;
all quality metrics, ids and file names are deterministic.

Training ablations are switched from the config: `training.train_aug`
(`geometric_d4`, `flips_only`, `intensity`, `none`), `training.data_fraction`
(seeded fraction of training videos) and `model.dropout_p` (encoder-feature
dropout during training only; state-dict keys are unchanged).

## Testing

```bash
python -m pytest -q
```

Coverage highlights: brute-force validation of surface distances (including
anisotropic spacing), HD95 robustness to a small outlier blob, empty-mask policies,
every transform/inverse pair plus every view in the shipped config, determinism of
group splits, leakage detection, TTA inversion/aggregation, bootstrap/Wilcoxon/Holm
behaviour, native-grid metric computation, checkpoint round-trips, CLI wiring, and
an end-to-end evaluation run that checks the long-format schema, provenance and
both split guards. The journal-extension additions are covered as well: HD95 parity
with `monai.metrics.compute_hausdorff_distance`, exact resume (atomic `last.pt`,
RNG round-trip, epoch continuation, per-run directories), grid state classification
(skip / evaluate-only / resume / fresh), timing gating (warm-ups only for fold 0 /
seed 0), probability-map selection, per-fold dataset statistics, pooled seed tables,
seed-averaged paired tests, the seed ensemble and 2D SwinUNETR forward passes, and
colour loading of the real TIFFs (3 distinct channels, native `288x384`, mask
binarisation at 127).

## Status / next steps

* CVC-ClinicDB (2D) pipeline is complete and tested end-to-end (5 folds x 3 seeds,
  resumable grid, pooled statistics, seed ensembles, uncertainty/cost analysis).
  Colour input (tifffile) is required for a reported number: the earlier grey runs
  were archived to `archive_grayscale/` and are invalid.
* `configs/cvc_clinicdb/swinunetr_2d.yaml` (MONAI SwinUNETR, `spatial_dims=2`, from
  scratch, per-fold training statistics) and its factory branch are implemented and
  unit-tested, but no SwinUNETR results have been produced yet.
* BraTS 2021 (3D) is **not started**: it needs an NIfTI reader and sliding-window
  inference. The `monai_segresnet` branch exists but has no dataset reader.
* The CVC-ClinicDB frame → sequence grouping is built reproducibly from the official
  end-frame table by `scripts/make_cvc_mapping.py` (`dataset.split.sequence_end_frames`)
  and validated against the image stems before any split is written;
  `dataset.split.strategy: contiguous_blocks` remains available only as an
  explicitly labelled proxy.
* Legacy `src/` code has been removed; `results/*.csv` moved unchanged to
  `legacy_results/`.

