# MELBA — geometry-aware TTA for medical image segmentation

Journal extension (MELBA) of the MICCAI UNSURE workshop paper on geometric
test-time augmentation (TTA) for medical image segmentation. This repository is a
clean rebuild: the previous code base (`github.com/Subkash2206/tta-medseg-evaluation`,
still present under `src/` for reference only) is **not** used for any reported
number.

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
| Hardcoded paths, seeds, device, hyper-parameters | everything is in `configs/*.yaml`; `device: auto`; only YAML is authoritative |

## Layout

```
configs/                     YAML experiment configs (base.yaml + dataset/model configs)
  base.yaml                  shared defaults: seed, paths, metrics, stats, runtime
  cvc_clinicdb/unet_resnet50.yaml
data/splits/                 frame -> video mapping (yours) and generated fold CSVs
docs/splits.md               how to provide the CVC-ClinicDB frame → video mapping
scripts/
  make_splits.py             group-level split creation (writes data/splits/*.csv)
  train.py                   training, val-only checkpoint selection
  evaluate.py                per-case evaluation of every configured TTA method
  run_stats.py               paired statistics over a per-case CSV
src/melba/
  config.py                  YAML loading, defaults, ${refs}, overrides, fingerprint
  utils/                     determinism (torch/numpy/random/cudnn), device, provenance
  metrics/                   overlap + spacing-aware surface metrics, empty-mask policy
  transforms/                invertible geometric transforms, YAML → TTA views
  inference/                 TTA inference with inverse-transformed aggregation
  data/                      video-level splits, CVC-ClinicDB dataset, data loaders
  models/                    smp UNet / MONAI SegResNet factory, checkpoint loading
  eval/                      long-format records, per-case runner, run metadata
  stats/                     Wilcoxon + bootstrap CI + effect size + Holm
tests/                       pytest suite (metrics, transforms, TTA, splits, stats, runner)
src/                         legacy code from the workshop paper — reference only
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

   Outputs, under `results/<experiment>/evaluation/`:
   * `cvc_clinicdb_test_fold0_per_case.csv` — long format, one row per
     case/method/metric (with `fold`, `empty_flag`, uncertainty and timing rows);
   * `cvc_clinicdb_test_fold0_summary.csv` — n/mean/std/median/IQR/count per metric;
   * `cvc_clinicdb_test_fold0_run.json` — config hash, seed, environment,
     checkpoint hash, timing setup.

5. **Statistics**, pooling all five folds so each video is a test unit exactly once:

   ```bash
   python scripts/run_stats.py \
       --config configs/cvc_clinicdb/unet_resnet50.yaml \
       --per-case results/cvc_clinicdb_unet_resnet50/evaluation/cvc_clinicdb_test_fold{0,1,2,3,4}_per_case.csv \
       --treatment tta8
   ```

   Pooling is validated: a case or video that appears in more than one fold aborts
   the analysis. With `unit: group` every video contributes one paired value, so
   the sample size is the number of videos (29 for CVC-ClinicDB), not frames.

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

`evaluation.save_probability_maps: true` additionally writes float16 mean and
variance maps (`*.npy`, ignored by git) to `evaluation.probability_map_dir`.

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
both split guards.

## Status / next steps

* CVC-ClinicDB (2D) pipeline is complete and tested end-to-end (5 folds, pooled
  statistics, uncertainty/cost analysis).
* BraTS 2021, SwinUNETR and SegResNet are **not started** (on hold per the review
  decision); the `monai_segresnet` factory branch exists but has no dataset reader.
* The CVC-ClinicDB frame → sequence grouping is built reproducibly from the official
  end-frame table by `scripts/make_cvc_mapping.py` (`dataset.split.sequence_end_frames`)
  and validated against the image stems before any split is written;
  `dataset.split.strategy: contiguous_blocks` remains available only as an
  explicitly labelled proxy.
* Legacy `src/` code has been removed; `results/*.csv` moved unchanged to
  `legacy_results/`.
* `scripts/dry_run.py` and `scripts/investigate_bf1.py` still import the removed
  legacy modules and are therefore non-functional; delete or port them on request.

