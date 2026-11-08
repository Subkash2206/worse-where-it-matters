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
  cvc_clinicdb/unet_resnet50_dropout.yaml  same + decoder dropout, MC-dropout ready
  cvc_clinicdb/unet_resnet50_data025.yaml  25 % of the training videos (sweep)
  cvc_clinicdb/unet_resnet50_data050.yaml  50 % of the training videos (sweep)
  cvc_clinicdb/swinunetr_2d.yaml     MONAI SwinUNETR (2D, from scratch), dataset statistics
data/splits/                 frame -> video mapping (yours) and generated fold CSVs
docs/splits.md               how to provide the CVC-ClinicDB frame → video mapping
scripts/
  make_splits.py             group-level split creation (writes data/splits/*.csv)
  compute_normalization.py   per-fold training-set mean/std (from-scratch models)
  train.py                   training, val-only checkpoint selection, resumable
  evaluate.py                per-case evaluation of every configured TTA method
  run_grid.py                folds x seeds grid; resumes whatever is incomplete
  run_one.py                 one fold/seed job in its own process (train + evaluate)
  aggregate.py               pooled fold/seed tables, paired stats, seed ensembles
  compare_models.py          two experiments side by side + residual-error test
  launch_fraction_sweep.py   data-fraction grids (dry run unless --execute)
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
moment and the next launch continues the same trajectory.

Each run executes in its **own child process** (`scripts/run_one.py`): a CUDA/cuDNN
crash (e.g. `CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH`) destroys only that child's CUDA
context, so it cannot poison the next run. The parent then frees GPU memory, waits
30 s and retries the run from its `last.pt`, up to **three** times; a run that still
fails is logged, marked failed, and the grid continues with the next run and prints a
done/skipped/failed summary at the end (`logs/cvc_unet_grid_summary.json`). Child
exit codes: `0` success, `2` configuration error, `3` CUDA/cuDNN (retryable), `4`
anything else.

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

Writes under `results/aggregate_<experiment>/` — one directory per experiment
(`results/aggregate_cvc_clinicdb_unet_resnet50/` for the default config), so the UNet
and SwinUNETR grids and the data-fraction variants never overwrite each other's tables;
`--output-dir` overrides it. Contents (CSV per table plus a `summary.md` report):

* `pooled_by_seed.csv` — per method and metric, the pooled 29-video result of each
  seed, the mean and the standard deviation across seeds;
* `method_summary.csv` — per method/metric at **frame level and video level** (mean ±
  std across seeds), including the two seed ensembles;
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
  < 5e-4 per case and distance metrics by < 0.6 px relative to the float32 runner;
* `tail_metrics.csv` — 90th/95th percentile HD95 and failure rates (HD95 > 50 px,
  Dice < 0.5) with a video-cluster bootstrap CI of the change vs the control;
* `sensitivity_penalized.csv` — all-cases comparison with empty predictions penalised
  by the image diagonal (survivorship-bias check);
* `uncertainty.csv` — Spearman rho of `view_disagreement` / `entropy_mean_roi`
  against HD95 and 1-Dice, plus AUROC for detecting failures;
* `uncertainty_audit_*.csv` — the audit of those descriptors: ROI diagnostics for the
  failures (empty / near-empty / size buckets), AUROC **and AURC** under three failure
  definitions (all, non-missed, missed-only) with video-cluster bootstrap CIs, explicit
  `n_used`/`n_undefined` counts and NaN bounds, Spearman for every predictor, and
  paired bootstrap tests against trivial baselines (predicted area, 1 − confidence,
  image entropy, single-view entropy) and the seed-ensemble entropy/variance. Every
  predictor is scored against the failures of **its own** prediction (`label_source:
  own`: baseline vs baseline, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4
  map vs ensemble3_tta4) and the table repeats the audit against the shared tta4
  labels (`label_source: tta4`) for comparison. The ROI descriptors are **undefined
  for an empty prediction**, so a plain AUROC silently drops missed detections — this
  table makes that explicit;
* AURC (`aurc`, `aurc_ci_low`, `aurc_ci_high`) is the area under the risk-coverage
  curve of a detector with `risk = 1 − Dice` of the prediction it scores, i.e. the mean
  selective risk when the most confident cases are kept first (ties are averaged inside
  the tie block, so the value does not depend on an arbitrary tie-break). Lower is
  better. `aurc_optimal` orders the same cohort by its own risk (the oracle lower
  bound), `aurc_excess` is the gap to it, and `aurc_if_missed_is_worst` repeats the
  point estimate with missed detections counted as risk 1 instead of being dropped
  (`n_used_aurc` / `n_undefined_aurc` say how many cases each value used);
* **agreement detectors**: `tta4_view_dice_disagreement` = 1 − mean pairwise Dice
  between the four tta4 views, `ensemble3_dice_disagreement` and
  `ensemble3_tta4_dice_disagreement` = the same over the three seeds' baseline / tta4
  maps. The seed-map versions come from the saved probability maps; the view version
  needs per-view masks, which the runner does not store, so it is read from a separate
  CPU re-run (below). The paired tests table compares
  `tta4_view_dice_disagreement` against the single-view ROI entropy and against
  1 − confidence on **both** AUROC and AURC with one shared video-cluster resampling
  (`delta_aurc < 0` means the first detector is the better one);
* `per_fold_baseline.csv` — fold-to-fold spread of the control method;
* `summary.md` — all of the above rendered as markdown with the protocol notes.

The view-agreement descriptor needs the individual view predictions, so it is produced
by re-running tta4 only, on CPU, into its own directory:

```bash
for fold in 0 1 2 3 4; do for seed in 0 1 2; do
  python scripts/evaluate.py --config configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml \
      --fold $fold --set seed=$seed --no-progress
done; done
```

`scripts/aggregate.py` picks up `results/<experiment>/view_agreement/evaluation/*.csv`
automatically (override with `--view-agreement-dir`, skip with `--no-view-agreement`)
and averages `view_pairwise_dice` per case over the seeds. Without it the detector is
NaN and a note in `summary.md` says why.

Any config value can be overridden without editing files:
`--set model.checkpoint=checkpoints/x.pt metrics.empty_policy=nan device=cpu`.

## Cross-model comparison (`scripts/compare_models.py`)

```bash
python scripts/compare_models.py \
    --configs configs/cvc_clinicdb/unet_resnet50.yaml configs/cvc_clinicdb/swinunetr_2d.yaml
```

Loads both grids (each fold's per-case CSVs plus the probability maps for
`ensemble3`, and the reference masks once, since both models are evaluated on the same
folds) and writes to `results/compare_models/`:

* `methods.csv` — `baseline` / `tta4` / `tta8` / `ensemble3`, Dice / BF1 / HD95, per
  model, as mean ± std over the 29 videos;
* `deltas.csv` — the paired tests of `scripts/aggregate.py` (Wilcoxon, bootstrap CI of
  the mean delta, rank-biserial effect size, Holm) against each model's own baseline;
* `detection.csv` — missed-detection rates, transitions and exact McNemar per model;
* `tails.csv` — HD95 percentiles and failure rates (HD95 > 50 px, Dice < 0.5) per model;
* `residual_error.csv` — the **residual-error test** reviewers ask for. Correlating the
  control error with `delta = tta4 − baseline` is biased towards a negative rho by
  regression to the mean, so the **primary** statistics are (a) **Oldham's method**,
  `Spearman((baseline + tta4)/2, tta4 − baseline)` — the x-axis becomes the mean of the
  pair, which is what removes that bias — and (b) a **paired-swap permutation null**:
  within each permutation whole *videos* are swapped (all of their cases exchange
  baseline and tta4, so the within-video dependence survives), `delta` flips sign and
  both statistics are recomputed (`n_permutations`, default 10 000); the table reports
  the null mean and the two-sided `|rho_perm| ≥ |rho_obs|` p-value. The raw
  `Spearman(baseline, delta)` is kept as the explicitly secondary `rho_naive` column.
  Both metrics are covered (HD95 on cases non-empty under both methods, Dice on all
  cases), rows are per model plus one row pooled over both models, and every CI is a
  **video-cluster bootstrap** (videos resampled as blocks). `delta = treatment −
  control`, so a negative rho means the gain grows as the control gets worse — for HD95
  and Dice alike;
* `gain_difference.csv` — the **between-model** question: is the mean video-level TTA
  gain of one model different from the other's? `gain = tta4 − baseline` per case,
  averaged per video inside each model, paired by video, tested with a Wilcoxon
  signed-rank test and a paired video bootstrap CI of `delta_gain = compared −
  reference` (`favours` names the model with the larger gain, `better_model` the model
  with the better score). Every metric appears on **two scales**, marked by the `scale`
  column and described in `relative_definition`:
  `scale=absolute` is the raw difference in metric units (dominated by whichever model
  works on the larger scale — SwinUNETR's HD95 is ~90 px against the UNet's ~40 px),
  while `scale=relative` is scale-free: `log_ratio` = `log(tta4 / baseline)` for distances
  (HD95, ASSD, …), so a 10 % reduction counts the same at 90 px and at 40 px, and
  `remaining_error` = `(tta4 − baseline) / (1 − baseline)` for Dice, the fraction of the
  error the treatment recovers. The relative rows need both values positive (distances)
  or `baseline < 1` (Dice), so their `n_units` can be one smaller. Needs two
  `--configs`; with one config the per-model residual-error rows are still produced;
* `comparison.md` — all of the above as one markdown report.

## Data-fraction sweep (configs only)

Two configs answer "how much of the TTA gain survives less training data?" by changing
only `training.data_fraction` (a seeded fraction of whole training **videos**, so no
frame of a held-out video leaks into training):

* `configs/cvc_clinicdb/unet_resnet50_data025.yaml` — 25 % of the training videos;
* `configs/cvc_clinicdb/unet_resnet50_data050.yaml` — 50 % of the training videos
  (100 % is the parent `unet_resnet50.yaml`).

Each variant has its own `experiment.name`, hence its own checkpoint, evaluation, stats
and `results/aggregate_<experiment>/` directories — nothing overwrites the full-data run.
The launcher is dry by design:

```bash
python scripts/launch_fraction_sweep.py            # prints the two run_grid commands
python scripts/launch_fraction_sweep.py --execute  # starts them detached, folds 0-4, seed 0
```

No fraction runs have been launched yet (one GPU job at a time).

## Overnight queue (`scripts/run_queue.py`)

One GPU job at a time is a hard rule, so unattended work is expressed as a queue in
`queues/overnight.yaml`: a `wait_for_pid_file` guard, then grid steps, then CPU-only
post-processing. Each grid step is a plain `scripts/run_grid.py` call with its own
`logs/<name>_grid.log` and `logs/<name>_grid.pid`, so per-run retries, `last.pt` resume
and "trained but not evaluated" recovery behave exactly as in a manual launch; a grid
that comes back non-zero is re-launched once (it resumes).

```bash
python scripts/run_queue.py --queue queues/overnight.yaml --dry-run   # plan only
python scripts/run_queue.py --queue queues/overnight.yaml             # run it
python scripts/run_queue.py --queue queues/overnight.yaml --only compare_unet_swin
```

```powershell
# check progress (queue + the grid it is currently running)
Get-Content logs\overnight.log -Tail 20
Get-Content logs\overnight_summary.json -ErrorAction SilentlyContinue
$step = (Get-ChildItem logs\*_grid.pid | Sort-Object LastWriteTime | Select-Object -Last 1)
"$($step.Name) -> pid $(Get-Content $step.FullName)"; Get-Content $step.BaseName.log -Tail 5

# stop everything the queue started (queue -> run_grid -> run_one, deepest first)
function Stop-Tree([int]$Id) { Get-CimInstance Win32_Process -Filter "ParentProcessId=$Id" | ForEach-Object { Stop-Tree $_.ProcessId }; Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue }
Stop-Tree (Get-Content logs\overnight.pid)

# resume: start the same queue again — finished runs are skipped, interrupted ones continue
python scripts\run_queue.py --queue queues\overnight.yaml
```

A grid can also be driven alone (`python scripts/run_grid.py --config ... --status-only`
prints the state of every run), and the queue file can be edited between runs; re-running
it is always safe.

## Status report (`scripts/status.py`)

```bash
python scripts/status.py                # everything
python scripts/status.py --tail 8       # more lines of the active job's log
python scripts/status.py --only swin dropout
```

One compact, read-only report — it imports neither torch nor the GPU stack, writes
nothing and takes a few seconds:

```
MELBA status  2026-11-01 16:42  (read-only, CPU only)
queue   overnight  ALIVE pid 31804      waiting for cvc_swin_grid.pid (pid 29940)
grid    cvc_swin_grid          ALIVE pid 29940 (python)
stale   view_agreement_unet (29860), view_agreement_unet_catchup (27840)

experiment                            runs  failed  current                            valDice      ETA
unet                                 15/15    0+1r  -                            0.8725/0.8094       0s
swin                                 11/15       0  fold3/seed2                  0.7021/0.5809  49m 11s
dropout                               0/15       0  -                                        -        ?
...
```

* `runs` = evaluated runs / planned runs (`evaluation` marker present, the same test
  `run_grid.py` uses to skip a run);
* `failed` counts failures that are *still* unresolved (`ERROR fold f seed s` in the grid
  log whose run has no evaluation yet); `+Nr` means the grid retried N of them and the
  retry/resume succeeded;
* `current` is the run the grid log points at, with its epoch (`training_log.csv` only
  exists once a run finishes, so mid-training progress is read from the log's
  `[train] epoch e/E` lines);
* `valDice` = best single run / mean of the per-run bests, both read from
  `run_done.json` (finished runs) and the log (the active run);
* `ETA` = median finished-run wall-clock × pending runs, except for the active run, which
  pays only for its remaining epochs (per-epoch time from the log);
* liveness comes from `logs/*.pid` **with the image name checked**
  (`melba.pidfile.pid_alive`), because Windows recycles pids — a stale pid file pointing
  at an unrelated process is reported as `stale` instead of `ALIVE`;
* the last `--tail` (default 3) non-empty lines of the active job's log are printed last.

`brats` is listed with `config missing` until the 3D milestone gets a config, so an
experiment can never silently disappear from the report.

## BraTS 2021 (3D, enhancing tumour)

`configs/brats2021/segresnet.yaml` points `dataset.root` at the real BraTS 2021 training
directory (`C:/Users/subka/Documents/tta_boundary_study/brats2021`, **read-only**: nothing
in the pipeline writes below it) and `configs/brats2021/segresnet_reduced.yaml` overrides
only the training budget.

```bash
python scripts/validate_brats.py --config configs/brats2021/segresnet.yaml   # data audit
python scripts/make_brats_splits.py --config configs/brats2021/segresnet.yaml  # fold CSVs
python scripts/make_brats_splits.py --config ... --force    # regenerate the CSVs
```

`scripts/validate_brats.py` is a read-only audit: 1251 cases, every one with
flair/t1/t1ce/t2/seg, one shape `240x240x155`, one spacing `1.0x1.0x1.0 mm` (from the
affine), labels `{0, 1, 2, 4}` only, **no bad or missing case**, and **33 cases without
enhancing tumour** (listed in `data/splits/brats2021_validation.json` together with the
label voxel counts). Those 33 are kept: an empty reference makes Dice undefined for that
case, so the aggregate reports it instead of dropping the patient silently.

`scripts/make_brats_splits.py` writes the case-level folds (`data/splits/brats2021_fold*.csv`,
committed) with test = fold *k*, validation = fold *k+1* (rotating), train = the rest, split
seed 2024 — fold 0: 750 train / 250 val / 251 test, fold 4: 754/251/250, each fold holding
250-251 patients. `data/splits/brats2021_empty_et.json` lists the empty-ET patients **per
fold** (e.g. fold 0: 8 in test, 4 in val).

Pipeline: 4 modalities in fixed order, per-modality 1-99 percentile clip + z-score over
brain voxels (background exactly 0), ET = label 4, distances in mm from the affine,
foreground-aware 128^3 training crops, MONAI sliding-window inference (128^3, overlap 0.5,
gaussian) for validation and test, `segresnet_3d` (SegResNet, `init_filters` 16, from
scratch), batch 1, AMP, and 1/2/4/8-view 3D TTA sets of the axial dihedral group times a
depth flip.

Measured on the real data (fold 0, RTX 4050 6 GB, AMP, 40 patches/epoch, 40 val volumes):

| measurement | value |
|---|---|
| peak VRAM | **2663 MiB allocated / 3212 MiB reserved** of 6 GB |
| training | **1.53 s per 128^3 patch** (includes reading + normalizing a 240^3 volume) |
| validation | **2.23 s per volume** (18 sliding windows of 128^3) |
| epoch | 152.9 s (median over two epochs) |

The reduced protocol (`segresnet_reduced.yaml`: 5 folds x 1 seed, 100 epochs max, patience
20 on val Dice, **250 random patches per epoch** via `training.max_patches_per_epoch`,
validation on a fixed seeded **40-case** subsample via `training.val_limit`, test on the
whole test fold with all four TTA methods, float16 mean maps for `baseline` and `tta4`)
therefore costs ~7.9 min per epoch, i.e. **~6.5 h of training for 50 epochs plus ~2.3 h of
test-time TTA evaluation per run, ~44 h for the five folds** (~31 h if early stopping lands
near 30 epochs, ~77 h if every fold runs the full 100). The grid has **not** been launched:

```bash
python scripts/run_grid.py --config configs/brats2021/segresnet_reduced.yaml \
    --folds 0 1 2 3 4 --seeds 0 --epochs 100 --patience 20 \
    --pid-file logs/cvc_brats_reduced_grid.pid
```

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
* `view_pairwise_dice` — mean Dice over every *pair* of binarised views (1.0 for a
  single view); the view-agreement detector used by the uncertainty audit;
* `latency_ms`, `peak_memory_mb` — wall-clock per case (3 warm-up passes, CUDA
  synchronised) and `torch.cuda.max_memory_allocated` for that view count.

`evaluation.save_probability_maps: true` writes the float16 mean probability map of
the methods listed in `evaluation.probability_map_methods` (default `[baseline, tta4]`)
to `evaluation.probability_map_dir/fold<k>_seed<s>/<case>__<method>__mean.npy`
(native grid, gitignored). Those maps are what `scripts/aggregate.py` averages into
the `ensemble3` / `ensemble3_tta4` predictions; `evaluation.save_variance_maps: true`
additionally writes `*__var.npy`.

### MC-dropout (`mcdropout_T`)

A method named `mcdropout` or `mcdropout_T` (default `T = 8`) is a **stochastic**
method: it expands into `T` identity views and the runner switches **decoder dropout**
on for those forward passes while keeping the model in `eval()` mode, so BatchNorm keeps
using its running statistics and never updates them. The `T` per-sample probability
maps go through exactly the same aggregation and uncertainty code as geometric views,
so `entropy_mean`, `variance_mean`, `view_disagreement`, ... are populated as usual and
`mcdropout_T` sits in the same tables as `tta4`/`tta8`.

Add it to a config without redeclaring the geometric methods:

```yaml
model:
  decoder_dropout_p: 0.2       # switchable decoder dropout (training + MC sampling)
tta:
  extra_methods:
    - mcdropout_8
```

`model.dropout_p` is the *encoder-feature* dropout used during training (state-dict
keys unchanged); `model.decoder_dropout_p` is the one MC sampling can switch on. A model
trained without it produces identical "samples", so the runner raises `ConfigError`
instead of writing a zero-variance uncertainty column. `configs/cvc_clinicdb/unet_resnet50_dropout.yaml`
is the ready-to-train variant (decoder dropout 0.2, `mcdropout_8` added); no MC-dropout
runs have been launched yet.

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
binarisation at 127). Grid robustness is covered too: a child process per run, the
CUDA-vs-other failure classification, retries with GPU cleanup, exhausted retries
being reported instead of raised, and `run_one.py` training then reporting
"nothing to do" end-to-end. The uncertainty extensions add: MC-dropout sampling that
keeps BatchNorm statistics frozen (and dropout off again after each run), `T` differing
samples, `mcdropout_T`/`extra_methods` expansion, the runner refusing stochastic methods
on a model without decoder dropout, the per-experiment aggregate directory, the derived
data-fraction/dropout configs, and the cross-model tables. The detector and
residual-error work adds: AURC of a ranking (hand-computed values, tie-invariance and
the oracle lower bound), the paired AUROC/AURC bootstrap, pairwise-Dice agreement over
views and seeds, `load_view_agreement` averaging seeds from a directory of per-case
CSVs, Oldham's mean-vs-change correlation and the paired-swap permutation null, plus a
zero-true-effect simulation in which the naive rho is significantly negative while
Oldham's rho and the swap null are not, and a between-model gain difference.

## Status / next steps

* CVC-ClinicDB (2D) pipeline is complete and tested end-to-end (5 folds x 3 seeds,
  resumable grid, pooled statistics, seed ensembles, uncertainty/cost analysis).
  Colour input (tifffile) is required for a reported number: the earlier grey runs
  were archived to `archive_grayscale/` and are invalid.
* `configs/cvc_clinicdb/swinunetr_2d.yaml` (MONAI SwinUNETR, `spatial_dims=2`, from
  scratch, per-fold training statistics) and its factory branch are implemented and
  unit-tested; the 5 folds x 3 seeds grid is running (same protocol and evaluation code
  as the UNet grid). Aggregate it separately with
  `python scripts/aggregate.py --config configs/cvc_clinicdb/swinunetr_2d.yaml`
  (default output `results/aggregate_cvc_clinicdb_swinunetr_2d/`) and compare with
  `python scripts/compare_models.py --configs configs/cvc_clinicdb/unet_resnet50.yaml
  configs/cvc_clinicdb/swinunetr_2d.yaml`.
* MC-dropout (`mcdropout_T`, `configs/cvc_clinicdb/unet_resnet50_dropout.yaml`) and the
  data-fraction sweep (`data025` / `data050`) are implemented, tested and ready to
  launch; neither has been run (GPU is busy with the SwinUNETR grid, one job at a time).
* BraTS 2021 (3D) is **not started**: it needs an NIfTI reader and sliding-window
  inference. The `monai_segresnet` branch exists but has no dataset reader.
* The CVC-ClinicDB frame → sequence grouping is built reproducibly from the official
  end-frame table by `scripts/make_cvc_mapping.py` (`dataset.split.sequence_end_frames`)
  and validated against the image stems before any split is written;
  `dataset.split.strategy: contiguous_blocks` remains available only as an
  explicitly labelled proxy.
* Legacy `src/` code has been removed; `results/*.csv` moved unchanged to
  `legacy_results/`.

