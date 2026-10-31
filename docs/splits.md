# Splits: why groups, and how to supply them

## The problem the old code had

CVC-ClinicDB contains 612 frames extracted from a small number of colonoscopy
videos. Consecutive frames of one video are near duplicates: the same polyp, the
same lighting, millimetre-level camera motion. The old code split the *frames*
(80/20 with `train_test_split`), so almost every "test" frame had a visually nearly
identical twin in training. That inflates Dice and shrinks HD95 for reasons that
have nothing to do with TTA.

## What this repository requires

1. A mapping from every frame to its source sequence (a `group_id`):

   ```csv
   case_id,group_id
   1,seq01
   2,seq01
   3,seq02
   ...
   ```

   This is referenced by `dataset.split.group_map_file` in the experiment config
   (column names configurable via `frame_column` / `group_column`).

   CVC-ClinicDB filenames are global frame numbers, so the mapping comes from the
   official end-frame table of each sequence (Bernal et al., 2015), stored in the
   config as `dataset.split.sequence_end_frames` (29 values, last one 612). Build
   the CSV reproducibly with

   ```bash
   python scripts/make_cvc_mapping.py --config configs/cvc_clinicdb/unet_resnet50.yaml
   ```

   The script expands `sequence k = (previous end + 1) .. end_k`, validates
   612 frames / 29 sequences, and **fails loudly** if the mapping does not match the
   image filename stems on disk (missing/unknown ids are printed).

2. Five fold files produced by

   ```bash
   python scripts/make_splits.py --config configs/cvc_clinicdb/unet_resnet50.yaml
   ```

   It writes `dataset.split.output_pattern`
   (`data/splits/cvc_clinicdb_fold{0..4}.csv`, columns
   `case_id,group_id,split,fold`) plus `cvc_clinicdb_folds.json` with the strategy,
   seed, per-fold group/case counts and the generated file names.

   For fold `k` the test set is fold `k`, the validation set is fold `k+1`
   (rotating) and the training set is the remaining folds. Every video is therefore
   a test unit in exactly one fold, so per-case results can be pooled over the five
   folds for the final statistics.

## Guarantees enforced in code

`melba.data.splits` refuses to proceed when:

* a frame has no group mapping (`grouped_split` / `folded_splits`), instead of
  silently falling back to frame-level splitting;
* the mapping's `case_id` values do not match the image filename stems exactly
  (missing or unknown ids) — `scripts/make_splits.py` prints both counts and the
  first 20 offending ids;
* fewer than three groups (or fewer groups than folds) are available;
* any group appears in more than one split (`assert_no_group_leakage`), checked
  both after splitting and when reading an existing split CSV;
* a group is a test unit in more than one fold, or a case is tested twice
  (`assert_folds_cover_once`);
* a case appears twice, a split name is unknown or a fold id is invalid
  (`validate_records`);
* the split file and the files on disk disagree (`load_cvc_items`), which would
  otherwise silently evaluate a different set of frames than the training run used;
* the fold files would be overwritten without `--force`.

## Pooling folds for statistics

`scripts/run_stats.py` accepts several per-case files and pools them:

```bash
python scripts/run_stats.py --config configs/cvc_clinicdb/unet_resnet50.yaml \
    --per-case results/*/evaluation/cvc_clinicdb_test_fold{0,1,2,3,4}_per_case.csv \
    --treatment tta8
```

`melba.stats.pool_folds` validates that every case and every video appears in
exactly one fold before testing, and `analyze_pooled_folds` reports `n_folds`,
`n_groups_pooled` and `n_cases_pooled`. With `unit: group` the paired test uses one
aggregated value per video (29 videos for CVC-ClinicDB), not per frame.

`scripts/evaluate.py` additionally refuses to report the split used for checkpoint
selection (`model.selection_split`, default `val`) and refuses the train split.

## Fractions

Fractions apply to **groups**, not frames: with 29 videos and
`{train: 0.7, val: 0.15, test: 0.15}` you get whole videos per split. Because video
lengths differ, the resulting *frame* fractions are reported by `make_splits` (and
stored in the sidecar JSON) rather than assumed.

## Fallback: `contiguous_blocks` (proxy only)

CVC-ClinicDB frames are stored in acquisition order, so grouping consecutive frames
in blocks of `dataset.split.block_size` (default 20) is a *proxy* for video
identity. It is not guaranteed to match the true videos. The script prints a
warning; if an experiment uses it, that limitation must be stated in the paper, and
the resulting numbers should be treated as exploratory.
