# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

15 runs: folds [0, 1, 2, 3, 4] x seeds [0, 1, 2]; config `configs/cvc_clinicdb/segformer_mit_b2.yaml`; dataset `cvc_clinicdb`; pairing unit `group` (one video); bootstrap draws `config`; Holm correction. Failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 9.9235 | 0.2279 |
| ensemble3 | assd | 612 | 29 | 0 | 9.2131 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 9.0705 |  |
| tta2 | assd | 612 | 29 | 3 | 9.5060 | 0.3616 |
| tta4 | assd | 612 | 29 | 3 | 9.2732 | 0.1836 |
| tta8 | assd | 612 | 29 | 3 | 9.1293 | 0.2009 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.5170 | 0.0023 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5372 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5607 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.5331 | 0.0074 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5472 | 0.0104 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5537 | 0.0093 |
| baseline | dice | 612 | 29 | 3 | 0.8437 | 0.0032 |
| ensemble3 | dice | 612 | 29 | 0 | 0.8543 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.8593 |  |
| tta2 | dice | 612 | 29 | 3 | 0.8513 | 0.0040 |
| tta4 | dice | 612 | 29 | 3 | 0.8541 | 0.0046 |
| tta8 | dice | 612 | 29 | 3 | 0.8547 | 0.0043 |
| baseline | hd95 | 612 | 29 | 3 | 39.1004 | 0.4668 |
| ensemble3 | hd95 | 612 | 29 | 0 | 36.5976 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 35.2489 |  |
| tta2 | hd95 | 612 | 29 | 3 | 38.1597 | 1.5174 |
| tta4 | hd95 | 612 | 29 | 3 | 36.6545 | 0.5066 |
| tta8 | hd95 | 612 | 29 | 3 | 36.1869 | 0.3029 |
| baseline | hd95_pooled | 612 | 29 | 3 | 36.0543 | 0.4760 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 33.8288 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 32.7523 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 35.1871 | 1.1352 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 33.8337 | 0.2164 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 33.4545 | 0.4739 |

Mean over all cases of the fold, pooled over the 5 folds, then mean ± std across the seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 10.8122 | 0.1077 |
| ensemble3 | assd | 612 | 29 | 0 | 9.8427 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 10.0307 |  |
| tta2 | assd | 612 | 29 | 3 | 10.2585 | 0.2789 |
| tta4 | assd | 612 | 29 | 3 | 10.1447 | 0.4537 |
| tta8 | assd | 612 | 29 | 3 | 9.8315 | 0.2813 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.5064 | 0.0019 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5290 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5519 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.5219 | 0.0062 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5368 | 0.0105 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5432 | 0.0091 |
| baseline | dice | 612 | 29 | 3 | 0.8351 | 0.0016 |
| ensemble3 | dice | 612 | 29 | 0 | 0.8499 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.8529 |  |
| tta2 | dice | 612 | 29 | 3 | 0.8429 | 0.0049 |
| tta4 | dice | 612 | 29 | 3 | 0.8459 | 0.0066 |
| tta8 | dice | 612 | 29 | 3 | 0.8455 | 0.0068 |
| baseline | hd95 | 612 | 29 | 3 | 41.8944 | 0.4977 |
| ensemble3 | hd95 | 612 | 29 | 0 | 39.1785 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 37.5931 |  |
| tta2 | hd95 | 612 | 29 | 3 | 40.5441 | 1.2524 |
| tta4 | hd95 | 612 | 29 | 3 | 39.3327 | 0.6844 |
| tta8 | hd95 | 612 | 29 | 3 | 38.6324 | 0.6217 |
| baseline | hd95_pooled | 612 | 29 | 3 | 38.7695 | 0.5211 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 35.9247 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 35.0409 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 37.5223 | 1.0898 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 36.3914 | 0.3328 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 35.7899 | 0.7465 |

Each video contributes one value (mean over its cases); mean ± std across the seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0077 | 0.0046 | 0.0114 | 0.0000 | 0.0001 | 0.8483 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0096 | 0.0060 | 0.0138 | 0.0000 | 0.0001 | 0.8621 | tta2 |
| tta2 | precision | 29 | 612 | all_cases | 0.0063 | 0.0026 | 0.0108 | 0.0044 | 0.0177 | 0.5908 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | 0.0062 | 0.0029 | 0.0099 | 0.0002 | 0.0011 | 0.7563 | tta2 |
| tta2 | hd | 29 | 612 | both_nonempty | -1.5671 | -3.2063 | -0.1711 | 0.0190 | 0.0571 | -0.4943 | tta2 |
| tta2 | hd95 | 29 | 612 | both_nonempty | -1.2379 | -2.9015 | 0.1794 | 0.0724 | 0.1447 | -0.3839 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | both_nonempty | -1.1334 | -2.7446 | 0.2224 | 0.0837 | 0.1447 | -0.3701 | tta2 |
| tta2 | assd | 29 | 612 | both_nonempty | -0.4169 | -0.8278 | -0.0031 | 0.0020 | 0.0099 | -0.6368 | tta2 |
| tta2 | boundary_f1 | 29 | 612 | all_cases | 0.0154 | 0.0083 | 0.0225 | 0.0002 | 0.0011 | 0.7563 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0108 | 0.0063 | 0.0158 | 0.0001 | 0.0004 | 0.7885 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0140 | 0.0088 | 0.0197 | 0.0000 | 0.0001 | 0.8391 | tta4 |
| tta4 | precision | 29 | 612 | all_cases | 0.0094 | 0.0028 | 0.0170 | 0.0025 | 0.0043 | 0.6230 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0101 | 0.0057 | 0.0152 | 0.0001 | 0.0004 | 0.7977 | tta4 |
| tta4 | hd | 29 | 612 | both_nonempty | -3.0137 | -4.6191 | -1.5260 | 0.0006 | 0.0022 | -0.7011 | tta4 |
| tta4 | hd95 | 29 | 612 | both_nonempty | -2.5311 | -4.1090 | -1.0617 | 0.0022 | 0.0043 | -0.6322 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | both_nonempty | -2.3492 | -3.7979 | -1.0476 | 0.0008 | 0.0024 | -0.6828 | tta4 |
| tta4 | assd | 29 | 612 | both_nonempty | -0.6376 | -0.9882 | -0.2934 | 0.0003 | 0.0017 | -0.7241 | tta4 |
| tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0301 | 0.0229 | 0.0383 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0103 | 0.0051 | 0.0158 | 0.0010 | 0.0049 | 0.6736 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0143 | 0.0076 | 0.0213 | 0.0007 | 0.0047 | 0.6920 | tta8 |
| tta8 | precision | 29 | 612 | all_cases | 0.0129 | 0.0055 | 0.0207 | 0.0002 | 0.0019 | 0.7379 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | 0.0066 | -0.0007 | 0.0128 | 0.0011 | 0.0049 | 0.6690 | tta8 |
| tta8 | hd | 29 | 612 | both_nonempty | -3.4007 | -5.4735 | -1.5131 | 0.0020 | 0.0049 | -0.6368 | tta8 |
| tta8 | hd95 | 29 | 612 | both_nonempty | -3.1108 | -4.9512 | -1.4394 | 0.0011 | 0.0049 | -0.6690 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | both_nonempty | -2.8274 | -4.4843 | -1.3589 | 0.0007 | 0.0047 | -0.6920 | tta8 |
| tta8 | assd | 29 | 612 | both_nonempty | -0.8394 | -1.3581 | -0.3752 | 0.0011 | 0.0049 | -0.6690 | tta8 |
| tta8 | boundary_f1 | 29 | 612 | all_cases | 0.0365 | 0.0277 | 0.0460 | 0.0000 | 0.0000 | 1.0000 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 0 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| tta2 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tta4 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tta8 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (c2) False positives on empty references - the mirrored failure mode

| method | n_cases | n_false_positive | false_positive_rate | ci_low | ci_high | correct_to_fp | fp_to_correct | mcnemar_p | mcnemar_p_adjusted | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 |  | 0.0000 | 0.0000 |  |  |  |  |  |  |  |
| tta2 | 0 | 0 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |  |  |  |
| tta4 | 0 | 0 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |  |  |  |
| tta8 | 0 | 0 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |  |  |  |

The other half of the error budget: a prediction with foreground where the reference has none. `n_cases` here is the number of **empty-reference** cases (BraTS ET: patients without enhancing tumour), so `false_positive_rate` is the fraction of them a method segments anyway; `correct_to_fp` counts empty-reference cases the baseline left empty and the method filled (a new false positive), `fp_to_correct` the opposite, and the exact McNemar p is computed on those two discordant cells (Holm-corrected). `pred_volume_mm3_*` is the ET volume the false positives invented, in mm^3 (mean / median / largest over the false-positive cases; blank when the per-case volume metric was not computed).

## (c3) Sensitivity: BraTS challenge-style scoring of the same predictions

| method | n_cases | n_scored_strict | n_empty_reference | n_false_positive | n_missed | strict_dice_mean | strict_dice_median | challenge_dice_mean | challenge_dice_median | delta_mean | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 612 | 0 | 0 | 0 | 0.8437 | 0.9065 | 0.8437 | 0.9065 | 0.0000 | 0.0000 | 0.0000 |
| tta2 | 612 | 612 | 0 | 0 | 0 | 0.8513 | 0.9137 | 0.8513 | 0.9137 | 0.0000 | 0.0000 | 0.0000 |
| tta4 | 612 | 612 | 0 | 0 | 0 | 0.8541 | 0.9182 | 0.8541 | 0.9182 | 0.0000 | 0.0000 | 0.0000 |
| tta8 | 612 | 612 | 0 | 0 | 0 | 0.8547 | 0.9197 | 0.8547 | 0.9197 | 0.0000 | 0.0000 | 0.0000 |

Sensitivity analysis for comparability with published BraTS results, which score **every** case: an empty reference with an empty prediction counts as Dice 1 and a prediction that finds something on an empty reference counts as 0. `strict_*` is our own protocol (those cases are flagged and left out of the mean, see `metrics.score_empty_reference`), computed on `n_scored_strict` cases; `challenge_*` is the same predictions over all `n_cases`, and `delta_mean` = challenge - strict with a cluster bootstrap CI over groups. A positive delta means the challenge convention flatters the method (correct empty calls scoring 1), a negative delta means its empty-reference false positives (`n_false_positive`) cost more than the correct empty calls gain. Delta is exactly 0 for a dataset without empty references.

## (c3-per-fold) Challenge-style vs strict Dice, fold by fold

| fold | method | n_cases | n_empty_reference | n_false_positive | strict_dice_mean | challenge_dice_mean | delta_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 118 | 0 | 0 | 0.7913 | 0.7913 | 0.0000 |
| 0 | tta2 | 118 | 0 | 0 | 0.8057 | 0.8057 | 0.0000 |
| 0 | tta4 | 118 | 0 | 0 | 0.8081 | 0.8081 | 0.0000 |
| 0 | tta8 | 118 | 0 | 0 | 0.8093 | 0.8093 | 0.0000 |
| 1 | baseline | 137 | 0 | 0 | 0.8554 | 0.8554 | 0.0000 |
| 1 | tta2 | 137 | 0 | 0 | 0.8661 | 0.8661 | 0.0000 |
| 1 | tta4 | 137 | 0 | 0 | 0.8727 | 0.8727 | 0.0000 |
| 1 | tta8 | 137 | 0 | 0 | 0.8696 | 0.8696 | 0.0000 |
| 2 | baseline | 126 | 0 | 0 | 0.8628 | 0.8628 | 0.0000 |
| 2 | tta2 | 126 | 0 | 0 | 0.8649 | 0.8649 | 0.0000 |
| 2 | tta4 | 126 | 0 | 0 | 0.8678 | 0.8678 | 0.0000 |
| 2 | tta8 | 126 | 0 | 0 | 0.8684 | 0.8684 | 0.0000 |
| 3 | baseline | 129 | 0 | 0 | 0.8191 | 0.8191 | 0.0000 |
| 3 | tta2 | 129 | 0 | 0 | 0.8272 | 0.8272 | 0.0000 |
| 3 | tta4 | 129 | 0 | 0 | 0.8294 | 0.8294 | 0.0000 |
| 3 | tta8 | 129 | 0 | 0 | 0.8335 | 0.8335 | 0.0000 |
| 4 | baseline | 102 | 0 | 0 | 0.8961 | 0.8961 | 0.0000 |
| 4 | tta2 | 102 | 0 | 0 | 0.8981 | 0.8981 | 0.0000 |
| 4 | tta4 | 102 | 0 | 0 | 0.8965 | 0.8965 | 0.0000 |
| 4 | tta8 | 102 | 0 | 0 | 0.8974 | 0.8974 | 0.0000 |

Same as (c3) per fold, so the shift cannot hide behind fold averaging.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 103.3190 | 141.0195 | 0.2778 | 0.0588 |  |  |  |  |  |  |  |  |  |
| ensemble3 | 612 | 112.5083 | 145.2537 | 0.2484 | 0.0605 | 4.2342 | -2.5097 | 18.1780 | -0.0294 | -0.0516 | -0.0098 | 0.0016 | -0.0053 | 0.0083 |
| ensemble3_tta4 | 612 | 105.6305 | 146.0472 | 0.2337 | 0.0523 | 5.0277 | -3.1527 | 14.7563 | -0.0441 | -0.0660 | -0.0234 | -0.0065 | -0.0176 | 0.0046 |
| tta2 | 612 | 100.0514 | 144.0022 | 0.2729 | 0.0507 | 2.9827 | -11.4429 | 8.6238 | -0.0049 | -0.0211 | 0.0106 | -0.0082 | -0.0192 | 0.0015 |
| tta4 | 612 | 99.3413 | 143.6046 | 0.2565 | 0.0539 | 2.5851 | -8.4435 | 9.7099 | -0.0212 | -0.0392 | -0.0034 | -0.0049 | -0.0154 | 0.0052 |
| tta8 | 612 | 101.1924 | 144.4709 | 0.2565 | 0.0556 | 3.4514 | -8.7106 | 11.2687 | -0.0212 | -0.0403 | -0.0017 | -0.0033 | -0.0118 | 0.0064 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -1.1438 | -2.5853 | 0.0480 | 0.0365 | 0.1096 | tta2 |
| tta2 | hd95 | 29 | 612 | -0.8082 | -2.3137 | 0.4607 | 0.1901 | 0.3803 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | -0.7034 | -2.2192 | 0.5444 | 0.2652 | 0.3803 | tta2 |
| tta2 | assd | 29 | 612 | 0.0143 | -1.2757 | 1.6099 | 0.0157 | 0.0630 | baseline |
| tta4 | hd | 29 | 612 | -3.0488 | -4.7366 | -1.4618 | 0.0014 | 0.0056 | tta4 |
| tta4 | hd95 | 29 | 612 | -2.5621 | -4.2423 | -0.9745 | 0.0048 | 0.0076 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -2.3774 | -3.9328 | -0.9538 | 0.0025 | 0.0076 | tta4 |
| tta4 | assd | 29 | 612 | -0.6669 | -1.3521 | 0.0049 | 0.0030 | 0.0076 | tta4 |
| tta8 | hd | 29 | 612 | -2.9194 | -4.8779 | -1.1790 | 0.0048 | 0.0144 | tta8 |
| tta8 | hd95 | 29 | 612 | -2.6238 | -4.3364 | -1.1103 | 0.0048 | 0.0144 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | -2.3404 | -3.7848 | -1.0455 | 0.0030 | 0.0120 | tta8 |
| tta8 | assd | 29 | 612 | -0.3051 | -1.3743 | 1.1561 | 0.0099 | 0.0144 | tta8 |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 33 | 0.7345 | 0.0000 | 0.5887 | 0.0000 | 0.7747 |
| -1 | entropy_mean_roi | 612 | 33 | 0.5565 | 0.0000 | 0.7954 | 0.0000 | 0.9179 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 3 | 118 | 6 | 0.7913 | 0.0107 | 0.7676 | 0.0252 |
| 1 | dice | 3 | 137 | 6 | 0.8554 | 0.0078 | 0.8593 | 0.0074 |
| 2 | dice | 3 | 126 | 6 | 0.8628 | 0.0063 | 0.8622 | 0.0070 |
| 3 | dice | 3 | 129 | 6 | 0.8191 | 0.0108 | 0.8089 | 0.0123 |
| 4 | dice | 3 | 102 | 5 | 0.8961 | 0.0154 | 0.8863 | 0.0184 |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4 | dice | 29 | 612 | all_cases | 0.0108 | 0.0063 | 0.0158 | 0.0001 | 0.0004 | 0.7885 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0140 | 0.0088 | 0.0197 | 0.0000 | 0.0001 | 0.8391 | tta4 |
| tta4 | precision | 29 | 612 | all_cases | 0.0094 | 0.0028 | 0.0170 | 0.0025 | 0.0043 | 0.6230 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0101 | 0.0057 | 0.0152 | 0.0001 | 0.0004 | 0.7977 | tta4 |
| tta4 | hd | 29 | 612 | both_nonempty | -3.0137 | -4.6191 | -1.5260 | 0.0006 | 0.0022 | -0.7011 | tta4 |
| tta4 | hd95 | 29 | 612 | both_nonempty | -2.5311 | -4.1090 | -1.0617 | 0.0022 | 0.0043 | -0.6322 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | both_nonempty | -2.3492 | -3.7979 | -1.0476 | 0.0008 | 0.0024 | -0.6828 | tta4 |
| tta4 | assd | 29 | 612 | both_nonempty | -0.6376 | -0.9882 | -0.2934 | 0.0003 | 0.0017 | -0.7241 | tta4 |
| tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0301 | 0.0229 | 0.0383 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0147 | 0.0075 | 0.0263 | 0.0000 | 0.0000 | 0.9954 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0186 | 0.0101 | 0.0319 | 0.0000 | 0.0000 | 0.9770 | ensemble3 |
| ensemble3 | precision | 29 | 612 | all_cases | 0.0140 | 0.0072 | 0.0235 | 0.0000 | 0.0000 | 0.9218 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0089 | 0.0045 | 0.0158 | 0.0000 | 0.0000 | 0.9172 | ensemble3 |
| ensemble3 | hd | 29 | 612 | both_nonempty | -3.2300 | -5.0248 | -1.5299 | 0.0011 | 0.0032 | -0.6690 | ensemble3 |
| ensemble3 | hd95 | 29 | 612 | both_nonempty | -2.8234 | -4.5594 | -1.2259 | 0.0018 | 0.0036 | -0.6414 | ensemble3 |
| ensemble3 | hd95_pooled | 29 | 612 | both_nonempty | -2.9526 | -5.0298 | -1.1192 | 0.0025 | 0.0036 | -0.6230 | ensemble3 |
| ensemble3 | assd | 29 | 612 | both_nonempty | -0.9961 | -1.7370 | -0.4207 | 0.0005 | 0.0020 | -0.7057 | ensemble3 |
| ensemble3 | boundary_f1 | 29 | 612 | all_cases | 0.0230 | 0.0159 | 0.0321 | 0.0000 | 0.0000 | 0.9862 | ensemble3 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0178 | 0.0101 | 0.0274 | 0.0000 | 0.0001 | 0.8391 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0242 | 0.0153 | 0.0353 | 0.0000 | 0.0000 | 0.8851 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 29 | 612 | all_cases | 0.0174 | 0.0065 | 0.0320 | 0.0004 | 0.0015 | 0.7149 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0124 | 0.0069 | 0.0185 | 0.0000 | 0.0002 | 0.8069 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 29 | 612 | both_nonempty | -5.1051 | -7.8019 | -2.7355 | 0.0001 | 0.0006 | -0.7701 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 29 | 612 | both_nonempty | -4.4087 | -6.9079 | -2.1938 | 0.0004 | 0.0015 | -0.7195 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 29 | 612 | both_nonempty | -3.8364 | -6.1850 | -1.8238 | 0.0005 | 0.0015 | -0.7057 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 29 | 612 | both_nonempty | -0.8081 | -1.3998 | -0.2439 | 0.0052 | 0.0052 | -0.5816 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0458 | 0.0371 | 0.0554 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0070 | 0.0030 | 0.0128 | 0.0000 | 0.0001 | 0.8391 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0102 | 0.0050 | 0.0181 | 0.0000 | 0.0000 | 0.8897 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 29 | 612 | all_cases | 0.0080 | 0.0024 | 0.0157 | 0.0007 | 0.0040 | 0.6920 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0022 | 0.0009 | 0.0036 | 0.0009 | 0.0044 | 0.6782 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 29 | 612 | both_nonempty | -2.0914 | -3.5386 | -0.8620 | 0.0035 | 0.0105 | -0.6046 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 29 | 612 | both_nonempty | -1.8777 | -3.0281 | -0.8491 | 0.0018 | 0.0073 | -0.6414 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 29 | 612 | both_nonempty | -1.4872 | -2.5446 | -0.5658 | 0.0138 | 0.0277 | -0.5172 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 29 | 612 | both_nonempty | -0.1705 | -0.5878 | 0.2719 | 0.0480 | 0.0480 | -0.4207 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0157 | 0.0105 | 0.0222 | 0.0000 | 0.0000 | 0.9448 | ensemble3_tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0039 | -0.0035 | 0.0139 | 0.8480 | 1.0000 | 0.0437 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0045 | -0.0046 | 0.0173 | 0.9152 | 1.0000 | 0.0253 | ensemble3 |
| ensemble3 | precision | 29 | 612 | all_cases | 0.0046 | -0.0003 | 0.0101 | 0.6239 | 1.0000 | 0.1080 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | -0.0012 | -0.0083 | 0.0078 | 0.1373 | 1.0000 | -0.3195 | tta4 |
| ensemble3 | hd | 29 | 612 | both_nonempty | -0.2163 | -1.8776 | 1.3285 | 0.9152 | 1.0000 | 0.0253 | ensemble3 |
| ensemble3 | hd95 | 29 | 612 | both_nonempty | -0.2923 | -1.8972 | 1.1784 | 0.9321 | 1.0000 | 0.0207 | ensemble3 |
| ensemble3 | hd95_pooled | 29 | 612 | both_nonempty | -0.6035 | -2.1566 | 0.8668 | 0.9491 | 1.0000 | -0.0161 | ensemble3 |
| ensemble3 | assd | 29 | 612 | both_nonempty | -0.3585 | -0.9417 | 0.1337 | 0.4679 | 1.0000 | -0.1586 | ensemble3 |
| ensemble3 | boundary_f1 | 29 | 612 | all_cases | -0.0072 | -0.0187 | 0.0044 | 0.2941 | 1.0000 | -0.2276 | tta4 |

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 0 | 1 | 0 | 7475.0000 | 0.1507 | 0.6680 |
| failures_all | 33 | 0 | 1 | 0 | 4740.0000 | 0.3164 | 0.6680 |
| failures_non_missed | 33 | 0 | 1 | 0 | 4740.0000 | 0.3164 | 0.6680 |
| failures_tiny_1_10px | 1 | 0 | 1 | 0 | 1.0000 | 0.6680 |  |
| failures_medium_101_1000px | 3 | 0 | 0 | 0 | 259.0000 | 0.6258 |  |
| failures_large_gt_1000px | 29 | 0 | 0 | 0 | 5991.0000 | 0.3015 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9179 | 0.8742 | 0.9789 | 0.0689 | 0.0535 | 0.0927 | 0.0557 | 612 | 612 | 33 | 0 | 0.9179 | 0.9179 | 0.0689 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.9179 | 0.8742 | 0.9789 | 0.0689 | 0.0535 | 0.0927 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0689 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0632 | 0.0501 | 0.0806 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0632 |
| tta4_variance_roi | own | tta4 | all_failures | 0.8919 | 0.8274 | 0.9736 | 0.0699 | 0.0541 | 0.0941 | 0.0557 | 612 | 612 | 33 | 0 | 0.8919 | 0.8919 | 0.0699 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.8919 | 0.8274 | 0.9736 | 0.0699 | 0.0541 | 0.0941 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0699 |
| tta4_variance_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0633 | 0.0509 | 0.0803 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0633 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.7747 | 0.6699 | 0.9053 | 0.0890 | 0.0683 | 0.1212 | 0.0557 | 612 | 612 | 33 | 0 | 0.7747 | 0.7747 | 0.0890 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.7747 | 0.6699 | 0.9053 | 0.0890 | 0.0683 | 0.1212 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0890 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean |  |  |  | 0.0783 | 0.0644 | 0.0976 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0783 |
| tta4_entropy_image | own | tta4 | all_failures | 0.8308 | 0.7324 | 0.9358 | 0.0841 | 0.0649 | 0.1124 | 0.0557 | 612 | 612 | 33 | 0 | 0.8308 | 0.8308 | 0.0841 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.8308 | 0.7324 | 0.9358 | 0.0841 | 0.0649 | 0.1124 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0841 |
| tta4_entropy_image | own | tta4 | missed_vs_clean |  |  |  | 0.0759 | 0.0613 | 0.0953 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0759 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.9092 | 0.8558 | 0.9778 | 0.0664 | 0.0512 | 0.0901 | 0.0520 | 612 | 612 | 32 | 0 | 0.9092 | 0.9092 | 0.0664 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.9092 | 0.8558 | 0.9778 | 0.0664 | 0.0512 | 0.0901 | 0.0520 | 612 | 612 | 32 | 0 |  |  | 0.0664 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0603 | 0.0480 | 0.0769 | 0.0481 | 580 | 580 | 0 | 0 |  |  | 0.0603 |
| map_entropy_roi_tta4 | tta4 | tta4 | all_failures | 0.9133 | 0.8616 | 0.9800 | 0.0692 | 0.0533 | 0.0938 | 0.0557 | 612 | 612 | 33 | 0 | 0.9133 | 0.9133 | 0.0692 |
| map_entropy_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.9133 | 0.8616 | 0.9800 | 0.0692 | 0.0533 | 0.0938 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0692 |
| map_entropy_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0629 | 0.0499 | 0.0800 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0629 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.8506 | 0.7784 | 0.9352 | 0.0706 | 0.0538 | 0.0978 | 0.0520 | 612 | 612 | 32 | 0 | 0.8506 | 0.8506 | 0.0706 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.8506 | 0.7784 | 0.9352 | 0.0706 | 0.0538 | 0.0978 | 0.0520 | 612 | 612 | 32 | 0 |  |  | 0.0706 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0627 | 0.0498 | 0.0806 | 0.0481 | 580 | 580 | 0 | 0 |  |  | 0.0627 |
| map_variance_roi_tta4 | tta4 | tta4 | all_failures | 0.8565 | 0.7861 | 0.9451 | 0.0735 | 0.0560 | 0.1012 | 0.0557 | 612 | 612 | 33 | 0 | 0.8565 | 0.8565 | 0.0735 |
| map_variance_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8565 | 0.7861 | 0.9451 | 0.0735 | 0.0560 | 0.1012 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0735 |
| map_variance_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0656 | 0.0517 | 0.0851 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0656 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | all_failures | 0.7755 | 0.6686 | 0.8995 | 0.0866 | 0.0666 | 0.1172 | 0.0520 | 612 | 612 | 32 | 0 | 0.7755 | 0.7755 | 0.0866 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.7755 | 0.6686 | 0.8995 | 0.0866 | 0.0666 | 0.1172 | 0.0520 | 612 | 612 | 32 | 0 |  |  | 0.0866 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0764 | 0.0631 | 0.0952 | 0.0481 | 580 | 580 | 0 | 0 |  |  | 0.0764 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | all_failures | 0.7826 | 0.6777 | 0.9052 | 0.0895 | 0.0688 | 0.1207 | 0.0557 | 612 | 612 | 33 | 0 | 0.7826 | 0.7826 | 0.0895 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | non_missed_failures | 0.7826 | 0.6777 | 0.9052 | 0.0895 | 0.0688 | 0.1207 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0895 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0793 | 0.0656 | 0.0989 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0793 |
| pred_area_px | own | ensemble3_tta4 | all_failures | 0.3479 | 0.1595 | 0.5276 | 0.1704 | 0.1217 | 0.2371 | 0.0520 | 612 | 612 | 32 | 0 | 0.3479 | 0.3479 | 0.1704 |
| pred_area_px | own | ensemble3_tta4 | non_missed_failures | 0.3479 | 0.1595 | 0.5276 | 0.1704 | 0.1217 | 0.2371 | 0.0520 | 612 | 612 | 32 | 0 |  |  | 0.1704 |
| pred_area_px | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1092 | 0.0892 | 0.1381 | 0.0481 | 580 | 580 | 0 | 0 |  |  | 0.1092 |
| pred_area_px | tta4 | tta4 | all_failures | 0.3398 | 0.1526 | 0.5158 | 0.1740 | 0.1259 | 0.2398 | 0.0557 | 612 | 612 | 33 | 0 | 0.3398 | 0.3398 | 0.1740 |
| pred_area_px | tta4 | tta4 | non_missed_failures | 0.3398 | 0.1526 | 0.5158 | 0.1740 | 0.1259 | 0.2398 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.1740 |
| pred_area_px | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1155 | 0.0945 | 0.1456 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.1155 |
| one_minus_confidence | own | ensemble3_tta4 | all_failures | 0.9067 | 0.8489 | 0.9804 | 0.0650 | 0.0502 | 0.0889 | 0.0520 | 612 | 612 | 32 | 0 | 0.9067 | 0.9067 | 0.0650 |
| one_minus_confidence | own | ensemble3_tta4 | non_missed_failures | 0.9067 | 0.8489 | 0.9804 | 0.0650 | 0.0502 | 0.0889 | 0.0520 | 612 | 612 | 32 | 0 |  |  | 0.0650 |
| one_minus_confidence | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0587 | 0.0468 | 0.0746 | 0.0481 | 580 | 580 | 0 | 0 |  |  | 0.0587 |
| one_minus_confidence | tta4 | tta4 | all_failures | 0.9108 | 0.8534 | 0.9830 | 0.0678 | 0.0523 | 0.0925 | 0.0557 | 612 | 612 | 33 | 0 | 0.9108 | 0.9108 | 0.0678 |
| one_minus_confidence | tta4 | tta4 | non_missed_failures | 0.9108 | 0.8534 | 0.9830 | 0.0678 | 0.0523 | 0.0925 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0678 |
| one_minus_confidence | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0613 | 0.0487 | 0.0781 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0613 |
| baseline_entropy_roi | own | baseline | all_failures | 0.9337 | 0.8998 | 0.9756 | 0.0777 | 0.0597 | 0.1037 | 0.0614 | 612 | 612 | 36 | 0 | 0.9337 | 0.9337 | 0.0777 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.9337 | 0.8998 | 0.9756 | 0.0777 | 0.0597 | 0.1037 | 0.0614 | 612 | 612 | 36 | 0 |  |  | 0.0777 |
| baseline_entropy_roi | own | baseline | missed_vs_clean |  |  |  | 0.0719 | 0.0562 | 0.0926 | 0.0565 | 576 | 576 | 0 | 0 |  |  | 0.0719 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.9168 | 0.8769 | 0.9681 | 0.0728 | 0.0554 | 0.0971 | 0.0557 | 612 | 612 | 33 | 0 | 0.9168 | 0.9168 | 0.0728 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.9168 | 0.8769 | 0.9681 | 0.0728 | 0.0554 | 0.0971 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0728 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0673 | 0.0522 | 0.0871 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0673 |
| baseline_entropy_image | own | baseline | all_failures | 0.8272 | 0.7248 | 0.9282 | 0.0942 | 0.0724 | 0.1226 | 0.0614 | 612 | 612 | 36 | 0 | 0.8272 | 0.8272 | 0.0942 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.8272 | 0.7248 | 0.9282 | 0.0942 | 0.0724 | 0.1226 | 0.0614 | 612 | 612 | 36 | 0 |  |  | 0.0942 |
| baseline_entropy_image | own | baseline | missed_vs_clean |  |  |  | 0.0851 | 0.0677 | 0.1072 | 0.0565 | 576 | 576 | 0 | 0 |  |  | 0.0851 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.8245 | 0.7201 | 0.9260 | 0.0873 | 0.0669 | 0.1153 | 0.0557 | 612 | 612 | 33 | 0 | 0.8245 | 0.8245 | 0.0873 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.8245 | 0.7201 | 0.9260 | 0.0873 | 0.0669 | 0.1153 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0873 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0790 | 0.0625 | 0.1009 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0790 |
| ensemble3_entropy_roi | own | ensemble3 | all_failures | 0.9160 | 0.8625 | 0.9772 | 0.0707 | 0.0541 | 0.0951 | 0.0549 | 612 | 612 | 37 | 0 | 0.9160 | 0.9160 | 0.0707 |
| ensemble3_entropy_roi | own | ensemble3 | non_missed_failures | 0.9160 | 0.8625 | 0.9772 | 0.0707 | 0.0541 | 0.0951 | 0.0549 | 612 | 612 | 37 | 0 |  |  | 0.0707 |
| ensemble3_entropy_roi | own | ensemble3 | missed_vs_clean |  |  |  | 0.0643 | 0.0503 | 0.0832 | 0.0503 | 575 | 575 | 0 | 0 |  |  | 0.0643 |
| ensemble3_entropy_roi | tta4 | tta4 | all_failures | 0.9190 | 0.8664 | 0.9760 | 0.0704 | 0.0539 | 0.0947 | 0.0557 | 612 | 612 | 33 | 0 | 0.9190 | 0.9190 | 0.0704 |
| ensemble3_entropy_roi | tta4 | tta4 | non_missed_failures | 0.9190 | 0.8664 | 0.9760 | 0.0704 | 0.0539 | 0.0947 | 0.0557 | 612 | 612 | 33 | 0 |  |  | 0.0704 |
| ensemble3_entropy_roi | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0648 | 0.0508 | 0.0835 | 0.0515 | 579 | 579 | 0 | 0 |  |  | 0.0648 |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9179 | 0.8742 | 0.9789 | 0.0689 | 0.0535 | 0.0927 | 0.0557 | 612 | 612 | 33 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.9179 | 0.8742 | 0.9789 | 0.0689 | 0.0535 | 0.0927 | 0.0557 | 612 | 612 | 33 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0632 | 0.0501 | 0.0806 | 0.0515 | 579 | 579 | 0 |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | ensemble3 | all_failures | 0.8988 | 0.8434 | 0.9725 | 0.0708 | 0.0542 | 0.0962 | 0.0549 | 612 | 612 | 37 |
| ensemble3_dice_disagreement | own | ensemble3 | non_missed_failures | 0.8988 | 0.8434 | 0.9725 | 0.0708 | 0.0542 | 0.0962 | 0.0549 | 612 | 612 | 37 |
| ensemble3_dice_disagreement | own | ensemble3 | missed_vs_clean |  |  |  | 0.0636 | 0.0503 | 0.0820 | 0.0503 | 575 | 575 | 0 |
| ensemble3_dice_disagreement | tta4 | tta4 | all_failures | 0.9035 | 0.8503 | 0.9698 | 0.0701 | 0.0534 | 0.0948 | 0.0557 | 612 | 612 | 33 |
| ensemble3_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.9035 | 0.8503 | 0.9698 | 0.0701 | 0.0534 | 0.0948 | 0.0557 | 612 | 612 | 33 |
| ensemble3_dice_disagreement | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0637 | 0.0501 | 0.0823 | 0.0515 | 579 | 579 | 0 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | all_failures | 0.8814 | 0.8162 | 0.9670 | 0.0678 | 0.0517 | 0.0930 | 0.0520 | 612 | 612 | 32 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | non_missed_failures | 0.8814 | 0.8162 | 0.9670 | 0.0678 | 0.0517 | 0.0930 | 0.0520 | 612 | 612 | 32 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0610 | 0.0482 | 0.0791 | 0.0481 | 580 | 580 | 0 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | all_failures | 0.8862 | 0.8207 | 0.9715 | 0.0696 | 0.0532 | 0.0953 | 0.0557 | 612 | 612 | 33 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8862 | 0.8207 | 0.9715 | 0.0696 | 0.0532 | 0.0953 | 0.0557 | 612 | 612 | 33 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0627 | 0.0494 | 0.0811 | 0.0515 | 579 | 579 | 0 |

`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the four tta4 views (per seed, averaged over seeds); `ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` are the same over the three seeds' baseline / tta4 maps. Higher = the members disagree = more uncertain. The tta4 entropy row is kept as the reference descriptor.

## (f-audit) Spearman against HD95 and 1 - Dice (all predictors)

| predictor | label_source | label_prediction | target | n_used | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | hd95 | 612 | 0.5565 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 612 | 0.7954 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 612 | 0.6540 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 612 | 0.8008 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 612 | 0.7345 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.5887 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 612 | 0.7132 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.6070 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.5386 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.7664 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | hd95 | 612 | 0.5578 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.7959 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.6418 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.7165 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | hd95 | 612 | 0.6563 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.7528 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.6990 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.5262 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | hd95 | 612 | 0.7399 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.5760 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | hd95 | 612 | 0.2836 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | one_minus_dice | 612 | -0.0672 | 0.0969 |
| pred_area_px | tta4 | tta4 | hd95 | 612 | 0.2439 | 0.0000 |
| pred_area_px | tta4 | tta4 | one_minus_dice | 612 | -0.0608 | 0.1332 |
| one_minus_confidence | own | ensemble3_tta4 | hd95 | 612 | 0.6098 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | one_minus_dice | 612 | 0.7958 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | hd95 | 612 | 0.6236 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | one_minus_dice | 612 | 0.8254 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 612 | 0.5337 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 612 | 0.7691 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 612 | 0.4896 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 612 | 0.7336 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 612 | 0.6534 | 0.0000 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.5529 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 612 | 0.6415 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.5368 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | hd95 | 612 | 0.5582 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | one_minus_dice | 612 | 0.7545 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | hd95 | 612 | 0.5487 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | one_minus_dice | 612 | 0.7707 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | hd95 | 612 | 0.6824 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | one_minus_dice | 612 | 0.7433 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | hd95 | 612 | 0.6551 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | one_minus_dice | 612 | 0.7571 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | hd95 | 612 | 0.6627 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | one_minus_dice | 612 | 0.5259 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | hd95 | 612 | 0.6843 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.5716 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | hd95 | 612 | 0.7202 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | one_minus_dice | 612 | 0.5612 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | hd95 | 612 | 0.7302 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | one_minus_dice | 612 | 0.5988 | 0.0000 |
| tta4_view_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| tta4_view_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_dice_disagreement | own | ensemble3 | hd95 | 612 | 0.6622 | 0.0000 |
| ensemble3_dice_disagreement | own | ensemble3 | one_minus_dice | 612 | 0.7558 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | hd95 | 612 | 0.6830 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | one_minus_dice | 612 | 0.7879 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | hd95 | 612 | 0.6338 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | one_minus_dice | 612 | 0.7515 | 0.0000 |

Targets also come from the predictor's own prediction (`label_source=own`); the `tta4` rows repeat the correlation against the tta4 prediction for comparison. `n_used` counts the finite pairs, so ROI descriptors lose the missed detections here.

## (f-audit) detector vs detector on AUROC and AURC (paired bootstrap)

| predictor | compared_with | label_prediction | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.0260 | -0.0061 | 0.0553 | 0.1540 | -0.0010 | -0.0041 | 0.0020 | 0.5640 | 612 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.1432 | 0.0482 | 0.2277 | 0.0000 | -0.0201 | -0.0323 | -0.0109 | 0.0000 | 612 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.0871 | 0.0128 | 0.1649 | 0.0180 | -0.0152 | -0.0255 | -0.0080 | 0.0000 | 612 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0047 | -0.0206 | 0.0243 | 0.7760 | -0.0002 | -0.0021 | 0.0009 | 0.7840 | 612 |
| tta4_entropy_roi | map_entropy_roi_tta4 | ensemble3_tta4 | 0.0048 | -0.0210 | 0.0247 | 0.7660 | 0.0001 | -0.0019 | 0.0015 | 0.7960 | 612 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | 0.0614 | 0.0072 | 0.1064 | 0.0240 | -0.0046 | -0.0110 | -0.0004 | 0.0320 | 612 |
| tta4_entropy_roi | map_variance_roi_tta4 | ensemble3_tta4 | 0.0634 | 0.0090 | 0.1087 | 0.0240 | -0.0040 | -0.0107 | 0.0003 | 0.0740 | 612 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.1353 | 0.0451 | 0.2210 | 0.0000 | -0.0206 | -0.0318 | -0.0118 | 0.0000 | 612 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | ensemble3_tta4 | 0.1386 | 0.0446 | 0.2240 | 0.0000 | -0.0201 | -0.0315 | -0.0110 | 0.0000 | 612 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.5781 | 0.4000 | 0.8164 | 0.0000 | -0.1051 | -0.1515 | -0.0670 | 0.0000 | 612 |
| tta4_entropy_roi | pred_area_px | ensemble3_tta4 | 0.5662 | 0.3824 | 0.8019 | 0.0000 | -0.1039 | -0.1543 | -0.0632 | 0.0000 | 612 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0071 | -0.0220 | 0.0333 | 0.7700 | 0.0011 | -0.0013 | 0.0030 | 0.2900 | 612 |
| tta4_entropy_roi | one_minus_confidence | ensemble3_tta4 | 0.0073 | -0.0225 | 0.0337 | 0.7680 | 0.0015 | -0.0009 | 0.0036 | 0.2120 | 612 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.0011 | -0.0148 | 0.0242 | 0.7760 | -0.0039 | -0.0075 | -0.0014 | 0.0020 | 612 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.0020 | -0.0146 | 0.0219 | 0.7080 | -0.0034 | -0.0067 | -0.0009 | 0.0060 | 612 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.0935 | 0.0156 | 0.1956 | 0.0160 | -0.0184 | -0.0310 | -0.0101 | 0.0000 | 612 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.1085 | 0.0339 | 0.1964 | 0.0040 | -0.0199 | -0.0327 | -0.0112 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | -0.0011 | -0.0213 | 0.0176 | 0.9740 | -0.0015 | -0.0037 | 0.0003 | 0.1100 | 612 |
| tta4_entropy_roi | ensemble3_entropy_roi | ensemble3 | 0.0021 | -0.0222 | 0.0204 | 0.8040 | -0.0008 | -0.0030 | 0.0012 | 0.4160 | 612 |
| tta4_entropy_roi | ensemble3_variance_roi | tta4 | 0.0294 | -0.0052 | 0.0676 | 0.1040 | -0.0033 | -0.0083 | 0.0001 | 0.0620 | 612 |
| tta4_entropy_roi | ensemble3_variance_roi | ensemble3 | 0.0411 | -0.0065 | 0.0838 | 0.0940 | -0.0024 | -0.0074 | 0.0012 | 0.2060 | 612 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.0883 | 0.0138 | 0.1787 | 0.0220 | -0.0165 | -0.0287 | -0.0090 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_entropy_image | ensemble3 | 0.0908 | 0.0206 | 0.1747 | 0.0160 | -0.0175 | -0.0297 | -0.0094 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_variance_image | tta4 | 0.1140 | 0.0360 | 0.1886 | 0.0000 | -0.0184 | -0.0300 | -0.0098 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_variance_image | ensemble3 | 0.1283 | 0.0475 | 0.1989 | 0.0000 | -0.0188 | -0.0308 | -0.0097 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_dice_disagreement | tta4 | 0.0144 | -0.0044 | 0.0386 | 0.1400 | -0.0011 | -0.0046 | 0.0013 | 0.4180 | 612 |
| tta4_entropy_roi | ensemble3_dice_disagreement | ensemble3 | 0.0193 | -0.0017 | 0.0425 | 0.0760 | -0.0010 | -0.0043 | 0.0015 | 0.4920 | 612 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | tta4 | 0.0317 | -0.0047 | 0.0640 | 0.1080 | -0.0007 | -0.0048 | 0.0021 | 0.7540 | 612 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | ensemble3_tta4 | 0.0327 | -0.0056 | 0.0665 | 0.1080 | -0.0012 | -0.0056 | 0.0016 | 0.4780 | 612 |
| ensemble3_dice_disagreement | ensemble3_entropy_roi | ensemble3 | -0.0172 | -0.0380 | -0.0004 | 0.0440 | 0.0002 | -0.0023 | 0.0032 | 0.9460 | 612 |
| ensemble3_tta4_dice_disagreement | map_entropy_roi_tta4 | ensemble3_tta4 | -0.0278 | -0.0538 | 0.0031 | 0.0900 | 0.0014 | -0.0012 | 0.0050 | 0.3200 | 612 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

map-derived descriptors replaced by the per-case CSV columns where the evaluation wrote an equivalent (map_entropy_roi_tta4 <- csv entropy_mean_roi; map_variance_roi_tta4 <- csv variance_mean_roi; tta4_view_pairwise_dice <- csv view_pairwise_dice).

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 33, baseline 36, ensemble3 37, ensemble3_tta4 32

## Audit note

tta4 shares 33 failures; 0 of them have no ROI value (empty prediction), 1 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice: mean over the 4 tta4 views per case, averaged over the seeds (1.0 means the four views produced the same binary mask).

## Audit note

seed agreement: 612 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 8.99e-05
