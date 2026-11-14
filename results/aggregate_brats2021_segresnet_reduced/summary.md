# MELBA BraTS 2021 - 5-fold x 1-seed TTA results

5 runs: folds [0, 1, 2, 3, 4] x seeds [0]; config `configs\brats2021\segresnet_reduced.yaml`; dataset `brats2021`; pairing unit `group` (one patient); bootstrap draws `config`; Holm correction.

Cases of one patient are not independent, so every CI and test resamples or pairs at the patient level. The empty-reference case class of this dataset (BraTS ET: patients without enhancing tumour) is scored `metrics.score_empty_reference: false`: those cases carry an `empty_flag` and are **left out** of the Dice/distance means instead of being counted as a perfect match. Section (c) reports missed detections on non-empty references, section (c2) the mirrored error - false positives on the empty ones - and section (c3) re-scores the same predictions with the challenge convention (two empty masks = Dice 1, foreground on an empty reference = 0) so the numbers are comparable with published BraTS results.

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 1251 | 1251 | 1 | 1.2119 |  |
| tta2 | assd | 1251 | 1251 | 1 | 1.0556 |  |
| tta4 | assd | 1251 | 1251 | 1 | 0.9718 |  |
| tta8 | assd | 1251 | 1251 | 1 | 1.0060 |  |
| baseline | boundary_f1 | 1251 | 1251 | 1 | 0.9280 |  |
| tta2 | boundary_f1 | 1251 | 1251 | 1 | 0.9311 |  |
| tta4 | boundary_f1 | 1251 | 1251 | 1 | 0.9336 |  |
| tta8 | boundary_f1 | 1251 | 1251 | 1 | 0.9323 |  |
| baseline | dice | 1251 | 1251 | 1 | 0.8547 |  |
| tta2 | dice | 1251 | 1251 | 1 | 0.8576 |  |
| tta4 | dice | 1251 | 1251 | 1 | 0.8599 |  |
| tta8 | dice | 1251 | 1251 | 1 | 0.8614 |  |
| baseline | hd95 | 1251 | 1251 | 1 | 4.8436 |  |
| tta2 | hd95 | 1251 | 1251 | 1 | 4.3267 |  |
| tta4 | hd95 | 1251 | 1251 | 1 | 4.1929 |  |
| tta8 | hd95 | 1251 | 1251 | 1 | 4.2275 |  |
| baseline | hd95_pooled | 1251 | 1251 | 1 | 3.7186 |  |
| tta2 | hd95_pooled | 1251 | 1251 | 1 | 3.3702 |  |
| tta4 | hd95_pooled | 1251 | 1251 | 1 | 3.2504 |  |
| tta8 | hd95_pooled | 1251 | 1251 | 1 | 3.2772 |  |

Mean over all cases of the fold, pooled over the 5 folds, then mean ± std across the seeds.

## (a) Per-method summary - patient level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 1251 | 1251 | 1 | 1.2119 |  |
| tta2 | assd | 1251 | 1251 | 1 | 1.0556 |  |
| tta4 | assd | 1251 | 1251 | 1 | 0.9718 |  |
| tta8 | assd | 1251 | 1251 | 1 | 1.0060 |  |
| baseline | boundary_f1 | 1251 | 1251 | 1 | 0.9280 |  |
| tta2 | boundary_f1 | 1251 | 1251 | 1 | 0.9311 |  |
| tta4 | boundary_f1 | 1251 | 1251 | 1 | 0.9336 |  |
| tta8 | boundary_f1 | 1251 | 1251 | 1 | 0.9323 |  |
| baseline | dice | 1251 | 1251 | 1 | 0.8547 |  |
| tta2 | dice | 1251 | 1251 | 1 | 0.8576 |  |
| tta4 | dice | 1251 | 1251 | 1 | 0.8599 |  |
| tta8 | dice | 1251 | 1251 | 1 | 0.8614 |  |
| baseline | hd95 | 1251 | 1251 | 1 | 4.8436 |  |
| tta2 | hd95 | 1251 | 1251 | 1 | 4.3267 |  |
| tta4 | hd95 | 1251 | 1251 | 1 | 4.1929 |  |
| tta8 | hd95 | 1251 | 1251 | 1 | 4.2275 |  |
| baseline | hd95_pooled | 1251 | 1251 | 1 | 3.7186 |  |
| tta2 | hd95_pooled | 1251 | 1251 | 1 | 3.3702 |  |
| tta4 | hd95_pooled | 1251 | 1251 | 1 | 3.2504 |  |
| tta8 | hd95_pooled | 1251 | 1251 | 1 | 3.2772 |  |

Each patient contributes one value (mean over its cases); mean ± std across the seeds.

## (b) Paired comparisons vs baseline - patient level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 1218 | 1218 | all_cases | 0.0029 | 0.0018 | 0.0040 | 0.0000 | 0.0000 | 0.5133 | tta2 |
| tta2 | jaccard | 1218 | 1218 | all_cases | 0.0041 | 0.0031 | 0.0051 | 0.0000 | 0.0000 | 0.5263 | tta2 |
| tta2 | precision | 1227 | 1227 | all_cases | 0.0045 | 0.0034 | 0.0058 | 0.0000 | 0.0000 | 0.4994 | tta2 |
| tta2 | recall | 1218 | 1218 | all_cases | 0.0012 | 0.0001 | 0.0024 | 0.0000 | 0.0000 | 0.2237 | tta2 |
| tta2 | hd | 1210 | 1210 | both_nonempty | -0.6654 | -1.0989 | -0.2599 | 0.0000 | 0.0000 | -0.2493 | tta2 |
| tta2 | hd95 | 1210 | 1210 | both_nonempty | -0.4361 | -0.7658 | -0.1600 | 0.0033 | 0.0067 | -0.1678 | tta2 |
| tta2 | hd95_pooled | 1210 | 1210 | both_nonempty | -0.2678 | -0.4765 | -0.1014 | 0.0046 | 0.0067 | -0.1781 | tta2 |
| tta2 | assd | 1210 | 1210 | both_nonempty | -0.1121 | -0.2056 | -0.0362 | 0.0000 | 0.0000 | -0.4503 | tta2 |
| tta2 | boundary_f1 | 1210 | 1210 | all_cases | 0.0017 | 0.0003 | 0.0030 | 0.0000 | 0.0000 | 0.1851 | tta2 |
| tta4 | dice | 1218 | 1218 | all_cases | 0.0052 | 0.0038 | 0.0067 | 0.0000 | 0.0000 | 0.6100 | tta4 |
| tta4 | jaccard | 1218 | 1218 | all_cases | 0.0073 | 0.0059 | 0.0087 | 0.0000 | 0.0000 | 0.6271 | tta4 |
| tta4 | precision | 1226 | 1226 | all_cases | 0.0072 | 0.0058 | 0.0088 | 0.0000 | 0.0000 | 0.5896 | tta4 |
| tta4 | recall | 1218 | 1218 | all_cases | 0.0028 | 0.0013 | 0.0044 | 0.0000 | 0.0000 | 0.2816 | tta4 |
| tta4 | hd | 1208 | 1208 | both_nonempty | -0.8784 | -1.3880 | -0.4150 | 0.0000 | 0.0000 | -0.2779 | tta4 |
| tta4 | hd95 | 1208 | 1208 | both_nonempty | -0.5377 | -0.9087 | -0.2324 | 0.0000 | 0.0000 | -0.2290 | tta4 |
| tta4 | hd95_pooled | 1208 | 1208 | both_nonempty | -0.3534 | -0.5920 | -0.1550 | 0.0000 | 0.0000 | -0.2753 | tta4 |
| tta4 | assd | 1208 | 1208 | both_nonempty | -0.1537 | -0.2960 | -0.0547 | 0.0000 | 0.0000 | -0.5432 | tta4 |
| tta4 | boundary_f1 | 1208 | 1208 | all_cases | 0.0033 | 0.0017 | 0.0051 | 0.0000 | 0.0000 | 0.2391 | tta4 |
| tta8 | dice | 1218 | 1218 | all_cases | 0.0067 | 0.0051 | 0.0083 | 0.0000 | 0.0000 | 0.6626 | tta8 |
| tta8 | jaccard | 1218 | 1218 | all_cases | 0.0098 | 0.0083 | 0.0113 | 0.0000 | 0.0000 | 0.6780 | tta8 |
| tta8 | precision | 1228 | 1228 | all_cases | 0.0091 | 0.0063 | 0.0117 | 0.0000 | 0.0000 | 0.6413 | tta8 |
| tta8 | recall | 1218 | 1218 | all_cases | 0.0037 | 0.0018 | 0.0057 | 0.0000 | 0.0000 | 0.3084 | tta8 |
| tta8 | hd | 1209 | 1209 | both_nonempty | -0.6743 | -1.1964 | -0.1687 | 0.0000 | 0.0000 | -0.2988 | tta8 |
| tta8 | hd95 | 1209 | 1209 | both_nonempty | -0.5031 | -0.8980 | -0.1504 | 0.0000 | 0.0000 | -0.2505 | tta8 |
| tta8 | hd95_pooled | 1209 | 1209 | both_nonempty | -0.3275 | -0.5779 | -0.1093 | 0.0000 | 0.0000 | -0.2823 | tta8 |
| tta8 | assd | 1209 | 1209 | both_nonempty | -0.1044 | -0.2107 | -0.0131 | 0.0000 | 0.0000 | -0.5615 | tta8 |
| tta8 | boundary_f1 | 1209 | 1209 | all_cases | 0.0036 | 0.0014 | 0.0057 | 0.0000 | 0.0000 | 0.2893 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1218 | 6 | 0.0049 | 0.0016 | 0.0090 |  |  |  |  |
| tta2 | 1218 | 8 | 0.0066 | 0.0025 | 0.0115 | 2.0000 | 0.0000 | 0.5000 | 1.0000 |
| tta4 | 1218 | 9 | 0.0074 | 0.0033 | 0.0123 | 4.0000 | 1.0000 | 0.3750 | 1.0000 |
| tta8 | 1218 | 7 | 0.0057 | 0.0016 | 0.0099 | 3.0000 | 2.0000 | 1.0000 | 1.0000 |

Cluster (patient) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (c2) False positives on empty references - the mirrored failure mode

| method | n_cases | n_false_positive | false_positive_rate | ci_low | ci_high | correct_to_fp | fp_to_correct | mcnemar_p | mcnemar_p_adjusted | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 33 | 22 | 0.6667 | 0.5152 | 0.8182 |  |  |  |  | 1129.9545 | 275.0000 | 5450.0000 |
| tta2 | 33 | 19 | 0.5758 | 0.3939 | 0.7273 | 2.0000 | 5.0000 | 0.4531 | 1.0000 | 996.5263 | 211.0000 | 5381.0000 |
| tta4 | 33 | 20 | 0.6061 | 0.4242 | 0.7576 | 2.0000 | 4.0000 | 0.6875 | 1.0000 | 916.8500 | 124.0000 | 4912.0000 |
| tta8 | 33 | 21 | 0.6364 | 0.4545 | 0.7879 | 2.0000 | 3.0000 | 1.0000 | 1.0000 | 831.2857 | 141.0000 | 4471.0000 |

The other half of the error budget: a prediction with foreground where the reference has none. `n_cases` here is the number of **empty-reference** cases (BraTS ET: patients without enhancing tumour), so `false_positive_rate` is the fraction of them a method segments anyway; `correct_to_fp` counts empty-reference cases the baseline left empty and the method filled (a new false positive), `fp_to_correct` the opposite, and the exact McNemar p is computed on those two discordant cells (Holm-corrected). `pred_volume_mm3_*` is the ET volume the false positives invented, in mm^3 (mean / median / largest over the false-positive cases; blank when the per-case volume metric was not computed).

## (c3) Sensitivity: BraTS challenge-style scoring of the same predictions

| method | n_cases | n_scored_strict | n_empty_reference | n_false_positive | n_missed | strict_dice_mean | strict_dice_median | challenge_dice_mean | challenge_dice_median | delta_mean | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1251 | 1218 | 33 | 22 | 6 | 0.8547 | 0.9131 | 0.8409 | 0.9127 | -0.0138 | -0.0202 | -0.0078 |
| tta2 | 1251 | 1218 | 33 | 19 | 8 | 0.8576 | 0.9162 | 0.8462 | 0.9157 | -0.0114 | -0.0175 | -0.0058 |
| tta4 | 1251 | 1218 | 33 | 20 | 9 | 0.8599 | 0.9173 | 0.8476 | 0.9169 | -0.0123 | -0.0185 | -0.0065 |
| tta8 | 1251 | 1218 | 33 | 21 | 7 | 0.8614 | 0.9181 | 0.8483 | 0.9175 | -0.0131 | -0.0194 | -0.0073 |

Sensitivity analysis for comparability with published BraTS results, which score **every** case: an empty reference with an empty prediction counts as Dice 1 and a prediction that finds something on an empty reference counts as 0. `strict_*` is our own protocol (those cases are flagged and left out of the mean, see `metrics.score_empty_reference`), computed on `n_scored_strict` cases; `challenge_*` is the same predictions over all `n_cases`, and `delta_mean` = challenge - strict with a cluster bootstrap CI over groups. A positive delta means the challenge convention flatters the method (correct empty calls scoring 1), a negative delta means its empty-reference false positives (`n_false_positive`) cost more than the correct empty calls gain. Delta is exactly 0 for a dataset without empty references.

## (c3-per-fold) Challenge-style vs strict Dice, fold by fold

| fold | method | n_cases | n_empty_reference | n_false_positive | strict_dice_mean | challenge_dice_mean | delta_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 251 | 8 | 5 | 0.8642 | 0.8486 | -0.0156 |
| 0 | tta2 | 251 | 8 | 4 | 0.8677 | 0.8560 | -0.0117 |
| 0 | tta4 | 251 | 8 | 4 | 0.8701 | 0.8583 | -0.0118 |
| 0 | tta8 | 251 | 8 | 4 | 0.8740 | 0.8621 | -0.0119 |
| 1 | baseline | 250 | 4 | 1 | 0.8512 | 0.8496 | -0.0016 |
| 1 | tta2 | 250 | 4 | 1 | 0.8536 | 0.8520 | -0.0017 |
| 1 | tta4 | 250 | 4 | 1 | 0.8558 | 0.8541 | -0.0017 |
| 1 | tta8 | 250 | 4 | 2 | 0.8568 | 0.8511 | -0.0057 |
| 2 | baseline | 250 | 5 | 5 | 0.8449 | 0.8280 | -0.0169 |
| 2 | tta2 | 250 | 5 | 4 | 0.8500 | 0.8370 | -0.0130 |
| 2 | tta4 | 250 | 5 | 4 | 0.8529 | 0.8398 | -0.0131 |
| 2 | tta8 | 250 | 5 | 5 | 0.8528 | 0.8357 | -0.0171 |
| 3 | baseline | 250 | 9 | 6 | 0.8644 | 0.8453 | -0.0191 |
| 3 | tta2 | 250 | 9 | 5 | 0.8678 | 0.8526 | -0.0152 |
| 3 | tta4 | 250 | 9 | 6 | 0.8716 | 0.8523 | -0.0194 |
| 3 | tta8 | 250 | 9 | 5 | 0.8734 | 0.8579 | -0.0154 |
| 4 | baseline | 250 | 7 | 5 | 0.8489 | 0.8332 | -0.0158 |
| 4 | tta2 | 250 | 7 | 5 | 0.8491 | 0.8333 | -0.0158 |
| 4 | tta4 | 250 | 7 | 5 | 0.8493 | 0.8335 | -0.0158 |
| 4 | tta8 | 250 | 7 | 5 | 0.8503 | 0.8344 | -0.0158 |

Same as (c3) per fold, so the shift cannot hide behind fold averaging.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1251 | 11.4368 | 21.6827 | 0.1335 | 0.0799 |  |  |  |  |  |  |  |  |  |
| tta2 | 1251 | 10.1536 | 18.4765 | 0.1215 | 0.0751 | -3.2062 | -5.3735 | -0.6346 | -0.0120 | -0.0208 | -0.0040 | -0.0048 | -0.0088 | -0.0016 |
| tta4 | 1251 | 9.9448 | 19.1150 | 0.1199 | 0.0735 | -2.5677 | -5.2631 | -0.1602 | -0.0136 | -0.0232 | -0.0040 | -0.0064 | -0.0112 | -0.0024 |
| tta8 | 1251 | 9.8387 | 18.5812 | 0.1175 | 0.0743 | -3.1015 | -6.1498 | -0.5895 | -0.0160 | -0.0264 | -0.0064 | -0.0056 | -0.0104 | -0.0016 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 1251 | 1251 | -1.0277 | -2.8029 | 0.6869 | 0.0000 | 0.0000 | tta2 |
| tta2 | hd95 | 1251 | 1251 | -0.7987 | -2.5513 | 0.9224 | 0.0034 | 0.0069 | tta2 |
| tta2 | hd95_pooled | 1251 | 1251 | -0.6348 | -2.3975 | 1.0764 | 0.0048 | 0.0069 | tta2 |
| tta2 | assd | 1251 | 1251 | -0.4489 | -2.2116 | 1.2883 | 0.0000 | 0.0000 | tta2 |
| tta4 | hd | 1251 | 1251 | -0.6810 | -2.5976 | 1.2205 | 0.0000 | 0.0000 | tta4 |
| tta4 | hd95 | 1251 | 1251 | -0.3425 | -2.2603 | 1.5301 | 0.0001 | 0.0001 | tta4 |
| tta4 | hd95_pooled | 1251 | 1251 | -0.1634 | -2.0514 | 1.6790 | 0.0000 | 0.0000 | tta4 |
| tta4 | assd | 1251 | 1251 | 0.0614 | -1.8863 | 1.9237 | 0.0000 | 0.0000 | baseline |
| tta8 | hd | 1251 | 1251 | -0.7703 | -2.5526 | 1.0001 | 0.0000 | 0.0000 | tta8 |
| tta8 | hd95 | 1251 | 1251 | -0.6023 | -2.3621 | 1.1496 | 0.0000 | 0.0000 | tta8 |
| tta8 | hd95_pooled | 1251 | 1251 | -0.4316 | -2.1948 | 1.2970 | 0.0000 | 0.0000 | tta8 |
| tta8 | assd | 1251 | 1251 | -0.2013 | -1.9625 | 1.5385 | 0.0000 | 0.0000 | tta8 |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 1251 | 59 | 0.1058 | 0.0002 | 0.0351 | 0.2204 | 0.3199 |
| -1 | entropy_mean_roi | 1251 | 59 | 0.4202 | 0.0000 | 0.5754 | 0.0000 | 0.8911 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 1 | 251 | 251 | 0.8642 |  | 0.8642 |  |
| 1 | dice | 1 | 250 | 250 | 0.8512 |  | 0.8512 |  |
| 2 | dice | 1 | 250 | 250 | 0.8449 |  | 0.8449 |  |
| 3 | dice | 1 | 250 | 250 | 0.8644 |  | 0.8644 |  |
| 4 | dice | 1 | 250 | 250 | 0.8489 |  | 0.8489 |  |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - patient level, seeds averaged per case

_(no rows)_

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 1251 | 22 | 19 | 22 | 17571.0000 | 0.1895 | 0.2765 |
| failures_all | 59 | 9 | 8 | 9 | 764.0000 | 0.2861 | 0.2798 |
| failures_non_missed | 50 | 0 | 8 | 0 | 1128.0000 | 0.2861 | 0.2798 |
| failures_missed | 9 | 9 | 0 | 9 | 0.0000 |  |  |
| failures_empty_0px | 9 | 9 | 9 | 9 | 0.0000 |  |  |
| failures_tiny_1_10px | 3 | 0 | 3 | 0 | 3.0000 | 0.2618 |  |
| failures_small_11_100px | 5 | 0 | 5 | 0 | 61.0000 | 0.2830 |  |
| failures_medium_101_1000px | 16 | 0 | 0 | 0 | 336.0000 | 0.2879 |  |
| failures_large_gt_1000px | 26 | 0 | 0 | 0 | 2888.0000 | 0.2978 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 | 22 | 0.7570 | 0.8986 | 0.0806 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 | 13 |  |  | 0.0806 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0757 | 0.0715 | 0.0803 | 0.0534 | 1159 | 1159 | 0 | 9 |  |  | 0.0757 |
| tta4_variance_roi | own | tta4 | all_failures | 0.9466 | 0.9222 | 0.9679 | 0.0724 | 0.0688 | 0.0760 | 0.0561 | 1229 | 1209 | 50 | 22 | 0.8035 | 0.9451 | 0.0724 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.9466 | 0.9222 | 0.9679 | 0.0724 | 0.0688 | 0.0760 | 0.0561 | 1229 | 1209 | 50 | 13 |  |  | 0.0724 |
| tta4_variance_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0692 | 0.0659 | 0.0725 | 0.0534 | 1159 | 1159 | 0 | 9 |  |  | 0.0692 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.3199 | 0.2407 | 0.4077 | 0.1914 | 0.1672 | 0.2189 | 0.0567 | 1251 | 1218 | 59 | 0 | 0.3199 | 0.3199 | 0.1914 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.3710 | 0.2784 | 0.4671 | 0.1638 | 0.1416 | 0.1838 | 0.0561 | 1242 | 1209 | 50 | 0 |  |  | 0.1638 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean | 0.0284 | 0.0000 | 0.0851 | 0.1469 | 0.1260 | 0.1696 | 0.0539 | 1168 | 1168 | 9 | 0 |  |  | 0.1469 |
| tta4_entropy_image | own | tta4 | all_failures | 0.4152 | 0.3464 | 0.4811 | 0.1530 | 0.1378 | 0.1716 | 0.0567 | 1251 | 1218 | 59 | 0 | 0.4152 | 0.4152 | 0.1530 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.4403 | 0.3619 | 0.5176 | 0.1420 | 0.1273 | 0.1575 | 0.0561 | 1242 | 1209 | 50 | 0 |  |  | 0.1420 |
| tta4_entropy_image | own | tta4 | missed_vs_clean | 0.2699 | 0.1282 | 0.4001 | 0.1204 | 0.1101 | 0.1326 | 0.0539 | 1168 | 1168 | 9 | 0 |  |  | 0.1204 |
| map_entropy_roi_tta4 | own | tta4 | all_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 | 22 | 0.7570 | 0.8986 | 0.0806 |
| map_entropy_roi_tta4 | own | tta4 | non_missed_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 | 13 |  |  | 0.0806 |
| map_entropy_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  | 0.0757 | 0.0715 | 0.0803 | 0.0534 | 1159 | 1159 | 0 | 9 |  |  | 0.0757 |
| map_variance_roi_tta4 | own | tta4 | all_failures | 0.9466 | 0.9222 | 0.9679 | 0.0724 | 0.0688 | 0.0760 | 0.0561 | 1229 | 1209 | 50 | 22 | 0.8035 | 0.9451 | 0.0724 |
| map_variance_roi_tta4 | own | tta4 | non_missed_failures | 0.9466 | 0.9222 | 0.9679 | 0.0724 | 0.0688 | 0.0760 | 0.0561 | 1229 | 1209 | 50 | 13 |  |  | 0.0724 |
| map_variance_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  | 0.0692 | 0.0659 | 0.0725 | 0.0534 | 1159 | 1159 | 0 | 9 |  |  | 0.0692 |
| map_ambiguous_fraction_tta4 | own | tta4 | all_failures | 0.0833 | 0.0204 | 0.1667 | 0.1572 | 0.0825 | 0.2527 | 0.0638 | 50 | 48 | 2 | 1201 | 0.4962 | 0.5026 | 0.1572 |
| map_ambiguous_fraction_tta4 | own | tta4 | non_missed_failures | 0.0833 | 0.0204 | 0.1667 | 0.1572 | 0.0825 | 0.2527 | 0.0638 | 50 | 48 | 2 | 1192 |  |  | 0.1572 |
| map_ambiguous_fraction_tta4 | own | tta4 | missed_vs_clean |  |  |  | 0.0919 | 0.0738 | 0.1147 | 0.0607 | 46 | 46 | 0 | 1122 |  |  | 0.0919 |
| pred_area_px | own | tta4 | all_failures | 0.0765 | 0.0493 | 0.1098 | 0.2523 | 0.2269 | 0.2814 | 0.0567 | 1251 | 1218 | 59 | 0 | 0.0765 | 0.0765 | 0.2523 |
| pred_area_px | own | tta4 | non_missed_failures | 0.0893 | 0.0565 | 0.1335 | 0.2277 | 0.2030 | 0.2512 | 0.0561 | 1242 | 1209 | 50 | 0 |  |  | 0.2277 |
| pred_area_px | own | tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.1766 | 0.1561 | 0.1997 | 0.0539 | 1168 | 1168 | 9 | 0 |  |  | 0.1766 |
| one_minus_confidence | own | tta4 | all_failures | 0.7917 | 0.5445 | 1.0000 | 0.0803 | 0.0599 | 0.1109 | 0.0638 | 50 | 48 | 2 | 1201 | 0.4972 | 0.5036 | 0.0803 |
| one_minus_confidence | own | tta4 | non_missed_failures | 0.7917 | 0.5445 | 1.0000 | 0.0803 | 0.0599 | 0.1109 | 0.0638 | 50 | 48 | 2 | 1192 |  |  | 0.0803 |
| one_minus_confidence | own | tta4 | missed_vs_clean |  |  |  | 0.0734 | 0.0566 | 0.0987 | 0.0607 | 46 | 46 | 0 | 1122 |  |  | 0.0734 |
| baseline_entropy_roi | own | baseline | all_failures | 0.8413 | 0.7852 | 0.8889 | 0.0888 | 0.0832 | 0.0948 | 0.0596 | 1234 | 1212 | 61 | 17 | 0.7677 | 0.8479 | 0.0888 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.8413 | 0.7852 | 0.8889 | 0.0888 | 0.0832 | 0.0948 | 0.0596 | 1234 | 1212 | 61 | 11 |  |  | 0.0888 |
| baseline_entropy_roi | own | baseline | missed_vs_clean |  |  |  | 0.0798 | 0.0756 | 0.0843 | 0.0563 | 1151 | 1151 | 0 | 6 |  |  | 0.0798 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.8526 | 0.7964 | 0.8981 | 0.0852 | 0.0798 | 0.0909 | 0.0563 | 1234 | 1212 | 53 | 17 | 0.7676 | 0.8601 | 0.0852 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8610 | 0.8060 | 0.9081 | 0.0840 | 0.0787 | 0.0898 | 0.0561 | 1230 | 1208 | 49 | 12 |  |  | 0.0840 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.7532 | 0.3930 | 1.0000 | 0.0793 | 0.0742 | 0.0847 | 0.0536 | 1163 | 1163 | 4 | 5 |  |  | 0.0793 |
| baseline_entropy_image | own | baseline | all_failures | 0.3782 | 0.3132 | 0.4431 | 0.1627 | 0.1464 | 0.1819 | 0.0601 | 1251 | 1218 | 67 | 0 | 0.3782 | 0.3782 | 0.1627 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.3888 | 0.3205 | 0.4624 | 0.1558 | 0.1400 | 0.1718 | 0.0596 | 1245 | 1212 | 61 | 0 |  |  | 0.1558 |
| baseline_entropy_image | own | baseline | missed_vs_clean | 0.2651 | 0.1093 | 0.4249 | 0.1168 | 0.1077 | 0.1270 | 0.0566 | 1157 | 1157 | 6 | 0 |  |  | 0.1168 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.4098 | 0.3418 | 0.4757 | 0.1546 | 0.1391 | 0.1738 | 0.0567 | 1251 | 1218 | 59 | 0 | 0.4098 | 0.4098 | 0.1546 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.4340 | 0.3567 | 0.5105 | 0.1432 | 0.1285 | 0.1590 | 0.0561 | 1242 | 1209 | 50 | 0 |  |  | 0.1432 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean | 0.2698 | 0.1197 | 0.4081 | 0.1215 | 0.1113 | 0.1340 | 0.0539 | 1168 | 1168 | 9 | 0 |  |  | 0.1215 |
| ensemble3_entropy_roi | own | tta4 | all_failures | 0.6562 | 0.2653 | 1.0000 | 0.0861 | 0.0604 | 0.1271 | 0.0638 | 50 | 48 | 2 | 1201 | 0.4970 | 0.5034 | 0.0861 |
| ensemble3_entropy_roi | own | tta4 | non_missed_failures | 0.6562 | 0.2653 | 1.0000 | 0.0861 | 0.0604 | 0.1271 | 0.0638 | 50 | 48 | 2 | 1192 |  |  | 0.0861 |
| ensemble3_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0718 | 0.0566 | 0.0940 | 0.0607 | 46 | 46 | 0 | 1122 |  |  | 0.0718 |
| ensemble3_variance_roi | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1251 |  |  |  |
| ensemble3_variance_roi | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1242 |  |  |  |
| ensemble3_variance_roi | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 1168 |  |  |  |
| ensemble3_entropy_image | own | tta4 | all_failures | 0.3021 | 0.1633 | 0.4681 | 0.1325 | 0.0887 | 0.1878 | 0.0638 | 50 | 48 | 2 | 1201 | 0.4965 | 0.5029 | 0.1325 |
| ensemble3_entropy_image | own | tta4 | non_missed_failures | 0.3021 | 0.1633 | 0.4681 | 0.1325 | 0.0887 | 0.1878 | 0.0638 | 50 | 48 | 2 | 1192 |  |  | 0.1325 |
| ensemble3_entropy_image | own | tta4 | missed_vs_clean |  |  |  | 0.1044 | 0.0795 | 0.1400 | 0.0607 | 46 | 46 | 0 | 1122 |  |  | 0.1044 |
| ensemble3_variance_image | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1251 |  |  |  |
| ensemble3_variance_image | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1242 |  |  |  |
| ensemble3_variance_image | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 1168 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1251 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1242 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 1168 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1251 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1242 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 1168 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1251 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 1242 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 1168 |  |  |  |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8911 | 0.8390 | 0.9347 | 0.0806 | 0.0759 | 0.0856 | 0.0561 | 1229 | 1209 | 50 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0757 | 0.0715 | 0.0803 | 0.0534 | 1159 | 1159 | 0 |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_tta4_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_tta4_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_tta4_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |

`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the four tta4 views (per seed, averaged over seeds); `ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` are the same over the three seeds' baseline / tta4 maps. Higher = the members disagree = more uncertain. The tta4 entropy row is kept as the reference descriptor.

## (f-audit) Spearman against HD95 and 1 - Dice (all predictors)

| predictor | label_source | label_prediction | target | n_used | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | hd95 | 1209 | 0.4202 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 1209 | 0.5754 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 1209 | 0.4861 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 1209 | 0.6818 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 1222 | 0.1058 | 0.0002 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 1218 | 0.0351 | 0.2204 |
| tta4_entropy_image | own | tta4 | hd95 | 1222 | 0.0916 | 0.0013 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 1218 | 0.0319 | 0.2666 |
| map_entropy_roi_tta4 | own | tta4 | hd95 | 1209 | 0.4202 | 0.0000 |
| map_entropy_roi_tta4 | own | tta4 | one_minus_dice | 1209 | 0.5754 | 0.0000 |
| map_variance_roi_tta4 | own | tta4 | hd95 | 1209 | 0.4861 | 0.0000 |
| map_variance_roi_tta4 | own | tta4 | one_minus_dice | 1209 | 0.6818 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | tta4 | hd95 | 48 | 0.2735 | 0.0600 |
| map_ambiguous_fraction_tta4 | own | tta4 | one_minus_dice | 48 | 0.2704 | 0.0630 |
| pred_area_px | own | tta4 | hd95 | 1222 | -0.1649 | 0.0000 |
| pred_area_px | own | tta4 | one_minus_dice | 1218 | -0.3823 | 0.0000 |
| one_minus_confidence | own | tta4 | hd95 | 48 | 0.4613 | 0.0010 |
| one_minus_confidence | own | tta4 | one_minus_dice | 48 | 0.7800 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 1212 | 0.3990 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 1212 | 0.5481 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 1212 | 0.3804 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 1212 | 0.5381 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 1223 | 0.0467 | 0.1024 |
| baseline_entropy_image | own | baseline | one_minus_dice | 1218 | 0.0113 | 0.6938 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 1222 | 0.0839 | 0.0033 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 1218 | 0.0241 | 0.4012 |
| ensemble3_entropy_roi | own | tta4 | hd95 | 48 | 0.5320 | 0.0001 |
| ensemble3_entropy_roi | own | tta4 | one_minus_dice | 48 | 0.7287 | 0.0000 |
| ensemble3_variance_roi | own | tta4 | hd95 | 0 |  |  |
| ensemble3_variance_roi | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_entropy_image | own | tta4 | hd95 | 48 | 0.4204 | 0.0029 |
| ensemble3_entropy_image | own | tta4 | one_minus_dice | 48 | 0.2572 | 0.0776 |
| ensemble3_variance_image | own | tta4 | hd95 | 0 |  |  |
| ensemble3_variance_image | own | tta4 | one_minus_dice | 0 |  |  |
| tta4_view_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| tta4_view_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| ensemble3_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |

Targets also come from the predictor's own prediction (`label_source=own`); the `tta4` rows repeat the correlation against the tta4 prediction for comparison. `n_used` counts the finite pairs, so ROI descriptors lose the missed detections here.

## (f-audit) detector vs detector on AUROC and AURC (paired bootstrap)

| predictor | compared_with | label_prediction | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | tta4_variance_roi | tta4 | -0.0578 | -0.0965 | -0.0246 | 0.0000 | 0.0082 | 0.0051 | 0.0116 | 0.0000 | 1209 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.5378 | 0.4466 | 0.6229 | 0.0000 | -0.0832 | -0.1024 | -0.0640 | 0.0000 | 1209 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.4620 | 0.3796 | 0.5461 | 0.0000 | -0.0614 | -0.0741 | -0.0491 | 0.0000 | 1209 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -0.0000 | 0.0000 | 0.6900 | 1209 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | -0.0000 | 0.0000 | 0.6900 | 1209 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | -0.0578 | -0.0965 | -0.0246 | 0.0000 | 0.0082 | 0.0051 | 0.0116 | 0.0000 | 1209 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | -0.0578 | -0.0965 | -0.0246 | 0.0000 | 0.0082 | 0.0051 | 0.0116 | 0.0000 | 1209 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.6522 | 0.2766 | 0.9574 | 0.0000 | -0.0741 | -0.1454 | -0.0171 | 0.0000 | 48 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.6522 | 0.2766 | 0.9574 | 0.0000 | -0.0741 | -0.1454 | -0.0171 | 0.0000 | 48 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.8287 | 0.7747 | 0.8852 | 0.0000 | -0.1471 | -0.1688 | -0.1236 | 0.0000 | 1209 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.8287 | 0.7747 | 0.8852 | 0.0000 | -0.1471 | -0.1688 | -0.1236 | 0.0000 | 1209 |
| tta4_entropy_roi | one_minus_confidence | tta4 | -0.1304 | -0.2979 | 0.0000 | 0.0661 | 0.0027 | -0.0075 | 0.0175 | 0.6800 | 48 |
| tta4_entropy_roi | one_minus_confidence | tta4 | -0.1304 | -0.2979 | 0.0000 | 0.0661 | 0.0027 | -0.0075 | 0.0175 | 0.6800 | 48 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.0318 | 0.0104 | 0.0552 | 0.0100 | -0.0035 | -0.0051 | -0.0021 | 0.0000 | 1208 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.0453 | 0.0206 | 0.0718 | 0.0000 | -0.0038 | -0.0056 | -0.0021 | 0.0000 | 1208 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.4684 | 0.3836 | 0.5532 | 0.0000 | -0.0626 | -0.0759 | -0.0505 | 0.0000 | 1209 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.5057 | 0.4199 | 0.5923 | 0.0000 | -0.0666 | -0.0814 | -0.0540 | 0.0000 | 1208 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0326 | 0.0000 | 0.1199 | 0.6431 | -0.0031 | -0.0102 | 0.0003 | 0.1200 | 48 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0326 | 0.0000 | 0.1199 | 0.6431 | -0.0031 | -0.0102 | 0.0003 | 0.1200 | 48 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.4239 | -0.0213 | 0.8099 | 0.0867 | -0.0495 | -0.0838 | -0.0228 | 0.0000 | 48 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.4239 | -0.0213 | 0.8099 | 0.0867 | -0.0495 | -0.0838 | -0.0228 | 0.0000 | 48 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

map-derived descriptors replaced by the per-case CSV columns where the evaluation wrote an equivalent (map_entropy_roi_tta4 <- csv entropy_mean_roi; map_variance_roi_tta4 <- csv variance_mean_roi; tta4_view_pairwise_dice <- csv view_pairwise_dice; pred_area_px <- csv pred_volume_mm3 / voxel volume).

## Audit note

probability maps present for 50 of 1251 cases (evaluation.probability_map_cases caps the saved maps); the map-only descriptors `one_minus_confidence`, `map_ambiguous_fraction_tta4`, `pred_roi_px` and every ensemble column are NaN for the cases without a map, and their rows drop out of the predictor tables (`n_used` / `n_undefined` show it).

## Audit note

`ensemble3_pairwise_dice` is undefined for every case: no view-agreement CSV and no map (ensembles need at least two seeds).

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 59, baseline 67, ensemble3 0, ensemble3_tta4 0

## Audit note

tta4 shares 59 failures; 9 of them have no ROI value (empty prediction), 17 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice: mean over the 4 tta4 views per case, averaged over the seeds (1.0 means the four views produced the same binary mask).

## Audit note

seed agreement: 0 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 1 case(s), max |diff| = 1.48e-05
