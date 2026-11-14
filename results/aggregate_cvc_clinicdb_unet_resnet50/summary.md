# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

15 runs: folds [0, 1, 2, 3, 4] x seeds [0, 1, 2]; config `configs\cvc_clinicdb\unet_resnet50.yaml`; pairing unit `group`; bootstrap draws `config`; Holm correction; failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 13.8831 | 0.5808 |
| ensemble3 | assd | 612 | 29 | 0 | 11.1300 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 10.1181 |  |
| tta2 | assd | 612 | 29 | 3 | 12.5206 | 0.7421 |
| tta4 | assd | 612 | 29 | 3 | 10.7957 | 0.3235 |
| tta8 | assd | 612 | 29 | 3 | 10.4782 | 0.3422 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.4964 | 0.0032 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5381 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5664 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.5191 | 0.0047 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5470 | 0.0046 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5559 | 0.0062 |
| baseline | dice | 612 | 29 | 3 | 0.7763 | 0.0156 |
| ensemble3 | dice | 612 | 29 | 0 | 0.8035 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.8132 |  |
| tta2 | dice | 612 | 29 | 3 | 0.7908 | 0.0173 |
| tta4 | dice | 612 | 29 | 3 | 0.7999 | 0.0167 |
| tta8 | dice | 612 | 29 | 3 | 0.8046 | 0.0100 |
| baseline | hd95 | 612 | 29 | 3 | 50.8592 | 3.4429 |
| ensemble3 | hd95 | 612 | 29 | 0 | 42.7198 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 38.4065 |  |
| tta2 | hd95 | 612 | 29 | 3 | 46.3581 | 3.1195 |
| tta4 | hd95 | 612 | 29 | 3 | 40.6978 | 1.5634 |
| tta8 | hd95 | 612 | 29 | 3 | 39.6879 | 1.7685 |
| baseline | hd95_pooled | 612 | 29 | 3 | 47.7237 | 3.2303 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 39.6407 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 36.1165 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 43.2999 | 2.8768 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 38.2140 | 1.3483 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 37.1529 | 1.6634 |

Mean over all frames of the fold, pooled over the 5 folds (612 cases), then mean ± std across the three seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 14.8814 | 0.6702 |
| ensemble3 | assd | 612 | 29 | 0 | 11.8730 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 10.7613 |  |
| tta2 | assd | 612 | 29 | 3 | 13.2503 | 0.7854 |
| tta4 | assd | 612 | 29 | 3 | 11.3921 | 0.5507 |
| tta8 | assd | 612 | 29 | 3 | 11.1326 | 0.3571 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.4838 | 0.0076 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5327 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5599 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.5136 | 0.0082 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5407 | 0.0034 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5495 | 0.0050 |
| baseline | dice | 612 | 29 | 3 | 0.7515 | 0.0174 |
| ensemble3 | dice | 612 | 29 | 0 | 0.7781 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.7874 |  |
| tta2 | dice | 612 | 29 | 3 | 0.7653 | 0.0169 |
| tta4 | dice | 612 | 29 | 3 | 0.7746 | 0.0147 |
| tta8 | dice | 612 | 29 | 3 | 0.7785 | 0.0081 |
| baseline | hd95 | 612 | 29 | 3 | 53.3672 | 3.0271 |
| ensemble3 | hd95 | 612 | 29 | 0 | 45.3409 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 40.9583 |  |
| tta2 | hd95 | 612 | 29 | 3 | 48.8901 | 3.1551 |
| tta4 | hd95 | 612 | 29 | 3 | 43.0153 | 1.6571 |
| tta8 | hd95 | 612 | 29 | 3 | 42.1365 | 1.0716 |
| baseline | hd95_pooled | 612 | 29 | 3 | 50.2909 | 2.8110 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 42.1892 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 38.6038 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 45.7896 | 2.8508 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 40.4613 | 1.4801 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 39.5470 | 0.9826 |

Each video contributes one value (mean over its frames), 29 videos in total; mean ± std across the three seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0138 | 0.0080 | 0.0199 | 0.0001 | 0.0008 | 0.7609 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0185 | 0.0126 | 0.0247 | 0.0000 | 0.0001 | 0.8667 | tta2 |
| tta2 | precision | 28 | 601 | all_cases | 0.0200 | 0.0108 | 0.0322 | 0.0000 | 0.0000 | 0.9163 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | 0.0053 | 0.0005 | 0.0102 | 0.0655 | 0.0655 | 0.3931 | tta2 |
| tta2 | hd | 28 | 592 | both_nonempty | -4.5681 | -6.9813 | -2.4057 | 0.0008 | 0.0040 | -0.6946 | tta2 |
| tta2 | hd95 | 28 | 592 | both_nonempty | -4.1732 | -6.5474 | -2.0588 | 0.0009 | 0.0040 | -0.6897 | tta2 |
| tta2 | hd95_pooled | 28 | 592 | both_nonempty | -4.1408 | -6.4902 | -2.0347 | 0.0009 | 0.0040 | -0.6897 | tta2 |
| tta2 | assd | 28 | 592 | both_nonempty | -1.3023 | -2.1031 | -0.5663 | 0.0044 | 0.0088 | -0.6010 | tta2 |
| tta2 | boundary_f1 | 28 | 601 | all_cases | 0.0218 | 0.0127 | 0.0302 | 0.0001 | 0.0006 | 0.7931 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0231 | 0.0150 | 0.0310 | 0.0000 | 0.0001 | 0.8345 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0314 | 0.0230 | 0.0395 | 0.0000 | 0.0000 | 0.9264 | tta4 |
| tta4 | precision | 28 | 598 | all_cases | 0.0356 | 0.0221 | 0.0535 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0115 | 0.0035 | 0.0200 | 0.0075 | 0.0075 | 0.5586 | tta4 |
| tta4 | hd | 28 | 591 | both_nonempty | -10.7023 | -14.5318 | -6.9648 | 0.0000 | 0.0001 | -0.8670 | tta4 |
| tta4 | hd95 | 28 | 591 | both_nonempty | -10.1203 | -13.9285 | -6.4125 | 0.0000 | 0.0001 | -0.8424 | tta4 |
| tta4 | hd95_pooled | 28 | 591 | both_nonempty | -9.5338 | -13.2985 | -5.8924 | 0.0000 | 0.0001 | -0.8325 | tta4 |
| tta4 | assd | 28 | 591 | both_nonempty | -3.1216 | -4.5188 | -1.8843 | 0.0000 | 0.0001 | -0.8374 | tta4 |
| tta4 | boundary_f1 | 28 | 598 | all_cases | 0.0497 | 0.0381 | 0.0613 | 0.0000 | 0.0000 | 0.9951 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0270 | 0.0183 | 0.0352 | 0.0000 | 0.0000 | 0.8575 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0374 | 0.0278 | 0.0470 | 0.0000 | 0.0000 | 0.9356 | tta8 |
| tta8 | precision | 28 | 598 | all_cases | 0.0467 | 0.0319 | 0.0643 | 0.0000 | 0.0000 | 1.0000 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | 0.0098 | 0.0019 | 0.0180 | 0.0345 | 0.0345 | 0.4483 | tta8 |
| tta8 | hd | 28 | 591 | both_nonempty | -11.6628 | -15.3710 | -7.9528 | 0.0000 | 0.0000 | -0.8867 | tta8 |
| tta8 | hd95 | 28 | 591 | both_nonempty | -10.7054 | -14.3268 | -7.0130 | 0.0000 | 0.0000 | -0.8621 | tta8 |
| tta8 | hd95_pooled | 28 | 591 | both_nonempty | -10.1554 | -13.8105 | -6.4880 | 0.0000 | 0.0000 | -0.8621 | tta8 |
| tta8 | assd | 28 | 591 | both_nonempty | -3.2294 | -4.4807 | -2.0575 | 0.0000 | 0.0000 | -0.8719 | tta8 |
| tta8 | boundary_f1 | 28 | 598 | all_cases | 0.0570 | 0.0422 | 0.0721 | 0.0000 | 0.0000 | 0.9803 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 15 | 0.0245 | 0.0079 | 0.0480 |  |  |  |  |
| tta2 | 612 | 18 | 0.0294 | 0.0083 | 0.0585 | 5.0000 | 2.0000 | 0.4531 | 0.8672 |
| tta4 | 612 | 19 | 0.0310 | 0.0094 | 0.0615 | 6.0000 | 2.0000 | 0.2891 | 0.8672 |
| tta8 | 612 | 19 | 0.0310 | 0.0105 | 0.0598 | 6.0000 | 2.0000 | 0.2891 | 0.8672 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 122.3093 | 149.8642 | 0.4101 | 0.1144 |  |  |  |  |  |  |  |  |  |
| ensemble3 | 612 | 130.0335 | 156.5783 | 0.3203 | 0.1062 | 6.7141 | -1.8759 | 11.9620 | -0.0899 | -0.1298 | -0.0530 | -0.0082 | -0.0179 | 0.0000 |
| ensemble3_tta4 | 612 | 120.4750 | 155.0230 | 0.2794 | 0.1095 | 5.1588 | -12.3966 | 9.3751 | -0.1307 | -0.1815 | -0.0845 | -0.0049 | -0.0174 | 0.0067 |
| tta2 | 612 | 116.6130 | 147.5756 | 0.3644 | 0.1144 | -2.2885 | -13.3299 | 3.2620 | -0.0458 | -0.0762 | -0.0182 | 0.0000 | -0.0132 | 0.0125 |
| tta4 | 612 | 112.0310 | 146.7790 | 0.3105 | 0.1111 | -3.0852 | -24.6512 | 4.0748 | -0.0997 | -0.1501 | -0.0542 | -0.0033 | -0.0162 | 0.0087 |
| tta8 | 612 | 113.2703 | 143.5993 | 0.2990 | 0.1144 | -6.2649 | -23.7102 | 6.3181 | -0.1111 | -0.1564 | -0.0694 | 0.0000 | -0.0132 | 0.0138 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -0.1131 | -5.2553 | 6.5519 | 0.0506 | 0.2026 | tta2 |
| tta2 | hd95 | 29 | 612 | 0.2902 | -4.8139 | 6.9388 | 0.0724 | 0.2066 | baseline |
| tta2 | hd95_pooled | 29 | 612 | 0.3305 | -4.7644 | 7.0056 | 0.0689 | 0.2066 | baseline |
| tta2 | assd | 29 | 612 | 3.5688 | -1.6094 | 10.2943 | 0.7172 | 0.7172 | baseline |
| tta4 | hd | 29 | 612 | -4.6449 | -10.9596 | 3.0228 | 0.0092 | 0.0369 | tta4 |
| tta4 | hd95 | 29 | 612 | -4.0057 | -10.1751 | 3.6191 | 0.0138 | 0.0415 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -3.4239 | -9.4769 | 4.0863 | 0.0179 | 0.0415 | tta4 |
| tta4 | assd | 29 | 612 | 3.1114 | -2.6980 | 10.5464 | 0.3357 | 0.3357 | baseline |
| tta8 | hd | 29 | 612 | -5.6196 | -11.9414 | 2.0674 | 0.0121 | 0.0484 | tta8 |
| tta8 | hd95 | 29 | 612 | -4.6646 | -10.8758 | 2.8569 | 0.0190 | 0.0571 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | -4.1148 | -10.2576 | 3.3137 | 0.0291 | 0.0581 | tta8 |
| tta8 | assd | 29 | 612 | 3.0583 | -2.4886 | 10.1181 | 0.5647 | 0.5647 | baseline |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 68 | 0.7118 | 0.0000 | 0.5051 | 0.0000 | 0.5675 |
| -1 | entropy_mean_roi | 612 | 68 | 0.5882 | 0.0000 | 0.7942 | 0.0000 | 0.9105 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 3 | 118 | 6 | 0.7270 | 0.0094 | 0.6417 | 0.0056 |
| 1 | dice | 3 | 137 | 6 | 0.7808 | 0.0408 | 0.7848 | 0.0379 |
| 2 | dice | 3 | 126 | 6 | 0.7680 | 0.0201 | 0.7695 | 0.0229 |
| 3 | dice | 3 | 129 | 6 | 0.7694 | 0.0310 | 0.7505 | 0.0307 |
| 4 | dice | 3 | 102 | 5 | 0.8461 | 0.0018 | 0.8229 | 0.0093 |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4 | dice | 29 | 612 | all_cases | 0.0231 | 0.0150 | 0.0310 | 0.0000 | 0.0001 | 0.8345 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0314 | 0.0230 | 0.0395 | 0.0000 | 0.0000 | 0.9264 | tta4 |
| tta4 | precision | 28 | 598 | all_cases | 0.0356 | 0.0221 | 0.0535 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0115 | 0.0035 | 0.0200 | 0.0075 | 0.0075 | 0.5586 | tta4 |
| tta4 | hd | 28 | 591 | both_nonempty | -10.7023 | -14.5318 | -6.9648 | 0.0000 | 0.0001 | -0.8670 | tta4 |
| tta4 | hd95 | 28 | 591 | both_nonempty | -10.1203 | -13.9285 | -6.4125 | 0.0000 | 0.0001 | -0.8424 | tta4 |
| tta4 | hd95_pooled | 28 | 591 | both_nonempty | -9.5338 | -13.2985 | -5.8924 | 0.0000 | 0.0001 | -0.8325 | tta4 |
| tta4 | assd | 28 | 591 | both_nonempty | -3.1216 | -4.5188 | -1.8843 | 0.0000 | 0.0001 | -0.8374 | tta4 |
| tta4 | boundary_f1 | 28 | 598 | all_cases | 0.0497 | 0.0381 | 0.0613 | 0.0000 | 0.0000 | 0.9951 | tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0266 | 0.0187 | 0.0348 | 0.0000 | 0.0000 | 0.9218 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0328 | 0.0245 | 0.0414 | 0.0000 | 0.0000 | 0.9586 | ensemble3 |
| ensemble3 | precision | 28 | 592 | all_cases | 0.0327 | 0.0234 | 0.0441 | 0.0000 | 0.0000 | 1.0000 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0130 | 0.0069 | 0.0196 | 0.0004 | 0.0004 | 0.7195 | ensemble3 |
| ensemble3 | hd | 28 | 592 | both_nonempty | -8.1383 | -10.9815 | -5.4419 | 0.0000 | 0.0000 | -0.8916 | ensemble3 |
| ensemble3 | hd95 | 28 | 592 | both_nonempty | -7.6848 | -10.3788 | -4.9808 | 0.0000 | 0.0000 | -0.8670 | ensemble3 |
| ensemble3 | hd95_pooled | 28 | 592 | both_nonempty | -7.6925 | -10.3361 | -5.0482 | 0.0000 | 0.0000 | -0.8670 | ensemble3 |
| ensemble3 | assd | 28 | 592 | both_nonempty | -2.6097 | -3.6121 | -1.7062 | 0.0000 | 0.0000 | -0.9261 | ensemble3 |
| ensemble3 | boundary_f1 | 28 | 592 | all_cases | 0.0440 | 0.0348 | 0.0538 | 0.0000 | 0.0000 | 0.9901 | ensemble3 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0359 | 0.0253 | 0.0458 | 0.0000 | 0.0000 | 0.8759 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0474 | 0.0360 | 0.0587 | 0.0000 | 0.0000 | 0.9632 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 28 | 590 | all_cases | 0.0482 | 0.0323 | 0.0697 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0179 | 0.0096 | 0.0268 | 0.0001 | 0.0001 | 0.7747 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 28 | 588 | both_nonempty | -13.0347 | -16.9161 | -9.3493 | 0.0000 | 0.0000 | -0.9409 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 28 | 588 | both_nonempty | -12.0675 | -15.9611 | -8.3143 | 0.0000 | 0.0000 | -0.9212 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 28 | 588 | both_nonempty | -11.2779 | -15.1382 | -7.5922 | 0.0000 | 0.0000 | -0.9163 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 28 | 588 | both_nonempty | -3.7214 | -5.1232 | -2.5379 | 0.0000 | 0.0000 | -0.9557 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 28 | 590 | all_cases | 0.0712 | 0.0568 | 0.0868 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0128 | 0.0076 | 0.0188 | 0.0000 | 0.0000 | 0.8818 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0160 | 0.0102 | 0.0227 | 0.0000 | 0.0000 | 0.9163 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 28 | 590 | all_cases | 0.0126 | 0.0090 | 0.0170 | 0.0000 | 0.0000 | 0.9704 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0064 | 0.0015 | 0.0119 | 0.0264 | 0.0264 | 0.4778 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 28 | 590 | both_nonempty | -2.3323 | -3.8715 | -0.5797 | 0.0020 | 0.0082 | -0.6453 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 28 | 590 | both_nonempty | -1.9472 | -3.4451 | -0.2746 | 0.0041 | 0.0122 | -0.6059 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 28 | 590 | both_nonempty | -1.7441 | -3.2340 | -0.0792 | 0.0089 | 0.0177 | -0.5567 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 28 | 590 | both_nonempty | -0.5998 | -1.0663 | -0.0609 | 0.0007 | 0.0036 | -0.6995 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 28 | 590 | all_cases | 0.0215 | 0.0117 | 0.0322 | 0.0000 | 0.0002 | 0.8325 | ensemble3_tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0035 | -0.0059 | 0.0132 | 0.6136 | 1.0000 | 0.1133 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0014 | -0.0078 | 0.0106 | 0.6618 | 1.0000 | 0.0985 | ensemble3 |
| ensemble3 | precision | 28 | 590 | all_cases | -0.0029 | -0.0109 | 0.0035 | 0.6456 | 1.0000 | 0.1034 | tta4 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0015 | -0.0089 | 0.0124 | 0.9197 | 1.0000 | 0.0246 | ensemble3 |
| ensemble3 | hd | 28 | 588 | both_nonempty | 2.5640 | -0.4761 | 5.8617 | 0.1858 | 1.0000 | 0.2906 | tta4 |
| ensemble3 | hd95 | 28 | 588 | both_nonempty | 2.4355 | -0.6431 | 5.6894 | 0.3620 | 1.0000 | 0.2020 | tta4 |
| ensemble3 | hd95_pooled | 28 | 588 | both_nonempty | 1.8413 | -1.2475 | 5.1456 | 0.6456 | 1.0000 | 0.1034 | tta4 |
| ensemble3 | assd | 28 | 588 | both_nonempty | 0.5118 | -0.4673 | 1.6170 | 0.7966 | 1.0000 | 0.0591 | tta4 |
| ensemble3 | boundary_f1 | 28 | 590 | all_cases | -0.0057 | -0.0170 | 0.0049 | 0.3161 | 1.0000 | -0.2217 | tta4 |

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 22 | 0 | 14 | 6860.5000 | 0.2472 | 0.6651 |
| failures_all | 68 | 22 | 0 | 14 | 2305.0000 | 0.4981 | 0.6651 |
| failures_non_missed | 49 | 3 | 0 | 0 | 4374.0000 | 0.4585 | 0.6703 |
| failures_missed | 19 | 19 | 0 | 14 | 0.0000 | 0.6599 | 0.6599 |
| failures_empty_0px | 22 | 22 | 22 | 14 | 0.0000 | 0.6651 |  |
| failures_medium_101_1000px | 2 | 0 | 0 | 0 | 233.0000 | 0.6524 |  |
| failures_large_gt_1000px | 44 | 0 | 0 | 0 | 4828.0000 | 0.4358 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9105 | 0.8522 | 0.9774 | 0.0828 | 0.0621 | 0.1153 | 0.0661 | 598 | 598 | 54 | 14 | 0.7231 | 0.9290 | 0.0828 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.9014 | 0.8421 | 0.9742 | 0.0820 | 0.0616 | 0.1139 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.0820 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean | 1.0000 | 1.0000 | 1.0000 | 0.0720 | 0.0562 | 0.0949 | 0.0581 | 549 | 549 | 5 | 14 |  |  | 0.0720 |
| tta4_variance_roi | own | tta4 | all_failures | 0.8591 | 0.7707 | 0.9608 | 0.0846 | 0.0621 | 0.1195 | 0.0661 | 598 | 598 | 54 | 14 | 0.6822 | 0.8881 | 0.0846 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.8463 | 0.7545 | 0.9565 | 0.0837 | 0.0616 | 0.1180 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.0837 |
| tta4_variance_roi | own | tta4 | missed_vs_clean | 0.9849 | 0.9677 | 0.9965 | 0.0713 | 0.0555 | 0.0939 | 0.0581 | 549 | 549 | 5 | 14 |  |  | 0.0713 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.5675 | 0.4112 | 0.7056 | 0.1882 | 0.1097 | 0.2879 | 0.0690 | 612 | 612 | 68 | 0 | 0.5675 | 0.5675 | 0.1890 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.7054 | 0.5964 | 0.8567 | 0.1074 | 0.0781 | 0.1515 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.1074 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean | 0.2121 | 0.0615 | 0.4503 | 0.1731 | 0.0935 | 0.2706 | 0.0604 | 563 | 563 | 19 | 0 |  |  | 0.1739 |
| tta4_entropy_image | own | tta4 | all_failures | 0.6331 | 0.4944 | 0.7450 | 0.1740 | 0.1173 | 0.2407 | 0.0690 | 612 | 612 | 68 | 0 | 0.6331 | 0.6331 | 0.1747 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.6834 | 0.5241 | 0.7893 | 0.1457 | 0.1040 | 0.1913 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.1457 |
| tta4_entropy_image | own | tta4 | missed_vs_clean | 0.5032 | 0.3048 | 0.6813 | 0.1527 | 0.1021 | 0.2131 | 0.0604 | 563 | 563 | 19 | 0 |  |  | 0.1534 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.8782 | 0.8061 | 0.9696 | 0.0752 | 0.0563 | 0.1058 | 0.0546 | 590 | 590 | 45 | 22 | 0.5898 | 0.9182 | 0.0752 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.8782 | 0.8061 | 0.9696 | 0.0752 | 0.0563 | 0.1058 | 0.0546 | 590 | 590 | 45 | 0 |  |  | 0.0752 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0652 | 0.0509 | 0.0855 | 0.0482 | 545 | 545 | 0 | 22 |  |  | 0.0652 |
| map_entropy_roi_tta4 | tta4 | tta4 | all_failures | 0.8816 | 0.8111 | 0.9701 | 0.0823 | 0.0619 | 0.1141 | 0.0646 | 590 | 590 | 46 | 22 | 0.5964 | 0.9199 | 0.0823 |
| map_entropy_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8816 | 0.8111 | 0.9701 | 0.0823 | 0.0619 | 0.1141 | 0.0646 | 590 | 590 | 46 | 3 |  |  | 0.0823 |
| map_entropy_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0719 | 0.0567 | 0.0932 | 0.0574 | 544 | 544 | 0 | 19 |  |  | 0.0719 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.7814 | 0.6898 | 0.9062 | 0.0835 | 0.0611 | 0.1190 | 0.0546 | 590 | 590 | 45 | 22 | 0.5248 | 0.8532 | 0.0835 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.7814 | 0.6898 | 0.9062 | 0.0835 | 0.0611 | 0.1190 | 0.0546 | 590 | 590 | 45 | 0 |  |  | 0.0835 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0694 | 0.0546 | 0.0906 | 0.0482 | 545 | 545 | 0 | 22 |  |  | 0.0694 |
| map_variance_roi_tta4 | tta4 | tta4 | all_failures | 0.7874 | 0.6988 | 0.9099 | 0.0903 | 0.0671 | 0.1254 | 0.0646 | 590 | 590 | 46 | 22 | 0.5327 | 0.8562 | 0.0903 |
| map_variance_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.7874 | 0.6988 | 0.9099 | 0.0903 | 0.0671 | 0.1254 | 0.0646 | 590 | 590 | 46 | 3 |  |  | 0.0903 |
| map_variance_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0763 | 0.0601 | 0.0988 | 0.0574 | 544 | 544 | 0 | 19 |  |  | 0.0763 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | all_failures | 0.5224 | 0.3725 | 0.6695 | 0.1900 | 0.1121 | 0.2923 | 0.0589 | 612 | 612 | 67 | 0 | 0.5224 | 0.5224 | 0.1900 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.6433 | 0.5241 | 0.8137 | 0.1092 | 0.0762 | 0.1592 | 0.0546 | 590 | 590 | 45 | 0 |  |  | 0.1092 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | missed_vs_clean | 0.2750 | 0.0776 | 0.5871 | 0.1723 | 0.0958 | 0.2701 | 0.0514 | 567 | 567 | 22 | 0 |  |  | 0.1723 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | all_failures | 0.5291 | 0.3733 | 0.6728 | 0.1955 | 0.1195 | 0.2968 | 0.0690 | 612 | 612 | 68 | 0 | 0.5291 | 0.5291 | 0.1965 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | non_missed_failures | 0.6543 | 0.5290 | 0.8169 | 0.1182 | 0.0837 | 0.1699 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.1182 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | missed_vs_clean | 0.2063 | 0.0708 | 0.4847 | 0.1769 | 0.1022 | 0.2759 | 0.0604 | 563 | 563 | 19 | 0 |  |  | 0.1779 |
| pred_area_px | own | ensemble3_tta4 | all_failures | 0.1910 | 0.0882 | 0.2979 | 0.3147 | 0.2256 | 0.4136 | 0.0589 | 612 | 612 | 67 | 0 | 0.1910 | 0.1910 | 0.3147 |
| pred_area_px | own | ensemble3_tta4 | non_missed_failures | 0.2844 | 0.1784 | 0.4114 | 0.1990 | 0.1397 | 0.2671 | 0.0546 | 590 | 590 | 45 | 0 |  |  | 0.1990 |
| pred_area_px | own | ensemble3_tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.2588 | 0.1665 | 0.3589 | 0.0514 | 567 | 567 | 22 | 0 |  |  | 0.2588 |
| pred_area_px | tta4 | tta4 | all_failures | 0.1992 | 0.1016 | 0.2990 | 0.3156 | 0.2317 | 0.4103 | 0.0690 | 612 | 612 | 68 | 0 | 0.1992 | 0.1992 | 0.3202 |
| pred_area_px | tta4 | tta4 | non_missed_failures | 0.2764 | 0.1793 | 0.3992 | 0.2244 | 0.1585 | 0.2964 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.2244 |
| pred_area_px | tta4 | tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.2538 | 0.1736 | 0.3470 | 0.0604 | 563 | 563 | 19 | 0 |  |  | 0.2588 |
| one_minus_confidence | own | ensemble3_tta4 | all_failures | 0.8789 | 0.8080 | 0.9682 | 0.0740 | 0.0554 | 0.1040 | 0.0546 | 590 | 590 | 45 | 22 | 0.5903 | 0.9187 | 0.0740 |
| one_minus_confidence | own | ensemble3_tta4 | non_missed_failures | 0.8789 | 0.8080 | 0.9682 | 0.0740 | 0.0554 | 0.1040 | 0.0546 | 590 | 590 | 45 | 0 |  |  | 0.0740 |
| one_minus_confidence | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0639 | 0.0503 | 0.0836 | 0.0482 | 545 | 545 | 0 | 22 |  |  | 0.0639 |
| one_minus_confidence | tta4 | tta4 | all_failures | 0.8824 | 0.8129 | 0.9695 | 0.0811 | 0.0612 | 0.1124 | 0.0646 | 590 | 590 | 46 | 22 | 0.5969 | 0.9204 | 0.0811 |
| one_minus_confidence | tta4 | tta4 | non_missed_failures | 0.8824 | 0.8129 | 0.9695 | 0.0811 | 0.0612 | 0.1124 | 0.0646 | 590 | 590 | 46 | 3 |  |  | 0.0811 |
| one_minus_confidence | tta4 | tta4 | missed_vs_clean |  |  |  | 0.0707 | 0.0559 | 0.0909 | 0.0574 | 544 | 544 | 0 | 19 |  |  | 0.0707 |
| baseline_entropy_roi | own | baseline | all_failures | 0.9187 | 0.8738 | 0.9718 | 0.1013 | 0.0768 | 0.1355 | 0.0846 | 605 | 605 | 63 | 7 | 0.8269 | 0.9269 | 0.1013 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.9069 | 0.8606 | 0.9661 | 0.0998 | 0.0758 | 0.1335 | 0.0829 | 597 | 597 | 55 | 0 |  |  | 0.0998 |
| baseline_entropy_roi | own | baseline | missed_vs_clean | 0.9998 | 0.9978 | 1.0000 | 0.0890 | 0.0709 | 0.1137 | 0.0744 | 550 | 550 | 8 | 7 |  |  | 0.0890 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.9273 | 0.8806 | 0.9778 | 0.0877 | 0.0649 | 0.1192 | 0.0675 | 605 | 605 | 61 | 7 | 0.8319 | 0.9348 | 0.0877 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.9100 | 0.8589 | 0.9689 | 0.0857 | 0.0647 | 0.1160 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.0857 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9982 | 0.9948 | 1.0000 | 0.0774 | 0.0598 | 0.1010 | 0.0592 | 556 | 556 | 12 | 7 |  |  | 0.0774 |
| baseline_entropy_image | own | baseline | all_failures | 0.5992 | 0.4543 | 0.7130 | 0.2108 | 0.1496 | 0.2733 | 0.0862 | 612 | 612 | 70 | 0 | 0.5992 | 0.5992 | 0.2114 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.6155 | 0.4540 | 0.7282 | 0.1909 | 0.1392 | 0.2392 | 0.0829 | 597 | 597 | 55 | 0 |  |  | 0.1909 |
| baseline_entropy_image | own | baseline | missed_vs_clean | 0.5395 | 0.3149 | 0.7050 | 0.1796 | 0.1277 | 0.2333 | 0.0757 | 557 | 557 | 15 | 0 |  |  | 0.1802 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.6272 | 0.4842 | 0.7366 | 0.1886 | 0.1309 | 0.2500 | 0.0690 | 612 | 612 | 68 | 0 | 0.6272 | 0.6272 | 0.1892 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.6574 | 0.4647 | 0.7675 | 0.1632 | 0.1185 | 0.2076 | 0.0652 | 593 | 593 | 49 | 0 |  |  | 0.1632 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean | 0.5494 | 0.3377 | 0.7353 | 0.1639 | 0.1096 | 0.2220 | 0.0604 | 563 | 563 | 19 | 0 |  |  | 0.1645 |
| ensemble3_entropy_roi | own | ensemble3 | all_failures | 0.8639 | 0.8058 | 0.9543 | 0.0816 | 0.0614 | 0.1121 | 0.0615 | 592 | 592 | 45 | 20 | 0.5981 | 0.9058 | 0.0816 |
| ensemble3_entropy_roi | own | ensemble3 | non_missed_failures | 0.8639 | 0.8058 | 0.9543 | 0.0816 | 0.0614 | 0.1121 | 0.0615 | 592 | 592 | 45 | 0 |  |  | 0.0816 |
| ensemble3_entropy_roi | own | ensemble3 | missed_vs_clean |  |  |  | 0.0710 | 0.0566 | 0.0921 | 0.0545 | 547 | 547 | 0 | 20 |  |  | 0.0710 |
| ensemble3_entropy_roi | tta4 | tta4 | all_failures | 0.8966 | 0.8410 | 0.9659 | 0.0835 | 0.0631 | 0.1140 | 0.0650 | 592 | 592 | 48 | 20 | 0.6329 | 0.9270 | 0.0835 |
| ensemble3_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8877 | 0.8321 | 0.9614 | 0.0828 | 0.0628 | 0.1127 | 0.0643 | 588 | 588 | 44 | 5 |  |  | 0.0828 |
| ensemble3_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9949 | 0.9880 | 1.0000 | 0.0739 | 0.0587 | 0.0961 | 0.0579 | 548 | 548 | 4 | 15 |  |  | 0.0739 |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9105 | 0.8522 | 0.9774 | 0.0828 | 0.0621 | 0.1153 | 0.0661 | 598 | 598 | 54 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.9014 | 0.8421 | 0.9742 | 0.0820 | 0.0616 | 0.1139 | 0.0652 | 593 | 593 | 49 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean | 1.0000 | 1.0000 | 1.0000 | 0.0720 | 0.0562 | 0.0949 | 0.0581 | 549 | 549 | 5 |
| tta4_view_dice_disagreement | own | tta4 | all_failures | 0.8220 | 0.7220 | 0.9378 | 0.1246 | 0.0755 | 0.1909 | 0.0690 | 612 | 612 | 68 |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures | 0.8651 | 0.7824 | 0.9708 | 0.0832 | 0.0615 | 0.1167 | 0.0652 | 593 | 593 | 49 |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean | 0.7110 | 0.4966 | 0.9533 | 0.1144 | 0.0667 | 0.1802 | 0.0604 | 563 | 563 | 19 |
| ensemble3_dice_disagreement | own | ensemble3 | all_failures | 0.7714 | 0.6440 | 0.8924 | 0.1473 | 0.0832 | 0.2254 | 0.0655 | 612 | 612 | 65 |
| ensemble3_dice_disagreement | own | ensemble3 | non_missed_failures | 0.8309 | 0.7681 | 0.9519 | 0.0868 | 0.0641 | 0.1213 | 0.0615 | 592 | 592 | 45 |
| ensemble3_dice_disagreement | own | ensemble3 | missed_vs_clean | 0.6376 | 0.3975 | 0.9825 | 0.1383 | 0.0749 | 0.2167 | 0.0576 | 567 | 567 | 20 |
| ensemble3_dice_disagreement | tta4 | tta4 | all_failures | 0.7876 | 0.6627 | 0.9187 | 0.1479 | 0.0842 | 0.2259 | 0.0690 | 612 | 612 | 68 |
| ensemble3_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8512 | 0.7728 | 0.9558 | 0.0876 | 0.0651 | 0.1209 | 0.0652 | 593 | 593 | 49 |
| ensemble3_dice_disagreement | tta4 | tta4 | missed_vs_clean | 0.6235 | 0.3561 | 0.9898 | 0.1389 | 0.0751 | 0.2172 | 0.0604 | 563 | 563 | 19 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | all_failures | 0.6960 | 0.4886 | 0.8391 | 0.1804 | 0.1050 | 0.2751 | 0.0589 | 612 | 612 | 67 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | non_missed_failures | 0.8608 | 0.7829 | 0.9755 | 0.0765 | 0.0560 | 0.1083 | 0.0546 | 590 | 590 | 45 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | missed_vs_clean | 0.3590 | 0.0577 | 0.6921 | 0.1759 | 0.0944 | 0.2710 | 0.0514 | 567 | 567 | 22 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | all_failures | 0.7013 | 0.4931 | 0.8429 | 0.1847 | 0.1096 | 0.2791 | 0.0690 | 612 | 612 | 68 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8730 | 0.7933 | 0.9793 | 0.0815 | 0.0600 | 0.1142 | 0.0652 | 593 | 593 | 49 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | missed_vs_clean | 0.2583 | 0.0472 | 0.5465 | 0.1800 | 0.0979 | 0.2767 | 0.0604 | 563 | 563 | 19 |

`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the four tta4 views (per seed, averaged over seeds); `ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` are the same over the three seeds' baseline / tta4 maps. Higher = the members disagree = more uncertain. The tta4 entropy row is kept as the reference descriptor.

## (f-audit) Spearman against HD95 and 1 - Dice (all predictors)

| predictor | label_source | label_prediction | target | n_used | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | hd95 | 598 | 0.5882 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 598 | 0.7942 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 598 | 0.6599 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 598 | 0.7908 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 598 | 0.7118 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.5051 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 598 | 0.2525 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.2194 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.5576 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 590 | 0.6885 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | hd95 | 590 | 0.5642 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | one_minus_dice | 590 | 0.7725 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.5609 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 590 | 0.6238 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | hd95 | 590 | 0.5645 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | one_minus_dice | 590 | 0.7171 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.5248 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.3210 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | hd95 | 598 | 0.6035 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.4024 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | hd95 | 590 | 0.2438 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | one_minus_dice | 612 | -0.1865 | 0.0000 |
| pred_area_px | tta4 | tta4 | hd95 | 598 | 0.2318 | 0.0000 |
| pred_area_px | tta4 | tta4 | one_minus_dice | 612 | -0.1664 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | hd95 | 590 | 0.6144 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | one_minus_dice | 590 | 0.7170 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | hd95 | 590 | 0.6147 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | one_minus_dice | 590 | 0.7967 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 605 | 0.5587 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 605 | 0.8205 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 598 | 0.4974 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 605 | 0.7457 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 605 | 0.2100 | 0.0000 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.1479 | 0.0002 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 598 | 0.1323 | 0.0012 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.1223 | 0.0024 |
| ensemble3_entropy_roi | own | ensemble3 | hd95 | 592 | 0.5577 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | one_minus_dice | 592 | 0.7192 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | hd95 | 590 | 0.5581 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | one_minus_dice | 592 | 0.7642 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | hd95 | 592 | 0.6094 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | one_minus_dice | 592 | 0.6937 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | hd95 | 590 | 0.5947 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | one_minus_dice | 592 | 0.7357 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | hd95 | 592 | 0.2116 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | one_minus_dice | 612 | 0.1842 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | hd95 | 598 | 0.2043 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.1848 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | hd95 | 592 | 0.5879 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | one_minus_dice | 612 | 0.4360 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | hd95 | 598 | 0.5915 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | one_minus_dice | 612 | 0.4813 | 0.0000 |
| tta4_view_dice_disagreement | own | tta4 | hd95 | 598 | 0.6363 | 0.0000 |
| tta4_view_dice_disagreement | own | tta4 | one_minus_dice | 612 | 0.7619 | 0.0000 |
| ensemble3_dice_disagreement | own | ensemble3 | hd95 | 592 | 0.5744 | 0.0000 |
| ensemble3_dice_disagreement | own | ensemble3 | one_minus_dice | 612 | 0.6320 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | hd95 | 598 | 0.5910 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | one_minus_dice | 612 | 0.6890 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | hd95 | 590 | 0.6289 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | one_minus_dice | 612 | 0.6085 | 0.0000 |

Targets also come from the predictor's own prediction (`label_source=own`); the `tta4` rows repeat the correlation against the tta4 prediction for comparison. `n_used` counts the finite pairs, so ROI descriptors lose the missed detections here.

## (f-audit) detector vs detector on AUROC and AURC (paired bootstrap)

| predictor | compared_with | label_prediction | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.0551 | 0.0059 | 0.0901 | 0.0160 | -0.0017 | -0.0073 | 0.0021 | 0.4360 | 593 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.1961 | 0.1009 | 0.2650 | 0.0000 | -0.0254 | -0.0417 | -0.0138 | 0.0000 | 593 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.2180 | 0.0951 | 0.4195 | 0.0000 | -0.0637 | -0.0954 | -0.0361 | 0.0000 | 593 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0134 | -0.0047 | 0.0275 | 0.1880 | -0.0007 | -0.0020 | 0.0006 | 0.2480 | 590 |
| tta4_entropy_roi | map_entropy_roi_tta4 | ensemble3_tta4 | 0.0161 | -0.0005 | 0.0301 | 0.0620 | -0.0003 | -0.0023 | 0.0012 | 0.7520 | 590 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | 0.1075 | 0.0341 | 0.1504 | 0.0000 | -0.0088 | -0.0160 | -0.0030 | 0.0020 | 590 |
| tta4_entropy_roi | map_variance_roi_tta4 | ensemble3_tta4 | 0.1129 | 0.0378 | 0.1544 | 0.0000 | -0.0086 | -0.0169 | -0.0025 | 0.0040 | 590 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.2471 | 0.1371 | 0.3400 | 0.0000 | -0.0362 | -0.0603 | -0.0193 | 0.0000 | 593 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | ensemble3_tta4 | 0.2510 | 0.1381 | 0.3375 | 0.0000 | -0.0343 | -0.0596 | -0.0173 | 0.0000 | 590 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.6250 | 0.4865 | 0.7741 | 0.0000 | -0.1424 | -0.1939 | -0.0913 | 0.0000 | 593 |
| tta4_entropy_roi | pred_area_px | ensemble3_tta4 | 0.6098 | 0.4740 | 0.7772 | 0.0000 | -0.1241 | -0.1781 | -0.0756 | 0.0000 | 590 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0126 | -0.0065 | 0.0248 | 0.2080 | 0.0004 | -0.0013 | 0.0024 | 0.7320 | 590 |
| tta4_entropy_roi | one_minus_confidence | ensemble3_tta4 | 0.0153 | -0.0007 | 0.0279 | 0.0700 | 0.0009 | -0.0009 | 0.0031 | 0.3280 | 590 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | -0.0086 | -0.0409 | 0.0404 | 0.7640 | -0.0037 | -0.0082 | 0.0006 | 0.0840 | 593 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | -0.0041 | -0.0324 | 0.0331 | 0.8840 | 0.0020 | -0.0022 | 0.0064 | 0.3260 | 593 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.2440 | 0.1071 | 0.4876 | 0.0000 | -0.0812 | -0.1099 | -0.0484 | 0.0000 | 593 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.2676 | 0.1268 | 0.4840 | 0.0000 | -0.0854 | -0.1192 | -0.0503 | 0.0000 | 593 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0027 | -0.0191 | 0.0255 | 0.7720 | -0.0016 | -0.0044 | 0.0014 | 0.2460 | 588 |
| tta4_entropy_roi | ensemble3_entropy_roi | ensemble3 | 0.0177 | -0.0107 | 0.0396 | 0.2220 | 0.0006 | -0.0026 | 0.0036 | 0.6020 | 590 |
| tta4_entropy_roi | ensemble3_variance_roi | tta4 | 0.0647 | 0.0196 | 0.0961 | 0.0000 | -0.0058 | -0.0117 | -0.0014 | 0.0100 | 588 |
| tta4_entropy_roi | ensemble3_variance_roi | ensemble3 | 0.0770 | 0.0127 | 0.1160 | 0.0000 | -0.0036 | -0.0102 | 0.0015 | 0.2140 | 590 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.2303 | 0.1047 | 0.4508 | 0.0000 | -0.0715 | -0.1020 | -0.0400 | 0.0000 | 593 |
| tta4_entropy_roi | ensemble3_entropy_image | ensemble3 | 0.2193 | 0.0835 | 0.4118 | 0.0020 | -0.0600 | -0.0877 | -0.0327 | 0.0000 | 590 |
| tta4_entropy_roi | ensemble3_variance_image | tta4 | 0.1649 | 0.0975 | 0.2667 | 0.0000 | -0.0309 | -0.0515 | -0.0177 | 0.0000 | 593 |
| tta4_entropy_roi | ensemble3_variance_image | ensemble3 | 0.1350 | 0.0678 | 0.2343 | 0.0000 | -0.0255 | -0.0445 | -0.0137 | 0.0000 | 590 |
| tta4_entropy_roi | tta4_view_dice_disagreement | tta4 | 0.0364 | -0.0017 | 0.0701 | 0.0760 | -0.0012 | -0.0066 | 0.0028 | 0.5640 | 593 |
| tta4_entropy_roi | ensemble3_dice_disagreement | tta4 | 0.0502 | 0.0059 | 0.0854 | 0.0120 | -0.0057 | -0.0126 | -0.0008 | 0.0220 | 593 |
| tta4_entropy_roi | ensemble3_dice_disagreement | ensemble3 | 0.0524 | -0.0045 | 0.0846 | 0.0880 | -0.0046 | -0.0120 | 0.0003 | 0.0760 | 590 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | tta4 | 0.0284 | -0.0154 | 0.0624 | 0.4880 | 0.0005 | -0.0046 | 0.0052 | 0.8480 | 593 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | ensemble3_tta4 | 0.0335 | -0.0133 | 0.0671 | 0.3620 | -0.0016 | -0.0070 | 0.0022 | 0.4560 | 590 |
| tta4_view_dice_disagreement | tta4_entropy_roi | tta4 | -0.0364 | -0.0701 | 0.0017 | 0.0760 | 0.0012 | -0.0028 | 0.0066 | 0.5640 | 593 |
| tta4_view_dice_disagreement | one_minus_confidence | tta4 | -0.0259 | -0.0520 | 0.0057 | 0.1620 | 0.0016 | -0.0022 | 0.0067 | 0.4260 | 590 |
| ensemble3_dice_disagreement | ensemble3_entropy_roi | ensemble3 | -0.0330 | -0.0627 | 0.0054 | 0.1280 | 0.0052 | 0.0001 | 0.0126 | 0.0440 | 592 |
| ensemble3_tta4_dice_disagreement | map_entropy_roi_tta4 | ensemble3_tta4 | -0.0174 | -0.0476 | 0.0220 | 0.6240 | 0.0013 | -0.0030 | 0.0065 | 0.5360 | 590 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 68, baseline 70, ensemble3 65, ensemble3_tta4 67

## Audit note

tta4 shares 68 failures; 14 of them have no ROI value (empty prediction), 22 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice: mean over the 4 tta4 views per case, averaged over the seeds (1.0 means the four views produced the same binary mask).

## Audit note

seed agreement: 612 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 5.72e-04
