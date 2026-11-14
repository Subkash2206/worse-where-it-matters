# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

5 runs: folds [0, 1, 2, 3, 4] x seeds [0]; config `configs/cvc_clinicdb/unet_resnet50_data025.yaml`; pairing unit `group`; bootstrap draws `config`; Holm correction; failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 1 | 20.8043 |  |
| tta2 | assd | 612 | 29 | 1 | 18.9322 |  |
| tta4 | assd | 612 | 29 | 1 | 17.5916 |  |
| tta8 | assd | 612 | 29 | 1 | 16.7722 |  |
| baseline | boundary_f1 | 612 | 29 | 1 | 0.2941 |  |
| tta2 | boundary_f1 | 612 | 29 | 1 | 0.3161 |  |
| tta4 | boundary_f1 | 612 | 29 | 1 | 0.3335 |  |
| tta8 | boundary_f1 | 612 | 29 | 1 | 0.3443 |  |
| baseline | dice | 612 | 29 | 1 | 0.6701 |  |
| tta2 | dice | 612 | 29 | 1 | 0.6843 |  |
| tta4 | dice | 612 | 29 | 1 | 0.6948 |  |
| tta8 | dice | 612 | 29 | 1 | 0.6953 |  |
| baseline | hd95 | 612 | 29 | 1 | 69.8064 |  |
| tta2 | hd95 | 612 | 29 | 1 | 62.9736 |  |
| tta4 | hd95 | 612 | 29 | 1 | 57.4316 |  |
| tta8 | hd95 | 612 | 29 | 1 | 53.3174 |  |
| baseline | hd95_pooled | 612 | 29 | 1 | 65.1813 |  |
| tta2 | hd95_pooled | 612 | 29 | 1 | 59.3979 |  |
| tta4 | hd95_pooled | 612 | 29 | 1 | 53.7939 |  |
| tta8 | hd95_pooled | 612 | 29 | 1 | 50.3765 |  |

Mean over all frames of the fold, pooled over the 5 folds (612 cases), then mean ± std across the three seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 1 | 22.8115 |  |
| tta2 | assd | 612 | 29 | 1 | 20.5005 |  |
| tta4 | assd | 612 | 29 | 1 | 22.5170 |  |
| tta8 | assd | 612 | 29 | 1 | 22.4418 |  |
| baseline | boundary_f1 | 612 | 29 | 1 | 0.2832 |  |
| tta2 | boundary_f1 | 612 | 29 | 1 | 0.2992 |  |
| tta4 | boundary_f1 | 612 | 29 | 1 | 0.3123 |  |
| tta8 | boundary_f1 | 612 | 29 | 1 | 0.3184 |  |
| baseline | dice | 612 | 29 | 1 | 0.6463 |  |
| tta2 | dice | 612 | 29 | 1 | 0.6548 |  |
| tta4 | dice | 612 | 29 | 1 | 0.6606 |  |
| tta8 | dice | 612 | 29 | 1 | 0.6597 |  |
| baseline | hd95 | 612 | 29 | 1 | 73.0219 |  |
| tta2 | hd95 | 612 | 29 | 1 | 66.2343 |  |
| tta4 | hd95 | 612 | 29 | 1 | 64.2079 |  |
| tta8 | hd95 | 612 | 29 | 1 | 61.9366 |  |
| baseline | hd95_pooled | 612 | 29 | 1 | 68.4800 |  |
| tta2 | hd95_pooled | 612 | 29 | 1 | 62.7674 |  |
| tta4 | hd95_pooled | 612 | 29 | 1 | 60.5094 |  |
| tta8 | hd95_pooled | 612 | 29 | 1 | 59.0941 |  |

Each video contributes one value (mean over its frames), 29 videos in total; mean ± std across the three seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0085 | -0.0143 | 0.0278 | 0.0562 | 0.1687 | 0.4069 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0128 | -0.0073 | 0.0307 | 0.0365 | 0.1461 | 0.4437 | tta2 |
| tta2 | precision | 29 | 587 | all_cases | 0.0291 | 0.0135 | 0.0475 | 0.0001 | 0.0011 | 0.7655 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | -0.0028 | -0.0231 | 0.0135 | 0.2560 | 0.2560 | 0.2460 | baseline |
| tta2 | hd | 29 | 587 | both_nonempty | -7.8931 | -11.8132 | -3.3918 | 0.0002 | 0.0016 | -0.7471 | tta2 |
| tta2 | hd95 | 29 | 587 | both_nonempty | -6.7876 | -10.6181 | -2.3702 | 0.0004 | 0.0026 | -0.7195 | tta2 |
| tta2 | hd95_pooled | 29 | 587 | both_nonempty | -5.7127 | -9.5988 | -1.3170 | 0.0006 | 0.0033 | -0.7011 | tta2 |
| tta2 | assd | 29 | 587 | both_nonempty | -2.3109 | -4.9423 | 0.1604 | 0.0006 | 0.0033 | -0.7011 | tta2 |
| tta2 | boundary_f1 | 29 | 587 | all_cases | 0.0160 | 0.0017 | 0.0307 | 0.0628 | 0.1687 | 0.4039 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0143 | -0.0223 | 0.0447 | 0.0431 | 0.1036 | 0.4299 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0254 | -0.0050 | 0.0527 | 0.0190 | 0.0762 | 0.4943 | tta4 |
| tta4 | precision | 29 | 583 | all_cases | 0.0364 | 0.0025 | 0.0638 | 0.0000 | 0.0004 | 0.8023 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | -0.0103 | -0.0369 | 0.0106 | 0.9152 | 0.9152 | 0.0253 | baseline |
| tta4 | hd | 29 | 583 | both_nonempty | -10.7812 | -17.8165 | -2.6654 | 0.0009 | 0.0071 | -0.6782 | tta4 |
| tta4 | hd95 | 29 | 583 | both_nonempty | -8.8141 | -15.6442 | -0.9227 | 0.0018 | 0.0127 | -0.6414 | tta4 |
| tta4 | hd95_pooled | 29 | 583 | both_nonempty | -7.9706 | -14.9759 | 0.0931 | 0.0028 | 0.0166 | -0.6184 | tta4 |
| tta4 | assd | 29 | 583 | both_nonempty | -0.2945 | -4.3258 | 4.7434 | 0.0345 | 0.1036 | -0.4483 | tta4 |
| tta4 | boundary_f1 | 29 | 583 | all_cases | 0.0291 | 0.0100 | 0.0476 | 0.0056 | 0.0280 | 0.5862 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0134 | -0.0306 | 0.0495 | 0.0920 | 0.1869 | 0.3609 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0278 | -0.0088 | 0.0605 | 0.0431 | 0.1725 | 0.4299 | tta8 |
| tta8 | precision | 29 | 575 | all_cases | 0.0453 | 0.0057 | 0.0795 | 0.0002 | 0.0020 | 0.7425 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | -0.0226 | -0.0548 | 0.0030 | 0.3042 | 0.3042 | -0.2230 | baseline |
| tta8 | hd | 29 | 575 | both_nonempty | -13.5920 | -23.1338 | -2.9409 | 0.0006 | 0.0044 | -0.7011 | tta8 |
| tta8 | hd95 | 29 | 575 | both_nonempty | -11.0853 | -20.3670 | -0.6836 | 0.0020 | 0.0138 | -0.6368 | tta8 |
| tta8 | hd95_pooled | 29 | 575 | both_nonempty | -9.3859 | -18.7594 | 1.2357 | 0.0060 | 0.0336 | -0.5724 | tta8 |
| tta8 | assd | 29 | 575 | both_nonempty | -0.3697 | -5.1321 | 5.4280 | 0.0623 | 0.1869 | -0.3977 | tta8 |
| tta8 | boundary_f1 | 29 | 575 | all_cases | 0.0352 | 0.0114 | 0.0574 | 0.0056 | 0.0336 | 0.5862 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 19 | 0.0310 | 0.0127 | 0.0551 |  |  |  |  |
| tta2 | 612 | 19 | 0.0310 | 0.0128 | 0.0543 | 6.0000 | 6.0000 | 1.0000 | 1.0000 |
| tta4 | 612 | 24 | 0.0392 | 0.0153 | 0.0702 | 10.0000 | 5.0000 | 0.3018 | 0.6035 |
| tta8 | 612 | 33 | 0.0539 | 0.0229 | 0.0936 | 18.0000 | 4.0000 | 0.0043 | 0.0130 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 151.2959 | 171.7942 | 0.5637 | 0.2255 |  |  |  |  |  |  |  |  |  |
| tta2 | 612 | 145.3075 | 165.5057 | 0.5114 | 0.2190 | -6.2885 | -21.3088 | 0.7902 | -0.0523 | -0.0764 | -0.0289 | -0.0065 | -0.0458 | 0.0333 |
| tta4 | 612 | 142.2114 | 166.7898 | 0.4510 | 0.2026 | -5.0045 | -25.9540 | 7.0164 | -0.1127 | -0.1552 | -0.0724 | -0.0229 | -0.0669 | 0.0188 |
| tta8 | 612 | 139.5467 | 171.7893 | 0.4167 | 0.1993 | -0.0049 | -26.4466 | 14.9274 | -0.1471 | -0.2048 | -0.0928 | -0.0261 | -0.0748 | 0.0186 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -4.7114 | -13.2084 | 6.3815 | 0.0092 | 0.0369 | tta2 |
| tta2 | hd95 | 29 | 612 | -3.6514 | -12.2629 | 7.3814 | 0.0291 | 0.0872 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | -2.6221 | -11.2460 | 8.4027 | 0.0431 | 0.0872 | tta2 |
| tta2 | assd | 29 | 612 | 0.8960 | -7.6967 | 12.3387 | 0.1155 | 0.1155 | baseline |
| tta4 | hd | 29 | 612 | -4.1807 | -18.3409 | 14.9180 | 0.0052 | 0.0207 | tta4 |
| tta4 | hd95 | 29 | 612 | -2.1476 | -16.3187 | 16.6038 | 0.0106 | 0.0317 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -1.2679 | -15.4549 | 17.5476 | 0.0179 | 0.0358 | tta4 |
| tta4 | assd | 29 | 612 | 6.9183 | -6.5283 | 25.9345 | 0.0837 | 0.0837 | baseline |
| tta8 | hd | 29 | 612 | -0.5705 | -20.1939 | 24.6631 | 0.0291 | 0.1163 | tta8 |
| tta8 | hd95 | 29 | 612 | 2.1000 | -17.4946 | 27.3705 | 0.0592 | 0.1776 | baseline |
| tta8 | hd95_pooled | 29 | 612 | 3.7689 | -15.8917 | 29.1792 | 0.1104 | 0.2208 | baseline |
| tta8 | assd | 29 | 612 | 14.9358 | -4.3624 | 40.4982 | 0.2383 | 0.2383 | baseline |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 124 | 0.5848 | 0.0000 | 0.3142 | 0.0000 | 0.4837 |
| -1 | entropy_mean_roi | 612 | 124 | 0.4521 | 0.0000 | 0.7236 | 0.0000 | 0.8201 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 1 | 118 | 6 | 0.6541 |  | 0.5899 |  |
| 1 | dice | 1 | 137 | 6 | 0.5459 |  | 0.5419 |  |
| 2 | dice | 1 | 126 | 6 | 0.7082 |  | 0.7075 |  |
| 3 | dice | 1 | 129 | 6 | 0.6919 |  | 0.6673 |  |
| 4 | dice | 1 | 102 | 5 | 0.7810 |  | 0.7405 |  |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

_(no rows)_

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 24 | 4 | 24 | 6303.5000 | 0.3720 | 0.6880 |
| failures_all | 124 | 24 | 4 | 24 | 2465.5000 | 0.5109 | 0.6880 |
| failures_non_missed | 100 | 0 | 4 | 0 | 3929.5000 | 0.5109 | 0.6880 |
| failures_missed | 24 | 24 | 0 | 24 | 0.0000 |  |  |
| failures_empty_0px | 24 | 24 | 24 | 24 | 0.0000 |  |  |
| failures_tiny_1_10px | 2 | 0 | 2 | 0 | 4.0000 | 0.6834 |  |
| failures_small_11_100px | 2 | 0 | 2 | 0 | 65.0000 | 0.6905 |  |
| failures_medium_101_1000px | 22 | 0 | 0 | 0 | 452.5000 | 0.6895 |  |
| failures_large_gt_1000px | 74 | 0 | 0 | 0 | 4564.0000 | 0.4622 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8201 | 0.7150 | 0.9325 | 0.1487 | 0.1089 | 0.2092 | 0.1140 | 588 | 588 | 100 | 24 | 0.6614 | 0.8549 | 0.1487 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8201 | 0.7150 | 0.9325 | 0.1487 | 0.1089 | 0.2092 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.1487 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.1142 | 0.0908 | 0.1484 | 0.0910 | 488 | 488 | 0 | 24 |  |  | 0.1142 |
| tta4_variance_roi | own | tta4 | all_failures | 0.7975 | 0.6707 | 0.9108 | 0.1674 | 0.1201 | 0.2288 | 0.1140 | 588 | 588 | 100 | 24 | 0.6431 | 0.8367 | 0.1674 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.7975 | 0.6707 | 0.9108 | 0.1674 | 0.1201 | 0.2288 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.1674 |
| tta4_variance_roi | own | tta4 | missed_vs_clean |  |  |  | 0.1298 | 0.0969 | 0.1766 | 0.0910 | 488 | 488 | 0 | 24 |  |  | 0.1298 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.4837 | 0.3666 | 0.6059 | 0.3124 | 0.2097 | 0.4257 | 0.1209 | 612 | 612 | 124 | 0 | 0.4837 | 0.4837 | 0.3124 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.5624 | 0.4479 | 0.6890 | 0.2101 | 0.1476 | 0.2948 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.2101 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean | 0.1558 | 0.0508 | 0.3279 | 0.2595 | 0.1635 | 0.3699 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.2595 |
| tta4_entropy_image | own | tta4 | all_failures | 0.4823 | 0.3257 | 0.6348 | 0.2764 | 0.1813 | 0.4134 | 0.1209 | 612 | 612 | 124 | 0 | 0.4823 | 0.4823 | 0.2764 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.5027 | 0.3373 | 0.6712 | 0.2246 | 0.1491 | 0.3401 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.2246 |
| tta4_entropy_image | own | tta4 | missed_vs_clean | 0.3974 | 0.1400 | 0.6250 | 0.2021 | 0.1362 | 0.3033 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.2021 |
| map_entropy_roi_tta4 | own | tta4 | all_failures | 0.8201 | 0.7146 | 0.9326 | 0.1487 | 0.1089 | 0.2093 | 0.1140 | 588 | 588 | 100 | 24 | 0.6614 | 0.8549 | 0.1487 |
| map_entropy_roi_tta4 | own | tta4 | non_missed_failures | 0.8201 | 0.7146 | 0.9326 | 0.1487 | 0.1089 | 0.2093 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.1487 |
| map_entropy_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  | 0.1142 | 0.0908 | 0.1483 | 0.0910 | 488 | 488 | 0 | 24 |  |  | 0.1142 |
| map_variance_roi_tta4 | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| map_variance_roi_tta4 | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| map_variance_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |
| map_ambiguous_fraction_tta4 | own | tta4 | all_failures | 0.4829 | 0.3291 | 0.6265 | 0.2969 | 0.1942 | 0.4243 | 0.1209 | 612 | 612 | 124 | 0 | 0.4829 | 0.4829 | 0.2969 |
| map_ambiguous_fraction_tta4 | own | tta4 | non_missed_failures | 0.5162 | 0.3594 | 0.6681 | 0.2233 | 0.1504 | 0.3296 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.2233 |
| map_ambiguous_fraction_tta4 | own | tta4 | missed_vs_clean | 0.3441 | 0.0575 | 0.6211 | 0.2297 | 0.1494 | 0.3365 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.2297 |
| pred_area_px | own | tta4 | all_failures | 0.2021 | 0.1115 | 0.3144 | 0.4749 | 0.3650 | 0.5837 | 0.1209 | 612 | 612 | 124 | 0 | 0.2021 | 0.2021 | 0.4749 |
| pred_area_px | own | tta4 | non_missed_failures | 0.2506 | 0.1512 | 0.3744 | 0.3964 | 0.2996 | 0.4945 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.3964 |
| pred_area_px | own | tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.3470 | 0.2503 | 0.4497 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.3470 |
| one_minus_confidence | own | tta4 | all_failures | 0.8183 | 0.7161 | 0.9354 | 0.1473 | 0.1068 | 0.2073 | 0.1140 | 588 | 588 | 100 | 24 | 0.6599 | 0.8535 | 0.1473 |
| one_minus_confidence | own | tta4 | non_missed_failures | 0.8183 | 0.7161 | 0.9354 | 0.1473 | 0.1068 | 0.2073 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.1473 |
| one_minus_confidence | own | tta4 | missed_vs_clean |  |  |  | 0.1122 | 0.0902 | 0.1470 | 0.0910 | 488 | 488 | 0 | 24 |  |  | 0.1122 |
| baseline_entropy_roi | own | baseline | all_failures | 0.7668 | 0.6445 | 0.8775 | 0.1910 | 0.1384 | 0.2665 | 0.1366 | 593 | 593 | 119 | 19 | 0.6612 | 0.7989 | 0.1910 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.7668 | 0.6445 | 0.8775 | 0.1910 | 0.1384 | 0.2665 | 0.1366 | 593 | 593 | 119 | 0 |  |  | 0.1910 |
| baseline_entropy_roi | own | baseline | missed_vs_clean |  |  |  | 0.1444 | 0.1122 | 0.1860 | 0.1073 | 474 | 474 | 0 | 19 |  |  | 0.1444 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.7076 | 0.5676 | 0.8435 | 0.1780 | 0.1251 | 0.2579 | 0.1160 | 593 | 593 | 107 | 19 | 0.6119 | 0.7449 | 0.1780 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.6825 | 0.5456 | 0.8215 | 0.1755 | 0.1233 | 0.2569 | 0.1132 | 583 | 583 | 97 | 5 |  |  | 0.1755 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9506 | 0.8883 | 1.0000 | 0.1278 | 0.1004 | 0.1671 | 0.0929 | 496 | 496 | 10 | 16 |  |  | 0.1278 |
| baseline_entropy_image | own | baseline | all_failures | 0.5459 | 0.3961 | 0.6810 | 0.2800 | 0.2014 | 0.3994 | 0.1423 | 612 | 612 | 138 | 0 | 0.5459 | 0.5459 | 0.2800 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.5595 | 0.3970 | 0.7059 | 0.2497 | 0.1724 | 0.3589 | 0.1366 | 593 | 593 | 119 | 0 |  |  | 0.2497 |
| baseline_entropy_image | own | baseline | missed_vs_clean | 0.4607 | 0.1979 | 0.6663 | 0.2025 | 0.1506 | 0.2795 | 0.1119 | 493 | 493 | 19 | 0 |  |  | 0.2025 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.4853 | 0.3274 | 0.6393 | 0.2645 | 0.1755 | 0.3976 | 0.1209 | 612 | 612 | 124 | 0 | 0.4853 | 0.4853 | 0.2645 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.4945 | 0.3203 | 0.6662 | 0.2276 | 0.1503 | 0.3436 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.2276 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean | 0.4471 | 0.1843 | 0.6789 | 0.1866 | 0.1306 | 0.2763 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.1866 |
| ensemble3_entropy_roi | own | tta4 | all_failures | 0.7076 | 0.5677 | 0.8436 | 0.1780 | 0.1251 | 0.2578 | 0.1160 | 593 | 593 | 107 | 19 | 0.6119 | 0.7449 | 0.1780 |
| ensemble3_entropy_roi | own | tta4 | non_missed_failures | 0.6826 | 0.5457 | 0.8215 | 0.1754 | 0.1233 | 0.2568 | 0.1132 | 583 | 583 | 97 | 5 |  |  | 0.1754 |
| ensemble3_entropy_roi | own | tta4 | missed_vs_clean | 0.9506 | 0.8883 | 1.0000 | 0.1278 | 0.1004 | 0.1671 | 0.0929 | 496 | 496 | 10 | 16 |  |  | 0.1278 |
| ensemble3_variance_roi | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_variance_roi | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| ensemble3_variance_roi | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |
| ensemble3_entropy_image | own | tta4 | all_failures | 0.4853 | 0.3274 | 0.6393 | 0.2645 | 0.1755 | 0.3976 | 0.1209 | 612 | 612 | 124 | 0 | 0.4853 | 0.4853 | 0.2645 |
| ensemble3_entropy_image | own | tta4 | non_missed_failures | 0.4945 | 0.3203 | 0.6662 | 0.2276 | 0.1503 | 0.3436 | 0.1140 | 588 | 588 | 100 | 0 |  |  | 0.2276 |
| ensemble3_entropy_image | own | tta4 | missed_vs_clean | 0.4471 | 0.1843 | 0.6789 | 0.1866 | 0.1306 | 0.2763 | 0.0964 | 512 | 512 | 24 | 0 |  |  | 0.1866 |
| ensemble3_variance_image | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_variance_image | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| ensemble3_variance_image | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 588 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 512 |  |  |  |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8201 | 0.7150 | 0.9325 | 0.1487 | 0.1089 | 0.2092 | 0.1140 | 588 | 588 | 100 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8201 | 0.7150 | 0.9325 | 0.1487 | 0.1089 | 0.2092 | 0.1140 | 588 | 588 | 100 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.1142 | 0.0908 | 0.1484 | 0.0910 | 488 | 488 | 0 |
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
| tta4_entropy_roi | own | tta4 | hd95 | 588 | 0.4521 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 588 | 0.7236 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 588 | 0.5115 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 588 | 0.6355 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 588 | 0.5848 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.3142 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 588 | 0.1826 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.2369 | 0.0000 |
| map_entropy_roi_tta4 | own | tta4 | hd95 | 588 | 0.4517 | 0.0000 |
| map_entropy_roi_tta4 | own | tta4 | one_minus_dice | 588 | 0.7235 | 0.0000 |
| map_variance_roi_tta4 | own | tta4 | hd95 | 0 |  |  |
| map_variance_roi_tta4 | own | tta4 | one_minus_dice | 0 |  |  |
| map_ambiguous_fraction_tta4 | own | tta4 | hd95 | 588 | 0.2572 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | tta4 | one_minus_dice | 612 | 0.2173 | 0.0000 |
| pred_area_px | own | tta4 | hd95 | 588 | 0.1127 | 0.0062 |
| pred_area_px | own | tta4 | one_minus_dice | 612 | -0.3245 | 0.0000 |
| one_minus_confidence | own | tta4 | hd95 | 588 | 0.4932 | 0.0000 |
| one_minus_confidence | own | tta4 | one_minus_dice | 588 | 0.7445 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 593 | 0.2825 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 593 | 0.5880 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 583 | 0.2675 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 593 | 0.5520 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 593 | 0.1255 | 0.0022 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.2240 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 588 | 0.1309 | 0.0015 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.2225 | 0.0000 |
| ensemble3_entropy_roi | own | tta4 | hd95 | 583 | 0.2676 | 0.0000 |
| ensemble3_entropy_roi | own | tta4 | one_minus_dice | 593 | 0.5522 | 0.0000 |
| ensemble3_variance_roi | own | tta4 | hd95 | 0 |  |  |
| ensemble3_variance_roi | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_entropy_image | own | tta4 | hd95 | 588 | 0.1309 | 0.0015 |
| ensemble3_entropy_image | own | tta4 | one_minus_dice | 612 | 0.2225 | 0.0000 |
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
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.0226 | -0.0946 | 0.1573 | 0.7020 | -0.0187 | -0.0563 | 0.0134 | 0.2800 | 588 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.2577 | 0.1509 | 0.3889 | 0.0000 | -0.0615 | -0.1091 | -0.0301 | 0.0000 | 588 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.3174 | 0.1716 | 0.4690 | 0.0000 | -0.0759 | -0.1410 | -0.0337 | 0.0000 | 588 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0000 | -0.0003 | 0.0003 | 1.0000 | -0.0000 | -0.0001 | 0.0000 | 0.2540 | 588 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | 0.0000 | -0.0003 | 0.0003 | 1.0000 | -0.0000 | -0.0001 | 0.0000 | 0.2540 | 588 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.3039 | 0.1808 | 0.4349 | 0.0000 | -0.0746 | -0.1314 | -0.0386 | 0.0000 | 588 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.3039 | 0.1808 | 0.4349 | 0.0000 | -0.0746 | -0.1314 | -0.0386 | 0.0000 | 588 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.5695 | 0.4062 | 0.7596 | 0.0000 | -0.2477 | -0.3166 | -0.1736 | 0.0000 | 588 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.5695 | 0.4062 | 0.7596 | 0.0000 | -0.2477 | -0.3166 | -0.1736 | 0.0000 | 588 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0018 | -0.0108 | 0.0157 | 0.8440 | 0.0014 | -0.0015 | 0.0044 | 0.3700 | 588 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0018 | -0.0108 | 0.0157 | 0.8440 | 0.0014 | -0.0015 | 0.0044 | 0.3700 | 588 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.1339 | 0.0492 | 0.2271 | 0.0000 | -0.0277 | -0.0522 | -0.0116 | 0.0000 | 583 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.0449 | -0.0161 | 0.1158 | 0.1560 | -0.0083 | -0.0247 | 0.0031 | 0.2260 | 583 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.3256 | 0.1821 | 0.4766 | 0.0000 | -0.0789 | -0.1461 | -0.0365 | 0.0000 | 588 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.2246 | 0.1005 | 0.3560 | 0.0000 | -0.0571 | -0.1067 | -0.0254 | 0.0000 | 583 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.1339 | 0.0492 | 0.2270 | 0.0000 | -0.0277 | -0.0522 | -0.0116 | 0.0000 | 583 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.1339 | 0.0492 | 0.2270 | 0.0000 | -0.0277 | -0.0522 | -0.0116 | 0.0000 | 583 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.3256 | 0.1821 | 0.4766 | 0.0000 | -0.0789 | -0.1461 | -0.0365 | 0.0000 | 588 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.3256 | 0.1821 | 0.4766 | 0.0000 | -0.0789 | -0.1461 | -0.0365 | 0.0000 | 588 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 124, baseline 138, ensemble3 0, ensemble3_tta4 0

## Audit note

tta4 shares 124 failures; 24 of them have no ROI value (empty prediction), 28 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice unavailable for 612 of 612 cases: the runner does not store per-view masks. Re-run tta4 on CPU with configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml so the `view_pairwise_dice` metric exists, then pass it as `view_agreement`.

## Audit note

seed agreement: 0 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 1.97e-04
