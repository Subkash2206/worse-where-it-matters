# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

5 runs: folds [0, 1, 2, 3, 4] x seeds [0]; config `configs/cvc_clinicdb/unet_resnet50_data050.yaml`; pairing unit `group`; bootstrap draws `config`; Holm correction; failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 1 | 17.6339 |  |
| tta2 | assd | 612 | 29 | 1 | 15.6948 |  |
| tta4 | assd | 612 | 29 | 1 | 14.9449 |  |
| tta8 | assd | 612 | 29 | 1 | 13.9568 |  |
| baseline | boundary_f1 | 612 | 29 | 1 | 0.3919 |  |
| tta2 | boundary_f1 | 612 | 29 | 1 | 0.4151 |  |
| tta4 | boundary_f1 | 612 | 29 | 1 | 0.4333 |  |
| tta8 | boundary_f1 | 612 | 29 | 1 | 0.4433 |  |
| baseline | dice | 612 | 29 | 1 | 0.7261 |  |
| tta2 | dice | 612 | 29 | 1 | 0.7357 |  |
| tta4 | dice | 612 | 29 | 1 | 0.7518 |  |
| tta8 | dice | 612 | 29 | 1 | 0.7609 |  |
| baseline | hd95 | 612 | 29 | 1 | 59.9762 |  |
| tta2 | hd95 | 612 | 29 | 1 | 51.7478 |  |
| tta4 | hd95 | 612 | 29 | 1 | 51.7508 |  |
| tta8 | hd95 | 612 | 29 | 1 | 47.8803 |  |
| baseline | hd95_pooled | 612 | 29 | 1 | 56.7292 |  |
| tta2 | hd95_pooled | 612 | 29 | 1 | 49.1771 |  |
| tta4 | hd95_pooled | 612 | 29 | 1 | 48.9299 |  |
| tta8 | hd95_pooled | 612 | 29 | 1 | 45.1846 |  |

Mean over all frames of the fold, pooled over the 5 folds (612 cases), then mean ± std across the three seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 1 | 20.1771 |  |
| tta2 | assd | 612 | 29 | 1 | 20.2154 |  |
| tta4 | assd | 612 | 29 | 1 | 18.9665 |  |
| tta8 | assd | 612 | 29 | 1 | 17.6860 |  |
| baseline | boundary_f1 | 612 | 29 | 1 | 0.3711 |  |
| tta2 | boundary_f1 | 612 | 29 | 1 | 0.3878 |  |
| tta4 | boundary_f1 | 612 | 29 | 1 | 0.4061 |  |
| tta8 | boundary_f1 | 612 | 29 | 1 | 0.4207 |  |
| baseline | dice | 612 | 29 | 1 | 0.7011 |  |
| tta2 | dice | 612 | 29 | 1 | 0.7104 |  |
| tta4 | dice | 612 | 29 | 1 | 0.7272 |  |
| tta8 | dice | 612 | 29 | 1 | 0.7345 |  |
| baseline | hd95 | 612 | 29 | 1 | 63.3106 |  |
| tta2 | hd95 | 612 | 29 | 1 | 58.6245 |  |
| tta4 | hd95 | 612 | 29 | 1 | 57.8061 |  |
| tta8 | hd95 | 612 | 29 | 1 | 53.0558 |  |
| baseline | hd95_pooled | 612 | 29 | 1 | 60.0803 |  |
| tta2 | hd95_pooled | 612 | 29 | 1 | 56.1050 |  |
| tta4 | hd95_pooled | 612 | 29 | 1 | 55.0182 |  |
| tta8 | hd95_pooled | 612 | 29 | 1 | 50.3968 |  |

Each video contributes one value (mean over its frames), 29 videos in total; mean ± std across the three seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0093 | -0.0023 | 0.0191 | 0.0086 | 0.0602 | 0.5494 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0152 | 0.0041 | 0.0258 | 0.0032 | 0.0292 | 0.6092 | tta2 |
| tta2 | precision | 29 | 581 | all_cases | 0.0163 | -0.0299 | 0.0528 | 0.0035 | 0.0292 | 0.6046 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | -0.0014 | -0.0141 | 0.0082 | 0.1104 | 0.3313 | 0.3425 | baseline |
| tta2 | hd | 29 | 581 | both_nonempty | -5.7109 | -14.0954 | 1.6149 | 0.0365 | 0.2192 | -0.4437 | tta2 |
| tta2 | hd95 | 29 | 581 | both_nonempty | -4.6860 | -12.4732 | 2.3296 | 0.0798 | 0.3190 | -0.3747 | tta2 |
| tta2 | hd95_pooled | 29 | 581 | both_nonempty | -3.9754 | -11.4899 | 2.8426 | 0.1155 | 0.3313 | -0.3379 | tta2 |
| tta2 | assd | 29 | 581 | both_nonempty | 0.0383 | -2.8379 | 3.6836 | 0.1557 | 0.3313 | -0.3057 | baseline |
| tta2 | boundary_f1 | 29 | 581 | all_cases | 0.0167 | 0.0028 | 0.0311 | 0.0506 | 0.2532 | 0.4161 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0261 | 0.0141 | 0.0390 | 0.0002 | 0.0014 | 0.7517 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0320 | 0.0187 | 0.0462 | 0.0001 | 0.0006 | 0.7885 | tta4 |
| tta4 | precision | 29 | 584 | all_cases | 0.0200 | -0.0274 | 0.0589 | 0.0075 | 0.0373 | 0.5586 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0193 | 0.0036 | 0.0353 | 0.0048 | 0.0287 | 0.5862 | tta4 |
| tta4 | hd | 29 | 584 | both_nonempty | -6.6340 | -14.4018 | 0.2408 | 0.0345 | 0.1036 | -0.4483 | tta4 |
| tta4 | hd95 | 29 | 584 | both_nonempty | -5.5045 | -12.6076 | 0.8915 | 0.0689 | 0.1310 | -0.3885 | tta4 |
| tta4 | hd95_pooled | 29 | 584 | both_nonempty | -5.0622 | -11.8277 | 1.1146 | 0.0655 | 0.1310 | -0.3931 | tta4 |
| tta4 | assd | 29 | 584 | both_nonempty | -1.2106 | -3.9862 | 2.3385 | 0.0075 | 0.0373 | -0.5586 | tta4 |
| tta4 | boundary_f1 | 29 | 584 | all_cases | 0.0349 | 0.0171 | 0.0528 | 0.0013 | 0.0090 | 0.6598 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0334 | 0.0191 | 0.0495 | 0.0000 | 0.0002 | 0.8299 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0404 | 0.0245 | 0.0580 | 0.0000 | 0.0001 | 0.8667 | tta8 |
| tta8 | precision | 29 | 586 | all_cases | 0.0226 | -0.0239 | 0.0595 | 0.0012 | 0.0053 | 0.6644 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | 0.0218 | 0.0050 | 0.0413 | 0.0326 | 0.0326 | 0.4529 | tta8 |
| tta8 | hd | 29 | 586 | both_nonempty | -11.6603 | -20.5444 | -3.2611 | 0.0011 | 0.0053 | -0.6690 | tta8 |
| tta8 | hd95 | 29 | 586 | both_nonempty | -10.2547 | -18.5700 | -2.1706 | 0.0020 | 0.0059 | -0.6368 | tta8 |
| tta8 | hd95_pooled | 29 | 586 | both_nonempty | -9.6836 | -17.8564 | -1.7569 | 0.0022 | 0.0059 | -0.6322 | tta8 |
| tta8 | assd | 29 | 586 | both_nonempty | -2.4911 | -6.2503 | 2.2580 | 0.0005 | 0.0027 | -0.7103 | tta8 |
| tta8 | boundary_f1 | 29 | 586 | all_cases | 0.0496 | 0.0283 | 0.0727 | 0.0001 | 0.0008 | 0.7701 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 20 | 0.0327 | 0.0083 | 0.0651 |  |  |  |  |
| tta2 | 612 | 28 | 0.0458 | 0.0096 | 0.1039 | 11.0000 | 3.0000 | 0.0574 | 0.1721 |
| tta4 | 612 | 21 | 0.0343 | 0.0047 | 0.0802 | 8.0000 | 7.0000 | 1.0000 | 1.0000 |
| tta8 | 612 | 18 | 0.0294 | 0.0092 | 0.0564 | 6.0000 | 8.0000 | 0.7905 | 1.0000 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 149.8201 | 168.6181 | 0.4444 | 0.1650 |  |  |  |  |  |  |  |  |  |
| tta2 | 612 | 146.1135 | 166.1192 | 0.3889 | 0.1650 | -2.4989 | -13.6164 | 4.6988 | -0.0556 | -0.1127 | -0.0119 | 0.0000 | -0.0164 | 0.0160 |
| tta4 | 612 | 137.6604 | 162.7103 | 0.3922 | 0.1389 | -5.9078 | -20.4941 | 1.0436 | -0.0523 | -0.1047 | -0.0077 | -0.0261 | -0.0452 | -0.0095 |
| tta8 | 612 | 132.7574 | 165.0298 | 0.3611 | 0.1356 | -3.5883 | -26.5084 | 5.9017 | -0.0833 | -0.1403 | -0.0309 | -0.0294 | -0.0570 | -0.0035 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -4.5807 | -13.4771 | 6.7901 | 0.0190 | 0.0762 | tta2 |
| tta2 | hd95 | 29 | 612 | -3.5402 | -12.1096 | 7.7354 | 0.0431 | 0.1294 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | -2.8376 | -11.3127 | 8.4202 | 0.0592 | 0.1294 | tta2 |
| tta2 | assd | 29 | 612 | 2.2311 | -5.7553 | 13.9437 | 0.3145 | 0.3145 | baseline |
| tta4 | hd | 29 | 612 | -6.8889 | -16.0070 | 3.2486 | 0.0157 | 0.0472 | tta4 |
| tta4 | hd95 | 29 | 612 | -5.8182 | -14.8056 | 4.0824 | 0.0229 | 0.0472 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -5.4028 | -14.2884 | 4.4658 | 0.0179 | 0.0472 | tta4 |
| tta4 | assd | 29 | 612 | -1.0394 | -9.5863 | 8.6326 | 0.0106 | 0.0423 | tta4 |
| tta8 | hd | 29 | 612 | -13.1899 | -20.1478 | -6.6558 | 0.0004 | 0.0016 | tta8 |
| tta8 | hd95 | 29 | 612 | -11.8458 | -18.5365 | -5.3615 | 0.0010 | 0.0029 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | -11.2899 | -17.9069 | -4.9932 | 0.0012 | 0.0029 | tta8 |
| tta8 | assd | 29 | 612 | -4.2933 | -9.0034 | 0.0283 | 0.0030 | 0.0030 | tta8 |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 85 | 0.6053 | 0.0000 | 0.4709 | 0.0000 | 0.5565 |
| -1 | entropy_mean_roi | 612 | 85 | 0.6091 | 0.0000 | 0.7712 | 0.0000 | 0.8823 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 1 | 118 | 6 | 0.6904 |  | 0.6065 |  |
| 1 | dice | 1 | 137 | 6 | 0.6636 |  | 0.6704 |  |
| 2 | dice | 1 | 126 | 6 | 0.7497 |  | 0.7509 |  |
| 3 | dice | 1 | 129 | 6 | 0.7133 |  | 0.6852 |  |
| 4 | dice | 1 | 102 | 5 | 0.8382 |  | 0.8108 |  |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

_(no rows)_

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 21 | 4 | 21 | 6573.0000 | 0.3421 | 0.6923 |
| failures_all | 85 | 21 | 4 | 21 | 1292.0000 | 0.6012 | 0.6923 |
| failures_non_missed | 64 | 0 | 4 | 0 | 3272.5000 | 0.6012 | 0.6923 |
| failures_missed | 21 | 21 | 0 | 21 | 0.0000 |  |  |
| failures_empty_0px | 21 | 21 | 21 | 21 | 0.0000 |  |  |
| failures_small_11_100px | 4 | 0 | 4 | 0 | 58.5000 | 0.6923 |  |
| failures_medium_101_1000px | 15 | 0 | 0 | 0 | 452.0000 | 0.6872 |  |
| failures_large_gt_1000px | 45 | 0 | 0 | 0 | 4435.0000 | 0.4794 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8823 | 0.7966 | 0.9604 | 0.1091 | 0.0817 | 0.1512 | 0.0856 | 591 | 591 | 64 | 21 | 0.6643 | 0.9114 | 0.1091 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8823 | 0.7966 | 0.9604 | 0.1091 | 0.0817 | 0.1512 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1091 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0918 | 0.0703 | 0.1244 | 0.0732 | 527 | 527 | 0 | 21 |  |  | 0.0918 |
| tta4_variance_roi | own | tta4 | all_failures | 0.8591 | 0.7438 | 0.9520 | 0.1164 | 0.0882 | 0.1580 | 0.0856 | 591 | 591 | 64 | 21 | 0.6468 | 0.8939 | 0.1164 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.8591 | 0.7438 | 0.9520 | 0.1164 | 0.0882 | 0.1580 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1164 |
| tta4_variance_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0980 | 0.0762 | 0.1282 | 0.0732 | 527 | 527 | 0 | 21 |  |  | 0.0980 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.5565 | 0.4021 | 0.6732 | 0.2174 | 0.1434 | 0.3167 | 0.0907 | 612 | 612 | 85 | 0 | 0.5565 | 0.5565 | 0.2174 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.6109 | 0.4639 | 0.7431 | 0.1488 | 0.1058 | 0.2112 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1488 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean | 0.3910 | 0.0544 | 0.6195 | 0.1835 | 0.1137 | 0.2750 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.1835 |
| tta4_entropy_image | own | tta4 | all_failures | 0.5217 | 0.3830 | 0.6622 | 0.1937 | 0.1251 | 0.3003 | 0.0907 | 612 | 612 | 85 | 0 | 0.5217 | 0.5217 | 0.1937 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.5654 | 0.4242 | 0.6843 | 0.1644 | 0.1140 | 0.2375 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1644 |
| tta4_entropy_image | own | tta4 | missed_vs_clean | 0.3883 | 0.2276 | 0.6006 | 0.1510 | 0.1009 | 0.2381 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.1510 |
| map_entropy_roi_tta4 | own | tta4 | all_failures | 0.8824 | 0.7967 | 0.9604 | 0.1091 | 0.0817 | 0.1513 | 0.0856 | 591 | 591 | 64 | 21 | 0.6644 | 0.9114 | 0.1091 |
| map_entropy_roi_tta4 | own | tta4 | non_missed_failures | 0.8824 | 0.7967 | 0.9604 | 0.1091 | 0.0817 | 0.1513 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1091 |
| map_entropy_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  | 0.0918 | 0.0703 | 0.1244 | 0.0732 | 527 | 527 | 0 | 21 |  |  | 0.0918 |
| map_variance_roi_tta4 | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| map_variance_roi_tta4 | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| map_variance_roi_tta4 | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |
| map_ambiguous_fraction_tta4 | own | tta4 | all_failures | 0.4993 | 0.3728 | 0.6251 | 0.2265 | 0.1427 | 0.3407 | 0.0907 | 612 | 612 | 85 | 0 | 0.4993 | 0.4993 | 0.2265 |
| map_ambiguous_fraction_tta4 | own | tta4 | non_missed_failures | 0.5644 | 0.4376 | 0.6903 | 0.1557 | 0.1086 | 0.2280 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1557 |
| map_ambiguous_fraction_tta4 | own | tta4 | missed_vs_clean | 0.3008 | 0.0495 | 0.4994 | 0.1859 | 0.1140 | 0.2864 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.1859 |
| pred_area_px | own | tta4 | all_failures | 0.1680 | 0.0786 | 0.2916 | 0.3977 | 0.2610 | 0.5375 | 0.0907 | 612 | 612 | 85 | 0 | 0.1680 | 0.1680 | 0.3977 |
| pred_area_px | own | tta4 | non_missed_failures | 0.2231 | 0.1219 | 0.3450 | 0.3157 | 0.2226 | 0.4146 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.3157 |
| pred_area_px | own | tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.2934 | 0.1819 | 0.4205 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.2934 |
| one_minus_confidence | own | tta4 | all_failures | 0.8820 | 0.7916 | 0.9621 | 0.1076 | 0.0809 | 0.1486 | 0.0856 | 591 | 591 | 64 | 21 | 0.6641 | 0.9111 | 0.1076 |
| one_minus_confidence | own | tta4 | non_missed_failures | 0.8820 | 0.7916 | 0.9621 | 0.1076 | 0.0809 | 0.1486 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1076 |
| one_minus_confidence | own | tta4 | missed_vs_clean |  |  |  | 0.0902 | 0.0696 | 0.1215 | 0.0732 | 527 | 527 | 0 | 21 |  |  | 0.0902 |
| baseline_entropy_roi | own | baseline | all_failures | 0.8830 | 0.8026 | 0.9577 | 0.1303 | 0.0984 | 0.1756 | 0.1012 | 592 | 592 | 81 | 20 | 0.7081 | 0.9061 | 0.1303 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.8830 | 0.8026 | 0.9577 | 0.1303 | 0.0984 | 0.1756 | 0.1012 | 592 | 592 | 81 | 0 |  |  | 0.1303 |
| baseline_entropy_roi | own | baseline | missed_vs_clean |  |  |  | 0.1088 | 0.0846 | 0.1459 | 0.0843 | 511 | 511 | 0 | 20 |  |  | 0.1088 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.8437 | 0.7273 | 0.9413 | 0.1192 | 0.0865 | 0.1663 | 0.0859 | 592 | 592 | 65 | 20 | 0.6452 | 0.8805 | 0.1192 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8224 | 0.7109 | 0.9281 | 0.1177 | 0.0853 | 0.1649 | 0.0841 | 584 | 584 | 57 | 7 |  |  | 0.1177 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9957 | 0.9879 | 1.0000 | 0.1015 | 0.0762 | 0.1372 | 0.0746 | 535 | 535 | 8 | 13 |  |  | 0.1015 |
| baseline_entropy_image | own | baseline | all_failures | 0.5393 | 0.4045 | 0.6730 | 0.2241 | 0.1515 | 0.3349 | 0.1065 | 612 | 612 | 101 | 0 | 0.5393 | 0.5393 | 0.2241 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.5662 | 0.4396 | 0.6839 | 0.1976 | 0.1403 | 0.2806 | 0.1012 | 592 | 592 | 81 | 0 |  |  | 0.1976 |
| baseline_entropy_image | own | baseline | missed_vs_clean | 0.4307 | 0.2237 | 0.7032 | 0.1713 | 0.1215 | 0.2599 | 0.0883 | 531 | 531 | 20 | 0 |  |  | 0.1713 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.5304 | 0.3854 | 0.6761 | 0.2001 | 0.1295 | 0.3128 | 0.0907 | 612 | 612 | 85 | 0 | 0.5304 | 0.5304 | 0.2001 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.5689 | 0.4263 | 0.6917 | 0.1714 | 0.1188 | 0.2499 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1714 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean | 0.4131 | 0.2328 | 0.7174 | 0.1568 | 0.1042 | 0.2465 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.1568 |
| ensemble3_entropy_roi | own | tta4 | all_failures | 0.8438 | 0.7274 | 0.9413 | 0.1192 | 0.0865 | 0.1663 | 0.0859 | 592 | 592 | 65 | 20 | 0.6452 | 0.8805 | 0.1192 |
| ensemble3_entropy_roi | own | tta4 | non_missed_failures | 0.8224 | 0.7110 | 0.9281 | 0.1177 | 0.0853 | 0.1649 | 0.0841 | 584 | 584 | 57 | 7 |  |  | 0.1177 |
| ensemble3_entropy_roi | own | tta4 | missed_vs_clean | 0.9957 | 0.9879 | 1.0000 | 0.1015 | 0.0762 | 0.1372 | 0.0746 | 535 | 535 | 8 | 13 |  |  | 0.1015 |
| ensemble3_variance_roi | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_variance_roi | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| ensemble3_variance_roi | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |
| ensemble3_entropy_image | own | tta4 | all_failures | 0.5304 | 0.3854 | 0.6761 | 0.2001 | 0.1295 | 0.3128 | 0.0907 | 612 | 612 | 85 | 0 | 0.5304 | 0.5304 | 0.2001 |
| ensemble3_entropy_image | own | tta4 | non_missed_failures | 0.5689 | 0.4263 | 0.6917 | 0.1714 | 0.1188 | 0.2499 | 0.0856 | 591 | 591 | 64 | 0 |  |  | 0.1714 |
| ensemble3_entropy_image | own | tta4 | missed_vs_clean | 0.4131 | 0.2328 | 0.7174 | 0.1568 | 0.1042 | 0.2465 | 0.0772 | 548 | 548 | 21 | 0 |  |  | 0.1568 |
| ensemble3_variance_image | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_variance_image | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| ensemble3_variance_image | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| ensemble3_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 612 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 | 591 |  |  |  |
| ensemble3_tta4_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 | 548 |  |  |  |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8823 | 0.7966 | 0.9604 | 0.1091 | 0.0817 | 0.1512 | 0.0856 | 591 | 591 | 64 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8823 | 0.7966 | 0.9604 | 0.1091 | 0.0817 | 0.1512 | 0.0856 | 591 | 591 | 64 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.0918 | 0.0703 | 0.1244 | 0.0732 | 527 | 527 | 0 |
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
| tta4_entropy_roi | own | tta4 | hd95 | 591 | 0.6091 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 591 | 0.7712 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 591 | 0.6197 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 591 | 0.7118 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 591 | 0.6053 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.4709 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 591 | 0.4174 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.3295 | 0.0000 |
| map_entropy_roi_tta4 | own | tta4 | hd95 | 591 | 0.6088 | 0.0000 |
| map_entropy_roi_tta4 | own | tta4 | one_minus_dice | 591 | 0.7711 | 0.0000 |
| map_variance_roi_tta4 | own | tta4 | hd95 | 0 |  |  |
| map_variance_roi_tta4 | own | tta4 | one_minus_dice | 0 |  |  |
| map_ambiguous_fraction_tta4 | own | tta4 | hd95 | 591 | 0.5932 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | tta4 | one_minus_dice | 612 | 0.4291 | 0.0000 |
| pred_area_px | own | tta4 | hd95 | 591 | 0.1715 | 0.0000 |
| pred_area_px | own | tta4 | one_minus_dice | 612 | -0.2142 | 0.0000 |
| one_minus_confidence | own | tta4 | hd95 | 591 | 0.6482 | 0.0000 |
| one_minus_confidence | own | tta4 | one_minus_dice | 591 | 0.7899 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 592 | 0.5456 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 592 | 0.7470 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 584 | 0.4976 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 592 | 0.6816 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 592 | 0.3006 | 0.0000 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.2369 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 591 | 0.3191 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.2632 | 0.0000 |
| ensemble3_entropy_roi | own | tta4 | hd95 | 584 | 0.4976 | 0.0000 |
| ensemble3_entropy_roi | own | tta4 | one_minus_dice | 592 | 0.6816 | 0.0000 |
| ensemble3_variance_roi | own | tta4 | hd95 | 0 |  |  |
| ensemble3_variance_roi | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_entropy_image | own | tta4 | hd95 | 591 | 0.3191 | 0.0000 |
| ensemble3_entropy_image | own | tta4 | one_minus_dice | 612 | 0.2632 | 0.0000 |
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
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.0232 | -0.0107 | 0.0642 | 0.1780 | -0.0073 | -0.0153 | 0.0006 | 0.0800 | 591 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.2714 | 0.1652 | 0.3648 | 0.0000 | -0.0397 | -0.0675 | -0.0189 | 0.0000 | 591 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.3168 | 0.1751 | 0.4943 | 0.0000 | -0.0553 | -0.0984 | -0.0263 | 0.0000 | 591 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | -0.0001 | -0.0004 | 0.0000 | 0.3580 | -0.0000 | -0.0000 | 0.0000 | 0.6260 | 591 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | -0.0001 | -0.0004 | 0.0000 | 0.3580 | -0.0000 | -0.0000 | 0.0000 | 0.6260 | 591 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.3179 | 0.1998 | 0.4271 | 0.0000 | -0.0466 | -0.0859 | -0.0215 | 0.0000 | 591 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.3179 | 0.1998 | 0.4271 | 0.0000 | -0.0466 | -0.0859 | -0.0215 | 0.0000 | 591 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.6592 | 0.4930 | 0.8218 | 0.0000 | -0.2066 | -0.2912 | -0.1268 | 0.0000 | 591 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.6592 | 0.4930 | 0.8218 | 0.0000 | -0.2066 | -0.2912 | -0.1268 | 0.0000 | 591 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0003 | -0.0099 | 0.0085 | 0.9720 | 0.0015 | -0.0003 | 0.0041 | 0.1300 | 591 |
| tta4_entropy_roi | one_minus_confidence | tta4 | 0.0003 | -0.0099 | 0.0085 | 0.9720 | 0.0015 | -0.0003 | 0.0041 | 0.1300 | 591 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.0454 | 0.0120 | 0.0766 | 0.0060 | -0.0099 | -0.0186 | -0.0033 | 0.0000 | 584 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.0086 | -0.0301 | 0.0380 | 0.5620 | 0.0009 | -0.0083 | 0.0118 | 0.8000 | 584 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.3133 | 0.1594 | 0.4970 | 0.0000 | -0.0623 | -0.1117 | -0.0306 | 0.0000 | 591 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.3006 | 0.1644 | 0.4638 | 0.0000 | -0.0585 | -0.1020 | -0.0283 | 0.0000 | 584 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0454 | 0.0120 | 0.0766 | 0.0060 | -0.0099 | -0.0186 | -0.0033 | 0.0000 | 584 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0454 | 0.0120 | 0.0766 | 0.0060 | -0.0099 | -0.0186 | -0.0033 | 0.0000 | 584 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.3133 | 0.1594 | 0.4970 | 0.0000 | -0.0623 | -0.1117 | -0.0306 | 0.0000 | 591 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.3133 | 0.1594 | 0.4970 | 0.0000 | -0.0623 | -0.1117 | -0.0306 | 0.0000 | 591 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 85, baseline 101, ensemble3 0, ensemble3_tta4 0

## Audit note

tta4 shares 85 failures; 21 of them have no ROI value (empty prediction), 25 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice unavailable for 612 of 612 cases: the runner does not store per-view masks. Re-run tta4 on CPU with configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml so the `view_pairwise_dice` metric exists, then pass it as `view_agreement`.

## Audit note

seed agreement: 0 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 1.11e-04
