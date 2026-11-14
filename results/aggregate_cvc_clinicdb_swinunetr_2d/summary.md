# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

15 runs: folds [0, 1, 2, 3, 4] x seeds [0, 1, 2]; config `configs/cvc_clinicdb/swinunetr_2d.yaml`; pairing unit `group`; bootstrap draws `config`; Holm correction; failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 31.9135 | 1.2244 |
| ensemble3 | assd | 612 | 29 | 0 | 29.3313 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 27.5591 |  |
| tta2 | assd | 612 | 29 | 3 | 30.5157 | 1.1410 |
| tta4 | assd | 612 | 29 | 3 | 29.4250 | 0.7342 |
| tta8 | assd | 612 | 29 | 3 | 28.7116 | 0.5746 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.1716 | 0.0107 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.1893 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.2055 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.1800 | 0.0112 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.1911 | 0.0095 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.1941 | 0.0113 |
| baseline | dice | 612 | 29 | 3 | 0.5471 | 0.0066 |
| ensemble3 | dice | 612 | 29 | 0 | 0.5720 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.5823 |  |
| tta2 | dice | 612 | 29 | 3 | 0.5600 | 0.0068 |
| tta4 | dice | 612 | 29 | 3 | 0.5654 | 0.0053 |
| tta8 | dice | 612 | 29 | 3 | 0.5680 | 0.0061 |
| baseline | hd95 | 612 | 29 | 3 | 92.8363 | 2.6294 |
| ensemble3 | hd95 | 612 | 29 | 0 | 85.5605 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 80.5637 |  |
| tta2 | hd95 | 612 | 29 | 3 | 89.8836 | 3.1373 |
| tta4 | hd95 | 612 | 29 | 3 | 86.0009 | 1.6499 |
| tta8 | hd95 | 612 | 29 | 3 | 83.8514 | 2.0962 |
| baseline | hd95_pooled | 612 | 29 | 3 | 87.7057 | 2.7414 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 81.0780 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 76.4413 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 85.2130 | 3.5038 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 81.3330 | 2.2186 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 79.3142 | 1.8847 |

Mean over all frames of the fold, pooled over the 5 folds (612 cases), then mean ± std across the three seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 31.8325 | 1.0250 |
| ensemble3 | assd | 612 | 29 | 0 | 29.1931 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 27.4084 |  |
| tta2 | assd | 612 | 29 | 3 | 30.5552 | 1.0348 |
| tta4 | assd | 612 | 29 | 3 | 29.3052 | 0.5822 |
| tta8 | assd | 612 | 29 | 3 | 28.5984 | 0.5357 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.1737 | 0.0103 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.1940 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.2098 |  |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.1813 | 0.0112 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.1934 | 0.0087 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.1964 | 0.0102 |
| baseline | dice | 612 | 29 | 3 | 0.5445 | 0.0050 |
| ensemble3 | dice | 612 | 29 | 0 | 0.5712 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.5833 |  |
| tta2 | dice | 612 | 29 | 3 | 0.5575 | 0.0039 |
| tta4 | dice | 612 | 29 | 3 | 0.5651 | 0.0038 |
| tta8 | dice | 612 | 29 | 3 | 0.5685 | 0.0043 |
| baseline | hd95 | 612 | 29 | 3 | 92.5643 | 2.8542 |
| ensemble3 | hd95 | 612 | 29 | 0 | 85.4627 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 80.3333 |  |
| tta2 | hd95 | 612 | 29 | 3 | 89.7840 | 3.4888 |
| tta4 | hd95 | 612 | 29 | 3 | 85.6092 | 1.5669 |
| tta8 | hd95 | 612 | 29 | 3 | 83.4319 | 1.9069 |
| baseline | hd95_pooled | 612 | 29 | 3 | 87.3939 | 2.7775 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 80.8892 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 76.2210 |  |
| tta2 | hd95_pooled | 612 | 29 | 3 | 85.1309 | 3.7411 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 80.9467 | 1.9502 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 78.9464 | 1.7627 |

Each video contributes one value (mean over its frames), 29 videos in total; mean ± std across the three seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0131 | 0.0065 | 0.0211 | 0.0002 | 0.0017 | 0.7379 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0124 | 0.0068 | 0.0189 | 0.0001 | 0.0006 | 0.7839 | tta2 |
| tta2 | precision | 29 | 612 | all_cases | 0.0149 | 0.0082 | 0.0225 | 0.0000 | 0.0003 | 0.8207 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | 0.0107 | 0.0032 | 0.0201 | 0.0030 | 0.0120 | 0.6138 | tta2 |
| tta2 | hd | 29 | 612 | both_nonempty | -3.9334 | -5.7833 | -2.1462 | 0.0003 | 0.0020 | -0.7241 | tta2 |
| tta2 | hd95 | 29 | 612 | both_nonempty | -2.7803 | -4.3911 | -1.1768 | 0.0030 | 0.0120 | -0.6138 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | both_nonempty | -2.2630 | -3.7833 | -0.7510 | 0.0138 | 0.0242 | -0.5172 | tta2 |
| tta2 | assd | 29 | 612 | both_nonempty | -1.2772 | -2.0857 | -0.5686 | 0.0008 | 0.0040 | -0.6828 | tta2 |
| tta2 | boundary_f1 | 29 | 612 | all_cases | 0.0076 | 0.0025 | 0.0131 | 0.0121 | 0.0242 | 0.5264 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0207 | 0.0104 | 0.0321 | 0.0002 | 0.0005 | 0.7379 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0230 | 0.0126 | 0.0344 | 0.0001 | 0.0003 | 0.7793 | tta4 |
| tta4 | precision | 29 | 612 | all_cases | 0.0313 | 0.0172 | 0.0477 | 0.0000 | 0.0001 | 0.8345 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0122 | 0.0019 | 0.0237 | 0.0138 | 0.0138 | 0.5172 | tta4 |
| tta4 | hd | 29 | 612 | both_nonempty | -9.8370 | -13.1567 | -6.6123 | 0.0000 | 0.0000 | -0.9080 | tta4 |
| tta4 | hd95 | 29 | 612 | both_nonempty | -6.9551 | -9.8021 | -4.1898 | 0.0000 | 0.0002 | -0.8161 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | both_nonempty | -6.4472 | -9.2662 | -3.7007 | 0.0001 | 0.0003 | -0.7747 | tta4 |
| tta4 | assd | 29 | 612 | both_nonempty | -2.5273 | -3.5604 | -1.6161 | 0.0000 | 0.0000 | -0.9126 | tta4 |
| tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0197 | 0.0111 | 0.0297 | 0.0000 | 0.0001 | 0.8437 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0241 | 0.0113 | 0.0388 | 0.0004 | 0.0007 | 0.7195 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0265 | 0.0138 | 0.0410 | 0.0001 | 0.0004 | 0.7609 | tta8 |
| tta8 | precision | 29 | 612 | all_cases | 0.0387 | 0.0212 | 0.0587 | 0.0000 | 0.0001 | 0.8345 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | 0.0130 | 0.0008 | 0.0280 | 0.0386 | 0.0386 | 0.4391 | tta8 |
| tta8 | hd | 29 | 612 | both_nonempty | -12.2898 | -16.6998 | -8.1488 | 0.0000 | 0.0000 | -0.9126 | tta8 |
| tta8 | hd95 | 29 | 612 | both_nonempty | -9.1237 | -13.0060 | -5.4786 | 0.0000 | 0.0001 | -0.8575 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | both_nonempty | -8.4384 | -12.3322 | -4.8756 | 0.0000 | 0.0001 | -0.8529 | tta8 |
| tta8 | assd | 29 | 612 | both_nonempty | -3.2320 | -4.8507 | -1.8795 | 0.0000 | 0.0001 | -0.8391 | tta8 |
| tta8 | boundary_f1 | 29 | 612 | all_cases | 0.0228 | 0.0129 | 0.0344 | 0.0000 | 0.0001 | 0.8529 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 0 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  |
| tta2 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tta4 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tta8 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 167.1760 | 183.5830 | 0.7598 | 0.3905 |  |  |  |  |  |  |  |  |  |
| ensemble3 | 612 | 165.4210 | 198.2790 | 0.6781 | 0.3644 | 14.6960 | -7.0467 | 17.9915 | -0.0817 | -0.1184 | -0.0457 | -0.0261 | -0.0443 | -0.0104 |
| ensemble3_tta4 | 612 | 157.5328 | 189.8761 | 0.6422 | 0.3513 | 6.2931 | -15.3810 | 12.9717 | -0.1176 | -0.1685 | -0.0713 | -0.0392 | -0.0730 | -0.0133 |
| tta2 | 612 | 165.2445 | 179.9065 | 0.7353 | 0.3807 | -3.6765 | -12.3760 | -0.0753 | -0.0245 | -0.0573 | 0.0069 | -0.0098 | -0.0313 | 0.0082 |
| tta4 | 612 | 158.1672 | 174.3871 | 0.7124 | 0.3791 | -9.1958 | -18.9917 | -1.1266 | -0.0474 | -0.0829 | -0.0136 | -0.0114 | -0.0334 | 0.0066 |
| tta8 | 612 | 157.0477 | 169.0857 | 0.6944 | 0.3627 | -14.4973 | -26.7573 | -4.9481 | -0.0654 | -0.1071 | -0.0258 | -0.0278 | -0.0665 | 0.0017 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -3.9334 | -5.7833 | -2.1462 | 0.0003 | 0.0013 | tta2 |
| tta2 | hd95 | 29 | 612 | -2.7803 | -4.3911 | -1.1768 | 0.0030 | 0.0060 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | -2.2630 | -3.7833 | -0.7510 | 0.0138 | 0.0138 | tta2 |
| tta2 | assd | 29 | 612 | -1.2772 | -2.0857 | -0.5686 | 0.0008 | 0.0024 | tta2 |
| tta4 | hd | 29 | 612 | -9.8370 | -13.1567 | -6.6123 | 0.0000 | 0.0000 | tta4 |
| tta4 | hd95 | 29 | 612 | -6.9551 | -9.8021 | -4.1898 | 0.0000 | 0.0001 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -6.4472 | -9.2662 | -3.7007 | 0.0001 | 0.0001 | tta4 |
| tta4 | assd | 29 | 612 | -2.5273 | -3.5604 | -1.6161 | 0.0000 | 0.0000 | tta4 |
| tta8 | hd | 29 | 612 | -12.0985 | -16.4609 | -8.0509 | 0.0000 | 0.0000 | tta8 |
| tta8 | hd95 | 29 | 612 | -8.9322 | -12.7651 | -5.4069 | 0.0000 | 0.0000 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | -8.2467 | -12.0601 | -4.8000 | 0.0000 | 0.0000 | tta8 |
| tta8 | assd | 29 | 612 | -3.0226 | -4.5860 | -1.7159 | 0.0000 | 0.0000 | tta8 |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 232 | 0.4498 | 0.0000 | 0.4696 | 0.0000 | 0.7109 |
| -1 | entropy_mean_roi | 612 | 232 | 0.5882 | 0.0000 | 0.7292 | 0.0000 | 0.8462 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 3 | 118 | 6 | 0.5930 | 0.0177 | 0.5697 | 0.0220 |
| 1 | dice | 3 | 137 | 6 | 0.5427 | 0.0327 | 0.5302 | 0.0333 |
| 2 | dice | 3 | 126 | 6 | 0.5639 | 0.0264 | 0.5594 | 0.0263 |
| 3 | dice | 3 | 129 | 6 | 0.3913 | 0.0313 | 0.4024 | 0.0276 |
| 4 | dice | 3 | 102 | 5 | 0.6760 | 0.0287 | 0.6838 | 0.0230 |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4 | dice | 29 | 612 | all_cases | 0.0207 | 0.0104 | 0.0321 | 0.0002 | 0.0005 | 0.7379 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0230 | 0.0126 | 0.0344 | 0.0001 | 0.0003 | 0.7793 | tta4 |
| tta4 | precision | 29 | 612 | all_cases | 0.0313 | 0.0172 | 0.0477 | 0.0000 | 0.0001 | 0.8345 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0122 | 0.0019 | 0.0237 | 0.0138 | 0.0138 | 0.5172 | tta4 |
| tta4 | hd | 29 | 612 | both_nonempty | -9.8370 | -13.1567 | -6.6123 | 0.0000 | 0.0000 | -0.9080 | tta4 |
| tta4 | hd95 | 29 | 612 | both_nonempty | -6.9551 | -9.8021 | -4.1898 | 0.0000 | 0.0002 | -0.8161 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | both_nonempty | -6.4472 | -9.2662 | -3.7007 | 0.0001 | 0.0003 | -0.7747 | tta4 |
| tta4 | assd | 29 | 612 | both_nonempty | -2.5273 | -3.5604 | -1.6161 | 0.0000 | 0.0000 | -0.9126 | tta4 |
| tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0197 | 0.0111 | 0.0297 | 0.0000 | 0.0001 | 0.8437 | tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0268 | 0.0183 | 0.0348 | 0.0000 | 0.0000 | 0.8759 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0290 | 0.0200 | 0.0378 | 0.0000 | 0.0000 | 0.8897 | ensemble3 |
| ensemble3 | precision | 29 | 612 | all_cases | 0.0404 | 0.0305 | 0.0525 | 0.0000 | 0.0000 | 1.0000 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0110 | 0.0011 | 0.0199 | 0.0048 | 0.0048 | 0.5862 | ensemble3 |
| ensemble3 | hd | 29 | 612 | both_nonempty | -9.5068 | -12.3676 | -6.4873 | 0.0000 | 0.0000 | -0.8805 | ensemble3 |
| ensemble3 | hd95 | 29 | 612 | both_nonempty | -7.1016 | -9.8221 | -4.2589 | 0.0001 | 0.0003 | -0.7701 | ensemble3 |
| ensemble3 | hd95_pooled | 29 | 612 | both_nonempty | -6.5047 | -8.8280 | -4.0525 | 0.0000 | 0.0001 | -0.8115 | ensemble3 |
| ensemble3 | assd | 29 | 612 | both_nonempty | -2.6393 | -3.5335 | -1.6520 | 0.0000 | 0.0001 | -0.8483 | ensemble3 |
| ensemble3 | boundary_f1 | 29 | 612 | all_cases | 0.0204 | 0.0118 | 0.0295 | 0.0002 | 0.0004 | 0.7471 | ensemble3 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0389 | 0.0271 | 0.0525 | 0.0000 | 0.0000 | 0.9402 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0430 | 0.0301 | 0.0573 | 0.0000 | 0.0000 | 0.9494 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 29 | 612 | all_cases | 0.0610 | 0.0424 | 0.0834 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0166 | 0.0051 | 0.0275 | 0.0009 | 0.0009 | 0.6782 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 29 | 612 | both_nonempty | -15.9233 | -21.1694 | -10.9943 | 0.0000 | 0.0000 | -0.9172 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 29 | 612 | both_nonempty | -12.2310 | -16.5667 | -8.0547 | 0.0000 | 0.0000 | -0.8621 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 29 | 612 | both_nonempty | -11.1729 | -15.2346 | -7.2974 | 0.0000 | 0.0000 | -0.8851 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 29 | 612 | both_nonempty | -4.4240 | -6.0647 | -3.0183 | 0.0000 | 0.0000 | -0.9218 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0362 | 0.0222 | 0.0520 | 0.0000 | 0.0000 | 0.8391 | ensemble3_tta4 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0182 | 0.0111 | 0.0257 | 0.0000 | 0.0002 | 0.8253 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0200 | 0.0123 | 0.0282 | 0.0000 | 0.0002 | 0.8115 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 29 | 612 | all_cases | 0.0297 | 0.0229 | 0.0378 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0043 | -0.0037 | 0.0117 | 0.0920 | 0.0920 | 0.3609 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 29 | 612 | both_nonempty | -6.0863 | -8.4986 | -3.7371 | 0.0000 | 0.0002 | -0.8207 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 29 | 612 | both_nonempty | -5.2759 | -7.2477 | -3.3576 | 0.0000 | 0.0001 | -0.8575 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 29 | 612 | both_nonempty | -4.7257 | -6.5478 | -2.9749 | 0.0000 | 0.0002 | -0.8253 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 29 | 612 | both_nonempty | -1.8967 | -2.8605 | -0.9903 | 0.0002 | 0.0007 | -0.7379 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 29 | 612 | all_cases | 0.0165 | 0.0090 | 0.0245 | 0.0003 | 0.0007 | 0.7333 | ensemble3_tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0061 | -0.0066 | 0.0180 | 0.1557 | 1.0000 | 0.3057 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0061 | -0.0067 | 0.0180 | 0.1828 | 1.0000 | 0.2874 | ensemble3 |
| ensemble3 | precision | 29 | 612 | all_cases | 0.0092 | -0.0022 | 0.0200 | 0.0562 | 0.5060 | 0.4069 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | -0.0012 | -0.0180 | 0.0130 | 0.3467 | 1.0000 | 0.2046 | tta4 |
| ensemble3 | hd | 29 | 612 | both_nonempty | 0.3302 | -2.0479 | 3.0845 | 0.9660 | 1.0000 | -0.0115 | tta4 |
| ensemble3 | hd95 | 29 | 612 | both_nonempty | -0.1465 | -2.6325 | 2.8493 | 0.3808 | 1.0000 | -0.1908 | ensemble3 |
| ensemble3 | hd95_pooled | 29 | 612 | both_nonempty | -0.0575 | -2.4857 | 2.7760 | 0.4420 | 1.0000 | -0.1678 | ensemble3 |
| ensemble3 | assd | 29 | 612 | both_nonempty | -0.1120 | -1.0941 | 0.9415 | 0.3357 | 1.0000 | -0.2092 | ensemble3 |
| ensemble3 | boundary_f1 | 29 | 612 | all_cases | 0.0007 | -0.0074 | 0.0086 | 0.9660 | 1.0000 | 0.0115 | ensemble3 |

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 0 | 0 | 0 | 7363.5000 | 0.4331 |  |
| failures_all | 232 | 0 | 0 | 0 | 5361.0000 | 0.5020 |  |
| failures_non_missed | 232 | 0 | 0 | 0 | 5361.0000 | 0.5020 |  |
| failures_medium_101_1000px | 10 | 0 | 0 | 0 | 479.5000 | 0.5921 |  |
| failures_large_gt_1000px | 222 | 0 | 0 | 0 | 5552.5000 | 0.4987 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8462 | 0.7609 | 0.9187 | 0.2851 | 0.2208 | 0.3710 | 0.2252 | 612 | 612 | 232 | 0 | 0.8462 | 0.8462 | 0.2851 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8462 | 0.7609 | 0.9187 | 0.2851 | 0.2208 | 0.3710 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.2851 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.1853 | 0.1617 | 0.2179 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1853 |
| tta4_variance_roi | own | tta4 | all_failures | 0.7184 | 0.6089 | 0.8168 | 0.3374 | 0.2727 | 0.4100 | 0.2252 | 612 | 612 | 232 | 0 | 0.7184 | 0.7184 | 0.3374 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.7184 | 0.6089 | 0.8168 | 0.3374 | 0.2727 | 0.4100 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3374 |
| tta4_variance_roi | own | tta4 | missed_vs_clean |  |  |  | 0.2312 | 0.1976 | 0.2661 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2312 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.7109 | 0.6151 | 0.7940 | 0.3266 | 0.2583 | 0.4033 | 0.2252 | 612 | 612 | 232 | 0 | 0.7109 | 0.7109 | 0.3266 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.7109 | 0.6151 | 0.7940 | 0.3266 | 0.2583 | 0.4033 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3266 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean |  |  |  | 0.2048 | 0.1788 | 0.2417 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2048 |
| tta4_entropy_image | own | tta4 | all_failures | 0.6292 | 0.4779 | 0.7783 | 0.3357 | 0.2526 | 0.4553 | 0.2252 | 612 | 612 | 232 | 0 | 0.6292 | 0.6292 | 0.3357 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.6292 | 0.4779 | 0.7783 | 0.3357 | 0.2526 | 0.4553 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3357 |
| tta4_entropy_image | own | tta4 | missed_vs_clean |  |  |  | 0.1974 | 0.1706 | 0.2342 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1974 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.8352 | 0.7504 | 0.9120 | 0.2744 | 0.2044 | 0.3674 | 0.2002 | 612 | 612 | 215 | 0 | 0.8352 | 0.8352 | 0.2744 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.8352 | 0.7504 | 0.9120 | 0.2744 | 0.2044 | 0.3674 | 0.2002 | 612 | 612 | 215 | 0 |  |  | 0.2744 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1718 | 0.1487 | 0.2021 | 0.1352 | 397 | 397 | 0 | 0 |  |  | 0.1718 |
| map_entropy_roi_tta4 | tta4 | tta4 | all_failures | 0.8515 | 0.7666 | 0.9234 | 0.2919 | 0.2235 | 0.3815 | 0.2252 | 612 | 612 | 232 | 0 | 0.8515 | 0.8515 | 0.2919 |
| map_entropy_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8515 | 0.7666 | 0.9234 | 0.2919 | 0.2235 | 0.3815 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.2919 |
| map_entropy_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1873 | 0.1630 | 0.2203 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1873 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.7842 | 0.7200 | 0.8437 | 0.2900 | 0.2265 | 0.3663 | 0.2002 | 612 | 612 | 215 | 0 | 0.7842 | 0.7842 | 0.2900 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.7842 | 0.7200 | 0.8437 | 0.2900 | 0.2265 | 0.3663 | 0.2002 | 612 | 612 | 215 | 0 |  |  | 0.2900 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1915 | 0.1602 | 0.2291 | 0.1352 | 397 | 397 | 0 | 0 |  |  | 0.1915 |
| map_variance_roi_tta4 | tta4 | tta4 | all_failures | 0.8082 | 0.7427 | 0.8675 | 0.3056 | 0.2411 | 0.3807 | 0.2252 | 612 | 612 | 232 | 0 | 0.8082 | 0.8082 | 0.3056 |
| map_variance_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8082 | 0.7427 | 0.8675 | 0.3056 | 0.2411 | 0.3807 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3056 |
| map_variance_roi_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.2058 | 0.1751 | 0.2476 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2058 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | all_failures | 0.6221 | 0.4723 | 0.7558 | 0.3339 | 0.2542 | 0.4498 | 0.2002 | 612 | 612 | 215 | 0 | 0.6221 | 0.6221 | 0.3339 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.6221 | 0.4723 | 0.7558 | 0.3339 | 0.2542 | 0.4498 | 0.2002 | 612 | 612 | 215 | 0 |  |  | 0.3339 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1897 | 0.1638 | 0.2230 | 0.1352 | 397 | 397 | 0 | 0 |  |  | 0.1897 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | all_failures | 0.6133 | 0.4660 | 0.7511 | 0.3495 | 0.2696 | 0.4637 | 0.2252 | 612 | 612 | 232 | 0 | 0.6133 | 0.6133 | 0.3495 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | non_missed_failures | 0.6133 | 0.4660 | 0.7511 | 0.3495 | 0.2696 | 0.4637 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3495 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | missed_vs_clean |  |  |  | 0.2038 | 0.1778 | 0.2386 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2038 |
| pred_area_px | own | ensemble3_tta4 | all_failures | 0.3201 | 0.2175 | 0.4409 | 0.5284 | 0.4234 | 0.6233 | 0.2002 | 612 | 612 | 215 | 0 | 0.3201 | 0.3201 | 0.5284 |
| pred_area_px | own | ensemble3_tta4 | non_missed_failures | 0.3201 | 0.2175 | 0.4409 | 0.5284 | 0.4234 | 0.6233 | 0.2002 | 612 | 612 | 215 | 0 |  |  | 0.5284 |
| pred_area_px | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.2501 | 0.2205 | 0.2800 | 0.1352 | 397 | 397 | 0 | 0 |  |  | 0.2501 |
| pred_area_px | tta4 | tta4 | all_failures | 0.2860 | 0.1932 | 0.3936 | 0.5403 | 0.4470 | 0.6295 | 0.2252 | 612 | 612 | 232 | 0 | 0.2860 | 0.2860 | 0.5403 |
| pred_area_px | tta4 | tta4 | non_missed_failures | 0.2860 | 0.1932 | 0.3936 | 0.5403 | 0.4470 | 0.6295 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.5403 |
| pred_area_px | tta4 | tta4 | missed_vs_clean |  |  |  | 0.2626 | 0.2375 | 0.2931 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2626 |
| one_minus_confidence | own | ensemble3_tta4 | all_failures | 0.8339 | 0.7469 | 0.9098 | 0.2753 | 0.2036 | 0.3681 | 0.2002 | 612 | 612 | 215 | 0 | 0.8339 | 0.8339 | 0.2753 |
| one_minus_confidence | own | ensemble3_tta4 | non_missed_failures | 0.8339 | 0.7469 | 0.9098 | 0.2753 | 0.2036 | 0.3681 | 0.2002 | 612 | 612 | 215 | 0 |  |  | 0.2753 |
| one_minus_confidence | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1704 | 0.1472 | 0.2007 | 0.1352 | 397 | 397 | 0 | 0 |  |  | 0.1704 |
| one_minus_confidence | tta4 | tta4 | all_failures | 0.8492 | 0.7631 | 0.9228 | 0.2926 | 0.2233 | 0.3831 | 0.2252 | 612 | 612 | 232 | 0 | 0.8492 | 0.8492 | 0.2926 |
| one_minus_confidence | tta4 | tta4 | non_missed_failures | 0.8492 | 0.7631 | 0.9228 | 0.2926 | 0.2233 | 0.3831 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.2926 |
| one_minus_confidence | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1856 | 0.1618 | 0.2185 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1856 |
| baseline_entropy_roi | own | baseline | all_failures | 0.7441 | 0.6105 | 0.8569 | 0.3306 | 0.2616 | 0.4276 | 0.2463 | 612 | 612 | 239 | 0 | 0.7441 | 0.7441 | 0.3306 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.7441 | 0.6105 | 0.8569 | 0.3306 | 0.2616 | 0.4276 | 0.2463 | 612 | 612 | 239 | 0 |  |  | 0.3306 |
| baseline_entropy_roi | own | baseline | missed_vs_clean |  |  |  | 0.2145 | 0.1902 | 0.2502 | 0.1752 | 373 | 373 | 0 | 0 |  |  | 0.2145 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.7496 | 0.6137 | 0.8635 | 0.3060 | 0.2380 | 0.4040 | 0.2252 | 612 | 612 | 232 | 0 | 0.7496 | 0.7496 | 0.3060 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.7496 | 0.6137 | 0.8635 | 0.3060 | 0.2380 | 0.4040 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3060 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1903 | 0.1664 | 0.2253 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1903 |
| baseline_entropy_image | own | baseline | all_failures | 0.6068 | 0.4525 | 0.7587 | 0.3649 | 0.2808 | 0.4842 | 0.2463 | 612 | 612 | 239 | 0 | 0.6068 | 0.6068 | 0.3649 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.6068 | 0.4525 | 0.7587 | 0.3649 | 0.2808 | 0.4842 | 0.2463 | 612 | 612 | 239 | 0 |  |  | 0.3649 |
| baseline_entropy_image | own | baseline | missed_vs_clean |  |  |  | 0.2242 | 0.1981 | 0.2619 | 0.1752 | 373 | 373 | 0 | 0 |  |  | 0.2242 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.6196 | 0.4654 | 0.7701 | 0.3394 | 0.2574 | 0.4567 | 0.2252 | 612 | 612 | 232 | 0 | 0.6196 | 0.6196 | 0.3394 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.6196 | 0.4654 | 0.7701 | 0.3394 | 0.2574 | 0.4567 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.3394 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean |  |  |  | 0.2019 | 0.1761 | 0.2408 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.2019 |
| ensemble3_entropy_roi | own | ensemble3 | all_failures | 0.8185 | 0.7412 | 0.8855 | 0.2921 | 0.2204 | 0.3814 | 0.2105 | 612 | 612 | 223 | 0 | 0.8185 | 0.8185 | 0.2921 |
| ensemble3_entropy_roi | own | ensemble3 | non_missed_failures | 0.8185 | 0.7412 | 0.8855 | 0.2921 | 0.2204 | 0.3814 | 0.2105 | 612 | 612 | 223 | 0 |  |  | 0.2921 |
| ensemble3_entropy_roi | own | ensemble3 | missed_vs_clean |  |  |  | 0.1792 | 0.1548 | 0.2109 | 0.1428 | 389 | 389 | 0 | 0 |  |  | 0.1792 |
| ensemble3_entropy_roi | tta4 | tta4 | all_failures | 0.8273 | 0.7477 | 0.8966 | 0.2995 | 0.2294 | 0.3901 | 0.2252 | 612 | 612 | 232 | 0 | 0.8273 | 0.8273 | 0.2995 |
| ensemble3_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8273 | 0.7477 | 0.8966 | 0.2995 | 0.2294 | 0.3901 | 0.2252 | 612 | 612 | 232 | 0 |  |  | 0.2995 |
| ensemble3_entropy_roi | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1866 | 0.1630 | 0.2193 | 0.1560 | 380 | 380 | 0 | 0 |  |  | 0.1866 |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.8462 | 0.7609 | 0.9187 | 0.2851 | 0.2208 | 0.3710 | 0.2252 | 612 | 612 | 232 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8462 | 0.7609 | 0.9187 | 0.2851 | 0.2208 | 0.3710 | 0.2252 | 612 | 612 | 232 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean |  |  |  | 0.1853 | 0.1617 | 0.2179 | 0.1560 | 380 | 380 | 0 |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | ensemble3 | all_failures | 0.8256 | 0.7555 | 0.8859 | 0.2753 | 0.2129 | 0.3509 | 0.2105 | 612 | 612 | 223 |
| ensemble3_dice_disagreement | own | ensemble3 | non_missed_failures | 0.8256 | 0.7555 | 0.8859 | 0.2753 | 0.2129 | 0.3509 | 0.2105 | 612 | 612 | 223 |
| ensemble3_dice_disagreement | own | ensemble3 | missed_vs_clean |  |  |  | 0.1807 | 0.1522 | 0.2180 | 0.1428 | 389 | 389 | 0 |
| ensemble3_dice_disagreement | tta4 | tta4 | all_failures | 0.8434 | 0.7729 | 0.9053 | 0.2856 | 0.2243 | 0.3604 | 0.2252 | 612 | 612 | 232 |
| ensemble3_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8434 | 0.7729 | 0.9053 | 0.2856 | 0.2243 | 0.3604 | 0.2252 | 612 | 612 | 232 |
| ensemble3_dice_disagreement | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1904 | 0.1634 | 0.2297 | 0.1560 | 380 | 380 | 0 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | all_failures | 0.8435 | 0.7656 | 0.9132 | 0.2615 | 0.2009 | 0.3415 | 0.2002 | 612 | 612 | 215 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | non_missed_failures | 0.8435 | 0.7656 | 0.9132 | 0.2615 | 0.2009 | 0.3415 | 0.2002 | 612 | 612 | 215 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.1707 | 0.1456 | 0.2059 | 0.1352 | 397 | 397 | 0 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | all_failures | 0.8732 | 0.8011 | 0.9361 | 0.2758 | 0.2143 | 0.3554 | 0.2252 | 612 | 612 | 232 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8732 | 0.8011 | 0.9361 | 0.2758 | 0.2143 | 0.3554 | 0.2252 | 612 | 612 | 232 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | missed_vs_clean |  |  |  | 0.1812 | 0.1564 | 0.2200 | 0.1560 | 380 | 380 | 0 |

`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the four tta4 views (per seed, averaged over seeds); `ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` are the same over the three seeds' baseline / tta4 maps. Higher = the members disagree = more uncertain. The tta4 entropy row is kept as the reference descriptor.

## (f-audit) Spearman against HD95 and 1 - Dice (all predictors)

| predictor | label_source | label_prediction | target | n_used | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | hd95 | 612 | 0.5882 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 612 | 0.7292 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 612 | 0.3538 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 612 | 0.3800 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 612 | 0.4498 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.4696 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 612 | 0.2860 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.3778 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.5250 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.6799 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | hd95 | 612 | 0.5482 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.7139 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.4078 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.5325 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | hd95 | 612 | 0.4501 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.5802 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | hd95 | 612 | 0.3020 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.3343 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | hd95 | 612 | 0.2837 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.3460 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | hd95 | 612 | -0.1854 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | one_minus_dice | 612 | -0.3368 | 0.0000 |
| pred_area_px | tta4 | tta4 | hd95 | 612 | -0.2729 | 0.0000 |
| pred_area_px | tta4 | tta4 | one_minus_dice | 612 | -0.3668 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | hd95 | 612 | 0.5261 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | one_minus_dice | 612 | 0.6812 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | hd95 | 612 | 0.5439 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | one_minus_dice | 612 | 0.7147 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 612 | 0.4020 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 612 | 0.5491 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 612 | 0.4544 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 612 | 0.5870 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 612 | 0.1752 | 0.0000 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.2977 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 612 | 0.2483 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.3520 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | hd95 | 612 | 0.4899 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | one_minus_dice | 612 | 0.6543 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | hd95 | 612 | 0.5070 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | one_minus_dice | 612 | 0.6784 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | hd95 | 612 | 0.3837 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | one_minus_dice | 612 | 0.4540 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | hd95 | 612 | 0.3528 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | one_minus_dice | 612 | 0.4431 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | hd95 | 612 | 0.2235 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | one_minus_dice | 612 | 0.3286 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | hd95 | 612 | 0.2872 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.3783 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | hd95 | 612 | 0.4137 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | one_minus_dice | 612 | 0.4300 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | hd95 | 612 | 0.4091 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | one_minus_dice | 612 | 0.4353 | 0.0000 |
| tta4_view_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| tta4_view_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_dice_disagreement | own | ensemble3 | hd95 | 612 | 0.5552 | 0.0000 |
| ensemble3_dice_disagreement | own | ensemble3 | one_minus_dice | 612 | 0.6689 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | hd95 | 612 | 0.5587 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | one_minus_dice | 612 | 0.6821 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | hd95 | 612 | 0.5622 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | one_minus_dice | 612 | 0.7026 | 0.0000 |

Targets also come from the predictor's own prediction (`label_source=own`); the `tta4` rows repeat the correlation against the tta4 prediction for comparison. `n_used` counts the finite pairs, so ROI descriptors lose the missed detections here.

## (f-audit) detector vs detector on AUROC and AURC (paired bootstrap)

| predictor | compared_with | label_prediction | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.1278 | -0.0120 | 0.2728 | 0.0820 | -0.0524 | -0.1072 | 0.0086 | 0.0940 | 612 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.1352 | 0.0520 | 0.2334 | 0.0020 | -0.0416 | -0.0825 | 0.0001 | 0.0520 | 612 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.2169 | 0.1101 | 0.3291 | 0.0000 | -0.0506 | -0.0944 | -0.0207 | 0.0000 | 612 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | -0.0053 | -0.0284 | 0.0170 | 0.6340 | -0.0068 | -0.0147 | 0.0006 | 0.0920 | 612 |
| tta4_entropy_roi | map_entropy_roi_tta4 | ensemble3_tta4 | -0.0061 | -0.0306 | 0.0184 | 0.5880 | -0.0067 | -0.0154 | 0.0015 | 0.1440 | 612 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | 0.0380 | -0.0261 | 0.1027 | 0.2560 | -0.0206 | -0.0462 | 0.0130 | 0.1520 | 612 |
| tta4_entropy_roi | map_variance_roi_tta4 | ensemble3_tta4 | 0.0449 | -0.0223 | 0.1108 | 0.1860 | -0.0223 | -0.0487 | 0.0120 | 0.1380 | 612 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.2329 | 0.1302 | 0.3476 | 0.0000 | -0.0645 | -0.1008 | -0.0379 | 0.0000 | 612 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | ensemble3_tta4 | 0.2069 | 0.1044 | 0.3257 | 0.0000 | -0.0662 | -0.1036 | -0.0390 | 0.0000 | 612 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.5601 | 0.4331 | 0.6808 | 0.0000 | -0.2552 | -0.3007 | -0.2018 | 0.0000 | 612 |
| tta4_entropy_roi | pred_area_px | ensemble3_tta4 | 0.5089 | 0.3670 | 0.6366 | 0.0000 | -0.2607 | -0.3119 | -0.1966 | 0.0000 | 612 |
| tta4_entropy_roi | one_minus_confidence | tta4 | -0.0031 | -0.0290 | 0.0222 | 0.8200 | -0.0076 | -0.0170 | 0.0019 | 0.1620 | 612 |
| tta4_entropy_roi | one_minus_confidence | ensemble3_tta4 | -0.0048 | -0.0324 | 0.0217 | 0.7080 | -0.0076 | -0.0179 | 0.0026 | 0.1980 | 612 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.0965 | 0.0378 | 0.1647 | 0.0020 | -0.0209 | -0.0381 | -0.0076 | 0.0020 | 612 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.1013 | 0.0409 | 0.1719 | 0.0000 | -0.0236 | -0.0431 | -0.0081 | 0.0000 | 612 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.2266 | 0.1192 | 0.3391 | 0.0000 | -0.0543 | -0.0999 | -0.0217 | 0.0000 | 612 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.2386 | 0.1238 | 0.3609 | 0.0000 | -0.0580 | -0.1056 | -0.0231 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | 0.0188 | -0.0135 | 0.0501 | 0.2520 | -0.0144 | -0.0244 | -0.0029 | 0.0080 | 612 |
| tta4_entropy_roi | ensemble3_entropy_roi | ensemble3 | 0.0066 | -0.0294 | 0.0407 | 0.6520 | -0.0130 | -0.0250 | 0.0008 | 0.0660 | 612 |
| tta4_entropy_roi | ensemble3_variance_roi | tta4 | 0.0920 | -0.0206 | 0.1923 | 0.0900 | -0.0420 | -0.0821 | 0.0087 | 0.1160 | 612 |
| tta4_entropy_roi | ensemble3_variance_roi | ensemble3 | 0.0756 | -0.0365 | 0.1761 | 0.1620 | -0.0331 | -0.0768 | 0.0178 | 0.1840 | 612 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.2170 | 0.1070 | 0.3297 | 0.0000 | -0.0508 | -0.0937 | -0.0210 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_entropy_image | ensemble3 | 0.2107 | 0.0987 | 0.3257 | 0.0000 | -0.0557 | -0.1031 | -0.0230 | 0.0000 | 612 |
| tta4_entropy_roi | ensemble3_variance_image | tta4 | 0.1274 | 0.0291 | 0.2356 | 0.0120 | -0.0421 | -0.0789 | -0.0042 | 0.0320 | 612 |
| tta4_entropy_roi | ensemble3_variance_image | ensemble3 | 0.0987 | 0.0020 | 0.2006 | 0.0420 | -0.0388 | -0.0780 | 0.0013 | 0.0660 | 612 |
| tta4_entropy_roi | ensemble3_dice_disagreement | tta4 | 0.0027 | -0.0659 | 0.0715 | 0.9300 | -0.0005 | -0.0261 | 0.0305 | 0.8920 | 612 |
| tta4_entropy_roi | ensemble3_dice_disagreement | ensemble3 | -0.0005 | -0.0691 | 0.0689 | 0.9880 | 0.0038 | -0.0236 | 0.0346 | 0.8640 | 612 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | tta4 | -0.0270 | -0.0688 | 0.0184 | 0.2200 | 0.0093 | -0.0066 | 0.0289 | 0.3080 | 612 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | ensemble3_tta4 | -0.0145 | -0.0573 | 0.0346 | 0.5420 | 0.0062 | -0.0107 | 0.0262 | 0.5540 | 612 |
| ensemble3_dice_disagreement | ensemble3_entropy_roi | ensemble3 | 0.0071 | -0.0653 | 0.0745 | 0.8260 | -0.0168 | -0.0523 | 0.0130 | 0.3660 | 612 |
| ensemble3_tta4_dice_disagreement | map_entropy_roi_tta4 | ensemble3_tta4 | 0.0084 | -0.0424 | 0.0513 | 0.6900 | -0.0130 | -0.0373 | 0.0074 | 0.2900 | 612 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 232, baseline 239, ensemble3 223, ensemble3_tta4 215

## Audit note

tta4 shares 232 failures; 0 of them have no ROI value (empty prediction), 0 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice unavailable for 612 of 612 cases: the runner does not store per-view masks. Re-run tta4 on CPU with configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml so the `view_pairwise_dice` metric exists, then pass it as `view_agreement`.

## Audit note

seed agreement: 612 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 6.19e-04
