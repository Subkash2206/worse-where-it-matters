# MELBA CVC-ClinicDB - 5-fold x 3-seed TTA results

15 runs: folds [0, 1, 2, 3, 4] x seeds [0, 1, 2]; config `configs/cvc_clinicdb/unet_resnet50_dropout.yaml`; pairing unit `group`; bootstrap draws `config`; Holm correction; failure thresholds HD95 > 50 px and Dice < 0.5.

Frames of one video are not independent, so every CI and test resamples or pairs at the video level (29 units). Distance metrics are restricted to cases non-empty under both methods (`n_cases_used`); section (e) repeats the comparison on all cases with empty predictions penalised by the image diagonal. `ensemble3` averages the three seeds' baseline maps and `ensemble3_tta4` their tta4 maps (3 x 4 views); both are a single realisation, so their across-seed std is undefined (blank).

## (a) Per-method summary - frame level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 13.2819 | 0.5242 |
| ensemble3 | assd | 612 | 29 | 0 | 11.5799 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 10.2045 |  |
| mcdropout_8 | assd | 612 | 29 | 3 | 13.1839 | 0.5519 |
| tta2 | assd | 612 | 29 | 3 | 12.2962 | 0.6553 |
| tta4 | assd | 612 | 29 | 3 | 11.0486 | 0.1872 |
| tta8 | assd | 612 | 29 | 3 | 10.4864 | 0.4725 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.4889 | 0.0164 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5257 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5573 |  |
| mcdropout_8 | boundary_f1 | 612 | 29 | 3 | 0.4835 | 0.0199 |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.5023 | 0.0206 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5298 | 0.0224 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5416 | 0.0214 |
| baseline | dice | 612 | 29 | 3 | 0.7738 | 0.0169 |
| ensemble3 | dice | 612 | 29 | 0 | 0.7970 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.8103 |  |
| mcdropout_8 | dice | 612 | 29 | 3 | 0.7728 | 0.0171 |
| tta2 | dice | 612 | 29 | 3 | 0.7846 | 0.0152 |
| tta4 | dice | 612 | 29 | 3 | 0.7964 | 0.0153 |
| tta8 | dice | 612 | 29 | 3 | 0.8078 | 0.0119 |
| baseline | hd95 | 612 | 29 | 3 | 50.1339 | 2.5342 |
| ensemble3 | hd95 | 612 | 29 | 0 | 42.7375 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 37.8545 |  |
| mcdropout_8 | hd95 | 612 | 29 | 3 | 49.4495 | 2.1519 |
| tta2 | hd95 | 612 | 29 | 3 | 45.8879 | 2.7033 |
| tta4 | hd95 | 612 | 29 | 3 | 41.7316 | 0.2186 |
| tta8 | hd95 | 612 | 29 | 3 | 39.2364 | 0.4303 |
| baseline | hd95_pooled | 612 | 29 | 3 | 47.0227 | 1.9650 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 40.2829 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 35.4127 |  |
| mcdropout_8 | hd95_pooled | 612 | 29 | 3 | 46.4881 | 1.7347 |
| tta2 | hd95_pooled | 612 | 29 | 3 | 42.7046 | 2.3483 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 39.3390 | 0.1709 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 36.9877 | 0.5996 |

Mean over all frames of the fold, pooled over the 5 folds (612 cases), then mean ± std across the three seeds.

## (a) Per-method summary - video level

| method | metric | n_cases | n_videos | n_seeds | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 612 | 29 | 3 | 14.2945 | 0.3854 |
| ensemble3 | assd | 612 | 29 | 0 | 12.4718 |  |
| ensemble3_tta4 | assd | 612 | 29 | 0 | 10.9320 |  |
| mcdropout_8 | assd | 612 | 29 | 3 | 14.1279 | 0.4198 |
| tta2 | assd | 612 | 29 | 3 | 13.3669 | 0.8640 |
| tta4 | assd | 612 | 29 | 3 | 11.9133 | 0.2520 |
| tta8 | assd | 612 | 29 | 3 | 11.1708 | 0.4041 |
| baseline | boundary_f1 | 612 | 29 | 3 | 0.4806 | 0.0174 |
| ensemble3 | boundary_f1 | 612 | 29 | 0 | 0.5203 |  |
| ensemble3_tta4 | boundary_f1 | 612 | 29 | 0 | 0.5495 |  |
| mcdropout_8 | boundary_f1 | 612 | 29 | 3 | 0.4758 | 0.0206 |
| tta2 | boundary_f1 | 612 | 29 | 3 | 0.4909 | 0.0290 |
| tta4 | boundary_f1 | 612 | 29 | 3 | 0.5208 | 0.0249 |
| tta8 | boundary_f1 | 612 | 29 | 3 | 0.5356 | 0.0230 |
| baseline | dice | 612 | 29 | 3 | 0.7484 | 0.0186 |
| ensemble3 | dice | 612 | 29 | 0 | 0.7702 |  |
| ensemble3_tta4 | dice | 612 | 29 | 0 | 0.7828 |  |
| mcdropout_8 | dice | 612 | 29 | 3 | 0.7475 | 0.0187 |
| tta2 | dice | 612 | 29 | 3 | 0.7574 | 0.0150 |
| tta4 | dice | 612 | 29 | 3 | 0.7698 | 0.0141 |
| tta8 | dice | 612 | 29 | 3 | 0.7811 | 0.0112 |
| baseline | hd95 | 612 | 29 | 3 | 52.6817 | 2.5058 |
| ensemble3 | hd95 | 612 | 29 | 0 | 45.6914 |  |
| ensemble3_tta4 | hd95 | 612 | 29 | 0 | 40.7746 |  |
| mcdropout_8 | hd95 | 612 | 29 | 3 | 52.0192 | 2.1789 |
| tta2 | hd95 | 612 | 29 | 3 | 48.8991 | 2.4686 |
| tta4 | hd95 | 612 | 29 | 3 | 44.6963 | 0.2396 |
| tta8 | hd95 | 612 | 29 | 3 | 41.9378 | 0.1975 |
| baseline | hd95_pooled | 612 | 29 | 3 | 49.5387 | 1.8696 |
| ensemble3 | hd95_pooled | 612 | 29 | 0 | 43.1467 |  |
| ensemble3_tta4 | hd95_pooled | 612 | 29 | 0 | 38.2525 |  |
| mcdropout_8 | hd95_pooled | 612 | 29 | 3 | 49.0134 | 1.6860 |
| tta2 | hd95_pooled | 612 | 29 | 3 | 45.7250 | 2.2646 |
| tta4 | hd95_pooled | 612 | 29 | 3 | 42.2391 | 0.2589 |
| tta8 | hd95_pooled | 612 | 29 | 3 | 39.6096 | 0.3263 |

Each video contributes one value (mean over its frames), 29 videos in total; mean ± std across the three seeds.

## (b) Paired comparisons vs baseline - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | dice | 29 | 612 | all_cases | 0.0090 | -0.0002 | 0.0188 | 0.0455 | 0.1366 | 0.4253 | tta2 |
| tta2 | jaccard | 29 | 612 | all_cases | 0.0125 | 0.0025 | 0.0231 | 0.0216 | 0.0862 | 0.4851 | tta2 |
| tta2 | precision | 29 | 600 | all_cases | 0.0251 | 0.0108 | 0.0437 | 0.0001 | 0.0006 | 0.7885 | tta2 |
| tta2 | recall | 29 | 612 | all_cases | 0.0018 | -0.0073 | 0.0110 | 0.2843 | 0.4107 | 0.2322 | tta2 |
| tta2 | hd | 28 | 590 | both_nonempty | -3.7831 | -6.3120 | -1.4640 | 0.0009 | 0.0063 | -0.6897 | tta2 |
| tta2 | hd95 | 28 | 590 | both_nonempty | -3.8946 | -6.4315 | -1.6280 | 0.0012 | 0.0070 | -0.6749 | tta2 |
| tta2 | hd95_pooled | 28 | 590 | both_nonempty | -3.9633 | -6.3184 | -1.7687 | 0.0008 | 0.0063 | -0.6946 | tta2 |
| tta2 | assd | 28 | 590 | both_nonempty | -1.0574 | -1.8686 | -0.2712 | 0.0027 | 0.0133 | -0.6305 | tta2 |
| tta2 | boundary_f1 | 29 | 600 | all_cases | 0.0049 | -0.0113 | 0.0191 | 0.2053 | 0.4107 | 0.2736 | tta2 |
| tta4 | dice | 29 | 612 | all_cases | 0.0214 | 0.0098 | 0.0329 | 0.0010 | 0.0019 | 0.6736 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0286 | 0.0157 | 0.0413 | 0.0002 | 0.0008 | 0.7471 | tta4 |
| tta4 | precision | 28 | 599 | all_cases | 0.0354 | 0.0240 | 0.0487 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0064 | -0.0038 | 0.0171 | 0.0308 | 0.0308 | 0.4575 | tta4 |
| tta4 | hd | 28 | 588 | both_nonempty | -8.8857 | -12.3960 | -5.4889 | 0.0000 | 0.0001 | -0.8621 | tta4 |
| tta4 | hd95 | 28 | 588 | both_nonempty | -8.2077 | -11.6058 | -4.8946 | 0.0000 | 0.0002 | -0.8325 | tta4 |
| tta4 | hd95_pooled | 28 | 588 | both_nonempty | -7.5219 | -10.7802 | -4.3586 | 0.0000 | 0.0002 | -0.8276 | tta4 |
| tta4 | assd | 28 | 588 | both_nonempty | -2.4213 | -3.5392 | -1.4248 | 0.0000 | 0.0001 | -0.8571 | tta4 |
| tta4 | boundary_f1 | 28 | 599 | all_cases | 0.0361 | 0.0202 | 0.0509 | 0.0002 | 0.0008 | 0.7537 | tta4 |
| tta8 | dice | 29 | 612 | all_cases | 0.0327 | 0.0203 | 0.0458 | 0.0000 | 0.0000 | 0.8989 | tta8 |
| tta8 | jaccard | 29 | 612 | all_cases | 0.0422 | 0.0283 | 0.0570 | 0.0000 | 0.0000 | 0.9356 | tta8 |
| tta8 | precision | 28 | 598 | all_cases | 0.0434 | 0.0296 | 0.0599 | 0.0000 | 0.0000 | 0.9951 | tta8 |
| tta8 | recall | 29 | 612 | all_cases | 0.0137 | 0.0027 | 0.0272 | 0.0203 | 0.0203 | 0.4897 | tta8 |
| tta8 | hd | 28 | 590 | both_nonempty | -11.7302 | -16.2801 | -7.5174 | 0.0000 | 0.0000 | -0.8867 | tta8 |
| tta8 | hd95 | 28 | 590 | both_nonempty | -10.8802 | -15.3344 | -6.7666 | 0.0000 | 0.0000 | -0.8621 | tta8 |
| tta8 | hd95_pooled | 28 | 590 | both_nonempty | -10.0632 | -14.3512 | -6.0426 | 0.0000 | 0.0001 | -0.8424 | tta8 |
| tta8 | assd | 28 | 590 | both_nonempty | -3.0699 | -4.5027 | -1.7667 | 0.0000 | 0.0001 | -0.8374 | tta8 |
| tta8 | boundary_f1 | 28 | 598 | all_cases | 0.0506 | 0.0340 | 0.0669 | 0.0000 | 0.0000 | 0.8966 | tta8 |

Wilcoxon signed-rank (primary), seeded percentile bootstrap CI of the mean delta, matched-pairs rank-biserial effect size, Holm across metrics. `restriction=both_nonempty` marks distance metrics and `n_cases_used` is the number of cases they were computed on.

## (c) Missed detections pooled over folds

| method | n_cases | n_missed | missed_rate | ci_low | ci_high | nonempty_to_empty | empty_to_nonempty | mcnemar_p | mcnemar_p_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 18 | 0.0294 | 0.0107 | 0.0567 |  |  |  |  |
| tta2 | 612 | 19 | 0.0310 | 0.0094 | 0.0615 | 4.0000 | 3.0000 | 1.0000 | 1.0000 |
| tta4 | 612 | 21 | 0.0343 | 0.0132 | 0.0634 | 6.0000 | 3.0000 | 0.5078 | 1.0000 |
| tta8 | 612 | 17 | 0.0278 | 0.0078 | 0.0567 | 4.0000 | 5.0000 | 1.0000 | 1.0000 |

Cluster (video) bootstrap CI of the missed rate; `nonempty_to_empty` counts new misses and `empty_to_nonempty` recovered cases relative to the baseline; exact McNemar p on the discordant pairs, Holm-corrected.

## (d) Tail metrics (HD95 percentiles, failure rates) and change vs baseline

| method | n_cases | hd95_p90 | hd95_p95 | hd95_failure_rate | dice_failure_rate | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 612 | 122.7901 | 150.8693 | 0.4134 | 0.1242 |  |  |  |  |  |  |  |  |  |
| ensemble3 | 612 | 127.7657 | 157.4608 | 0.3137 | 0.1095 | 6.5915 | -0.5455 | 14.5212 | -0.0997 | -0.1396 | -0.0613 | -0.0147 | -0.0285 | -0.0033 |
| ensemble3_tta4 | 612 | 121.4953 | 152.0156 | 0.2810 | 0.0980 | 1.1463 | -10.0164 | 8.7473 | -0.1324 | -0.1848 | -0.0831 | -0.0261 | -0.0468 | -0.0088 |
| mcdropout_8 | 612 | 122.5814 | 151.3541 | 0.4003 | 0.1258 | 0.4848 | -1.5476 | 2.5431 | -0.0131 | -0.0209 | -0.0062 | 0.0016 | 0.0000 | 0.0051 |
| tta2 | 612 | 121.1698 | 153.3517 | 0.3513 | 0.1160 | 2.4824 | -10.9418 | 7.2430 | -0.0621 | -0.0909 | -0.0335 | -0.0082 | -0.0209 | 0.0048 |
| tta4 | 612 | 116.1544 | 147.6718 | 0.3219 | 0.1111 | -3.1975 | -16.4921 | 2.6028 | -0.0915 | -0.1399 | -0.0446 | -0.0131 | -0.0316 | 0.0080 |
| tta8 | 612 | 114.8641 | 150.1739 | 0.2908 | 0.1062 | -0.6954 | -21.2316 | 6.5491 | -0.1225 | -0.1785 | -0.0695 | -0.0180 | -0.0327 | -0.0049 |

Percentiles over all cases (an undefined distance counts as a failure); the change is a video-cluster paired bootstrap CI of the difference.

## (e) Sensitivity: all cases, empty predictions penalised

| treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta2 | hd | 29 | 612 | -3.2711 | -7.0913 | 0.5768 | 0.0345 | 0.1381 | tta2 |
| tta2 | hd95 | 29 | 612 | -3.3681 | -7.2065 | 0.4721 | 0.0431 | 0.1381 | tta2 |
| tta2 | hd95_pooled | 29 | 612 | -3.4415 | -7.2199 | 0.4175 | 0.0345 | 0.1381 | tta2 |
| tta2 | assd | 29 | 612 | -0.6427 | -3.7668 | 2.6886 | 0.1373 | 0.1381 | tta2 |
| tta4 | hd | 29 | 612 | -5.4175 | -10.5071 | 0.5443 | 0.0044 | 0.0177 | tta4 |
| tta4 | hd95 | 29 | 612 | -4.7303 | -9.7447 | 1.0938 | 0.0080 | 0.0240 | tta4 |
| tta4 | hd95_pooled | 29 | 612 | -4.0591 | -8.9551 | 1.7128 | 0.0129 | 0.0259 | tta4 |
| tta4 | assd | 29 | 612 | 1.0505 | -2.8449 | 6.0298 | 0.1373 | 0.1373 | baseline |
| tta8 | hd | 29 | 612 | -9.9525 | -15.8789 | -3.8399 | 0.0003 | 0.0012 | tta8 |
| tta8 | hd95 | 29 | 612 | -9.1339 | -14.9816 | -3.1299 | 0.0005 | 0.0015 | tta8 |
| tta8 | hd95_pooled | 29 | 612 | -8.3616 | -14.1154 | -2.4129 | 0.0009 | 0.0018 | tta8 |
| tta8 | assd | 29 | 612 | -1.7164 | -5.9636 | 3.3126 | 0.0455 | 0.0455 | tta8 |

Uses the `*_penalized` columns (empty prediction -> image diagonal), so a method that misses lesions cannot look good by dropping them.

## (f) Do the TTA uncertainty descriptors rank errors?

| seed | predictor | n_cases | n_failures | spearman_vs_hd95 | spearman_vs_hd95_p | spearman_vs_one_minus_dice | spearman_vs_one_minus_dice_p | auroc_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | view_disagreement | 612 | 68 | 0.6876 | 0.0000 | 0.4923 | 0.0000 | 0.5602 |
| -1 | entropy_mean_roi | 612 | 68 | 0.5670 | 0.0000 | 0.8030 | 0.0000 | 0.9095 |

Spearman rho against HD95 and 1-Dice, and AUROC for detecting a failure (Dice < 0.5 or missed detection); row `seed=-1` is the seed-averaged set. `n_cases` is the cohort size, NOT the number of cases used: the ROI descriptors are undefined (NaN) for empty predictions and are silently dropped, which is what the (f-audit) tables below quantify.

## (g) Baseline Dice per fold

| fold | metric | n_seeds | n_cases | n_videos | frame_mean | frame_std | video_mean | video_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dice | 3 | 118 | 6 | 0.7200 | 0.0210 | 0.6397 | 0.0129 |
| 1 | dice | 3 | 137 | 6 | 0.7914 | 0.0366 | 0.7956 | 0.0342 |
| 2 | dice | 3 | 126 | 6 | 0.7727 | 0.0089 | 0.7710 | 0.0095 |
| 3 | dice | 3 | 129 | 6 | 0.7463 | 0.0373 | 0.7249 | 0.0416 |
| 4 | dice | 3 | 102 | 5 | 0.8485 | 0.0203 | 0.8231 | 0.0315 |

Fold-to-fold spread of the control method (seeds averaged).

## Ensemble comparisons - video level, seeds averaged per case

| treatment | metric | n_units | n_cases_used | restriction | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | effect_size | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4 | dice | 29 | 612 | all_cases | 0.0214 | 0.0098 | 0.0329 | 0.0010 | 0.0019 | 0.6736 | tta4 |
| tta4 | jaccard | 29 | 612 | all_cases | 0.0286 | 0.0157 | 0.0413 | 0.0002 | 0.0008 | 0.7471 | tta4 |
| tta4 | precision | 28 | 599 | all_cases | 0.0354 | 0.0240 | 0.0487 | 0.0000 | 0.0000 | 1.0000 | tta4 |
| tta4 | recall | 29 | 612 | all_cases | 0.0064 | -0.0038 | 0.0171 | 0.0308 | 0.0308 | 0.4575 | tta4 |
| tta4 | hd | 28 | 588 | both_nonempty | -8.8857 | -12.3960 | -5.4889 | 0.0000 | 0.0001 | -0.8621 | tta4 |
| tta4 | hd95 | 28 | 588 | both_nonempty | -8.2077 | -11.6058 | -4.8946 | 0.0000 | 0.0002 | -0.8325 | tta4 |
| tta4 | hd95_pooled | 28 | 588 | both_nonempty | -7.5219 | -10.7802 | -4.3586 | 0.0000 | 0.0002 | -0.8276 | tta4 |
| tta4 | assd | 28 | 588 | both_nonempty | -2.4213 | -3.5392 | -1.4248 | 0.0000 | 0.0001 | -0.8571 | tta4 |
| tta4 | boundary_f1 | 28 | 599 | all_cases | 0.0361 | 0.0202 | 0.0509 | 0.0002 | 0.0008 | 0.7537 | tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0218 | 0.0140 | 0.0292 | 0.0000 | 0.0000 | 0.8391 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | 0.0283 | 0.0198 | 0.0364 | 0.0000 | 0.0000 | 0.9034 | ensemble3 |
| ensemble3 | precision | 28 | 594 | all_cases | 0.0292 | 0.0213 | 0.0375 | 0.0000 | 0.0000 | 0.9458 | ensemble3 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0070 | 0.0020 | 0.0121 | 0.0007 | 0.0007 | 0.6874 | ensemble3 |
| ensemble3 | hd | 28 | 594 | both_nonempty | -7.4958 | -9.8681 | -5.2114 | 0.0000 | 0.0000 | -0.9360 | ensemble3 |
| ensemble3 | hd95 | 28 | 594 | both_nonempty | -7.2263 | -9.6929 | -4.8355 | 0.0000 | 0.0000 | -0.9212 | ensemble3 |
| ensemble3 | hd95_pooled | 28 | 594 | both_nonempty | -6.6283 | -8.8419 | -4.4799 | 0.0000 | 0.0000 | -0.9360 | ensemble3 |
| ensemble3 | assd | 28 | 594 | both_nonempty | -1.8486 | -2.6297 | -1.0891 | 0.0001 | 0.0001 | -0.8030 | ensemble3 |
| ensemble3 | boundary_f1 | 28 | 594 | all_cases | 0.0387 | 0.0291 | 0.0484 | 0.0000 | 0.0000 | 0.9606 | ensemble3 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0344 | 0.0204 | 0.0490 | 0.0001 | 0.0001 | 0.7885 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0463 | 0.0307 | 0.0625 | 0.0000 | 0.0000 | 0.8805 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 28 | 590 | all_cases | 0.0533 | 0.0373 | 0.0725 | 0.0000 | 0.0000 | 1.0000 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0111 | -0.0003 | 0.0236 | 0.0121 | 0.0121 | 0.5264 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 28 | 588 | both_nonempty | -12.8105 | -17.1075 | -8.8683 | 0.0000 | 0.0000 | -0.9704 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 28 | 588 | both_nonempty | -12.1431 | -16.3437 | -8.2157 | 0.0000 | 0.0000 | -0.9458 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 28 | 588 | both_nonempty | -11.5225 | -15.5168 | -7.7421 | 0.0000 | 0.0000 | -0.9261 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 28 | 588 | both_nonempty | -3.3884 | -4.5614 | -2.2779 | 0.0000 | 0.0000 | -0.9212 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 28 | 590 | all_cases | 0.0680 | 0.0509 | 0.0853 | 0.0000 | 0.0000 | 0.9852 | ensemble3_tta4 |
| ensemble3_tta4 | dice | 29 | 612 | all_cases | 0.0129 | 0.0068 | 0.0199 | 0.0002 | 0.0013 | 0.7537 | ensemble3_tta4 |
| ensemble3_tta4 | jaccard | 29 | 612 | all_cases | 0.0178 | 0.0112 | 0.0253 | 0.0000 | 0.0000 | 0.8768 | ensemble3_tta4 |
| ensemble3_tta4 | precision | 28 | 590 | all_cases | 0.0179 | 0.0114 | 0.0253 | 0.0000 | 0.0000 | 0.9852 | ensemble3_tta4 |
| ensemble3_tta4 | recall | 29 | 612 | all_cases | 0.0047 | 0.0003 | 0.0094 | 0.0095 | 0.0095 | 0.5517 | ensemble3_tta4 |
| ensemble3_tta4 | hd | 28 | 589 | both_nonempty | -3.9248 | -6.3226 | -1.8959 | 0.0005 | 0.0021 | -0.7143 | ensemble3_tta4 |
| ensemble3_tta4 | hd95 | 28 | 589 | both_nonempty | -3.9354 | -6.2322 | -1.9557 | 0.0004 | 0.0021 | -0.7241 | ensemble3_tta4 |
| ensemble3_tta4 | hd95_pooled | 28 | 589 | both_nonempty | -4.0006 | -6.2303 | -2.0357 | 0.0005 | 0.0021 | -0.7192 | ensemble3_tta4 |
| ensemble3_tta4 | assd | 28 | 589 | both_nonempty | -0.9672 | -1.5283 | -0.4445 | 0.0029 | 0.0058 | -0.6256 | ensemble3_tta4 |
| ensemble3_tta4 | boundary_f1 | 28 | 590 | all_cases | 0.0319 | 0.0233 | 0.0408 | 0.0000 | 0.0000 | 0.9606 | ensemble3_tta4 |
| ensemble3 | dice | 29 | 612 | all_cases | 0.0004 | -0.0096 | 0.0105 | 0.9553 | 1.0000 | -0.0148 | ensemble3 |
| ensemble3 | jaccard | 29 | 612 | all_cases | -0.0003 | -0.0117 | 0.0103 | 0.9732 | 1.0000 | -0.0099 | tta4 |
| ensemble3 | precision | 28 | 594 | all_cases | -0.0062 | -0.0173 | 0.0044 | 0.4932 | 1.0000 | -0.1527 | tta4 |
| ensemble3 | recall | 29 | 612 | all_cases | 0.0006 | -0.0102 | 0.0098 | 0.9019 | 1.0000 | 0.0296 | ensemble3 |
| ensemble3 | hd | 28 | 588 | both_nonempty | 1.3899 | -0.7987 | 3.5122 | 0.2270 | 1.0000 | 0.2660 | tta4 |
| ensemble3 | hd95 | 28 | 588 | both_nonempty | 0.9815 | -1.1197 | 2.9911 | 0.2741 | 1.0000 | 0.2414 | tta4 |
| ensemble3 | hd95_pooled | 28 | 588 | both_nonempty | 0.8936 | -1.1608 | 2.9031 | 0.3161 | 1.0000 | 0.2217 | tta4 |
| ensemble3 | assd | 28 | 588 | both_nonempty | 0.5727 | -0.2410 | 1.3742 | 0.2946 | 1.0000 | 0.2315 | tta4 |
| ensemble3 | boundary_f1 | 28 | 594 | all_cases | 0.0026 | -0.0117 | 0.0182 | 0.9197 | 1.0000 | -0.0246 | ensemble3 |

`ensemble3` and `ensemble3_tta4` against `baseline` and `tta4`.

## (f-audit) ROI diagnostics for failures

| cohort | n | n_empty_prediction | n_near_empty_le_100px | n_entropy_roi_nan | median_area_px | median_entropy_roi | median_entropy_roi_empty_or_near |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_cases | 612 | 22 | 3 | 11 | 6646.0000 | 0.2196 | 0.6494 |
| failures_all | 68 | 22 | 3 | 11 | 1800.5000 | 0.5283 | 0.6494 |
| failures_non_missed | 47 | 2 | 2 | 0 | 4197.0000 | 0.4952 | 0.6188 |
| failures_missed | 21 | 20 | 1 | 11 | 0.0000 | 0.6568 | 0.6568 |
| failures_empty_0px | 22 | 22 | 22 | 11 | 0.0000 | 0.6595 |  |
| failures_small_11_100px | 3 | 0 | 3 | 0 | 17.0000 | 0.5925 |  |
| failures_medium_101_1000px | 4 | 0 | 0 | 0 | 262.0000 | 0.6599 |  |
| failures_large_gt_1000px | 39 | 0 | 0 | 0 | 4977.0000 | 0.4113 |  |

The ROI descriptors are undefined when the predicted mask is empty (`n_entropy_roi_nan`) and can be near zero when it is tiny, so the NaNs would silently drop missed detections from any correlation/AUROC.

## (f-audit) AUROC and AURC for failure detection, three definitions

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures | n_undefined | auroc_if_undefined_certain | auroc_if_undefined_uncertain | aurc_if_missed_is_worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9095 | 0.8216 | 0.9864 | 0.0852 | 0.0628 | 0.1203 | 0.0674 | 601 | 601 | 57 | 11 | 0.7624 | 0.9241 | 0.0852 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8905 | 0.7999 | 0.9819 | 0.0835 | 0.0618 | 0.1188 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.0835 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean | 0.9987 | 0.9960 | 1.0000 | 0.0737 | 0.0570 | 0.0972 | 0.0593 | 554 | 554 | 10 | 11 |  |  | 0.0738 |
| tta4_variance_roi | own | tta4 | all_failures | 0.9056 | 0.8326 | 0.9751 | 0.0862 | 0.0638 | 0.1198 | 0.0674 | 601 | 601 | 57 | 11 | 0.7591 | 0.9209 | 0.0863 |
| tta4_variance_roi | own | tta4 | non_missed_failures | 0.8866 | 0.8070 | 0.9662 | 0.0845 | 0.0629 | 0.1168 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.0845 |
| tta4_variance_roi | own | tta4 | missed_vs_clean | 0.9950 | 0.9888 | 0.9995 | 0.0749 | 0.0589 | 0.0973 | 0.0593 | 554 | 554 | 10 | 11 |  |  | 0.0750 |
| tta4_view_disagreement | own | tta4 | all_failures | 0.5602 | 0.3783 | 0.7047 | 0.1913 | 0.1136 | 0.2903 | 0.0697 | 612 | 612 | 68 | 0 | 0.5602 | 0.5602 | 0.1938 |
| tta4_view_disagreement | own | tta4 | non_missed_failures | 0.7044 | 0.5823 | 0.8523 | 0.1087 | 0.0772 | 0.1512 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.1087 |
| tta4_view_disagreement | own | tta4 | missed_vs_clean | 0.2375 | 0.0804 | 0.4513 | 0.1771 | 0.1020 | 0.2714 | 0.0612 | 565 | 565 | 21 | 0 |  |  | 0.1796 |
| tta4_entropy_image | own | tta4 | all_failures | 0.4902 | 0.2732 | 0.7509 | 0.2027 | 0.1136 | 0.3234 | 0.0697 | 612 | 612 | 68 | 0 | 0.4902 | 0.4902 | 0.2040 |
| tta4_entropy_image | own | tta4 | non_missed_failures | 0.5432 | 0.2787 | 0.8366 | 0.1352 | 0.0855 | 0.2259 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.1352 |
| tta4_entropy_image | own | tta4 | missed_vs_clean | 0.3718 | 0.1272 | 0.7000 | 0.1723 | 0.0960 | 0.2763 | 0.0612 | 565 | 565 | 21 | 0 |  |  | 0.1738 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.8811 | 0.8030 | 0.9814 | 0.0754 | 0.0558 | 0.1073 | 0.0549 | 590 | 590 | 38 | 22 | 0.5580 | 0.9247 | 0.0754 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.8811 | 0.8030 | 0.9814 | 0.0754 | 0.0558 | 0.1073 | 0.0549 | 590 | 590 | 38 | 0 |  |  | 0.0754 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0666 | 0.0513 | 0.0888 | 0.0491 | 552 | 552 | 0 | 22 |  |  | 0.0666 |
| map_entropy_roi_tta4 | tta4 | tta4 | all_failures | 0.8971 | 0.8191 | 0.9856 | 0.0845 | 0.0631 | 0.1185 | 0.0652 | 590 | 590 | 46 | 22 | 0.6068 | 0.9304 | 0.0845 |
| map_entropy_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8948 | 0.8113 | 0.9856 | 0.0844 | 0.0631 | 0.1180 | 0.0651 | 589 | 589 | 45 | 2 |  |  | 0.0844 |
| map_entropy_roi_tta4 | tta4 | tta4 | missed_vs_clean | 0.9982 | 0.9939 | 1.0000 | 0.0743 | 0.0579 | 0.0978 | 0.0579 | 545 | 545 | 1 | 20 |  |  | 0.0743 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | all_failures | 0.8280 | 0.7605 | 0.9352 | 0.0807 | 0.0591 | 0.1144 | 0.0549 | 590 | 590 | 38 | 22 | 0.5244 | 0.8911 | 0.0807 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.8280 | 0.7605 | 0.9352 | 0.0807 | 0.0591 | 0.1144 | 0.0549 | 590 | 590 | 38 | 0 |  |  | 0.0807 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0697 | 0.0549 | 0.0915 | 0.0491 | 552 | 552 | 0 | 22 |  |  | 0.0697 |
| map_variance_roi_tta4 | tta4 | tta4 | all_failures | 0.8530 | 0.7866 | 0.9467 | 0.0896 | 0.0669 | 0.1247 | 0.0652 | 590 | 590 | 46 | 22 | 0.5770 | 0.9005 | 0.0896 |
| map_variance_roi_tta4 | tta4 | tta4 | non_missed_failures | 0.8509 | 0.7797 | 0.9467 | 0.0894 | 0.0668 | 0.1239 | 0.0651 | 589 | 589 | 45 | 2 |  |  | 0.0894 |
| map_variance_roi_tta4 | tta4 | tta4 | missed_vs_clean | 0.9485 | 0.9010 | 0.9817 | 0.0776 | 0.0615 | 0.1008 | 0.0579 | 545 | 545 | 1 | 20 |  |  | 0.0776 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | all_failures | 0.4466 | 0.2754 | 0.6318 | 0.2041 | 0.1167 | 0.3141 | 0.0592 | 612 | 612 | 60 | 0 | 0.4466 | 0.4466 | 0.2041 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | non_missed_failures | 0.5340 | 0.3658 | 0.7572 | 0.1187 | 0.0759 | 0.1910 | 0.0549 | 590 | 590 | 38 | 0 |  |  | 0.1187 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | missed_vs_clean | 0.2957 | 0.0681 | 0.5944 | 0.1784 | 0.0964 | 0.2823 | 0.0526 | 574 | 574 | 22 | 0 |  |  | 0.1784 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | all_failures | 0.4909 | 0.2998 | 0.6923 | 0.2090 | 0.1216 | 0.3157 | 0.0697 | 612 | 612 | 68 | 0 | 0.4909 | 0.4909 | 0.2114 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | non_missed_failures | 0.6033 | 0.3988 | 0.8239 | 0.1235 | 0.0826 | 0.1874 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.1235 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | missed_vs_clean | 0.2393 | 0.0468 | 0.5164 | 0.1860 | 0.1020 | 0.2894 | 0.0612 | 565 | 565 | 21 | 0 |  |  | 0.1885 |
| pred_area_px | own | ensemble3_tta4 | all_failures | 0.1426 | 0.0441 | 0.2431 | 0.3270 | 0.2359 | 0.4261 | 0.0592 | 612 | 612 | 60 | 0 | 0.1426 | 0.1426 | 0.3270 |
| pred_area_px | own | ensemble3_tta4 | non_missed_failures | 0.2251 | 0.1108 | 0.3222 | 0.2204 | 0.1463 | 0.2971 | 0.0549 | 590 | 590 | 38 | 0 |  |  | 0.2204 |
| pred_area_px | own | ensemble3_tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.2661 | 0.1827 | 0.3578 | 0.0526 | 574 | 574 | 22 | 0 |  |  | 0.2661 |
| pred_area_px | tta4 | tta4 | all_failures | 0.1819 | 0.0706 | 0.2828 | 0.3269 | 0.2407 | 0.4228 | 0.0697 | 612 | 612 | 68 | 0 | 0.1819 | 0.1819 | 0.3368 |
| pred_area_px | tta4 | tta4 | non_missed_failures | 0.2632 | 0.1395 | 0.3725 | 0.2353 | 0.1676 | 0.3074 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.2353 |
| pred_area_px | tta4 | tta4 | missed_vs_clean | 0.0000 | 0.0000 | 0.0000 | 0.2663 | 0.1903 | 0.3552 | 0.0612 | 565 | 565 | 21 | 0 |  |  | 0.2770 |
| one_minus_confidence | own | ensemble3_tta4 | all_failures | 0.8833 | 0.8046 | 0.9835 | 0.0741 | 0.0547 | 0.1053 | 0.0549 | 590 | 590 | 38 | 22 | 0.5595 | 0.9261 | 0.0741 |
| one_minus_confidence | own | ensemble3_tta4 | non_missed_failures | 0.8833 | 0.8046 | 0.9835 | 0.0741 | 0.0547 | 0.1053 | 0.0549 | 590 | 590 | 38 | 0 |  |  | 0.0741 |
| one_minus_confidence | own | ensemble3_tta4 | missed_vs_clean |  |  |  | 0.0652 | 0.0503 | 0.0865 | 0.0491 | 552 | 552 | 0 | 22 |  |  | 0.0652 |
| one_minus_confidence | tta4 | tta4 | all_failures | 0.9001 | 0.8266 | 0.9867 | 0.0833 | 0.0623 | 0.1163 | 0.0652 | 590 | 590 | 46 | 22 | 0.6089 | 0.9324 | 0.0833 |
| one_minus_confidence | tta4 | tta4 | non_missed_failures | 0.8979 | 0.8193 | 0.9867 | 0.0832 | 0.0622 | 0.1160 | 0.0651 | 589 | 589 | 45 | 2 |  |  | 0.0832 |
| one_minus_confidence | tta4 | tta4 | missed_vs_clean | 1.0000 | 1.0000 | 1.0000 | 0.0731 | 0.0572 | 0.0958 | 0.0579 | 545 | 545 | 1 | 20 |  |  | 0.0731 |
| baseline_entropy_roi | own | baseline | all_failures | 0.8814 | 0.8024 | 0.9681 | 0.1057 | 0.0774 | 0.1457 | 0.0829 | 601 | 601 | 65 | 11 | 0.7538 | 0.8985 | 0.1058 |
| baseline_entropy_roi | own | baseline | non_missed_failures | 0.8680 | 0.7922 | 0.9646 | 0.1045 | 0.0773 | 0.1450 | 0.0814 | 594 | 594 | 58 | 0 |  |  | 0.1045 |
| baseline_entropy_roi | own | baseline | missed_vs_clean | 0.9920 | 0.9714 | 1.0000 | 0.0903 | 0.0703 | 0.1159 | 0.0721 | 543 | 543 | 7 | 11 |  |  | 0.0903 |
| baseline_entropy_roi | tta4 | tta4 | all_failures | 0.8894 | 0.7892 | 0.9831 | 0.0909 | 0.0657 | 0.1295 | 0.0674 | 601 | 601 | 57 | 11 | 0.7456 | 0.9073 | 0.0910 |
| baseline_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8679 | 0.7580 | 0.9769 | 0.0892 | 0.0653 | 0.1282 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.0892 |
| baseline_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9906 | 0.9804 | 1.0000 | 0.0781 | 0.0593 | 0.1020 | 0.0593 | 554 | 554 | 10 | 11 |  |  | 0.0782 |
| baseline_entropy_image | own | baseline | all_failures | 0.4670 | 0.2939 | 0.6716 | 0.2215 | 0.1341 | 0.3377 | 0.0853 | 612 | 612 | 76 | 0 | 0.4670 | 0.4670 | 0.2231 |
| baseline_entropy_image | own | baseline | non_missed_failures | 0.5052 | 0.3061 | 0.7457 | 0.1614 | 0.1064 | 0.2556 | 0.0814 | 594 | 594 | 58 | 0 |  |  | 0.1614 |
| baseline_entropy_image | own | baseline | missed_vs_clean | 0.3439 | 0.1113 | 0.6631 | 0.1835 | 0.1103 | 0.2788 | 0.0742 | 554 | 554 | 18 | 0 |  |  | 0.1852 |
| baseline_entropy_image | tta4 | tta4 | all_failures | 0.4828 | 0.2744 | 0.7252 | 0.2029 | 0.1165 | 0.3239 | 0.0697 | 612 | 612 | 68 | 0 | 0.4828 | 0.4828 | 0.2041 |
| baseline_entropy_image | tta4 | tta4 | non_missed_failures | 0.5187 | 0.2696 | 0.8029 | 0.1390 | 0.0888 | 0.2282 | 0.0654 | 591 | 591 | 47 | 0 |  |  | 0.1390 |
| baseline_entropy_image | tta4 | tta4 | missed_vs_clean | 0.4025 | 0.1522 | 0.7296 | 0.1709 | 0.0969 | 0.2711 | 0.0612 | 565 | 565 | 21 | 0 |  |  | 0.1721 |
| ensemble3_entropy_roi | own | ensemble3 | all_failures | 0.8962 | 0.8307 | 0.9786 | 0.0838 | 0.0619 | 0.1191 | 0.0627 | 594 | 594 | 49 | 18 | 0.6554 | 0.9241 | 0.0838 |
| ensemble3_entropy_roi | own | ensemble3 | non_missed_failures | 0.8962 | 0.8307 | 0.9786 | 0.0838 | 0.0619 | 0.1191 | 0.0627 | 594 | 594 | 49 | 0 |  |  | 0.0838 |
| ensemble3_entropy_roi | own | ensemble3 | missed_vs_clean |  |  |  | 0.0725 | 0.0562 | 0.0950 | 0.0547 | 545 | 545 | 0 | 18 |  |  | 0.0725 |
| ensemble3_entropy_roi | tta4 | tta4 | all_failures | 0.9039 | 0.8350 | 0.9824 | 0.0855 | 0.0636 | 0.1197 | 0.0660 | 594 | 594 | 51 | 18 | 0.6783 | 0.9264 | 0.0856 |
| ensemble3_entropy_roi | tta4 | tta4 | non_missed_failures | 0.8923 | 0.8196 | 0.9786 | 0.0845 | 0.0630 | 0.1179 | 0.0649 | 588 | 588 | 45 | 3 |  |  | 0.0845 |
| ensemble3_entropy_roi | tta4 | tta4 | missed_vs_clean | 0.9908 | 0.9758 | 1.0000 | 0.0748 | 0.0587 | 0.0978 | 0.0585 | 549 | 549 | 6 | 16 |  |  | 0.0748 |

`label_source=own` scores each predictor against the failures of **its own** prediction (baseline descriptor vs baseline failures, tta4 vs tta4, ensemble3 vs ensemble3, seed-averaged tta4 map vs ensemble3_tta4); `label_source=tta4` repeats the audit against the shared tta4 labels for comparison. `all_failures` = Dice < 0.5 or missed; `non_missed_failures` excludes missed detections (every predictor is defined there); `missed_vs_clean` contrasts missed detections with clean cases only. `n_used`/`n_undefined` make the NaN drop visible and the two AUROC bounds score undefined cases as maximally certain/uncertain. **AURC** is the area under the risk-coverage curve with risk = 1 - Dice (lower = better); `aurc_optimal` is the same cohort sorted by its own risk (the oracle lower bound) and `aurc_if_missed_is_worst` repeats the point estimate with missed detections counted as risk 1 instead of being dropped. CIs resample videos (29 clusters).

## (f-audit) Agreement detectors: pairwise Dice between views/seeds

| predictor | label_source | label_prediction | failure_definition | auroc | ci_low | ci_high | aurc | aurc_ci_low | aurc_ci_high | aurc_optimal | n_used | n_used_aurc | n_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | all_failures | 0.9095 | 0.8216 | 0.9864 | 0.0852 | 0.0628 | 0.1203 | 0.0674 | 601 | 601 | 57 |
| tta4_entropy_roi | own | tta4 | non_missed_failures | 0.8905 | 0.7999 | 0.9819 | 0.0835 | 0.0618 | 0.1188 | 0.0654 | 591 | 591 | 47 |
| tta4_entropy_roi | own | tta4 | missed_vs_clean | 0.9987 | 0.9960 | 1.0000 | 0.0737 | 0.0570 | 0.0972 | 0.0593 | 554 | 554 | 10 |
| tta4_view_dice_disagreement | own | tta4 | all_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | non_missed_failures |  |  |  |  |  |  |  | 0 | 0 | 0 |
| tta4_view_dice_disagreement | own | tta4 | missed_vs_clean |  |  |  |  |  |  |  | 0 | 0 | 0 |
| ensemble3_dice_disagreement | own | ensemble3 | all_failures | 0.7329 | 0.6002 | 0.8283 | 0.1730 | 0.1145 | 0.2464 | 0.0665 | 612 | 612 | 67 |
| ensemble3_dice_disagreement | own | ensemble3 | non_missed_failures | 0.8616 | 0.7916 | 0.9647 | 0.0877 | 0.0631 | 0.1228 | 0.0627 | 594 | 594 | 49 |
| ensemble3_dice_disagreement | own | ensemble3 | missed_vs_clean | 0.3825 | 0.1646 | 0.5948 | 0.1661 | 0.1043 | 0.2443 | 0.0576 | 563 | 563 | 18 |
| ensemble3_dice_disagreement | tta4 | tta4 | all_failures | 0.7433 | 0.6026 | 0.8329 | 0.1701 | 0.1129 | 0.2436 | 0.0697 | 612 | 612 | 68 |
| ensemble3_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8651 | 0.7787 | 0.9657 | 0.0872 | 0.0639 | 0.1207 | 0.0654 | 591 | 591 | 47 |
| ensemble3_dice_disagreement | tta4 | tta4 | missed_vs_clean | 0.4707 | 0.2489 | 0.6932 | 0.1631 | 0.1018 | 0.2394 | 0.0612 | 565 | 565 | 21 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | all_failures | 0.7232 | 0.4842 | 0.8777 | 0.1634 | 0.0912 | 0.2555 | 0.0592 | 612 | 612 | 60 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | non_missed_failures | 0.8555 | 0.7739 | 0.9736 | 0.0770 | 0.0554 | 0.1113 | 0.0549 | 590 | 590 | 38 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | missed_vs_clean | 0.4947 | 0.1413 | 0.8603 | 0.1576 | 0.0828 | 0.2523 | 0.0526 | 574 | 574 | 22 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | all_failures | 0.7495 | 0.5192 | 0.8994 | 0.1682 | 0.0960 | 0.2613 | 0.0697 | 612 | 612 | 68 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | non_missed_failures | 0.8728 | 0.7701 | 0.9830 | 0.0821 | 0.0600 | 0.1167 | 0.0654 | 591 | 591 | 47 |
| ensemble3_tta4_dice_disagreement | tta4 | tta4 | missed_vs_clean | 0.4736 | 0.1527 | 0.8532 | 0.1618 | 0.0867 | 0.2585 | 0.0612 | 565 | 565 | 21 |

`tta4_view_dice_disagreement` = 1 - mean pairwise Dice between the four tta4 views (per seed, averaged over seeds); `ensemble3_dice_disagreement` and `ensemble3_tta4_dice_disagreement` are the same over the three seeds' baseline / tta4 maps. Higher = the members disagree = more uncertain. The tta4 entropy row is kept as the reference descriptor.

## (f-audit) Spearman against HD95 and 1 - Dice (all predictors)

| predictor | label_source | label_prediction | target | n_used | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | own | tta4 | hd95 | 601 | 0.5670 | 0.0000 |
| tta4_entropy_roi | own | tta4 | one_minus_dice | 601 | 0.8030 | 0.0000 |
| tta4_variance_roi | own | tta4 | hd95 | 601 | 0.6771 | 0.0000 |
| tta4_variance_roi | own | tta4 | one_minus_dice | 601 | 0.7912 | 0.0000 |
| tta4_view_disagreement | own | tta4 | hd95 | 601 | 0.6876 | 0.0000 |
| tta4_view_disagreement | own | tta4 | one_minus_dice | 612 | 0.4923 | 0.0000 |
| tta4_entropy_image | own | tta4 | hd95 | 601 | 0.2680 | 0.0000 |
| tta4_entropy_image | own | tta4 | one_minus_dice | 612 | 0.3057 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.5282 | 0.0000 |
| map_entropy_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 590 | 0.6984 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | hd95 | 590 | 0.5459 | 0.0000 |
| map_entropy_roi_tta4 | tta4 | tta4 | one_minus_dice | 590 | 0.7720 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.5609 | 0.0000 |
| map_variance_roi_tta4 | own | ensemble3_tta4 | one_minus_dice | 590 | 0.6533 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | hd95 | 590 | 0.5865 | 0.0000 |
| map_variance_roi_tta4 | tta4 | tta4 | one_minus_dice | 590 | 0.7384 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | hd95 | 590 | 0.3887 | 0.0000 |
| map_ambiguous_fraction_tta4 | own | ensemble3_tta4 | one_minus_dice | 612 | 0.3054 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | hd95 | 601 | 0.4669 | 0.0000 |
| map_ambiguous_fraction_tta4 | tta4 | tta4 | one_minus_dice | 612 | 0.3807 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | hd95 | 590 | 0.2024 | 0.0000 |
| pred_area_px | own | ensemble3_tta4 | one_minus_dice | 612 | -0.2209 | 0.0000 |
| pred_area_px | tta4 | tta4 | hd95 | 601 | 0.1596 | 0.0001 |
| pred_area_px | tta4 | tta4 | one_minus_dice | 612 | -0.2316 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | hd95 | 590 | 0.5773 | 0.0000 |
| one_minus_confidence | own | ensemble3_tta4 | one_minus_dice | 590 | 0.7218 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | hd95 | 590 | 0.5923 | 0.0000 |
| one_minus_confidence | tta4 | tta4 | one_minus_dice | 590 | 0.7913 | 0.0000 |
| baseline_entropy_roi | own | baseline | hd95 | 601 | 0.4655 | 0.0000 |
| baseline_entropy_roi | own | baseline | one_minus_dice | 601 | 0.7749 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | hd95 | 599 | 0.4724 | 0.0000 |
| baseline_entropy_roi | tta4 | tta4 | one_minus_dice | 601 | 0.7453 | 0.0000 |
| baseline_entropy_image | own | baseline | hd95 | 601 | 0.1701 | 0.0000 |
| baseline_entropy_image | own | baseline | one_minus_dice | 612 | 0.2576 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | hd95 | 601 | 0.2001 | 0.0000 |
| baseline_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.2662 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | hd95 | 594 | 0.5474 | 0.0000 |
| ensemble3_entropy_roi | own | ensemble3 | one_minus_dice | 594 | 0.7394 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | hd95 | 594 | 0.5558 | 0.0000 |
| ensemble3_entropy_roi | tta4 | tta4 | one_minus_dice | 594 | 0.7844 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | hd95 | 594 | 0.6422 | 0.0000 |
| ensemble3_variance_roi | own | ensemble3 | one_minus_dice | 594 | 0.7210 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | hd95 | 594 | 0.6413 | 0.0000 |
| ensemble3_variance_roi | tta4 | tta4 | one_minus_dice | 594 | 0.7570 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | hd95 | 594 | 0.1779 | 0.0000 |
| ensemble3_entropy_image | own | ensemble3 | one_minus_dice | 612 | 0.2270 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | hd95 | 601 | 0.2428 | 0.0000 |
| ensemble3_entropy_image | tta4 | tta4 | one_minus_dice | 612 | 0.2872 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | hd95 | 594 | 0.5248 | 0.0000 |
| ensemble3_variance_image | own | ensemble3 | one_minus_dice | 612 | 0.4101 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | hd95 | 601 | 0.5782 | 0.0000 |
| ensemble3_variance_image | tta4 | tta4 | one_minus_dice | 612 | 0.4670 | 0.0000 |
| tta4_view_dice_disagreement | own | tta4 | hd95 | 0 |  |  |
| tta4_view_dice_disagreement | own | tta4 | one_minus_dice | 0 |  |  |
| ensemble3_dice_disagreement | own | ensemble3 | hd95 | 594 | 0.5722 | 0.0000 |
| ensemble3_dice_disagreement | own | ensemble3 | one_minus_dice | 612 | 0.6126 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | hd95 | 601 | 0.6104 | 0.0000 |
| ensemble3_dice_disagreement | tta4 | tta4 | one_minus_dice | 612 | 0.6669 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | hd95 | 590 | 0.5867 | 0.0000 |
| ensemble3_tta4_dice_disagreement | own | ensemble3_tta4 | one_minus_dice | 612 | 0.6312 | 0.0000 |

Targets also come from the predictor's own prediction (`label_source=own`); the `tta4` rows repeat the correlation against the tta4 prediction for comparison. `n_used` counts the finite pairs, so ROI descriptors lose the missed detections here.

## (f-audit) detector vs detector on AUROC and AURC (paired bootstrap)

| predictor | compared_with | label_prediction | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tta4_entropy_roi | tta4_variance_roi | tta4 | 0.0039 | -0.0224 | 0.0318 | 0.6620 | -0.0010 | -0.0052 | 0.0041 | 0.6480 | 591 |
| tta4_entropy_roi | tta4_view_disagreement | tta4 | 0.1861 | 0.1032 | 0.2623 | 0.0000 | -0.0252 | -0.0391 | -0.0131 | 0.0000 | 591 |
| tta4_entropy_roi | tta4_entropy_image | tta4 | 0.3473 | 0.1362 | 0.5453 | 0.0000 | -0.0517 | -0.1127 | -0.0193 | 0.0000 | 591 |
| tta4_entropy_roi | map_entropy_roi_tta4 | tta4 | -0.0091 | -0.0221 | 0.0072 | 0.2440 | -0.0012 | -0.0034 | 0.0010 | 0.2860 | 589 |
| tta4_entropy_roi | map_entropy_roi_tta4 | ensemble3_tta4 | -0.0019 | -0.0143 | 0.0107 | 0.7220 | 0.0010 | -0.0003 | 0.0026 | 0.1420 | 590 |
| tta4_entropy_roi | map_variance_roi_tta4 | tta4 | 0.0348 | -0.0061 | 0.0948 | 0.0780 | -0.0062 | -0.0125 | -0.0017 | 0.0080 | 589 |
| tta4_entropy_roi | map_variance_roi_tta4 | ensemble3_tta4 | 0.0512 | 0.0132 | 0.1136 | 0.0060 | -0.0042 | -0.0078 | -0.0011 | 0.0220 | 590 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | tta4 | 0.2872 | 0.1514 | 0.4402 | 0.0000 | -0.0400 | -0.0751 | -0.0178 | 0.0000 | 591 |
| tta4_entropy_roi | map_ambiguous_fraction_tta4 | ensemble3_tta4 | 0.3453 | 0.2097 | 0.4720 | 0.0000 | -0.0422 | -0.0865 | -0.0164 | 0.0000 | 590 |
| tta4_entropy_roi | pred_area_px | tta4 | 0.6274 | 0.4514 | 0.8247 | 0.0000 | -0.1518 | -0.2014 | -0.0988 | 0.0000 | 591 |
| tta4_entropy_roi | pred_area_px | ensemble3_tta4 | 0.6541 | 0.4997 | 0.8555 | 0.0000 | -0.1439 | -0.2003 | -0.0837 | 0.0000 | 590 |
| tta4_entropy_roi | one_minus_confidence | tta4 | -0.0122 | -0.0297 | 0.0082 | 0.2180 | 0.0000 | -0.0030 | 0.0030 | 0.9420 | 589 |
| tta4_entropy_roi | one_minus_confidence | ensemble3_tta4 | -0.0041 | -0.0180 | 0.0110 | 0.5840 | 0.0024 | 0.0007 | 0.0044 | 0.0040 | 590 |
| tta4_entropy_roi | baseline_entropy_roi | tta4 | 0.0226 | -0.0117 | 0.0571 | 0.2840 | -0.0057 | -0.0126 | -0.0017 | 0.0060 | 591 |
| tta4_entropy_roi | baseline_entropy_roi | baseline | 0.0202 | -0.0148 | 0.0534 | 0.3540 | -0.0025 | -0.0090 | 0.0017 | 0.3400 | 594 |
| tta4_entropy_roi | baseline_entropy_image | tta4 | 0.3719 | 0.1648 | 0.5651 | 0.0000 | -0.0555 | -0.1189 | -0.0218 | 0.0000 | 591 |
| tta4_entropy_roi | baseline_entropy_image | baseline | 0.3830 | 0.1990 | 0.5454 | 0.0000 | -0.0594 | -0.1260 | -0.0233 | 0.0000 | 594 |
| tta4_entropy_roi | ensemble3_entropy_roi | tta4 | -0.0052 | -0.0305 | 0.0182 | 0.7520 | -0.0014 | -0.0043 | 0.0018 | 0.2960 | 588 |
| tta4_entropy_roi | ensemble3_entropy_roi | ensemble3 | -0.0147 | -0.0356 | 0.0046 | 0.1400 | 0.0015 | -0.0017 | 0.0049 | 0.4120 | 594 |
| tta4_entropy_roi | ensemble3_variance_roi | tta4 | 0.0066 | -0.0515 | 0.0763 | 0.7840 | -0.0035 | -0.0101 | 0.0029 | 0.2720 | 588 |
| tta4_entropy_roi | ensemble3_variance_roi | ensemble3 | -0.0135 | -0.0518 | 0.0413 | 0.6300 | -0.0007 | -0.0066 | 0.0056 | 0.7200 | 594 |
| tta4_entropy_roi | ensemble3_entropy_image | tta4 | 0.3558 | 0.1462 | 0.5481 | 0.0000 | -0.0570 | -0.1224 | -0.0212 | 0.0000 | 591 |
| tta4_entropy_roi | ensemble3_entropy_image | ensemble3 | 0.4078 | 0.2264 | 0.5388 | 0.0000 | -0.0601 | -0.1316 | -0.0218 | 0.0000 | 594 |
| tta4_entropy_roi | ensemble3_variance_image | tta4 | 0.2108 | 0.1333 | 0.2829 | 0.0000 | -0.0317 | -0.0588 | -0.0142 | 0.0000 | 591 |
| tta4_entropy_roi | ensemble3_variance_image | ensemble3 | 0.2268 | 0.1202 | 0.3132 | 0.0000 | -0.0345 | -0.0697 | -0.0135 | 0.0000 | 594 |
| tta4_entropy_roi | ensemble3_dice_disagreement | tta4 | 0.0254 | -0.0061 | 0.0529 | 0.1020 | -0.0037 | -0.0097 | 0.0014 | 0.1420 | 591 |
| tta4_entropy_roi | ensemble3_dice_disagreement | ensemble3 | 0.0199 | -0.0103 | 0.0474 | 0.1940 | -0.0025 | -0.0087 | 0.0030 | 0.3760 | 594 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | tta4 | 0.0177 | -0.0133 | 0.0494 | 0.3440 | 0.0014 | -0.0023 | 0.0047 | 0.4600 | 591 |
| tta4_entropy_roi | ensemble3_tta4_dice_disagreement | ensemble3_tta4 | 0.0238 | -0.0068 | 0.0502 | 0.1260 | -0.0005 | -0.0050 | 0.0032 | 0.8220 | 590 |
| ensemble3_dice_disagreement | ensemble3_entropy_roi | ensemble3 | -0.0346 | -0.0562 | -0.0102 | 0.0040 | 0.0040 | -0.0014 | 0.0105 | 0.1320 | 594 |
| ensemble3_tta4_dice_disagreement | map_entropy_roi_tta4 | ensemble3_tta4 | -0.0256 | -0.0551 | 0.0065 | 0.1140 | 0.0015 | -0.0026 | 0.0071 | 0.4960 | 590 |

Definition `non_missed_failures`, one shared video-cluster resampling for both statistics. `delta_auroc` is positive when the first predictor is the better classifier; `delta_aurc = AURC(first) - AURC(second)` is a difference of selective risks, so it is **negative** when the first predictor is the better detector. The `tta4_entropy_roi` rows compare the reference descriptor with every alternative (under the tta4 labels and under the alternative's own labels); the `tta4_view_dice_disagreement` rows are the required comparisons of the view-pairwise-Dice detector against the single-view ROI entropy and against 1 - confidence.

## Audit note

failures per prediction (Dice < 0.5 or missed): tta4 68, baseline 76, ensemble3 67, ensemble3_tta4 60

## Audit note

tta4 shares 68 failures; 11 of them have no ROI value (empty prediction), 25 have a ROI of <= 100 px.

## Audit note

AURC uses risk = 1 - Dice of the prediction a descriptor is scored against (lower is better); `aurc_optimal` orders the same cases by their own risk, so `aurc_excess` is what the ranking loses against the oracle. Missed detections have no Dice and are dropped (`n_undefined_aurc`); `aurc_if_missed_is_worst` repeats the point estimate with their risk set to 1.

## Audit note

view pairwise Dice unavailable for 612 of 612 cases: the runner does not store per-view masks. Re-run tta4 on CPU with configs/cvc_clinicdb/unet_resnet50_view_agreement.yaml so the `view_pairwise_dice` metric exists, then pass it as `view_agreement`.

## Audit note

seed agreement: 612 cases have a pairwise Dice between the three seeds' baseline maps (NaN when an ensemble has one seed); `ensemble3_tta4_pairwise_dice` is the same over the three tta4 maps.

## Audit note

validation: recomputed tta4 ROI entropy matches the runner's CSV on 5 case(s), max |diff| = 1.98e-04
