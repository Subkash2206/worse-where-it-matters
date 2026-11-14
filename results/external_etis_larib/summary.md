# MELBA external validation - ETIS-LaribPolypDB

External dataset `ETIS-LaribPolypDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/ETIS-LaribPolypDB/images`) scored by 15 internal checkpoints. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 196 | 76.0696 | 8.1630 | 66.4970 | 93.0119 | 26.5150 | external |
| baseline | boundary_f1 | 15 | 196 | 0.1569 | 0.0111 | 0.1366 | 0.1744 | 0.0378 | external |
| baseline | dice | 15 | 196 | 0.6906 | 0.0165 | 0.6646 | 0.7169 | 0.0522 | external |
| baseline | hd95 | 15 | 196 | 222.4704 | 27.3310 | 171.9760 | 275.0081 | 103.0321 | external |
| tta2 | assd | 15 | 196 | 72.6494 | 7.2721 | 61.3297 | 82.7466 | 21.4168 | external |
| tta2 | boundary_f1 | 15 | 196 | 0.1710 | 0.0153 | 0.1479 | 0.2028 | 0.0549 | external |
| tta2 | dice | 15 | 196 | 0.7019 | 0.0184 | 0.6705 | 0.7396 | 0.0691 | external |
| tta2 | hd95 | 15 | 196 | 214.5053 | 22.7157 | 182.8265 | 252.8208 | 69.9943 | external |
| tta4 | assd | 15 | 196 | 69.3023 | 6.9586 | 58.5539 | 83.6393 | 25.0853 | external |
| tta4 | boundary_f1 | 15 | 196 | 0.1866 | 0.0168 | 0.1545 | 0.2107 | 0.0562 | external |
| tta4 | dice | 15 | 196 | 0.7128 | 0.0190 | 0.6715 | 0.7442 | 0.0727 | external |
| tta4 | hd95 | 15 | 196 | 208.0047 | 21.4395 | 176.1468 | 246.4886 | 70.3418 | external |
| tta8 | assd | 15 | 196 | 64.8592 | 6.1390 | 57.2190 | 78.3621 | 21.1431 | external |
| tta8 | boundary_f1 | 15 | 196 | 0.1933 | 0.0190 | 0.1453 | 0.2200 | 0.0747 | external |
| tta8 | dice | 15 | 196 | 0.7209 | 0.0163 | 0.6865 | 0.7501 | 0.0635 | external |
| tta8 | hd95 | 15 | 196 | 199.7927 | 23.1604 | 157.8620 | 234.4222 | 76.5602 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.6906 | 0.7019 | 0.0113 | 0.0050 | 0.0065 | 0.0161 | 0.4050 | rank_biserial | 5512.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 4.5381 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 196 | both_nonempty | 196 | 196 | 0 | 223.0989 | 215.5564 | -7.5425 | -2.4898 | -12.9222 | -2.0226 | -0.3361 | rank_biserial | 6409.0000 | wilcoxon | 0.0000 | 0.0002 | tta2 | two-sided | -2.6929 | 0.0077 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.6906 | 0.7128 | 0.0222 | 0.0140 | 0.0156 | 0.0289 | 0.6006 | rank_biserial | 3700.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 6.5503 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 196 | both_nonempty | 196 | 196 | 0 | 223.0989 | 208.8853 | -14.2136 | -4.8844 | -21.9477 | -6.5485 | -0.3923 | rank_biserial | 5866.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -3.6826 | 0.0003 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.6906 | 0.7209 | 0.0303 | 0.0158 | 0.0223 | 0.0385 | 0.6215 | rank_biserial | 3543.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 7.2460 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 196 | both_nonempty | 196 | 196 | 0 | 223.0989 | 200.5103 | -22.5886 | -7.1510 | -31.9632 | -13.5374 | -0.4681 | rank_biserial | 5134.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -4.8311 | 0.0000 | t_test_rel |

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 196 | 196 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 196 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 196 | 196 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 196 | 196.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta4 | baseline | 196 | 196 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 196 | 196.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta8 | baseline | 196 | 196 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 196 | 196.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
