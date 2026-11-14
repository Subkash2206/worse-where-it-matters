# MELBA external validation - CVC-ColonDB

External dataset `CVC-ColonDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/CVC-ColonDB/images`) scored by 15 internal checkpoints over dice, jaccard, precision, recall, hd, hd95, hd95_pooled, assd, boundary_f1, pred_volume_mm3. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 380 | 28.9855 | 1.9583 | 25.3041 | 31.7120 | 6.4079 | external |
| baseline | boundary_f1 | 15 | 380 | 0.2930 | 0.0130 | 0.2641 | 0.3132 | 0.0491 | external |
| baseline | dice | 15 | 380 | 0.7379 | 0.0079 | 0.7270 | 0.7564 | 0.0294 | external |
| baseline | hd95 | 15 | 380 | 85.5968 | 6.9564 | 76.7999 | 101.5936 | 24.7937 | external |
| tta2 | assd | 15 | 380 | 28.2660 | 1.6389 | 25.3522 | 31.0441 | 5.6919 | external |
| tta2 | boundary_f1 | 15 | 380 | 0.2998 | 0.0125 | 0.2693 | 0.3178 | 0.0485 | external |
| tta2 | dice | 15 | 380 | 0.7448 | 0.0086 | 0.7322 | 0.7609 | 0.0287 | external |
| tta2 | hd95 | 15 | 380 | 83.1759 | 6.1035 | 76.8903 | 101.1256 | 24.2353 | external |
| tta4 | assd | 15 | 380 | 27.1377 | 1.9329 | 23.4000 | 30.1826 | 6.7826 | external |
| tta4 | boundary_f1 | 15 | 380 | 0.3142 | 0.0151 | 0.2753 | 0.3348 | 0.0595 | external |
| tta4 | dice | 15 | 380 | 0.7511 | 0.0091 | 0.7358 | 0.7652 | 0.0294 | external |
| tta4 | hd95 | 15 | 380 | 79.1443 | 6.5854 | 70.5686 | 96.8597 | 26.2910 | external |
| tta4_anyview | assd | 15 | 380 | 28.2943 | 1.5564 | 25.6446 | 30.6029 | 4.9583 | external |
| tta4_anyview | boundary_f1 | 15 | 380 | 0.3114 | 0.0150 | 0.2753 | 0.3296 | 0.0543 | external |
| tta4_anyview | dice | 15 | 380 | 0.7515 | 0.0089 | 0.7358 | 0.7652 | 0.0294 | external |
| tta4_anyview | hd95 | 15 | 380 | 80.9220 | 5.4395 | 74.2361 | 96.8597 | 22.6235 | external |
| tta4_lowdet_20 | assd | 15 | 380 | 27.9808 | 1.6006 | 25.0172 | 30.3667 | 5.3495 | external |
| tta4_lowdet_20 | boundary_f1 | 15 | 380 | 0.3115 | 0.0150 | 0.2753 | 0.3300 | 0.0547 | external |
| tta4_lowdet_20 | dice | 15 | 380 | 0.7520 | 0.0088 | 0.7358 | 0.7652 | 0.0295 | external |
| tta4_lowdet_20 | hd95 | 15 | 380 | 80.6115 | 5.5769 | 73.3743 | 96.8597 | 23.4854 | external |
| tta4_lowdet_30 | assd | 15 | 380 | 27.9387 | 1.5264 | 25.1841 | 30.4495 | 5.2653 | external |
| tta4_lowdet_30 | boundary_f1 | 15 | 380 | 0.3121 | 0.0151 | 0.2753 | 0.3336 | 0.0582 | external |
| tta4_lowdet_30 | dice | 15 | 380 | 0.7516 | 0.0088 | 0.7358 | 0.7652 | 0.0294 | external |
| tta4_lowdet_30 | hd95 | 15 | 380 | 80.3767 | 5.6825 | 72.9828 | 96.8597 | 23.8769 | external |
| tta4_lowdet_40 | assd | 15 | 380 | 27.6979 | 1.6008 | 24.8561 | 30.6191 | 5.7630 | external |
| tta4_lowdet_40 | boundary_f1 | 15 | 380 | 0.3129 | 0.0153 | 0.2753 | 0.3339 | 0.0586 | external |
| tta4_lowdet_40 | dice | 15 | 380 | 0.7512 | 0.0090 | 0.7358 | 0.7652 | 0.0294 | external |
| tta4_lowdet_40 | hd95 | 15 | 380 | 79.9121 | 6.0665 | 72.5532 | 96.8597 | 24.3064 | external |
| tta8 | assd | 15 | 380 | 26.4189 | 1.6553 | 23.8838 | 29.1206 | 5.2368 | external |
| tta8 | boundary_f1 | 15 | 380 | 0.3186 | 0.0173 | 0.2756 | 0.3462 | 0.0706 | external |
| tta8 | dice | 15 | 380 | 0.7522 | 0.0083 | 0.7396 | 0.7657 | 0.0261 | external |
| tta8 | hd95 | 15 | 380 | 77.1240 | 6.1417 | 70.1984 | 93.7496 | 23.5512 | external |
| tta8_anyview | assd | 15 | 380 | 27.8335 | 1.6365 | 24.6790 | 31.6475 | 6.9686 | external |
| tta8_anyview | boundary_f1 | 15 | 380 | 0.3149 | 0.0163 | 0.2756 | 0.3369 | 0.0613 | external |
| tta8_anyview | dice | 15 | 380 | 0.7530 | 0.0082 | 0.7396 | 0.7657 | 0.0261 | external |
| tta8_anyview | hd95 | 15 | 380 | 79.3128 | 5.2287 | 71.1801 | 93.7496 | 22.5696 | external |
| tta8_lowdet_20 | assd | 15 | 380 | 27.5506 | 1.6649 | 24.5660 | 31.3607 | 6.7948 | external |
| tta8_lowdet_20 | boundary_f1 | 15 | 380 | 0.3150 | 0.0165 | 0.2756 | 0.3384 | 0.0628 | external |
| tta8_lowdet_20 | dice | 15 | 380 | 0.7537 | 0.0082 | 0.7396 | 0.7657 | 0.0261 | external |
| tta8_lowdet_20 | hd95 | 15 | 380 | 79.0134 | 5.2774 | 71.1069 | 93.7496 | 22.6427 | external |
| tta8_lowdet_30 | assd | 15 | 380 | 27.3190 | 1.5550 | 24.6006 | 29.4696 | 4.8689 | external |
| tta8_lowdet_30 | boundary_f1 | 15 | 380 | 0.3159 | 0.0164 | 0.2756 | 0.3389 | 0.0633 | external |
| tta8_lowdet_30 | dice | 15 | 380 | 0.7529 | 0.0083 | 0.7396 | 0.7657 | 0.0261 | external |
| tta8_lowdet_30 | hd95 | 15 | 380 | 78.5060 | 5.5008 | 71.1459 | 93.7496 | 22.6037 | external |
| tta8_lowdet_40 | assd | 15 | 380 | 27.1559 | 1.5760 | 24.7472 | 30.1064 | 5.3593 | external |
| tta8_lowdet_40 | boundary_f1 | 15 | 380 | 0.3169 | 0.0167 | 0.2756 | 0.3425 | 0.0669 | external |
| tta8_lowdet_40 | dice | 15 | 380 | 0.7523 | 0.0083 | 0.7396 | 0.7657 | 0.0261 | external |
| tta8_lowdet_40 | hd95 | 15 | 380 | 78.1383 | 5.6755 | 71.1984 | 93.7496 | 22.5513 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.7379 | 0.7448 | 0.0069 | 0.0029 | 0.0044 | 0.0095 | 0.3156 | rank_biserial | 24382.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 5.1400 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 380 | both_nonempty | 380 | 380 | 0 | 86.6858 | 84.3219 | -2.3638 | -0.5283 | -3.8744 | -0.8835 | -0.2340 | rank_biserial | 27726.0000 | wilcoxon | 0.0001 | 0.0004 | tta2 | two-sided | -3.0684 | 0.0023 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.7379 | 0.7511 | 0.0132 | 0.0061 | 0.0095 | 0.0171 | 0.4558 | rank_biserial | 19388.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 6.7578 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 379 | both_nonempty | 379 | 379 | 0 | 86.3729 | 80.4554 | -5.9175 | -1.0134 | -8.1219 | -3.8178 | -0.3763 | rank_biserial | 22458.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -5.3843 | 0.0000 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.7379 | 0.7522 | 0.0143 | 0.0066 | 0.0097 | 0.0190 | 0.4800 | rank_biserial | 18527.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 5.9757 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 379 | both_nonempty | 379 | 379 | 0 | 86.3729 | 78.7500 | -7.6228 | -1.5343 | -10.0759 | -5.2282 | -0.4571 | rank_biserial | 19546.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -6.1936 | 0.0000 | t_test_rel |

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 380 | 380 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 380 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 380 | 380 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 380 | 380.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta4 | baseline | 380 | 380 | 1 | 0.0026 | 0.0000 | 0.0079 | cluster_bootstrap_video | 380 | 379.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta8 | baseline | 380 | 380 | 1 | 0.0026 | 0.0000 | 0.0079 | cluster_bootstrap_video | 380 | 379.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
