# MELBA external validation - Kvasir-SEG

External dataset `Kvasir-SEG` (`C:/Users/subka/Documents/datasets/polyp/Kvasir-SEG/images`) scored by 15 internal checkpoints. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 1000 | 51.8099 | 11.1171 | 29.5726 | 69.0590 | 39.4864 | external |
| baseline | boundary_f1 | 15 | 1000 | 0.2718 | 0.0359 | 0.2076 | 0.3346 | 0.1270 | external |
| baseline | dice | 15 | 1000 | 0.7159 | 0.0483 | 0.6439 | 0.8027 | 0.1589 | external |
| baseline | hd95 | 15 | 1000 | 170.6398 | 26.4310 | 114.2281 | 208.2592 | 94.0311 | external |
| tta2 | assd | 15 | 1000 | 49.3393 | 9.9632 | 29.7111 | 63.2985 | 33.5874 | external |
| tta2 | boundary_f1 | 15 | 1000 | 0.2811 | 0.0347 | 0.2168 | 0.3404 | 0.1235 | external |
| tta2 | dice | 15 | 1000 | 0.7326 | 0.0447 | 0.6711 | 0.8127 | 0.1416 | external |
| tta2 | hd95 | 15 | 1000 | 162.7740 | 25.1445 | 110.5716 | 206.1802 | 95.6086 | external |
| tta4 | assd | 15 | 1000 | 44.9172 | 10.0452 | 26.2601 | 59.0340 | 32.7739 | external |
| tta4 | boundary_f1 | 15 | 1000 | 0.3034 | 0.0377 | 0.2313 | 0.3660 | 0.1348 | external |
| tta4 | dice | 15 | 1000 | 0.7474 | 0.0448 | 0.6851 | 0.8244 | 0.1393 | external |
| tta4 | hd95 | 15 | 1000 | 151.3117 | 25.5873 | 99.5930 | 193.2111 | 93.6181 | external |
| tta4_anyview | assd | 15 | 1000 | 44.9838 | 9.9890 | 26.6221 | 59.0339 | 32.4118 | external |
| tta4_anyview | boundary_f1 | 15 | 1000 | 0.3032 | 0.0375 | 0.2313 | 0.3648 | 0.1335 | external |
| tta4_anyview | dice | 15 | 1000 | 0.7475 | 0.0450 | 0.6851 | 0.8250 | 0.1399 | external |
| tta4_anyview | hd95 | 15 | 1000 | 151.4203 | 25.4629 | 100.2980 | 193.2113 | 92.9133 | external |
| tta4_lowdet_20 | assd | 15 | 1000 | 44.9630 | 10.0028 | 26.5741 | 59.0339 | 32.4599 | external |
| tta4_lowdet_20 | boundary_f1 | 15 | 1000 | 0.3032 | 0.0375 | 0.2313 | 0.3648 | 0.1336 | external |
| tta4_lowdet_20 | dice | 15 | 1000 | 0.7476 | 0.0451 | 0.6851 | 0.8257 | 0.1406 | external |
| tta4_lowdet_20 | hd95 | 15 | 1000 | 151.3953 | 25.4835 | 100.2563 | 193.2113 | 92.9550 | external |
| tta4_lowdet_30 | assd | 15 | 1000 | 44.9338 | 10.0157 | 26.4564 | 59.0339 | 32.5775 | external |
| tta4_lowdet_30 | boundary_f1 | 15 | 1000 | 0.3032 | 0.0375 | 0.2313 | 0.3653 | 0.1340 | external |
| tta4_lowdet_30 | dice | 15 | 1000 | 0.7475 | 0.0449 | 0.6851 | 0.8247 | 0.1396 | external |
| tta4_lowdet_30 | hd95 | 15 | 1000 | 151.3243 | 25.5581 | 99.7905 | 193.2113 | 93.4208 | external |
| tta4_lowdet_40 | assd | 15 | 1000 | 44.9451 | 10.0025 | 26.4917 | 59.0339 | 32.5422 | external |
| tta4_lowdet_40 | boundary_f1 | 15 | 1000 | 0.3032 | 0.0375 | 0.2313 | 0.3653 | 0.1340 | external |
| tta4_lowdet_40 | dice | 15 | 1000 | 0.7474 | 0.0448 | 0.6851 | 0.8245 | 0.1394 | external |
| tta4_lowdet_40 | hd95 | 15 | 1000 | 151.3397 | 25.5423 | 99.8215 | 193.2113 | 93.3898 | external |
| tta8 | assd | 15 | 1000 | 41.9048 | 9.7328 | 24.5792 | 55.4218 | 30.8426 | external |
| tta8 | boundary_f1 | 15 | 1000 | 0.3158 | 0.0393 | 0.2341 | 0.3778 | 0.1436 | external |
| tta8 | dice | 15 | 1000 | 0.7573 | 0.0448 | 0.6902 | 0.8292 | 0.1390 | external |
| tta8 | hd95 | 15 | 1000 | 143.4249 | 25.8476 | 93.1444 | 179.9197 | 86.7753 | external |
| tta8_anyview | assd | 15 | 1000 | 41.9697 | 9.6853 | 24.9399 | 55.4218 | 30.4819 | external |
| tta8_anyview | boundary_f1 | 15 | 1000 | 0.3154 | 0.0391 | 0.2341 | 0.3757 | 0.1416 | external |
| tta8_anyview | dice | 15 | 1000 | 0.7576 | 0.0450 | 0.6902 | 0.8311 | 0.1409 | external |
| tta8_anyview | hd95 | 15 | 1000 | 143.5353 | 25.7309 | 93.9261 | 179.9197 | 85.9935 | external |
| tta8_lowdet_20 | assd | 15 | 1000 | 41.9765 | 9.6903 | 24.9694 | 55.4218 | 30.4524 | external |
| tta8_lowdet_20 | boundary_f1 | 15 | 1000 | 0.3154 | 0.0391 | 0.2341 | 0.3759 | 0.1417 | external |
| tta8_lowdet_20 | dice | 15 | 1000 | 0.7576 | 0.0450 | 0.6902 | 0.8301 | 0.1399 | external |
| tta8_lowdet_20 | hd95 | 15 | 1000 | 143.5398 | 25.7545 | 93.7590 | 179.9197 | 86.1607 | external |
| tta8_lowdet_30 | assd | 15 | 1000 | 41.9545 | 9.6782 | 24.9685 | 55.4218 | 30.4533 | external |
| tta8_lowdet_30 | boundary_f1 | 15 | 1000 | 0.3155 | 0.0391 | 0.2341 | 0.3759 | 0.1418 | external |
| tta8_lowdet_30 | dice | 15 | 1000 | 0.7574 | 0.0448 | 0.6902 | 0.8295 | 0.1393 | external |
| tta8_lowdet_30 | hd95 | 15 | 1000 | 143.4946 | 25.7501 | 93.8194 | 179.9197 | 86.1002 | external |
| tta8_lowdet_40 | assd | 15 | 1000 | 41.9425 | 9.7122 | 24.7018 | 55.4218 | 30.7200 | external |
| tta8_lowdet_40 | boundary_f1 | 15 | 1000 | 0.3156 | 0.0393 | 0.2341 | 0.3774 | 0.1432 | external |
| tta8_lowdet_40 | dice | 15 | 1000 | 0.7573 | 0.0448 | 0.6902 | 0.8292 | 0.1390 | external |
| tta8_lowdet_40 | hd95 | 15 | 1000 | 143.4664 | 25.8307 | 93.3210 | 179.9197 | 86.5986 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.7159 | 0.7326 | 0.0168 | 0.0132 | 0.0153 | 0.0183 | 0.7238 | rank_biserial | 69118.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 21.6980 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 170.6482 | 162.7878 | -7.8604 | -5.8069 | -9.0590 | -6.6961 | -0.4963 | rank_biserial | 126056.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | -13.2322 | 0.0000 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.7159 | 0.7474 | 0.0315 | 0.0243 | 0.0294 | 0.0337 | 0.8605 | rank_biserial | 34913.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 28.7980 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 170.6482 | 151.3471 | -19.3011 | -14.3858 | -20.9782 | -17.7008 | -0.7525 | rank_biserial | 61935.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -23.0891 | 0.0000 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.7159 | 0.7573 | 0.0414 | 0.0326 | 0.0388 | 0.0440 | 0.9091 | rank_biserial | 22755.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 31.0765 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 170.6482 | 143.4853 | -27.1629 | -20.6712 | -29.1594 | -25.2815 | -0.8494 | rank_biserial | 37692.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -27.2040 | 0.0000 | t_test_rel |

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 1000 | 1000 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 1000 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 1000 | 1000 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 1000 | 1000.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta4 | baseline | 1000 | 1000 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 1000 | 1000.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta8 | baseline | 1000 | 1000 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 1000 | 1000.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
