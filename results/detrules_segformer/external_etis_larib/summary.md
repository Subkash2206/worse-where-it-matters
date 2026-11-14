# MELBA external validation - ETIS-LaribPolypDB

External dataset `ETIS-LaribPolypDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/ETIS-LaribPolypDB/images`) scored by 15 internal checkpoints over dice, jaccard, precision, recall, hd, hd95, hd95_pooled, assd, boundary_f1, pred_volume_mm3. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

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
| tta4_anyview | assd | 15 | 196 | 70.9798 | 7.2353 | 62.0528 | 86.5286 | 24.4758 | external |
| tta4_anyview | boundary_f1 | 15 | 196 | 0.1852 | 0.0162 | 0.1542 | 0.2085 | 0.0543 | external |
| tta4_anyview | dice | 15 | 196 | 0.7138 | 0.0188 | 0.6745 | 0.7442 | 0.0696 | external |
| tta4_anyview | hd95 | 15 | 196 | 209.8118 | 21.0510 | 178.5168 | 249.0272 | 70.5104 | external |
| tta4_lowdet_20 | assd | 15 | 196 | 70.7205 | 7.5462 | 62.3831 | 89.0289 | 26.6458 | external |
| tta4_lowdet_20 | boundary_f1 | 15 | 196 | 0.1852 | 0.0162 | 0.1545 | 0.2085 | 0.0541 | external |
| tta4_lowdet_20 | dice | 15 | 196 | 0.7141 | 0.0190 | 0.6735 | 0.7442 | 0.0707 | external |
| tta4_lowdet_20 | hd95 | 15 | 196 | 210.7309 | 21.7617 | 179.2640 | 256.1794 | 76.9154 | external |
| tta4_lowdet_30 | assd | 15 | 196 | 70.6496 | 7.2873 | 61.8238 | 87.4232 | 25.5993 | external |
| tta4_lowdet_30 | boundary_f1 | 15 | 196 | 0.1854 | 0.0164 | 0.1542 | 0.2096 | 0.0554 | external |
| tta4_lowdet_30 | dice | 15 | 196 | 0.7134 | 0.0187 | 0.6743 | 0.7442 | 0.0699 | external |
| tta4_lowdet_30 | hd95 | 15 | 196 | 209.6698 | 21.2762 | 178.8178 | 251.1521 | 72.3343 | external |
| tta4_lowdet_40 | assd | 15 | 196 | 70.1946 | 7.2418 | 61.9029 | 86.7608 | 24.8579 | external |
| tta4_lowdet_40 | boundary_f1 | 15 | 196 | 0.1857 | 0.0163 | 0.1545 | 0.2107 | 0.0562 | external |
| tta4_lowdet_40 | dice | 15 | 196 | 0.7130 | 0.0189 | 0.6727 | 0.7442 | 0.0715 | external |
| tta4_lowdet_40 | hd95 | 15 | 196 | 208.8853 | 21.2527 | 176.1468 | 248.4129 | 72.2661 | external |
| tta8 | assd | 15 | 196 | 64.8592 | 6.1390 | 57.2190 | 78.3621 | 21.1431 | external |
| tta8 | boundary_f1 | 15 | 196 | 0.1933 | 0.0190 | 0.1453 | 0.2200 | 0.0747 | external |
| tta8 | dice | 15 | 196 | 0.7209 | 0.0163 | 0.6865 | 0.7501 | 0.0635 | external |
| tta8 | hd95 | 15 | 196 | 199.7927 | 23.1604 | 157.8620 | 234.4222 | 76.5602 | external |
| tta8_anyview | assd | 15 | 196 | 66.5187 | 6.7067 | 59.1949 | 80.5456 | 21.3507 | external |
| tta8_anyview | boundary_f1 | 15 | 196 | 0.1916 | 0.0187 | 0.1453 | 0.2168 | 0.0715 | external |
| tta8_anyview | dice | 15 | 196 | 0.7219 | 0.0152 | 0.6920 | 0.7501 | 0.0581 | external |
| tta8_anyview | hd95 | 15 | 196 | 201.8431 | 23.3316 | 160.8783 | 234.4222 | 73.5438 | external |
| tta8_lowdet_20 | assd | 15 | 196 | 65.9500 | 6.3299 | 58.6956 | 81.1389 | 22.4433 | external |
| tta8_lowdet_20 | boundary_f1 | 15 | 196 | 0.1917 | 0.0187 | 0.1453 | 0.2169 | 0.0716 | external |
| tta8_lowdet_20 | dice | 15 | 196 | 0.7226 | 0.0149 | 0.6913 | 0.7501 | 0.0588 | external |
| tta8_lowdet_20 | hd95 | 15 | 196 | 202.3511 | 23.0096 | 160.2451 | 239.0935 | 78.8484 | external |
| tta8_lowdet_30 | assd | 15 | 196 | 65.8781 | 6.2460 | 59.0388 | 80.8940 | 21.8552 | external |
| tta8_lowdet_30 | boundary_f1 | 15 | 196 | 0.1918 | 0.0189 | 0.1453 | 0.2178 | 0.0725 | external |
| tta8_lowdet_30 | dice | 15 | 196 | 0.7220 | 0.0152 | 0.6910 | 0.7501 | 0.0591 | external |
| tta8_lowdet_30 | hd95 | 15 | 196 | 201.0187 | 22.5895 | 158.6473 | 234.4222 | 75.7749 | external |
| tta8_lowdet_40 | assd | 15 | 196 | 65.6528 | 6.0446 | 59.1520 | 79.5174 | 20.3654 | external |
| tta8_lowdet_40 | boundary_f1 | 15 | 196 | 0.1922 | 0.0189 | 0.1453 | 0.2189 | 0.0736 | external |
| tta8_lowdet_40 | dice | 15 | 196 | 0.7211 | 0.0161 | 0.6872 | 0.7501 | 0.0629 | external |
| tta8_lowdet_40 | hd95 | 15 | 196 | 200.4009 | 22.8421 | 159.2183 | 234.4222 | 75.2039 | external |

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
