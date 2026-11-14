# MELBA external validation - ETIS-LaribPolypDB

External dataset `ETIS-LaribPolypDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/ETIS-LaribPolypDB/images`) scored by 15 internal checkpoints. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 195 | 174.2904 | 43.6392 | 104.9116 | 230.6339 | 125.7223 | external |
| baseline | boundary_f1 | 15 | 195 | 0.1209 | 0.0253 | 0.0879 | 0.1553 | 0.0674 | external |
| baseline | dice | 15 | 196 | 0.4627 | 0.0603 | 0.3836 | 0.5985 | 0.2149 | external |
| baseline | hd95 | 15 | 195 | 412.9317 | 88.9574 | 252.3550 | 542.8106 | 290.4557 | external |
| tta2 | assd | 15 | 193 | 152.6183 | 42.6533 | 92.1097 | 220.5496 | 128.4399 | external |
| tta2 | boundary_f1 | 15 | 193 | 0.1331 | 0.0241 | 0.1003 | 0.1762 | 0.0759 | external |
| tta2 | dice | 15 | 196 | 0.5034 | 0.0542 | 0.4246 | 0.6188 | 0.1943 | external |
| tta2 | hd95 | 15 | 193 | 375.0547 | 85.0896 | 236.1382 | 520.9973 | 284.8592 | external |
| tta4 | assd | 15 | 190 | 130.4417 | 40.0918 | 74.9505 | 191.6298 | 116.6794 | external |
| tta4 | boundary_f1 | 15 | 190 | 0.1574 | 0.0302 | 0.1167 | 0.2015 | 0.0849 | external |
| tta4 | dice | 15 | 196 | 0.5284 | 0.0591 | 0.4395 | 0.6202 | 0.1807 | external |
| tta4 | hd95 | 15 | 190 | 333.6749 | 83.8425 | 195.1832 | 469.9704 | 274.7872 | external |
| tta4_anyview | assd | 15 | 196 | 148.8338 | 33.9880 | 100.4917 | 203.3245 | 102.8328 | external |
| tta4_anyview | boundary_f1 | 15 | 196 | 0.1468 | 0.0264 | 0.1084 | 0.1904 | 0.0821 | external |
| tta4_anyview | dice | 15 | 196 | 0.5366 | 0.0599 | 0.4477 | 0.6243 | 0.1766 | external |
| tta4_anyview | hd95 | 15 | 196 | 351.1219 | 74.4510 | 227.1383 | 474.4877 | 247.3493 | external |
| tta4_lowdet_20 | assd | 15 | 196 | 151.6296 | 32.1303 | 101.3816 | 211.2168 | 109.8353 | external |
| tta4_lowdet_20 | boundary_f1 | 15 | 196 | 0.1463 | 0.0261 | 0.1084 | 0.1908 | 0.0824 | external |
| tta4_lowdet_20 | dice | 15 | 196 | 0.5385 | 0.0605 | 0.4443 | 0.6241 | 0.1799 | external |
| tta4_lowdet_20 | hd95 | 15 | 196 | 360.4311 | 71.3513 | 254.4407 | 492.9821 | 238.5414 | external |
| tta4_lowdet_30 | assd | 15 | 196 | 141.8897 | 38.7758 | 93.1648 | 207.9799 | 114.8151 | external |
| tta4_lowdet_30 | boundary_f1 | 15 | 196 | 0.1501 | 0.0286 | 0.1084 | 0.1922 | 0.0839 | external |
| tta4_lowdet_30 | dice | 15 | 196 | 0.5326 | 0.0595 | 0.4425 | 0.6223 | 0.1798 | external |
| tta4_lowdet_30 | hd95 | 15 | 196 | 344.8760 | 83.0537 | 214.2097 | 492.6565 | 278.4468 | external |
| tta4_lowdet_40 | assd | 15 | 195 | 136.2594 | 38.9142 | 84.9461 | 195.0057 | 110.0597 | external |
| tta4_lowdet_40 | boundary_f1 | 15 | 195 | 0.1536 | 0.0292 | 0.1124 | 0.1942 | 0.0818 | external |
| tta4_lowdet_40 | dice | 15 | 196 | 0.5295 | 0.0591 | 0.4412 | 0.6202 | 0.1790 | external |
| tta4_lowdet_40 | hd95 | 15 | 195 | 338.3596 | 82.3108 | 204.2819 | 469.9704 | 265.6885 | external |
| tta8 | assd | 15 | 191 | 116.6796 | 40.4399 | 70.9438 | 191.5218 | 120.5781 | external |
| tta8 | boundary_f1 | 15 | 191 | 0.1709 | 0.0335 | 0.1248 | 0.2148 | 0.0900 | external |
| tta8 | dice | 15 | 196 | 0.5469 | 0.0583 | 0.4521 | 0.6306 | 0.1785 | external |
| tta8 | hd95 | 15 | 191 | 300.4533 | 86.1202 | 193.0569 | 439.3588 | 246.3019 | external |
| tta8_anyview | assd | 15 | 196 | 141.1115 | 31.2430 | 86.4203 | 197.6151 | 111.1947 | external |
| tta8_anyview | boundary_f1 | 15 | 196 | 0.1557 | 0.0272 | 0.1133 | 0.2014 | 0.0881 | external |
| tta8_anyview | dice | 15 | 196 | 0.5573 | 0.0599 | 0.4523 | 0.6385 | 0.1862 | external |
| tta8_anyview | hd95 | 15 | 196 | 329.2545 | 70.7460 | 216.3995 | 444.2140 | 227.8145 | external |
| tta8_lowdet_20 | assd | 15 | 196 | 139.4585 | 32.1701 | 81.8207 | 195.3284 | 113.5077 | external |
| tta8_lowdet_20 | boundary_f1 | 15 | 196 | 0.1579 | 0.0287 | 0.1131 | 0.2011 | 0.0880 | external |
| tta8_lowdet_20 | dice | 15 | 196 | 0.5592 | 0.0603 | 0.4533 | 0.6402 | 0.1869 | external |
| tta8_lowdet_20 | hd95 | 15 | 196 | 330.2686 | 73.8689 | 205.2674 | 443.8529 | 238.5855 | external |
| tta8_lowdet_30 | assd | 15 | 196 | 127.9883 | 38.0481 | 76.5571 | 194.1442 | 117.5871 | external |
| tta8_lowdet_30 | boundary_f1 | 15 | 196 | 0.1627 | 0.0317 | 0.1133 | 0.2073 | 0.0940 | external |
| tta8_lowdet_30 | dice | 15 | 196 | 0.5537 | 0.0590 | 0.4523 | 0.6327 | 0.1804 | external |
| tta8_lowdet_30 | hd95 | 15 | 196 | 313.4787 | 81.8760 | 204.9256 | 442.8108 | 237.8852 | external |
| tta8_lowdet_40 | assd | 15 | 193 | 121.2140 | 38.4435 | 74.9643 | 194.5576 | 119.5933 | external |
| tta8_lowdet_40 | boundary_f1 | 15 | 193 | 0.1667 | 0.0318 | 0.1208 | 0.2104 | 0.0896 | external |
| tta8_lowdet_40 | dice | 15 | 196 | 0.5491 | 0.0585 | 0.4521 | 0.6306 | 0.1784 | external |
| tta8_lowdet_40 | hd95 | 15 | 193 | 304.1175 | 83.7285 | 196.9784 | 441.4097 | 244.4313 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.4627 | 0.5034 | 0.0407 | 0.0365 | 0.0346 | 0.0468 | 0.8562 | rank_biserial | 1305.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 13.1866 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 188 | both_nonempty | 188 | 188 | 0 | 414.4050 | 379.5546 | -34.8505 | -27.6618 | -42.8844 | -26.9509 | -0.6430 | rank_biserial | 3171.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | -8.6206 | 0.0000 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.4627 | 0.5284 | 0.0657 | 0.0641 | 0.0567 | 0.0747 | 0.8738 | rank_biserial | 1145.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 14.1615 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 180 | both_nonempty | 180 | 180 | 0 | 410.0682 | 337.2965 | -72.7717 | -65.4728 | -83.9423 | -62.0784 | -0.8462 | rank_biserial | 1253.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -12.9644 | 0.0000 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 196 | all_cases | 196 | 196 | 0 | 0.4627 | 0.5469 | 0.0842 | 0.0797 | 0.0731 | 0.0953 | 0.8945 | rank_biserial | 957.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 14.8861 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 177 | both_nonempty | 177 | 177 | 0 | 408.5048 | 304.3080 | -104.1967 | -99.0008 | -117.6322 | -91.0476 | -0.9257 | rank_biserial | 585.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -15.3669 | 0.0000 | t_test_rel |

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 196 | 196 | 5 | 0.0255 | 0.0051 | 0.0510 | cluster_bootstrap_video | 196 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 196 | 196 | 8 | 0.0408 | 0.0153 | 0.0714 | cluster_bootstrap_video | 196 | 188.0000 | 3.0000 | 0.0000 | 5.0000 | 0.2500 | 0.2500 | case |
| external | tta4 | baseline | 196 | 196 | 16 | 0.0816 | 0.0459 | 0.1224 | cluster_bootstrap_video | 196 | 180.0000 | 11.0000 | 0.0000 | 5.0000 | 0.0010 | 0.0029 | case |
| external | tta8 | baseline | 196 | 196 | 18 | 0.0918 | 0.0561 | 0.1327 | cluster_bootstrap_video | 196 | 177.0000 | 14.0000 | 1.0000 | 4.0000 | 0.0010 | 0.0029 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
