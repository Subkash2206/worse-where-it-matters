# MELBA external validation - CVC-ColonDB

External dataset `CVC-ColonDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/CVC-ColonDB/images`) scored by 15 internal checkpoints. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 344 | 31.8797 | 7.8104 | 20.5903 | 47.3116 | 26.7214 | external |
| baseline | boundary_f1 | 15 | 344 | 0.3010 | 0.0313 | 0.2240 | 0.3459 | 0.1220 | external |
| baseline | dice | 15 | 380 | 0.6307 | 0.0235 | 0.5854 | 0.6715 | 0.0860 | external |
| baseline | hd95 | 15 | 344 | 93.8897 | 20.4245 | 59.7421 | 133.9072 | 74.1650 | external |
| tta2 | assd | 15 | 350 | 27.4609 | 5.4878 | 19.1330 | 39.2406 | 20.1076 | external |
| tta2 | boundary_f1 | 15 | 350 | 0.3107 | 0.0311 | 0.2239 | 0.3505 | 0.1266 | external |
| tta2 | dice | 15 | 380 | 0.6443 | 0.0236 | 0.5936 | 0.6721 | 0.0785 | external |
| tta2 | hd95 | 15 | 350 | 84.0837 | 16.5951 | 56.7960 | 118.3478 | 61.5518 | external |
| tta4 | assd | 15 | 337 | 22.6300 | 3.2243 | 17.3702 | 27.1357 | 9.7656 | external |
| tta4 | boundary_f1 | 15 | 337 | 0.3336 | 0.0289 | 0.2483 | 0.3666 | 0.1183 | external |
| tta4 | dice | 15 | 380 | 0.6546 | 0.0253 | 0.6158 | 0.6883 | 0.0725 | external |
| tta4 | hd95 | 15 | 337 | 71.8714 | 10.1659 | 52.3589 | 85.8270 | 33.4681 | external |
| tta4_anyview | assd | 15 | 367 | 32.4826 | 4.3869 | 23.5271 | 38.7826 | 15.2555 | external |
| tta4_anyview | boundary_f1 | 15 | 367 | 0.3084 | 0.0276 | 0.2270 | 0.3379 | 0.1110 | external |
| tta4_anyview | dice | 15 | 380 | 0.6696 | 0.0237 | 0.6316 | 0.7022 | 0.0705 | external |
| tta4_anyview | hd95 | 15 | 367 | 85.5982 | 10.8806 | 61.9522 | 101.0675 | 39.1153 | external |
| tta4_lowdet_20 | assd | 15 | 380 | 37.6475 | 10.1182 | 22.6656 | 56.9060 | 34.2404 | external |
| tta4_lowdet_20 | boundary_f1 | 15 | 380 | 0.3014 | 0.0338 | 0.2176 | 0.3389 | 0.1213 | external |
| tta4_lowdet_20 | dice | 15 | 380 | 0.6714 | 0.0243 | 0.6316 | 0.7070 | 0.0754 | external |
| tta4_lowdet_20 | hd95 | 15 | 380 | 95.1047 | 21.0061 | 61.3731 | 130.3955 | 69.0224 | external |
| tta4_lowdet_30 | assd | 15 | 380 | 28.9510 | 9.3337 | 18.3490 | 54.3302 | 35.9812 | external |
| tta4_lowdet_30 | boundary_f1 | 15 | 380 | 0.3186 | 0.0356 | 0.2181 | 0.3607 | 0.1425 | external |
| tta4_lowdet_30 | dice | 15 | 380 | 0.6605 | 0.0258 | 0.6179 | 0.6942 | 0.0762 | external |
| tta4_lowdet_30 | hd95 | 15 | 380 | 80.5078 | 17.9022 | 54.1142 | 122.1985 | 68.0844 | external |
| tta4_lowdet_40 | assd | 15 | 346 | 24.9683 | 3.9288 | 18.5606 | 31.5551 | 12.9946 | external |
| tta4_lowdet_40 | boundary_f1 | 15 | 346 | 0.3267 | 0.0299 | 0.2376 | 0.3608 | 0.1232 | external |
| tta4_lowdet_40 | dice | 15 | 380 | 0.6567 | 0.0257 | 0.6168 | 0.6917 | 0.0748 | external |
| tta4_lowdet_40 | hd95 | 15 | 346 | 74.7866 | 10.5427 | 54.2757 | 92.3863 | 38.1106 | external |
| tta8 | assd | 15 | 334 | 20.3095 | 3.1060 | 16.3487 | 26.4360 | 10.0873 | external |
| tta8 | boundary_f1 | 15 | 334 | 0.3478 | 0.0290 | 0.2682 | 0.3825 | 0.1143 | external |
| tta8 | dice | 15 | 380 | 0.6557 | 0.0281 | 0.6087 | 0.6977 | 0.0890 | external |
| tta8 | hd95 | 15 | 334 | 66.2723 | 10.1903 | 51.1672 | 85.6290 | 34.4617 | external |
| tta8_anyview | assd | 15 | 374 | 33.1188 | 3.4372 | 27.1729 | 39.6414 | 12.4685 | external |
| tta8_anyview | boundary_f1 | 15 | 374 | 0.3115 | 0.0261 | 0.2410 | 0.3368 | 0.0958 | external |
| tta8_anyview | dice | 15 | 380 | 0.6826 | 0.0221 | 0.6404 | 0.7131 | 0.0727 | external |
| tta8_anyview | hd95 | 15 | 374 | 85.4338 | 9.1222 | 68.9937 | 100.6515 | 31.6578 | external |
| tta8_lowdet_20 | assd | 15 | 380 | 34.7795 | 11.5807 | 21.8405 | 55.6174 | 33.7769 | external |
| tta8_lowdet_20 | boundary_f1 | 15 | 380 | 0.3159 | 0.0370 | 0.2313 | 0.3592 | 0.1280 | external |
| tta8_lowdet_20 | dice | 15 | 380 | 0.6755 | 0.0252 | 0.6333 | 0.7127 | 0.0794 | external |
| tta8_lowdet_20 | hd95 | 15 | 380 | 88.8013 | 22.4794 | 63.5652 | 121.4773 | 57.9121 | external |
| tta8_lowdet_30 | assd | 15 | 380 | 27.3001 | 9.2840 | 18.5967 | 52.3816 | 33.7850 | external |
| tta8_lowdet_30 | boundary_f1 | 15 | 380 | 0.3292 | 0.0369 | 0.2320 | 0.3669 | 0.1348 | external |
| tta8_lowdet_30 | dice | 15 | 380 | 0.6652 | 0.0260 | 0.6225 | 0.7057 | 0.0831 | external |
| tta8_lowdet_30 | hd95 | 15 | 380 | 76.5133 | 17.9789 | 54.4965 | 117.2169 | 62.7204 | external |
| tta8_lowdet_40 | assd | 15 | 347 | 22.4040 | 3.1440 | 18.4170 | 27.6787 | 9.2617 | external |
| tta8_lowdet_40 | boundary_f1 | 15 | 347 | 0.3401 | 0.0309 | 0.2515 | 0.3699 | 0.1184 | external |
| tta8_lowdet_40 | dice | 15 | 380 | 0.6588 | 0.0274 | 0.6120 | 0.7016 | 0.0896 | external |
| tta8_lowdet_40 | hd95 | 15 | 347 | 69.1368 | 9.7409 | 54.4158 | 86.5256 | 32.1097 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.6307 | 0.6443 | 0.0135 | 0.0080 | 0.0097 | 0.0173 | 0.4832 | rank_biserial | 17355.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 6.9824 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 318 | both_nonempty | 318 | 318 | 0 | 90.2934 | 81.6598 | -8.6336 | -4.5630 | -10.5493 | -6.7466 | -0.5782 | rank_biserial | 10698.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | -9.0054 | 0.0000 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.6307 | 0.6546 | 0.0238 | 0.0170 | 0.0183 | 0.0293 | 0.6026 | rank_biserial | 13344.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 8.5223 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 313 | both_nonempty | 313 | 313 | 0 | 88.4591 | 69.1226 | -19.3365 | -13.6048 | -22.1342 | -16.5853 | -0.8509 | rank_biserial | 3664.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -13.7834 | 0.0000 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 380 | all_cases | 380 | 380 | 0 | 0.6307 | 0.6557 | 0.0250 | 0.0210 | 0.0187 | 0.0311 | 0.5721 | rank_biserial | 14369.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 7.8700 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 310 | both_nonempty | 310 | 310 | 0 | 88.1888 | 64.7641 | -23.4248 | -15.8876 | -26.6774 | -20.1842 | -0.8475 | rank_biserial | 3675.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -14.0165 | 0.0000 | t_test_rel |

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 380 | 380 | 55 | 0.1447 | 0.1105 | 0.1816 | cluster_bootstrap_video | 380 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 380 | 380 | 59 | 0.1553 | 0.1184 | 0.1921 | cluster_bootstrap_video | 380 | 318.0000 | 7.0000 | 3.0000 | 52.0000 | 0.3438 | 0.3438 | case |
| external | tta4 | baseline | 380 | 380 | 63 | 0.1658 | 0.1289 | 0.2053 | cluster_bootstrap_video | 380 | 313.0000 | 12.0000 | 4.0000 | 51.0000 | 0.0768 | 0.1536 | case |
| external | tta8 | baseline | 380 | 380 | 68 | 0.1789 | 0.1421 | 0.2184 | cluster_bootstrap_video | 380 | 310.0000 | 15.0000 | 2.0000 | 53.0000 | 0.0023 | 0.0070 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
