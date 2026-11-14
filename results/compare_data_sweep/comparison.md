# Data-fraction sweep: 100pct vs 50pct vs 25pct

3 fractions (100pct, 50pct, 25pct), folds [0, 1, 2, 3, 4], seeds [0]; the variants differ only in `training.data_fraction` (whole training videos, seeded).

## (1) Methods side by side

| model | method | metric | n_cases | n_units | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| 100pct | baseline | dice | 612 | 29 | 0.7567 | 0.1958 |
| 100pct | baseline | boundary_f1 | 612 | 29 | 0.4925 | 0.1434 |
| 100pct | baseline | hd95 | 612 | 29 | 56.4869 | 33.5930 |
| 100pct | tta4 | dice | 612 | 29 | 0.7850 | 0.1941 |
| 100pct | tta4 | boundary_f1 | 612 | 29 | 0.5440 | 0.1539 |
| 100pct | tta4 | hd95 | 612 | 29 | 42.3592 | 32.9854 |
| 100pct | tta8 | dice | 612 | 29 | 0.7841 | 0.2009 |
| 100pct | tta8 | boundary_f1 | 612 | 29 | 0.5509 | 0.1651 |
| 100pct | tta8 | hd95 | 612 | 29 | 42.7429 | 34.5310 |
| 50pct | baseline | dice | 612 | 29 | 0.7011 | 0.2193 |
| 50pct | baseline | boundary_f1 | 612 | 29 | 0.3711 | 0.1522 |
| 50pct | baseline | hd95 | 612 | 29 | 63.3106 | 34.1259 |
| 50pct | tta4 | dice | 612 | 29 | 0.7272 | 0.2243 |
| 50pct | tta4 | boundary_f1 | 612 | 29 | 0.4061 | 0.1776 |
| 50pct | tta4 | hd95 | 612 | 29 | 57.8061 | 39.5453 |
| 50pct | tta8 | dice | 612 | 29 | 0.7345 | 0.2129 |
| 50pct | tta8 | boundary_f1 | 612 | 29 | 0.4207 | 0.1807 |
| 50pct | tta8 | hd95 | 612 | 29 | 53.0558 | 39.6638 |
| 25pct | baseline | dice | 612 | 29 | 0.6463 | 0.1924 |
| 25pct | baseline | boundary_f1 | 612 | 29 | 0.2832 | 0.1455 |
| 25pct | baseline | hd95 | 612 | 29 | 73.0219 | 34.6536 |
| 25pct | tta4 | dice | 612 | 29 | 0.6606 | 0.2383 |
| 25pct | tta4 | boundary_f1 | 612 | 29 | 0.3123 | 0.1760 |
| 25pct | tta4 | hd95 | 612 | 29 | 64.2079 | 42.4367 |
| 25pct | tta8 | dice | 612 | 29 | 0.6597 | 0.2501 |
| 25pct | tta8 | boundary_f1 | 612 | 29 | 0.3184 | 0.1810 |
| 25pct | tta8 | hd95 | 612 | 29 | 61.9366 | 45.2938 |

Video level, mean ± std.

## (2) TTA gains of each fraction vs its own baseline

| model | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100pct | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7567 | 0.7850 | 0.0283 | 0.0252 | 0.0103 | 0.0446 | 0.6946 | rank_biserial | 62.0000 | wilcoxon | 0.0008 | 0.0008 | tta4 | two-sided | 3.3008 | 0.0026 | t_test_rel |
| 100pct | boundary_f1 | higher | tta4 | baseline | group | 28 | all_cases | 594 | 28 | 0 | 0.4925 | 0.5440 | 0.0515 | 0.0393 | 0.0316 | 0.0728 | 0.8571 | rank_biserial | 29.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 4.7484 | 0.0001 | t_test_rel |
| 100pct | hd95 | lower | tta4 | baseline | group | 28 | both_nonempty | 594 | 28 | 0 | 56.4869 | 42.3592 | -14.1278 | -11.1026 | -19.6388 | -9.2153 | -0.9064 | rank_biserial | 19.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -5.1321 | 0.0000 | t_test_rel |
| 100pct | hd95_pooled | lower | tta4 | baseline | group | 28 | both_nonempty | 594 | 28 | 0 | 53.1305 | 39.7414 | -13.3891 | -9.3839 | -18.8116 | -8.6436 | -0.8768 | rank_biserial | 25.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -4.9939 | 0.0000 | t_test_rel |
| 100pct | assd | lower | tta4 | baseline | group | 28 | both_nonempty | 594 | 28 | 0 | 14.9943 | 10.9227 | -4.0716 | -2.9551 | -6.5641 | -2.1214 | -0.7882 | rank_biserial | 43.0000 | wilcoxon | 0.0001 | 0.0002 | tta4 | two-sided | -3.5694 | 0.0014 | t_test_rel |
| 100pct | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7567 | 0.7841 | 0.0274 | 0.0292 | 0.0067 | 0.0455 | 0.7143 | rank_biserial | 58.0000 | wilcoxon | 0.0005 | 0.0005 | tta8 | two-sided | 2.8426 | 0.0083 | t_test_rel |
| 100pct | boundary_f1 | higher | tta8 | baseline | group | 28 | all_cases | 590 | 28 | 0 | 0.4925 | 0.5509 | 0.0584 | 0.0514 | 0.0371 | 0.0816 | 0.8818 | rank_biserial | 24.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 5.0814 | 0.0000 | t_test_rel |
| 100pct | hd95 | lower | tta8 | baseline | group | 28 | both_nonempty | 590 | 28 | 0 | 56.4869 | 42.7429 | -13.7441 | -13.8511 | -18.3173 | -9.3280 | -0.8818 | rank_biserial | 24.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.8023 | 0.0000 | t_test_rel |
| 100pct | hd95_pooled | lower | tta8 | baseline | group | 28 | both_nonempty | 590 | 28 | 0 | 53.1305 | 40.1026 | -13.0279 | -13.5921 | -17.4269 | -8.8809 | -0.8768 | rank_biserial | 25.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.7608 | 0.0000 | t_test_rel |
| 100pct | assd | lower | tta8 | baseline | group | 28 | both_nonempty | 590 | 28 | 0 | 14.9943 | 11.0518 | -3.9425 | -3.6034 | -5.5485 | -2.5432 | -0.8818 | rank_biserial | 24.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.1441 | 0.0000 | t_test_rel |
| 50pct | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7011 | 0.7272 | 0.0261 | 0.0196 | 0.0146 | 0.0385 | 0.7517 | rank_biserial | 54.0000 | wilcoxon | 0.0002 | 0.0009 | tta4 | two-sided | 4.0225 | 0.0004 | t_test_rel |
| 50pct | boundary_f1 | higher | tta4 | baseline | group | 29 | all_cases | 584 | 29 | 0 | 0.3711 | 0.4061 | 0.0349 | 0.0424 | 0.0179 | 0.0526 | 0.6598 | rank_biserial | 74.0000 | wilcoxon | 0.0013 | 0.0051 | tta4 | two-sided | 3.8275 | 0.0007 | t_test_rel |
| 50pct | hd95 | lower | tta4 | baseline | group | 29 | both_nonempty | 584 | 29 | 0 | 63.3106 | 57.8061 | -5.5045 | -4.4291 | -12.5466 | 0.7850 | -0.3885 | rank_biserial | 133.0000 | wilcoxon | 0.0689 | 0.1310 | tta4 | two-sided | -1.5837 | 0.1245 | t_test_rel |
| 50pct | hd95_pooled | lower | tta4 | baseline | group | 29 | both_nonempty | 584 | 29 | 0 | 60.0803 | 55.0182 | -5.0622 | -3.1289 | -11.7264 | 0.9727 | -0.3931 | rank_biserial | 132.0000 | wilcoxon | 0.0655 | 0.1310 | tta4 | two-sided | -1.5027 | 0.1441 | t_test_rel |
| 50pct | assd | lower | tta4 | baseline | group | 29 | both_nonempty | 584 | 29 | 0 | 20.1771 | 18.9665 | -1.2106 | -1.1328 | -4.1176 | 2.6021 | -0.5586 | rank_biserial | 96.0000 | wilcoxon | 0.0075 | 0.0224 | tta4 | two-sided | -0.7332 | 0.4695 | t_test_rel |
| 50pct | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7011 | 0.7345 | 0.0334 | 0.0285 | 0.0188 | 0.0489 | 0.8299 | rank_biserial | 37.0000 | wilcoxon | 0.0000 | 0.0001 | tta8 | two-sided | 4.2414 | 0.0002 | t_test_rel |
| 50pct | boundary_f1 | higher | tta8 | baseline | group | 29 | all_cases | 586 | 29 | 0 | 0.3711 | 0.4207 | 0.0496 | 0.0434 | 0.0289 | 0.0730 | 0.7701 | rank_biserial | 50.0000 | wilcoxon | 0.0001 | 0.0004 | tta8 | two-sided | 4.3103 | 0.0002 | t_test_rel |
| 50pct | hd95 | lower | tta8 | baseline | group | 29 | both_nonempty | 586 | 29 | 0 | 63.3106 | 53.0558 | -10.2547 | -9.9608 | -18.7417 | -1.8180 | -0.6368 | rank_biserial | 79.0000 | wilcoxon | 0.0020 | 0.0040 | tta8 | two-sided | -2.4119 | 0.0227 | t_test_rel |
| 50pct | hd95_pooled | lower | tta8 | baseline | group | 29 | both_nonempty | 586 | 29 | 0 | 60.0803 | 50.3968 | -9.6836 | -9.5646 | -18.0340 | -1.2675 | -0.6322 | rank_biserial | 80.0000 | wilcoxon | 0.0022 | 0.0040 | tta8 | two-sided | -2.3292 | 0.0273 | t_test_rel |
| 50pct | assd | lower | tta8 | baseline | group | 29 | both_nonempty | 586 | 29 | 0 | 20.1771 | 17.6860 | -2.4911 | -2.6055 | -6.3358 | 2.6918 | -0.7103 | rank_biserial | 63.0000 | wilcoxon | 0.0005 | 0.0014 | tta8 | two-sided | -1.1138 | 0.2748 | t_test_rel |
| 25pct | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.6463 | 0.6606 | 0.0143 | 0.0229 | -0.0221 | 0.0432 | 0.4299 | rank_biserial | 124.0000 | wilcoxon | 0.0431 | 0.0691 | tta4 | two-sided | 0.8100 | 0.4248 | t_test_rel |
| 25pct | boundary_f1 | higher | tta4 | baseline | group | 29 | all_cases | 583 | 29 | 0 | 0.2832 | 0.3123 | 0.0291 | 0.0414 | 0.0099 | 0.0468 | 0.5862 | rank_biserial | 84.0000 | wilcoxon | 0.0056 | 0.0168 | tta4 | two-sided | 2.9462 | 0.0064 | t_test_rel |
| 25pct | hd95 | lower | tta4 | baseline | group | 29 | both_nonempty | 583 | 29 | 0 | 73.0219 | 64.2079 | -8.8141 | -11.8322 | -15.0966 | -0.9879 | -0.6414 | rank_biserial | 78.0000 | wilcoxon | 0.0018 | 0.0091 | tta4 | two-sided | -2.3100 | 0.0285 | t_test_rel |
| 25pct | hd95_pooled | lower | tta4 | baseline | group | 29 | both_nonempty | 583 | 29 | 0 | 68.4800 | 60.5094 | -7.9706 | -9.0773 | -14.3265 | -0.0195 | -0.6184 | rank_biserial | 83.0000 | wilcoxon | 0.0028 | 0.0110 | tta4 | two-sided | -2.0452 | 0.0503 | t_test_rel |
| 25pct | assd | lower | tta4 | baseline | group | 29 | both_nonempty | 583 | 29 | 0 | 22.8115 | 22.5170 | -0.2945 | -1.3740 | -4.1076 | 4.7310 | -0.4483 | rank_biserial | 120.0000 | wilcoxon | 0.0345 | 0.0691 | tta4 | two-sided | -0.1248 | 0.9016 | t_test_rel |
| 25pct | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.6463 | 0.6597 | 0.0134 | 0.0254 | -0.0306 | 0.0472 | 0.3609 | rank_biserial | 139.0000 | wilcoxon | 0.0920 | 0.1246 | tta8 | two-sided | 0.6388 | 0.5281 | t_test_rel |
| 25pct | boundary_f1 | higher | tta8 | baseline | group | 29 | all_cases | 575 | 29 | 0 | 0.2832 | 0.3184 | 0.0352 | 0.0306 | 0.0121 | 0.0567 | 0.5862 | rank_biserial | 84.0000 | wilcoxon | 0.0056 | 0.0224 | tta8 | two-sided | 2.9397 | 0.0065 | t_test_rel |
| 25pct | hd95 | lower | tta8 | baseline | group | 29 | both_nonempty | 575 | 29 | 0 | 73.0219 | 61.9366 | -11.0853 | -9.4346 | -19.8829 | -0.9670 | -0.6368 | rank_biserial | 79.0000 | wilcoxon | 0.0020 | 0.0099 | tta8 | two-sided | -2.1699 | 0.0387 | t_test_rel |
| 25pct | hd95_pooled | lower | tta8 | baseline | group | 29 | both_nonempty | 575 | 29 | 0 | 68.4800 | 59.0941 | -9.3859 | -9.1218 | -18.2326 | 0.9475 | -0.5724 | rank_biserial | 93.0000 | wilcoxon | 0.0060 | 0.0224 | tta8 | two-sided | -1.8096 | 0.0811 | t_test_rel |
| 25pct | assd | lower | tta8 | baseline | group | 29 | both_nonempty | 575 | 29 | 0 | 22.8115 | 22.4418 | -0.3697 | -1.7763 | -4.9411 | 5.3768 | -0.3977 | rank_biserial | 131.0000 | wilcoxon | 0.0623 | 0.1246 | tta8 | two-sided | -0.1344 | 0.8940 | t_test_rel |

Wilcoxon signed-rank, video bootstrap CI of the mean delta, Holm correction.

## (3) Between-fraction gain difference

| metric | scale | relative_definition | reference | compared | treatment | control | metric_direction | n_units | gain_reference | gain_compared | delta_gain | ci_low | ci_high | p_value | favours | score_reference | score_compared | better_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | absolute |  | 100pct | 50pct | tta4 | baseline | higher | 29 | 0.0283 | 0.0261 | -0.0022 | -0.0204 | 0.0164 | 0.9830 | 100pct | 0.7850 | 0.7272 | 100pct |
| dice | relative | remaining_error | 100pct | 50pct | tta4 | baseline | higher | 29 | 0.0727 | 0.0684 | -0.0044 | -0.0774 | 0.0722 | 0.7332 | 100pct | 0.7850 | 0.7272 | 100pct |
| hd95 | absolute |  | 100pct | 50pct | tta4 | baseline | lower | 28 | -14.3491 | -7.6523 | 6.6968 | 0.9052 | 13.4115 | 0.1782 | 100pct | 42.3592 | 55.2026 | 100pct |
| hd95 | relative | log_ratio | 100pct | 50pct | tta4 | baseline | lower | 28 | -0.3818 | -0.2134 | 0.1684 | 0.0613 | 0.2852 | 0.0118 | 100pct | 42.3592 | 55.2026 | 100pct |
| assd | absolute |  | 100pct | 50pct | tta4 | baseline | lower | 28 | -3.7228 | -2.2660 | 1.4567 | -0.2768 | 3.5789 | 0.3989 | 100pct | 10.9227 | 16.1379 | 100pct |
| assd | relative | log_ratio | 100pct | 50pct | tta4 | baseline | lower | 28 | -0.3371 | -0.1965 | 0.1406 | 0.0533 | 0.2372 | 0.0127 | 100pct | 10.9227 | 16.1379 | 100pct |
| dice | absolute |  | 100pct | 25pct | tta4 | baseline | higher | 29 | 0.0283 | 0.0143 | -0.0140 | -0.0519 | 0.0191 | 0.5940 | 100pct | 0.7850 | 0.6606 | 100pct |
| dice | relative | remaining_error | 100pct | 25pct | tta4 | baseline | higher | 29 | 0.0727 | 0.0445 | -0.0282 | -0.1331 | 0.0717 | 0.5504 | 100pct | 0.7850 | 0.6606 | 100pct |
| hd95 | absolute |  | 100pct | 25pct | tta4 | baseline | lower | 28 | -14.3491 | -10.5034 | 3.8457 | -4.2307 | 12.9100 | 0.3052 | 100pct | 42.3592 | 61.1696 | 100pct |
| hd95 | relative | log_ratio | 100pct | 25pct | tta4 | baseline | lower | 28 | -0.3818 | -0.2602 | 0.1216 | -0.0384 | 0.2948 | 0.2098 | 100pct | 42.3592 | 61.1696 | 100pct |
| assd | absolute |  | 100pct | 25pct | tta4 | baseline | lower | 28 | -3.7228 | -2.2074 | 1.5154 | -1.9837 | 5.7065 | 0.5670 | 100pct | 10.9227 | 19.2526 | 100pct |
| assd | relative | log_ratio | 100pct | 25pct | tta4 | baseline | lower | 28 | -0.3371 | -0.2156 | 0.1215 | -0.0448 | 0.2975 | 0.1093 | 100pct | 10.9227 | 19.2526 | 100pct |
| dice | absolute |  | 50pct | 25pct | tta4 | baseline | higher | 29 | 0.0261 | 0.0143 | -0.0119 | -0.0510 | 0.0205 | 0.6391 | 50pct | 0.7272 | 0.6606 | 50pct |
| dice | relative | remaining_error | 50pct | 25pct | tta4 | baseline | higher | 29 | 0.0684 | 0.0445 | -0.0239 | -0.1284 | 0.0702 | 0.6856 | 50pct | 0.7272 | 0.6606 | 50pct |
| hd95 | absolute |  | 50pct | 25pct | tta4 | baseline | lower | 29 | -7.0299 | -9.3581 | -2.3283 | -9.5763 | 5.4191 | 0.4420 | 25pct | 57.8061 | 64.2079 | 50pct |
| hd95 | relative | log_ratio | 50pct | 25pct | tta4 | baseline | lower | 29 | -0.2032 | -0.2456 | -0.0424 | -0.1823 | 0.1063 | 0.5647 | 25pct | 57.8061 | 64.2079 | 50pct |
| assd | absolute |  | 50pct | 25pct | tta4 | baseline | lower | 29 | -1.8741 | 0.2704 | 2.1446 | -2.1499 | 8.2014 | 1.0000 | 50pct | 18.9665 | 22.5170 | 50pct |
| assd | relative | log_ratio | 50pct | 25pct | tta4 | baseline | lower | 29 | -0.1864 | -0.1756 | 0.0108 | -0.1346 | 0.1714 | 0.9830 | 50pct | 18.9665 | 22.5170 | 50pct |

Paired by video. `scale=absolute` is metric units; `scale=relative` is scale-free: `log(treatment / control)` for distances and `(treatment - control) / (1 - control)` for Dice.

## (4) Residual-error test per fraction

| scope | metric | restriction | n_cases | n_clusters | delta_mean | rho_naive | p_naive | ci_low_naive | ci_high_naive | p_bootstrap_naive | null_mean_naive | p_permutation_naive | rho_oldham | p_oldham | ci_low_oldham | ci_high_oldham | p_bootstrap_oldham | null_mean_oldham | p_permutation_oldham | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model:100pct | hd95 | both_nonempty | 594 | 28 | -13.8126 | -0.3978 | 0.0000 | -0.5187 | -0.2857 | 0.0000 | -0.2788 | 0.0116 | -0.2207 | 0.0000 | -0.3651 | -0.0973 | 0.0000 | -0.0002 | 0.0008 | 10000 |
| model:100pct | dice | all_cases | 612 | 29 | 0.0295 | -0.3467 | 0.0000 | -0.4571 | -0.2210 | 0.0000 | -0.2224 | 0.0546 | -0.1817 | 0.0000 | -0.3109 | -0.0354 | 0.0190 | -0.0000 | 0.0341 | 10000 |
| model:50pct | hd95 | both_nonempty | 584 | 29 | -8.3453 | -0.2637 | 0.0000 | -0.3968 | -0.1450 | 0.0000 | -0.2454 | 0.3692 | -0.0715 | 0.0841 | -0.2096 | 0.0491 | 0.2720 | 0.0013 | 0.2817 | 10000 |
| model:50pct | dice | all_cases | 612 | 29 | 0.0258 | -0.1617 | 0.0001 | -0.3506 | 0.0054 | 0.0570 | -0.1764 | 0.5979 | -0.0166 | 0.6817 | -0.1962 | 0.1347 | 0.8040 | 0.0000 | 0.8058 | 10000 |
| model:25pct | hd95 | both_nonempty | 583 | 29 | -11.8215 | -0.3059 | 0.0000 | -0.4479 | -0.1541 | 0.0000 | -0.2873 | 0.3872 | -0.0710 | 0.0866 | -0.2253 | 0.0858 | 0.3600 | 0.0004 | 0.3663 | 10000 |
| model:25pct | dice | all_cases | 612 | 29 | 0.0247 | -0.1191 | 0.0032 | -0.3135 | 0.0518 | 0.1950 | -0.1879 | 0.8126 | 0.0465 | 0.2507 | -0.1613 | 0.2256 | 0.6440 | 0.0007 | 0.5934 | 10000 |
| pooled_models | hd95 | both_nonempty | 1761 | 29 | -11.3403 | -0.3161 | 0.0000 | -0.4215 | -0.2221 | 0.0000 | -0.2680 | 0.0912 | -0.1161 | 0.0000 | -0.2298 | -0.0186 | 0.0160 | 0.0006 | 0.0093 | 10000 |
| pooled_models | dice | all_cases | 1836 | 29 | 0.0267 | -0.2032 | 0.0000 | -0.3458 | -0.0763 | 0.0010 | -0.1905 | 0.4084 | -0.0476 | 0.0415 | -0.1912 | 0.0828 | 0.4880 | 0.0003 | 0.3884 | 10000 |

Oldham's `Spearman((control + treatment)/2, delta)` with a video-cluster bootstrap CI and a paired-swap permutation null; `rho_naive` is the regression-to-the-mean biased version, kept for comparison.
