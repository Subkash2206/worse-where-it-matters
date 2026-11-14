# Cross-model comparison: UNet vs SwinUNETR vs SegFormer

3 models, folds [0, 1, 2, 3, 4], seeds [0, 1, 2]; statistics identical to scripts/aggregate.py (video-level pairing, Holm correction).

## (1) Methods side by side

| model | method | metric | n_cases | n_units | mean | std |
| --- | --- | --- | --- | --- | --- | --- |
| UNet | baseline | dice | 612 | 29 | 0.7515 | 0.1922 |
| UNet | baseline | boundary_f1 | 612 | 29 | 0.4752 | 0.1536 |
| UNet | baseline | hd95 | 612 | 29 | 53.6903 | 30.5946 |
| UNet | tta4 | dice | 612 | 29 | 0.7746 | 0.1986 |
| UNet | tta4 | boundary_f1 | 612 | 29 | 0.5384 | 0.1541 |
| UNet | tta4 | hd95 | 612 | 29 | 43.1376 | 32.4668 |
| UNet | tta8 | dice | 612 | 29 | 0.7785 | 0.2033 |
| UNet | tta8 | boundary_f1 | 612 | 29 | 0.5457 | 0.1636 |
| UNet | tta8 | hd95 | 612 | 29 | 42.3750 | 34.0604 |
| UNet | ensemble3 | dice | 612 | 29 | 0.7781 | 0.2000 |
| UNet | ensemble3 | boundary_f1 | 612 | 29 | 0.5327 | 0.1518 |
| UNet | ensemble3 | hd95 | 612 | 29 | 45.3409 | 35.4574 |
| SwinUNETR | baseline | dice | 612 | 29 | 0.5445 | 0.1964 |
| SwinUNETR | baseline | boundary_f1 | 612 | 29 | 0.1737 | 0.0770 |
| SwinUNETR | baseline | hd95 | 612 | 29 | 92.5643 | 33.4822 |
| SwinUNETR | tta4 | dice | 612 | 29 | 0.5651 | 0.1984 |
| SwinUNETR | tta4 | boundary_f1 | 612 | 29 | 0.1934 | 0.0901 |
| SwinUNETR | tta4 | hd95 | 612 | 29 | 85.6092 | 33.4498 |
| SwinUNETR | tta8 | dice | 612 | 29 | 0.5685 | 0.1962 |
| SwinUNETR | tta8 | boundary_f1 | 612 | 29 | 0.1965 | 0.0933 |
| SwinUNETR | tta8 | hd95 | 612 | 29 | 83.4406 | 33.0019 |
| SwinUNETR | ensemble3 | dice | 612 | 29 | 0.5712 | 0.2088 |
| SwinUNETR | ensemble3 | boundary_f1 | 612 | 29 | 0.1940 | 0.0937 |
| SwinUNETR | ensemble3 | hd95 | 612 | 29 | 85.4627 | 35.9846 |
| SegFormer | baseline | dice | 612 | 29 | 0.8351 | 0.1207 |
| SegFormer | baseline | boundary_f1 | 612 | 29 | 0.5061 | 0.1536 |
| SegFormer | baseline | hd95 | 612 | 29 | 42.0019 | 28.8685 |
| SegFormer | tta4 | dice | 612 | 29 | 0.8459 | 0.1170 |
| SegFormer | tta4 | boundary_f1 | 612 | 29 | 0.5362 | 0.1601 |
| SegFormer | tta4 | hd95 | 612 | 29 | 39.4708 | 28.9890 |
| SegFormer | tta8 | dice | 612 | 29 | 0.8455 | 0.1221 |
| SegFormer | tta8 | boundary_f1 | 612 | 29 | 0.5425 | 0.1658 |
| SegFormer | tta8 | hd95 | 612 | 29 | 38.8911 | 29.1117 |
| SegFormer | ensemble3 | dice | 612 | 29 | 0.8499 | 0.1119 |
| SegFormer | ensemble3 | boundary_f1 | 612 | 29 | 0.5290 | 0.1526 |
| SegFormer | ensemble3 | hd95 | 612 | 29 | 39.1785 | 29.8009 |

Per-video values pooled over the folds, mean ± std over the videos.

## (2) Paired comparisons vs baseline

| model | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7515 | 0.7746 | 0.0231 | 0.0217 | 0.0150 | 0.0311 | 0.8345 | rank_biserial | 36.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | 5.5389 | 0.0000 | t_test_rel |
| UNet | boundary_f1 | higher | tta4 | baseline | group | 28 | all_cases | 598 | 28 | 0 | 0.4887 | 0.5384 | 0.0497 | 0.0413 | 0.0385 | 0.0613 | 0.9951 | rank_biserial | 1.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 8.2476 | 0.0000 | t_test_rel |
| UNet | hd95 | lower | tta4 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 53.0257 | 42.9054 | -10.1203 | -7.1892 | -14.1515 | -6.6242 | -0.8424 | rank_biserial | 32.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | -5.1560 | 0.0000 | t_test_rel |
| UNet | hd95_pooled | lower | tta4 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 49.8818 | 40.3479 | -9.5338 | -7.1660 | -13.4579 | -5.9980 | -0.8325 | rank_biserial | 34.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | -4.9082 | 0.0000 | t_test_rel |
| UNet | assd | lower | tta4 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 14.4827 | 11.3611 | -3.1216 | -2.0869 | -4.5677 | -1.9385 | -0.8374 | rank_biserial | 33.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | -4.4897 | 0.0001 | t_test_rel |
| UNet | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7515 | 0.7785 | 0.0270 | 0.0260 | 0.0181 | 0.0353 | 0.8575 | rank_biserial | 31.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 6.1510 | 0.0000 | t_test_rel |
| UNet | boundary_f1 | higher | tta8 | baseline | group | 28 | all_cases | 598 | 28 | 0 | 0.4887 | 0.5457 | 0.0570 | 0.0492 | 0.0426 | 0.0724 | 0.9803 | rank_biserial | 4.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 7.3259 | 0.0000 | t_test_rel |
| UNet | hd95 | lower | tta8 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 53.0257 | 42.3203 | -10.7054 | -9.7014 | -14.4781 | -7.0455 | -0.8621 | rank_biserial | 28.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.6085 | 0.0000 | t_test_rel |
| UNet | hd95_pooled | lower | tta8 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 49.8818 | 39.7264 | -10.1554 | -9.5192 | -13.8929 | -6.5452 | -0.8621 | rank_biserial | 28.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.3134 | 0.0000 | t_test_rel |
| UNet | assd | lower | tta8 | baseline | group | 28 | both_nonempty | 591 | 28 | 0 | 14.4827 | 11.2533 | -3.2294 | -3.1883 | -4.5151 | -2.0714 | -0.8719 | rank_biserial | 26.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -5.0943 | 0.0000 | t_test_rel |
| UNet | dice | higher | ensemble3 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7515 | 0.7781 | 0.0266 | 0.0248 | 0.0184 | 0.0348 | 0.9218 | rank_biserial | 17.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | 6.3333 | 0.0000 | t_test_rel |
| UNet | boundary_f1 | higher | ensemble3 | baseline | group | 28 | all_cases | 592 | 28 | 0 | 0.4887 | 0.5327 | 0.0440 | 0.0444 | 0.0350 | 0.0539 | 0.9901 | rank_biserial | 2.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | 8.7516 | 0.0000 | t_test_rel |
| UNet | hd95 | lower | ensemble3 | baseline | group | 28 | both_nonempty | 592 | 28 | 0 | 53.0257 | 45.3409 | -7.6848 | -6.9487 | -10.4638 | -4.9363 | -0.8670 | rank_biserial | 27.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | -5.4669 | 0.0000 | t_test_rel |
| UNet | hd95_pooled | lower | ensemble3 | baseline | group | 28 | both_nonempty | 592 | 28 | 0 | 49.8818 | 42.1892 | -7.6925 | -6.7986 | -10.3949 | -5.0247 | -0.8670 | rank_biserial | 27.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | -5.5622 | 0.0000 | t_test_rel |
| UNet | assd | lower | ensemble3 | baseline | group | 28 | both_nonempty | 592 | 28 | 0 | 14.4827 | 11.8730 | -2.6097 | -1.7217 | -3.6353 | -1.6999 | -0.9261 | rank_biserial | 15.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | -5.2014 | 0.0000 | t_test_rel |
| SwinUNETR | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5445 | 0.5651 | 0.0207 | 0.0138 | 0.0104 | 0.0324 | 0.7379 | rank_biserial | 57.0000 | wilcoxon | 0.0002 | 0.0002 | tta4 | two-sided | 3.6763 | 0.0010 | t_test_rel |
| SwinUNETR | boundary_f1 | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.1737 | 0.1934 | 0.0197 | 0.0126 | 0.0113 | 0.0295 | 0.8437 | rank_biserial | 34.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | 4.0228 | 0.0004 | t_test_rel |
| SwinUNETR | hd95 | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 92.5643 | 85.6092 | -6.9551 | -5.9331 | -9.7391 | -4.2474 | -0.8161 | rank_biserial | 40.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | -4.7502 | 0.0001 | t_test_rel |
| SwinUNETR | hd95_pooled | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 87.3939 | 80.9467 | -6.4472 | -4.8066 | -9.2189 | -3.7583 | -0.7747 | rank_biserial | 49.0000 | wilcoxon | 0.0001 | 0.0002 | tta4 | two-sided | -4.4258 | 0.0001 | t_test_rel |
| SwinUNETR | assd | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 31.8325 | 29.3052 | -2.5273 | -1.7737 | -3.5656 | -1.6446 | -0.9126 | rank_biserial | 19.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -4.9922 | 0.0000 | t_test_rel |
| SwinUNETR | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5445 | 0.5685 | 0.0241 | 0.0175 | 0.0113 | 0.0390 | 0.7195 | rank_biserial | 61.0000 | wilcoxon | 0.0004 | 0.0004 | tta8 | two-sided | 3.3436 | 0.0024 | t_test_rel |
| SwinUNETR | boundary_f1 | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.1737 | 0.1965 | 0.0228 | 0.0141 | 0.0129 | 0.0342 | 0.8529 | rank_biserial | 32.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 4.0315 | 0.0004 | t_test_rel |
| SwinUNETR | hd95 | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 92.5643 | 83.4406 | -9.1237 | -7.4083 | -12.9736 | -5.5805 | -0.8575 | rank_biserial | 31.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -4.6511 | 0.0001 | t_test_rel |
| SwinUNETR | hd95_pooled | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 87.3939 | 78.9555 | -8.4384 | -6.6311 | -12.3235 | -5.0285 | -0.8529 | rank_biserial | 32.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -4.3409 | 0.0002 | t_test_rel |
| SwinUNETR | assd | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 31.8325 | 28.6005 | -3.2320 | -2.3649 | -4.8610 | -1.9095 | -0.8391 | rank_biserial | 35.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -4.1954 | 0.0002 | t_test_rel |
| SwinUNETR | dice | higher | ensemble3 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5445 | 0.5712 | 0.0268 | 0.0299 | 0.0183 | 0.0348 | 0.8759 | rank_biserial | 27.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | 6.3022 | 0.0000 | t_test_rel |
| SwinUNETR | boundary_f1 | higher | ensemble3 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.1737 | 0.1940 | 0.0204 | 0.0120 | 0.0119 | 0.0300 | 0.7471 | rank_biserial | 55.0000 | wilcoxon | 0.0002 | 0.0002 | ensemble3 | two-sided | 4.3858 | 0.0001 | t_test_rel |
| SwinUNETR | hd95 | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 92.5643 | 85.4627 | -7.1016 | -8.2462 | -9.9689 | -4.3938 | -0.7701 | rank_biserial | 50.0000 | wilcoxon | 0.0001 | 0.0002 | ensemble3 | two-sided | -4.8852 | 0.0000 | t_test_rel |
| SwinUNETR | hd95_pooled | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 87.3939 | 80.8892 | -6.5047 | -7.1728 | -8.9423 | -4.0806 | -0.8115 | rank_biserial | 41.0000 | wilcoxon | 0.0000 | 0.0001 | ensemble3 | two-sided | -5.2398 | 0.0000 | t_test_rel |
| SwinUNETR | assd | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 31.8325 | 29.1931 | -2.6393 | -2.5811 | -3.5650 | -1.6853 | -0.8483 | rank_biserial | 33.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | -5.3857 | 0.0000 | t_test_rel |
| SegFormer | dice | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.8351 | 0.8459 | 0.0108 | 0.0097 | 0.0062 | 0.0157 | 0.7885 | rank_biserial | 46.0000 | wilcoxon | 0.0001 | 0.0003 | tta4 | two-sided | 4.3032 | 0.0002 | t_test_rel |
| SegFormer | boundary_f1 | higher | tta4 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5061 | 0.5362 | 0.0301 | 0.0245 | 0.0228 | 0.0382 | 1.0000 | rank_biserial | 0.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 7.5894 | 0.0000 | t_test_rel |
| SegFormer | hd95 | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 42.0019 | 39.4708 | -2.5311 | -1.5719 | -4.0780 | -1.0860 | -0.6322 | rank_biserial | 80.0000 | wilcoxon | 0.0022 | 0.0022 | tta4 | two-sided | -3.2006 | 0.0034 | t_test_rel |
| SegFormer | hd95_pooled | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 38.8773 | 36.5281 | -2.3492 | -1.5036 | -3.7907 | -1.0753 | -0.6828 | rank_biserial | 69.0000 | wilcoxon | 0.0008 | 0.0016 | tta4 | two-sided | -3.2907 | 0.0027 | t_test_rel |
| SegFormer | assd | lower | tta4 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 10.8388 | 10.2012 | -0.6376 | -0.5106 | -0.9764 | -0.2966 | -0.7241 | rank_biserial | 60.0000 | wilcoxon | 0.0003 | 0.0010 | tta4 | two-sided | -3.5004 | 0.0016 | t_test_rel |
| SegFormer | dice | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.8351 | 0.8455 | 0.0103 | 0.0124 | 0.0051 | 0.0159 | 0.6736 | rank_biserial | 71.0000 | wilcoxon | 0.0010 | 0.0029 | tta8 | two-sided | 3.6820 | 0.0010 | t_test_rel |
| SegFormer | boundary_f1 | higher | tta8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5061 | 0.5425 | 0.0365 | 0.0341 | 0.0277 | 0.0457 | 1.0000 | rank_biserial | 0.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 7.6876 | 0.0000 | t_test_rel |
| SegFormer | hd95 | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 42.0019 | 38.8911 | -3.1108 | -1.5698 | -4.9847 | -1.4406 | -0.6690 | rank_biserial | 72.0000 | wilcoxon | 0.0011 | 0.0029 | tta8 | two-sided | -3.3768 | 0.0022 | t_test_rel |
| SegFormer | hd95_pooled | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 38.8773 | 36.0499 | -2.8274 | -1.6447 | -4.5270 | -1.3735 | -0.6920 | rank_biserial | 67.0000 | wilcoxon | 0.0007 | 0.0027 | tta8 | two-sided | -3.4553 | 0.0018 | t_test_rel |
| SegFormer | assd | lower | tta8 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 10.8388 | 9.9994 | -0.8394 | -0.7028 | -1.3613 | -0.3781 | -0.6690 | rank_biserial | 72.0000 | wilcoxon | 0.0011 | 0.0029 | tta8 | two-sided | -3.2791 | 0.0028 | t_test_rel |
| SegFormer | dice | higher | ensemble3 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.8351 | 0.8499 | 0.0147 | 0.0063 | 0.0076 | 0.0270 | 0.9954 | rank_biserial | 1.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | 2.7931 | 0.0093 | t_test_rel |
| SegFormer | boundary_f1 | higher | ensemble3 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.5061 | 0.5290 | 0.0230 | 0.0210 | 0.0159 | 0.0324 | 0.9862 | rank_biserial | 3.0000 | wilcoxon | 0.0000 | 0.0000 | ensemble3 | two-sided | 5.3429 | 0.0000 | t_test_rel |
| SegFormer | hd95 | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 42.0019 | 39.1785 | -2.8234 | -2.0232 | -4.4980 | -1.2683 | -0.6414 | rank_biserial | 78.0000 | wilcoxon | 0.0018 | 0.0036 | ensemble3 | two-sided | -3.2582 | 0.0029 | t_test_rel |
| SegFormer | hd95_pooled | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 38.8773 | 35.9247 | -2.9526 | -1.8173 | -5.0341 | -1.1700 | -0.6230 | rank_biserial | 82.0000 | wilcoxon | 0.0025 | 0.0036 | ensemble3 | two-sided | -2.9125 | 0.0070 | t_test_rel |
| SegFormer | assd | lower | ensemble3 | baseline | group | 29 | both_nonempty | 612 | 29 | 0 | 10.8388 | 9.8427 | -0.9961 | -0.4551 | -1.7255 | -0.4222 | -0.7057 | rank_biserial | 64.0000 | wilcoxon | 0.0005 | 0.0015 | ensemble3 | two-sided | -2.8803 | 0.0075 | t_test_rel |

Wilcoxon signed-rank, bootstrap CI of the mean delta, rank-biserial effect size, Holm correction; distance metrics on both-non-empty cases.

## (3) Missed detections

| model | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet | baseline | baseline | 612 | 29 | 15 | 0.0245 | 0.0082 | 0.0491 | cluster_bootstrap_video | 612 |  |  |  |  |  |  | case |
| UNet | tta4 | baseline | 612 | 29 | 19 | 0.0310 | 0.0094 | 0.0621 | cluster_bootstrap_video | 612 | 591.0000 | 6.0000 | 2.0000 | 13.0000 | 0.2891 | 0.5781 | case |
| UNet | tta8 | baseline | 612 | 29 | 19 | 0.0310 | 0.0108 | 0.0614 | cluster_bootstrap_video | 612 | 591.0000 | 6.0000 | 2.0000 | 13.0000 | 0.2891 | 0.5781 | case |
| UNet | ensemble3 | baseline | 612 | 29 | 20 | 0.0327 | 0.0128 | 0.0613 | cluster_bootstrap_video | 612 | 592.0000 | 5.0000 | 0.0000 | 15.0000 | 0.0625 | 0.1875 | case |
| SwinUNETR | baseline | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 |  |  |  |  |  |  | case |
| SwinUNETR | tta4 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| SwinUNETR | tta8 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| SwinUNETR | ensemble3 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| SegFormer | baseline | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 |  |  |  |  |  |  | case |
| SegFormer | tta4 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| SegFormer | tta8 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| SegFormer | ensemble3 | baseline | 612 | 29 | 0 | 0.0000 | 0.0000 | 0.0000 | cluster_bootstrap_video | 612 | 612.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |

Cluster bootstrap CI of the missed rate, transitions vs the model's own baseline and exact McNemar.

## (4) Tail metrics

| model | method | n_cases | n_videos | n_cases_hd_defined | hd95_p90 | hd95_p95 | hd95_median | hd95_failure_rate | dice_failure_rate | delta_hd95_p90_estimate | delta_hd95_p90_ci_low | delta_hd95_p90_ci_high | delta_hd95_p95_estimate | delta_hd95_p95_ci_low | delta_hd95_p95_ci_high | delta_hd95_failure_rate_estimate | delta_hd95_failure_rate_ci_low | delta_hd95_failure_rate_ci_high | delta_dice_failure_rate_estimate | delta_dice_failure_rate_ci_low | delta_dice_failure_rate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNet | baseline | 612 | 29 | 605 | 122.3093 | 149.8642 | 36.8100 | 0.4101 | 0.1144 |  |  |  |  |  |  |  |  |  |  |  |  |
| UNet | ensemble3 | 612 | 29 | 592 | 130.0335 | 156.5783 | 17.6201 | 0.3203 | 0.1062 | 7.7242 | -9.1429 | 13.4664 | 6.7141 | -1.6700 | 11.8303 | -0.0899 | -0.1300 | -0.0514 | -0.0082 | -0.0179 | 0.0000 |
| UNet | tta2 | 612 | 29 | 601 | 116.6130 | 147.5756 | 29.0121 | 0.3644 | 0.1144 | -5.6963 | -14.0493 | 0.3802 | -2.2885 | -13.3875 | 3.5673 | -0.0458 | -0.0760 | -0.0190 | 0.0000 | -0.0130 | 0.0121 |
| UNet | tta4 | 612 | 29 | 598 | 112.0310 | 146.7790 | 23.7922 | 0.3105 | 0.1111 | -10.2783 | -25.5623 | 1.8625 | -3.0852 | -24.5075 | 4.1437 | -0.0997 | -0.1479 | -0.0522 | -0.0033 | -0.0162 | 0.0082 |
| UNet | tta8 | 612 | 29 | 598 | 113.2703 | 143.5993 | 21.3033 | 0.2990 | 0.1144 | -9.0390 | -28.7004 | 1.3407 | -6.2649 | -23.6209 | 6.2855 | -0.1111 | -0.1564 | -0.0689 | 0.0000 | -0.0131 | 0.0134 |
| SwinUNETR | baseline | 612 | 29 | 612 | 167.1760 | 183.5830 | 86.2495 | 0.7598 | 0.3905 |  |  |  |  |  |  |  |  |  |  |  |  |
| SwinUNETR | ensemble3 | 612 | 29 | 612 | 165.4210 | 198.2790 | 75.5559 | 0.6781 | 0.3644 | -1.7550 | -7.7972 | 18.4873 | 14.6960 | -7.4216 | 18.1270 | -0.0817 | -0.1182 | -0.0484 | -0.0261 | -0.0449 | -0.0098 |
| SwinUNETR | tta2 | 612 | 29 | 612 | 165.2445 | 179.9065 | 83.3671 | 0.7353 | 0.3807 | -1.9315 | -7.7480 | 1.5525 | -3.6765 | -12.2280 | -0.1175 | -0.0245 | -0.0550 | 0.0069 | -0.0098 | -0.0305 | 0.0084 |
| SwinUNETR | tta4 | 612 | 29 | 612 | 158.1672 | 174.3871 | 78.6482 | 0.7124 | 0.3791 | -9.0088 | -14.7256 | -1.5231 | -9.1958 | -18.7738 | -1.1266 | -0.0474 | -0.0822 | -0.0150 | -0.0114 | -0.0334 | 0.0065 |
| SwinUNETR | tta8 | 612 | 29 | 612 | 157.0477 | 169.0857 | 75.7500 | 0.6944 | 0.3627 | -10.1283 | -20.0300 | -3.1620 | -14.4973 | -26.6046 | -5.1505 | -0.0654 | -0.1069 | -0.0272 | -0.0278 | -0.0661 | 0.0016 |
| SegFormer | baseline | 612 | 29 | 612 | 103.3190 | 141.0195 | 23.1797 | 0.2778 | 0.0588 |  |  |  |  |  |  |  |  |  |  |  |  |
| SegFormer | ensemble3 | 612 | 29 | 612 | 112.5083 | 145.2537 | 15.6602 | 0.2484 | 0.0605 | 9.1893 | -5.0760 | 19.1493 | 4.2342 | -2.9406 | 17.8250 | -0.0294 | -0.0511 | -0.0098 | 0.0016 | -0.0054 | 0.0081 |
| SegFormer | tta2 | 612 | 29 | 612 | 100.0514 | 144.0022 | 21.6898 | 0.2729 | 0.0507 | -3.2676 | -11.6882 | 3.6115 | 2.9827 | -12.2602 | 8.3424 | -0.0049 | -0.0216 | 0.0101 | -0.0082 | -0.0188 | 0.0015 |
| SegFormer | tta4 | 612 | 29 | 612 | 99.3413 | 143.6046 | 19.1628 | 0.2565 | 0.0539 | -3.9777 | -14.4703 | 11.7797 | 2.5851 | -7.8835 | 9.7100 | -0.0212 | -0.0379 | -0.0035 | -0.0049 | -0.0157 | 0.0051 |
| SegFormer | tta8 | 612 | 29 | 612 | 101.1924 | 144.4709 | 18.6261 | 0.2565 | 0.0556 | -2.1266 | -18.7894 | 10.5779 | 3.4514 | -9.1973 | 11.0841 | -0.0212 | -0.0406 | -0.0030 | -0.0033 | -0.0119 | 0.0054 |

HD95 percentiles and failure rates (HD95 > 50 px, Dice < 0.5).

## (5) Residual-error test

| scope | metric | restriction | n_cases | n_clusters | delta_mean | rho_naive | p_naive | ci_low_naive | ci_high_naive | p_bootstrap_naive | null_mean_naive | p_permutation_naive | rho_oldham | p_oldham | ci_low_oldham | ci_high_oldham | p_bootstrap_oldham | null_mean_oldham | p_permutation_oldham | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model:UNet | hd95 | both_nonempty | 598 | 28 | -10.3751 | -0.2527 | 0.0000 | -0.4390 | -0.0787 | 0.0080 | -0.2629 | 0.5638 | -0.0732 | 0.0736 | -0.2634 | 0.0917 | 0.3860 | 0.0009 | 0.3399 | 10000 |
| model:UNet | dice | all_cases | 612 | 29 | 0.0237 | -0.2338 | 0.0000 | -0.3956 | -0.0776 | 0.0010 | -0.1765 | 0.2362 | -0.1121 | 0.0055 | -0.2811 | 0.0521 | 0.1910 | -0.0003 | 0.1617 | 10000 |
| model:SwinUNETR | hd95 | both_nonempty | 612 | 29 | -6.8355 | -0.1315 | 0.0011 | -0.2516 | -0.0242 | 0.0190 | -0.1589 | 0.6470 | 0.0204 | 0.6140 | -0.0942 | 0.1221 | 0.8060 | 0.0002 | 0.7946 | 10000 |
| model:SwinUNETR | dice | all_cases | 612 | 29 | 0.0183 | 0.0913 | 0.0239 | -0.0422 | 0.2088 | 0.1730 | -0.0917 | 0.5235 | 0.1723 | 0.0000 | 0.0498 | 0.2847 | 0.0090 | -0.0002 | 0.0389 | 10000 |
| model:SegFormer | hd95 | both_nonempty | 612 | 29 | -2.3921 | -0.2133 | 0.0000 | -0.3376 | -0.0818 | 0.0020 | -0.1243 | 0.0961 | -0.1094 | 0.0067 | -0.2324 | 0.0205 | 0.1030 | -0.0001 | 0.1153 | 10000 |
| model:SegFormer | dice | all_cases | 612 | 29 | 0.0104 | -0.2357 | 0.0000 | -0.3682 | -0.0868 | 0.0000 | -0.1019 | 0.0627 | -0.1524 | 0.0002 | -0.2864 | 0.0007 | 0.0520 | -0.0005 | 0.0780 | 10000 |
| pooled_models | hd95 | both_nonempty | 1822 | 29 | -6.5047 | -0.1816 | 0.0000 | -0.2851 | -0.0960 | 0.0000 | -0.1627 | 0.2673 | -0.0484 | 0.0388 | -0.1531 | 0.0388 | 0.2700 | -0.0001 | 0.1349 | 10000 |
| pooled_models | dice | all_cases | 1836 | 29 | 0.0175 | -0.1222 | 0.0000 | -0.2226 | -0.0270 | 0.0100 | -0.1077 | 0.3911 | -0.0379 | 0.1043 | -0.1388 | 0.0645 | 0.4460 | 0.0000 | 0.4696 | 10000 |

Primary: Oldham's `Spearman((control + treatment)/2, treatment - control)`, which removes the regression-to-the-mean bias of the naive `Spearman(control, delta)` (kept as `*_naive`, secondary), plus a paired-swap permutation null (`null_mean_*`, `p_permutation_*`: whole videos swapped, 10k draws). `delta = treatment - control`, so a negative rho means the gain is larger where the control was worse, for HD95 and Dice alike. `pooled_models` repeats it over both models; every CI resamples videos.

## (6) Between-model gain difference

| metric | scale | relative_definition | reference | compared | treatment | control | metric_direction | n_units | gain_reference | gain_compared | delta_gain | ci_low | ci_high | p_value | favours | score_reference | score_compared | better_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | absolute |  | UNet | SwinUNETR | tta4 | baseline | higher | 29 | 0.0231 | 0.0207 | -0.0024 | -0.0162 | 0.0134 | 0.3692 | UNet | 0.7746 | 0.5651 | UNet |
| dice | relative | remaining_error | UNet | SwinUNETR | tta4 | baseline | higher | 29 | 0.1300 | 0.0625 | -0.0674 | -0.1106 | -0.0216 | 0.0065 | UNet | 0.7746 | 0.5651 | UNet |
| hd95 | absolute |  | UNet | SwinUNETR | tta4 | baseline | lower | 28 | -10.5596 | -6.5637 | 3.9959 | 0.0016 | 8.2419 | 0.0900 | UNet | 43.1376 | 86.0420 | UNet |
| hd95 | relative | log_ratio | UNet | SwinUNETR | tta4 | baseline | lower | 28 | -0.3910 | -0.1113 | 0.2797 | 0.1609 | 0.4092 | 0.0000 | UNet | 43.1376 | 86.0420 | UNet |
| assd | absolute |  | UNet | SwinUNETR | tta4 | baseline | lower | 28 | -3.1242 | -2.4205 | 0.7037 | -0.8922 | 2.4885 | 0.4791 | UNet | 11.5054 | 29.4562 | UNet |
| assd | relative | log_ratio | UNet | SwinUNETR | tta4 | baseline | lower | 28 | -0.3447 | -0.1181 | 0.2266 | 0.1247 | 0.3358 | 0.0002 | UNet | 11.5054 | 29.4562 | UNet |
| dice | absolute |  | UNet | SegFormer | tta4 | baseline | higher | 29 | 0.0231 | 0.0108 | -0.0123 | -0.0210 | -0.0034 | 0.0017 | UNet | 0.7746 | 0.8459 | SegFormer |
| dice | relative | remaining_error | UNet | SegFormer | tta4 | baseline | higher | 29 | 0.1300 | 0.0711 | -0.0589 | -0.0931 | -0.0261 | 0.0038 | UNet | 0.7746 | 0.8459 | SegFormer |
| hd95 | absolute |  | UNet | SegFormer | tta4 | baseline | lower | 28 | -10.5596 | -2.1118 | 8.4477 | 5.2211 | 12.1313 | 0.0000 | UNet | 43.1376 | 38.2489 | SegFormer |
| hd95 | relative | log_ratio | UNet | SegFormer | tta4 | baseline | lower | 28 | -0.3910 | -0.1085 | 0.2825 | 0.1747 | 0.4177 | 0.0000 | UNet | 43.1376 | 38.2489 | SegFormer |
| assd | absolute |  | UNet | SegFormer | tta4 | baseline | lower | 28 | -3.1242 | -0.5614 | 2.5628 | 1.4902 | 3.8242 | 0.0000 | UNet | 11.5054 | 9.4271 | SegFormer |
| assd | relative | log_ratio | UNet | SegFormer | tta4 | baseline | lower | 28 | -0.3447 | -0.1040 | 0.2406 | 0.1532 | 0.3496 | 0.0000 | UNet | 11.5054 | 9.4271 | SegFormer |

Per video, `gain = treatment - control` inside each model; the models are paired by video and `delta_gain = compared - reference` is tested with a Wilcoxon signed-rank test and a paired video bootstrap CI. `favours` names the model with the larger gain, `better_model` the model with the better mean score (the two differ when both gains are negative). Every metric appears twice: `scale=absolute` (metric units, dominated by whichever model works on the larger scale) and `scale=relative` — `relative_definition=log_ratio` is `log(treatment / control)` for distances, so a 10 % reduction counts the same at 90 px and at 40 px, and `relative_definition=remaining_error` is `(treatment - control) / (1 - control)` for Dice, the fraction of the error the treatment recovers. Needs two configs.
