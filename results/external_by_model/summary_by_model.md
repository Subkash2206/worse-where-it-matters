# External polyp validation, split by architecture

One architecture at a time (15 UNet checkpoints, 15 SwinUNetR checkpoints), so the external numbers are no longer an average over two different models. Metrics are averaged per case over the 15 checkpoints of the architecture and then summarised over cases; the paired tests use those per-case values, so one image stays one independent unit.

## kvasir_seg / unet_resnet50

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 1000 | 0.7159 | 0.1980 | 0.0000 |
| dice | tta2 | 1000 | 0.7326 | 0.1990 | 2.3403 |
| dice | tta4 | 1000 | 0.7474 | 0.1993 | 4.4060 |
| dice | tta8 | 1000 | 0.7573 | 0.1976 | 5.7845 |
| boundary_f1 | baseline | 1000 | 0.2716 | 0.1239 | 0.0000 |
| boundary_f1 | tta2 | 1000 | 0.2809 | 0.1335 | 3.4499 |
| boundary_f1 | tta4 | 1000 | 0.3032 | 0.1434 | 11.6455 |
| boundary_f1 | tta8 | 1000 | 0.3155 | 0.1489 | 16.1661 |
| hd95 | baseline | 1000 | 170.6482 | 93.9310 | 0.0000 |
| hd95 | tta2 | 1000 | 162.7878 | 96.9361 | -4.6062 |
| hd95 | tta4 | 1000 | 151.3471 | 97.8868 | -11.3105 |
| hd95 | tta8 | 1000 | 143.4853 | 97.7722 | -15.9175 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 1000 | 0.0168 | 2.3403 | 0.0153 | 0.0183 | 0.0000 | 0.0000 | 0.7238 |
| dice | tta4 vs baseline | 1000 | 0.0315 | 4.4060 | 0.0294 | 0.0337 | 0.0000 | 0.0000 | 0.8605 |
| dice | tta8 vs baseline | 1000 | 0.0414 | 5.7845 | 0.0388 | 0.0440 | 0.0000 | 0.0000 | 0.9091 |
| boundary_f1 | tta2 vs baseline | 1000 | 0.0094 | 3.4499 | 0.0077 | 0.0110 | 0.0000 | 0.0000 | 0.3591 |
| boundary_f1 | tta4 vs baseline | 1000 | 0.0316 | 11.6455 | 0.0291 | 0.0342 | 0.0000 | 0.0000 | 0.7376 |
| boundary_f1 | tta8 vs baseline | 1000 | 0.0439 | 16.1661 | 0.0409 | 0.0470 | 0.0000 | 0.0000 | 0.8175 |
| hd95 | tta2 vs baseline | 1000 | -7.8604 | -4.6062 | -9.0590 | -6.6961 | 0.0000 | 0.0000 | -0.4963 |
| hd95 | tta4 vs baseline | 1000 | -19.3011 | -11.3105 | -20.9782 | -17.7008 | 0.0000 | 0.0000 | -0.7525 |
| hd95 | tta8 vs baseline | 1000 | -27.1629 | -15.9175 | -29.1594 | -25.2815 | 0.0000 | 0.0000 | -0.8494 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1000 | 15 | 20 | 1.3333 | 0 | 4 | 13 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 1000 | 15 | 11 | 0.7333 | 0 | 3 | 7 | 0 | 3.0000 | 12.0000 | 0.0352 | 0.1235 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 1000 | 15 | 10 | 0.6667 | 0 | 4 | 8 | 0 | 4.0000 | 14.0000 | 0.0309 | 0.1235 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 1000 | 15 | 19 | 1.2667 | 0 | 7 | 13 | 0 | 11.0000 | 12.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7763 | 0.7159 | -0.0604 | -7.7791 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.7908 | 0.7326 | -0.0581 | -7.3512 | 0.0145 | 0.0168 | 1.8676893843473927 | 2.340318529569006 | 0.0023 | larger externally |
| tta4 | 0.7999 | 0.7474 | -0.0525 | -6.5630 | 0.0237 | 0.0315 | 3.0471782024802123 | 4.405991595006498 | 0.0079 | larger externally |
| tta8 | 0.8046 | 0.7573 | -0.0473 | -5.8843 | 0.0284 | 0.0414 | 3.65473183788675 | 5.784498718625247 | 0.0130 | larger externally |
| tta4 -> tta8 | 0.0047 | 0.0099 | 0.0052 |  | 0.0047 | 0.0099 | <NA> | <NA> | 0.0052 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 334 | 0.5986 | 0.2338 | 0.0000 |
| tta2 | small | 334 | 0.6217 | 0.2404 | 0.0000 |
| tta4 | small | 334 | 0.6426 | 0.2450 | 0.0000 |
| tta8 | small | 334 | 0.6597 | 0.2451 | 0.0000 |
| baseline | medium | 333 | 0.7698 | 0.1474 | 0.0000 |
| tta2 | medium | 333 | 0.7872 | 0.1463 | 0.0000 |
| tta4 | medium | 333 | 0.8024 | 0.1470 | 0.0000 |
| tta8 | medium | 333 | 0.8102 | 0.1462 | 0.0000 |
| baseline | large | 333 | 0.7796 | 0.1435 | 0.0000 |
| tta2 | large | 333 | 0.7894 | 0.1457 | 0.0000 |
| tta4 | large | 333 | 0.7976 | 0.1454 | 0.0000 |
| tta8 | large | 333 | 0.8023 | 0.1464 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 334 | 0.0231 | 3.8544 | 0.0198 | 0.0263 | 0.0000 | 0.0000 |
| dice | small | tta4 vs baseline | 334 | 0.0440 | 7.3577 | 0.0396 | 0.0486 | 0.0000 | 0.0000 |
| dice | small | tta8 vs baseline | 334 | 0.0611 | 10.2110 | 0.0558 | 0.0667 | 0.0000 | 0.0000 |
| dice | medium | tta2 vs baseline | 333 | 0.0173 | 2.2520 | 0.0152 | 0.0196 | 0.0000 | 0.0000 |
| dice | medium | tta4 vs baseline | 333 | 0.0325 | 4.2239 | 0.0294 | 0.0358 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 333 | 0.0403 | 5.2400 | 0.0367 | 0.0442 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 333 | 0.0098 | 1.2614 | 0.0079 | 0.0118 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 333 | 0.0180 | 2.3126 | 0.0155 | 0.0207 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 333 | 0.0227 | 2.9132 | 0.0198 | 0.0257 | 0.0000 | 0.0000 |
| hd95 | small | tta2 vs baseline | 334 | -9.1711 | -4.4626 | -11.6827 | -6.6847 | 0.0000 | 0.0000 |
| hd95 | small | tta4 vs baseline | 334 | -25.2310 | -12.2772 | -28.5552 | -21.9505 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 334 | -35.6877 | -17.3653 | -39.5806 | -31.8676 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 333 | -8.4689 | -5.4874 | -10.4196 | -6.5735 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 333 | -20.4322 | -13.2388 | -23.0618 | -17.9135 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 333 | -29.6085 | -19.1845 | -32.5728 | -26.7250 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 333 | -5.9371 | -3.9062 | -7.3744 | -4.5280 | 0.0000 | 0.0000 |
| hd95 | large | tta4 vs baseline | 333 | -12.2224 | -8.0414 | -14.6176 | -9.9565 | 0.0000 | 0.0000 |
| hd95 | large | tta8 vs baseline | 333 | -16.1670 | -10.6367 | -19.1987 | -13.4153 | 0.0000 | 0.0000 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 50.7156 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta4 vs baseline | 86.7976 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta8 vs baseline | 119.6718 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta2 vs baseline | 6.0710 | 0.0481 | 0.0481 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta4 vs baseline | 44.3593 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta8 vs baseline | 78.2594 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |



## kvasir_seg / swinunetr_2d

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 1000 | 0.3969 | 0.2127 | 0.0000 |
| dice | tta2 | 1000 | 0.4055 | 0.2174 | 2.1722 |
| dice | tta4 | 1000 | 0.4046 | 0.2224 | 1.9332 |
| dice | tta8 | 1000 | 0.4033 | 0.2250 | 1.6061 |
| boundary_f1 | baseline | 1000 | 0.0634 | 0.0474 | 0.0000 |
| boundary_f1 | tta2 | 1000 | 0.0656 | 0.0512 | 3.4616 |
| boundary_f1 | tta4 | 1000 | 0.0681 | 0.0561 | 7.4284 |
| boundary_f1 | tta8 | 1000 | 0.0690 | 0.0580 | 8.9529 |
| hd95 | baseline | 1000 | 229.6400 | 85.0779 | 0.0000 |
| hd95 | tta2 | 1000 | 226.0676 | 84.9113 | -1.5556 |
| hd95 | tta4 | 1000 | 222.7720 | 84.6486 | -2.9908 |
| hd95 | tta8 | 1000 | 220.4175 | 84.1324 | -4.0161 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 1000 | 0.0086 | 2.1722 | 0.0075 | 0.0097 | 0.0000 | 0.0000 | 0.5129 |
| dice | tta4 vs baseline | 1000 | 0.0077 | 1.9332 | 0.0061 | 0.0093 | 0.0000 | 0.0000 | 0.3011 |
| dice | tta8 vs baseline | 1000 | 0.0064 | 1.6061 | 0.0045 | 0.0082 | 0.0000 | 0.0000 | 0.1919 |
| boundary_f1 | tta2 vs baseline | 1000 | 0.0022 | 3.4616 | 0.0016 | 0.0028 | 0.0000 | 0.0000 | 0.1898 |
| boundary_f1 | tta4 vs baseline | 1000 | 0.0047 | 7.4284 | 0.0038 | 0.0056 | 0.0000 | 0.0000 | 0.2705 |
| boundary_f1 | tta8 vs baseline | 1000 | 0.0057 | 8.9529 | 0.0047 | 0.0068 | 0.0000 | 0.0000 | 0.2679 |
| hd95 | tta2 vs baseline | 1000 | -3.5723 | -1.5556 | -4.3292 | -2.8719 | 0.0000 | 0.0000 | -0.3420 |
| hd95 | tta4 vs baseline | 1000 | -6.8680 | -2.9908 | -7.9107 | -5.8345 | 0.0000 | 0.0000 | -0.4655 |
| hd95 | tta8 vs baseline | 1000 | -9.2225 | -4.0161 | -10.4931 | -7.9600 | 0.0000 | 0.0000 | -0.5125 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1000 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 1000 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 1000 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 1000 | 15 | 1 | 0.0667 | 0 | 1 | 1 | 0 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.5471 | 0.3969 | -0.1502 | -27.4493 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.5600 | 0.4055 | -0.1545 | -27.5875 | 0.0130 | 0.0086 | 2.3672680016008556 | 2.172231739703382 | -0.0043 | smaller externally |
| tta4 | 0.5654 | 0.4046 | -0.1608 | -28.4407 | 0.0183 | 0.0077 | 3.3454477994831624 | 1.933222687499348 | -0.0106 | smaller externally |
| tta8 | 0.5680 | 0.4033 | -0.1647 | -28.9960 | 0.0209 | 0.0064 | 3.8194110430798576 | 1.6060882456230912 | -0.0145 | smaller externally |
| tta4 -> tta8 | 0.0026 | -0.0013 | -0.0039 |  | 0.0026 | -0.0013 | <NA> | <NA> | -0.0039 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 334 | 0.3126 | 0.2080 | 0.0000 |
| tta2 | small | 334 | 0.3234 | 0.2157 | 0.0000 |
| tta4 | small | 334 | 0.3266 | 0.2224 | 0.0000 |
| tta8 | small | 334 | 0.3287 | 0.2259 | 0.0000 |
| baseline | medium | 333 | 0.4473 | 0.2205 | 0.0000 |
| tta2 | medium | 333 | 0.4576 | 0.2256 | 0.0000 |
| tta4 | medium | 333 | 0.4559 | 0.2321 | 0.0000 |
| tta8 | medium | 333 | 0.4533 | 0.2359 | 0.0000 |
| baseline | large | 333 | 0.4311 | 0.1823 | 0.0000 |
| tta2 | large | 333 | 0.4358 | 0.1851 | 0.0000 |
| tta4 | large | 333 | 0.4314 | 0.1896 | 0.0000 |
| tta8 | large | 333 | 0.4281 | 0.1916 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 334 | 0.0108 | 3.4561 | 0.0087 | 0.0129 | 0.0000 | 0.0000 |
| dice | small | tta4 vs baseline | 334 | 0.0141 | 4.5093 | 0.0110 | 0.0172 | 0.0000 | 0.0000 |
| dice | small | tta8 vs baseline | 334 | 0.0161 | 5.1544 | 0.0125 | 0.0197 | 0.0000 | 0.0000 |
| dice | medium | tta2 vs baseline | 333 | 0.0103 | 2.3068 | 0.0082 | 0.0125 | 0.0000 | 0.0000 |
| dice | medium | tta4 vs baseline | 333 | 0.0086 | 1.9181 | 0.0056 | 0.0115 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 333 | 0.0060 | 1.3308 | 0.0026 | 0.0093 | 0.0004 | 0.0004 |
| dice | large | tta2 vs baseline | 333 | 0.0047 | 1.0990 | 0.0034 | 0.0061 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 333 | 0.0003 | 0.0756 | -0.0018 | 0.0025 | 0.5505 | 0.5505 |
| dice | large | tta8 vs baseline | 333 | -0.0030 | -0.6885 | -0.0053 | -0.0006 | 0.0002 | 0.0003 |
| hd95 | small | tta2 vs baseline | 334 | -4.7027 | -1.7224 | -6.1015 | -3.4323 | 0.0000 | 0.0000 |
| hd95 | small | tta4 vs baseline | 334 | -10.3059 | -3.7746 | -12.0334 | -8.6573 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 334 | -14.7072 | -5.3865 | -16.8508 | -12.6705 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 333 | -4.3199 | -2.0008 | -5.4723 | -3.1982 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 333 | -9.1238 | -4.2257 | -10.7680 | -7.5457 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 333 | -12.0526 | -5.5822 | -13.9614 | -10.2301 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 333 | -1.6909 | -0.8461 | -2.9622 | -0.4634 | 0.0611 | 0.0611 |
| hd95 | large | tta4 vs baseline | 333 | -1.1639 | -0.5824 | -3.1028 | 0.7279 | 0.7447 | 0.7447 |
| hd95 | large | tta8 vs baseline | 333 | -0.8912 | -0.4460 | -3.1786 | 1.3343 | 0.8447 | 0.8447 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 23.4050 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta4 vs baseline | 52.1219 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta8 vs baseline | 69.5283 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta2 vs baseline | 19.6968 | 0.0001 | 0.0001 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta4 vs baseline | 80.3040 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta8 vs baseline | 109.8672 | 0.0000 | 0.0000 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |



## kvasir_seg / segformer_mit_b2

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 1000 | 0.8428 | 0.1555 | 0.0000 |
| dice | tta2 | 1000 | 0.8474 | 0.1557 | 0.5412 |
| dice | tta4 | 1000 | 0.8511 | 0.1560 | 0.9843 |
| dice | tta8 | 1000 | 0.8528 | 0.1566 | 1.1927 |
| boundary_f1 | baseline | 1000 | 0.3345 | 0.1526 | 0.0000 |
| boundary_f1 | tta2 | 1000 | 0.3483 | 0.1611 | 4.1150 |
| boundary_f1 | tta4 | 1000 | 0.3616 | 0.1698 | 8.1125 |
| boundary_f1 | tta8 | 1000 | 0.3655 | 0.1737 | 9.2620 |
| hd95 | baseline | 1000 | 84.0048 | 75.3965 | 0.0000 |
| hd95 | tta2 | 1000 | 81.3946 | 76.8433 | -3.1072 |
| hd95 | tta4 | 1000 | 78.3163 | 76.1935 | -6.7716 |
| hd95 | tta8 | 1000 | 76.3208 | 75.4985 | -9.1471 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 1000 | 0.0046 | 0.5412 | 0.0036 | 0.0055 | 0.0000 | 0.0000 | 0.4503 |
| dice | tta4 vs baseline | 1000 | 0.0083 | 0.9843 | 0.0069 | 0.0097 | 0.0000 | 0.0000 | 0.5406 |
| dice | tta8 vs baseline | 1000 | 0.0101 | 1.1927 | 0.0086 | 0.0116 | 0.0000 | 0.0000 | 0.5830 |
| boundary_f1 | tta2 vs baseline | 1000 | 0.0138 | 4.1150 | 0.0117 | 0.0159 | 0.0000 | 0.0000 | 0.4446 |
| boundary_f1 | tta4 vs baseline | 1000 | 0.0271 | 8.1125 | 0.0243 | 0.0300 | 0.0000 | 0.0000 | 0.6237 |
| boundary_f1 | tta8 vs baseline | 1000 | 0.0310 | 9.2620 | 0.0279 | 0.0341 | 0.0000 | 0.0000 | 0.6365 |
| hd95 | tta2 vs baseline | 1000 | -2.6102 | -3.1072 | -3.3451 | -1.8841 | 0.0000 | 0.0000 | -0.3564 |
| hd95 | tta4 vs baseline | 1000 | -5.6885 | -6.7716 | -6.7163 | -4.6838 | 0.0000 | 0.0000 | -0.4775 |
| hd95 | tta8 vs baseline | 1000 | -7.6840 | -9.1471 | -8.8574 | -6.5563 | 0.0000 | 0.0000 | -0.5413 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1000 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 1000 | 15 | 2 | 0.1333 | 0 | 1 | 1 | 0 | 2.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 1000 | 15 | 2 | 0.1333 | 0 | 1 | 1 | 0 | 2.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 1000 | 15 | 3 | 0.2000 | 0 | 1 | 1 | 0 | 3.0000 | 0.0000 | 0.2500 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8437 | 0.8428 | -0.0009 | -0.1038 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.8513 | 0.8474 | -0.0040 | -0.4684 | 0.0077 | 0.0046 | 0.9095220225793348 | 0.5411851207194789 | -0.0031 | smaller externally |
| tta4 | 0.8541 | 0.8511 | -0.0030 | -0.3524 | 0.0104 | 0.0083 | 1.236278004755835 | 0.9843342070132083 | -0.0021 | smaller externally |
| tta8 | 0.8547 | 0.8528 | -0.0019 | -0.2221 | 0.0111 | 0.0101 | 1.3126764304154008 | 1.192713085494152 | -0.0010 | within 0.002 |
| tta4 -> tta8 | 0.0006 | 0.0018 | 0.0011 |  | 0.0006 | 0.0018 | <NA> | <NA> | 0.0011 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 334 | 0.8526 | 0.1581 | 0.0000 |
| tta2 | small | 334 | 0.8577 | 0.1578 | 0.0000 |
| tta4 | small | 334 | 0.8635 | 0.1564 | 0.0000 |
| tta8 | small | 334 | 0.8660 | 0.1574 | 0.0000 |
| baseline | medium | 333 | 0.8829 | 0.1077 | 0.0000 |
| tta2 | medium | 333 | 0.8879 | 0.1059 | 0.0000 |
| tta4 | medium | 333 | 0.8920 | 0.1043 | 0.0000 |
| tta8 | medium | 333 | 0.8930 | 0.1051 | 0.0000 |
| baseline | large | 333 | 0.7928 | 0.1787 | 0.0000 |
| tta2 | large | 333 | 0.7964 | 0.1801 | 0.0000 |
| tta4 | large | 333 | 0.7977 | 0.1820 | 0.0000 |
| tta8 | large | 333 | 0.7995 | 0.1823 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 334 | 0.0051 | 0.5996 | 0.0031 | 0.0071 | 0.0000 | 0.0000 |
| dice | small | tta4 vs baseline | 334 | 0.0109 | 1.2791 | 0.0083 | 0.0136 | 0.0000 | 0.0000 |
| dice | small | tta8 vs baseline | 334 | 0.0133 | 1.5625 | 0.0105 | 0.0163 | 0.0000 | 0.0000 |
| dice | medium | tta2 vs baseline | 333 | 0.0050 | 0.5650 | 0.0036 | 0.0064 | 0.0000 | 0.0000 |
| dice | medium | tta4 vs baseline | 333 | 0.0091 | 1.0308 | 0.0069 | 0.0115 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 333 | 0.0101 | 1.1478 | 0.0078 | 0.0126 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 333 | 0.0036 | 0.4517 | 0.0020 | 0.0052 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 333 | 0.0049 | 0.6147 | 0.0027 | 0.0071 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 333 | 0.0067 | 0.8439 | 0.0042 | 0.0092 | 0.0000 | 0.0000 |
| hd95 | small | tta2 vs baseline | 334 | -2.4871 | -3.7822 | -3.8792 | -1.0919 | 0.0000 | 0.0000 |
| hd95 | small | tta4 vs baseline | 334 | -6.4923 | -9.8728 | -8.4480 | -4.5790 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 334 | -8.5815 | -13.0500 | -10.7084 | -6.4823 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 333 | -3.7646 | -5.3807 | -4.9683 | -2.5776 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 333 | -6.6891 | -9.5607 | -8.2959 | -5.1246 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 333 | -9.1958 | -13.1435 | -11.3220 | -7.2439 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 333 | -1.5794 | -1.3575 | -2.6736 | -0.4656 | 0.0000 | 0.0000 |
| hd95 | large | tta4 vs baseline | 333 | -3.8816 | -3.3362 | -5.5342 | -2.2717 | 0.0000 | 0.0000 |
| hd95 | large | tta8 vs baseline | 333 | -5.2720 | -4.5313 | -7.1315 | -3.5284 | 0.0000 | 0.0000 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 2.0683 | 0.3555 | 0.3555 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta4 vs baseline | 12.0549 | 0.0024 | 0.0145 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| dice | tta8 vs baseline | 11.7678 | 0.0028 | 0.0145 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta2 vs baseline | 3.8524 | 0.1457 | 0.2914 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta4 vs baseline | 8.1770 | 0.0168 | 0.0503 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |
| hd95 | tta8 vs baseline | 9.1015 | 0.0106 | 0.0422 | 334 | 333 | 333 | 25204.0000 | 55261.0000 |



## cvc_colondb / unet_resnet50

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 380 | 0.6307 | 0.3183 | 0.0000 |
| dice | tta2 | 380 | 0.6443 | 0.3222 | 2.1447 |
| dice | tta4 | 380 | 0.6546 | 0.3305 | 3.7804 |
| dice | tta8 | 380 | 0.6557 | 0.3346 | 3.9563 |
| boundary_f1 | baseline | 378 | 0.2697 | 0.1572 | 0.0000 |
| boundary_f1 | tta2 | 373 | 0.2808 | 0.1626 | 4.0866 |
| boundary_f1 | tta4 | 365 | 0.3035 | 0.1706 | 12.5221 |
| boundary_f1 | tta8 | 357 | 0.3205 | 0.1743 | 18.8021 |
| hd95 | baseline | 378 | 111.0489 | 96.0295 | 0.0000 |
| hd95 | tta2 | 373 | 99.8721 | 97.0105 | -10.0647 |
| hd95 | tta4 | 365 | 86.5394 | 96.7306 | -22.0708 |
| hd95 | tta8 | 357 | 76.7453 | 89.4440 | -30.8905 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 380 | 0.0135 | 2.1447 | 0.0097 | 0.0174 | 0.0000 | 0.0000 | 0.4757 |
| dice | tta4 vs baseline | 380 | 0.0238 | 3.7804 | 0.0182 | 0.0295 | 0.0000 | 0.0000 | 0.5931 |
| dice | tta8 vs baseline | 380 | 0.0250 | 3.9563 | 0.0187 | 0.0312 | 0.0000 | 0.0000 | 0.5675 |
| boundary_f1 | tta2 vs baseline | 371 | 0.0075 | 2.7196 | 0.0035 | 0.0114 | 0.0000 | 0.0000 | 0.2572 |
| boundary_f1 | tta4 vs baseline | 365 | 0.0242 | 8.6662 | 0.0190 | 0.0294 | 0.0000 | 0.0000 | 0.5177 |
| boundary_f1 | tta8 vs baseline | 357 | 0.0354 | 12.4061 | 0.0287 | 0.0419 | 0.0000 | 0.0000 | 0.6038 |
| hd95 | tta2 vs baseline | 371 | -9.8171 | -9.0676 | -12.7189 | -7.0828 | 0.0000 | 0.0000 | -0.5141 |
| hd95 | tta4 vs baseline | 365 | -18.7150 | -17.7807 | -22.0311 | -15.4098 | 0.0000 | 0.0000 | -0.7495 |
| hd95 | tta8 vs baseline | 357 | -24.5254 | -24.2177 | -28.0870 | -21.0075 | 0.0000 | 0.0000 | -0.8002 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 380 | 15 | 815 | 54.3333 | 36 | 79 | 119 | 55 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 380 | 15 | 853 | 56.8667 | 30 | 78 | 117 | 59 | 145.0000 | 107.0000 | 0.0196 | 0.0392 | 7.0000 | 3.0000 | 0.3438 |
| tta4 | 380 | 15 | 915 | 61.0000 | 43 | 87 | 109 | 63 | 203.0000 | 103.0000 | 0.0000 | 0.0000 | 12.0000 | 4.0000 | 0.0768 |
| tta8 | 380 | 15 | 983 | 65.5333 | 46 | 89 | 114 | 68 | 259.0000 | 91.0000 | 0.0000 | 0.0000 | 15.0000 | 2.0000 | 0.0023 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7763 | 0.6307 | -0.1455 | -18.7462 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.7908 | 0.6443 | -0.1465 | -18.5252 | 0.0145 | 0.0135 | 1.8676893843473927 | 2.144731826527133 | -0.0010 | within 0.002 |
| tta4 | 0.7999 | 0.6546 | -0.1453 | -18.1680 | 0.0237 | 0.0238 | 3.0471782024802123 | 3.7803534632979403 | 0.0002 | within 0.002 |
| tta8 | 0.8046 | 0.6557 | -0.1489 | -18.5097 | 0.0284 | 0.0250 | 3.65473183788675 | 3.9563375649644534 | -0.0034 | smaller externally |
| tta4 -> tta8 | 0.0047 | 0.0011 | -0.0036 |  | 0.0047 | 0.0011 | <NA> | <NA> | -0.0036 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 127 | 0.5576 | 0.3303 | 0.1811 |
| tta2 | small | 127 | 0.5744 | 0.3387 | 0.1890 |
| tta4 | small | 127 | 0.5880 | 0.3467 | 0.2047 |
| tta8 | small | 127 | 0.5912 | 0.3516 | 0.2362 |
| baseline | medium | 126 | 0.7651 | 0.2516 | 0.0952 |
| tta2 | medium | 126 | 0.7819 | 0.2459 | 0.1032 |
| tta4 | medium | 126 | 0.7977 | 0.2487 | 0.0952 |
| tta8 | medium | 126 | 0.8017 | 0.2489 | 0.1032 |
| baseline | large | 127 | 0.5705 | 0.3252 | 0.1575 |
| tta2 | large | 127 | 0.5775 | 0.3303 | 0.1732 |
| tta4 | large | 127 | 0.5792 | 0.3410 | 0.1969 |
| tta8 | large | 127 | 0.5753 | 0.3459 | 0.1969 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 127 | 0.0168 | 3.0139 | 0.0084 | 0.0251 | 0.0000 | 0.0000 |
| dice | small | tta4 vs baseline | 127 | 0.0303 | 5.4417 | 0.0200 | 0.0407 | 0.0000 | 0.0000 |
| dice | small | tta8 vs baseline | 127 | 0.0335 | 6.0148 | 0.0220 | 0.0448 | 0.0000 | 0.0000 |
| dice | medium | tta2 vs baseline | 126 | 0.0168 | 2.1923 | 0.0117 | 0.0220 | 0.0000 | 0.0000 |
| dice | medium | tta4 vs baseline | 126 | 0.0326 | 4.2576 | 0.0261 | 0.0394 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 126 | 0.0366 | 4.7838 | 0.0284 | 0.0452 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 127 | 0.0070 | 1.2319 | 0.0013 | 0.0127 | 0.0029 | 0.0029 |
| dice | large | tta4 vs baseline | 127 | 0.0087 | 1.5216 | -0.0019 | 0.0184 | 0.0037 | 0.0037 |
| dice | large | tta8 vs baseline | 127 | 0.0048 | 0.8434 | -0.0071 | 0.0157 | 0.0452 | 0.0452 |
| hd95 | small | tta2 vs baseline | 122 | -12.5790 | -14.6059 | -18.5094 | -7.2442 | 0.0000 | 0.0000 |
| hd95 | small | tta4 vs baseline | 119 | -22.9909 | -28.5416 | -30.0836 | -15.8831 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 116 | -32.9511 | -43.3729 | -40.3303 | -26.0073 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 126 | -8.5644 | -11.1061 | -12.3499 | -4.8816 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 126 | -18.1605 | -23.5499 | -22.6561 | -13.5442 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 125 | -22.6082 | -29.5753 | -27.2892 | -18.1572 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 123 | -8.3610 | -5.1566 | -13.0769 | -3.2716 | 0.0000 | 0.0000 |
| hd95 | large | tta4 vs baseline | 120 | -15.0570 | -9.4522 | -20.3500 | -9.8772 | 0.0000 | 0.0000 |
| hd95 | large | tta8 vs baseline | 116 | -18.1656 | -11.8479 | -24.5524 | -11.9438 | 0.0000 | 0.0000 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 6.0456 | 0.0487 | 0.1223 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta4 vs baseline | 19.0649 | 0.0001 | 0.0004 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta8 vs baseline | 26.2335 | 0.0000 | 0.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta2 vs baseline | 0.1898 | 0.9095 | 0.9095 | 122 | 126 | 123 | 7471.3333 | 16726.3333 |
| hd95 | tta4 vs baseline | 6.3990 | 0.0408 | 0.1223 | 119 | 126 | 120 | 7471.3333 | 16726.3333 |
| hd95 | tta8 vs baseline | 13.4492 | 0.0012 | 0.0048 | 116 | 125 | 116 | 7471.3333 | 16726.3333 |



## cvc_colondb / swinunetr_2d

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 380 | 0.2513 | 0.2204 | 0.0000 |
| dice | tta2 | 380 | 0.2575 | 0.2276 | 2.4620 |
| dice | tta4 | 380 | 0.2591 | 0.2330 | 3.0888 |
| dice | tta8 | 380 | 0.2608 | 0.2364 | 3.7586 |
| boundary_f1 | baseline | 380 | 0.0453 | 0.0424 | 0.0000 |
| boundary_f1 | tta2 | 380 | 0.0477 | 0.0474 | 5.2511 |
| boundary_f1 | tta4 | 380 | 0.0500 | 0.0514 | 10.2266 |
| boundary_f1 | tta8 | 380 | 0.0514 | 0.0540 | 13.3913 |
| hd95 | baseline | 380 | 245.3504 | 73.2207 | 0.0000 |
| hd95 | tta2 | 380 | 241.4596 | 74.3574 | -1.5858 |
| hd95 | tta4 | 380 | 236.6432 | 75.7780 | -3.5489 |
| hd95 | tta8 | 380 | 231.4762 | 77.0051 | -5.6548 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 380 | 0.0062 | 2.4620 | 0.0042 | 0.0083 | 0.0000 | 0.0000 | 0.2597 |
| dice | tta4 vs baseline | 380 | 0.0078 | 3.0888 | 0.0047 | 0.0110 | 0.0065 | 0.0141 | 0.1526 |
| dice | tta8 vs baseline | 380 | 0.0094 | 3.7586 | 0.0057 | 0.0134 | 0.0059 | 0.0141 | 0.1536 |
| boundary_f1 | tta2 vs baseline | 380 | 0.0024 | 5.2511 | 0.0014 | 0.0034 | 0.0047 | 0.0141 | 0.1623 |
| boundary_f1 | tta4 vs baseline | 380 | 0.0046 | 10.2266 | 0.0031 | 0.0063 | 0.0003 | 0.0013 | 0.2048 |
| boundary_f1 | tta8 vs baseline | 380 | 0.0061 | 13.3913 | 0.0043 | 0.0080 | 0.0000 | 0.0002 | 0.2363 |
| hd95 | tta2 vs baseline | 380 | -3.8907 | -1.5858 | -4.9693 | -2.8130 | 0.0000 | 0.0000 | -0.4333 |
| hd95 | tta4 vs baseline | 380 | -8.7072 | -3.5489 | -10.4039 | -7.0748 | 0.0000 | 0.0000 | -0.5823 |
| hd95 | tta8 vs baseline | 380 | -13.8742 | -5.6548 | -15.9686 | -11.8009 | 0.0000 | 0.0000 | -0.6881 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 380 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 380 | 15 | 2 | 0.1333 | 0 | 1 | 2 | 0 | 2.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 380 | 15 | 10 | 0.6667 | 0 | 5 | 8 | 0 | 10.0000 | 0.0000 | 0.0020 | 0.0059 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 380 | 15 | 18 | 1.2000 | 0 | 6 | 9 | 0 | 18.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.5471 | 0.2513 | -0.2957 | -54.0561 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.5600 | 0.2575 | -0.3025 | -54.0135 | 0.0130 | 0.0062 | 2.3672680016008556 | 2.461983004655219 | -0.0068 | smaller externally |
| tta4 | 0.5654 | 0.2591 | -0.3063 | -54.1702 | 0.0183 | 0.0078 | 3.3454477994831624 | 3.0887511860241443 | -0.0105 | smaller externally |
| tta8 | 0.5680 | 0.2608 | -0.3072 | -54.0830 | 0.0209 | 0.0094 | 3.8194110430798576 | 3.758603349233372 | -0.0114 | smaller externally |
| tta4 -> tta8 | 0.0026 | 0.0017 | -0.0009 |  | 0.0026 | 0.0017 | <NA> | <NA> | -0.0009 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 127 | 0.1123 | 0.1427 | 0.0000 |
| tta2 | small | 127 | 0.1207 | 0.1534 | 0.0000 |
| tta4 | small | 127 | 0.1262 | 0.1652 | 0.0000 |
| tta8 | small | 127 | 0.1306 | 0.1724 | 0.0000 |
| baseline | medium | 126 | 0.2882 | 0.2234 | 0.0000 |
| tta2 | medium | 126 | 0.2943 | 0.2366 | 0.0000 |
| tta4 | medium | 126 | 0.3007 | 0.2455 | 0.0000 |
| tta8 | medium | 126 | 0.3063 | 0.2511 | 0.0000 |
| baseline | large | 127 | 0.3538 | 0.2115 | 0.0000 |
| tta2 | large | 127 | 0.3578 | 0.2153 | 0.0000 |
| tta4 | large | 127 | 0.3507 | 0.2195 | 0.0000 |
| tta8 | large | 127 | 0.3459 | 0.2221 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 127 | 0.0085 | 7.5312 | 0.0053 | 0.0118 | 0.0001 | 0.0002 |
| dice | small | tta4 vs baseline | 127 | 0.0140 | 12.4348 | 0.0085 | 0.0198 | 0.0007 | 0.0021 |
| dice | small | tta8 vs baseline | 127 | 0.0183 | 16.3139 | 0.0115 | 0.0256 | 0.0006 | 0.0006 |
| dice | medium | tta2 vs baseline | 126 | 0.0061 | 2.1266 | 0.0018 | 0.0105 | 0.1582 | 0.1582 |
| dice | medium | tta4 vs baseline | 126 | 0.0125 | 4.3373 | 0.0065 | 0.0185 | 0.0009 | 0.0021 |
| dice | medium | tta8 vs baseline | 126 | 0.0180 | 6.2601 | 0.0110 | 0.0249 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 127 | 0.0040 | 1.1245 | 0.0011 | 0.0068 | 0.0050 | 0.0100 |
| dice | large | tta4 vs baseline | 127 | -0.0031 | -0.8858 | -0.0071 | 0.0010 | 0.0262 | 0.0262 |
| dice | large | tta8 vs baseline | 127 | -0.0080 | -2.2469 | -0.0125 | -0.0033 | 0.0002 | 0.0004 |
| hd95 | small | tta2 vs baseline | 127 | -5.3796 | -1.8553 | -7.5602 | -3.2853 | 0.0000 | 0.0000 |
| hd95 | small | tta4 vs baseline | 127 | -11.1236 | -3.8362 | -14.0420 | -8.2377 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 127 | -17.2909 | -5.9631 | -20.4788 | -14.2498 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 126 | -3.7849 | -1.5342 | -5.3898 | -2.2133 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 126 | -12.0931 | -4.9018 | -15.2562 | -9.1607 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 126 | -20.1539 | -8.1691 | -24.1713 | -16.2066 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 127 | -2.5069 | -1.2573 | -4.3089 | -0.7324 | 0.0149 | 0.0149 |
| hd95 | large | tta4 vs baseline | 127 | -2.9316 | -1.4703 | -5.6563 | -0.3324 | 0.0622 | 0.0622 |
| hd95 | large | tta8 vs baseline | 127 | -4.2272 | -2.1201 | -7.5640 | -1.0002 | 0.0252 | 0.0252 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 3.3781 | 0.1847 | 0.1847 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta4 vs baseline | 21.3584 | 0.0000 | 0.0001 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta8 vs baseline | 44.0498 | 0.0000 | 0.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta2 vs baseline | 5.0419 | 0.0804 | 0.1608 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta4 vs baseline | 26.4434 | 0.0000 | 0.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta8 vs baseline | 41.5134 | 0.0000 | 0.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |



## cvc_colondb / segformer_mit_b2

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 380 | 0.7379 | 0.2665 | 0.0000 |
| dice | tta2 | 380 | 0.7448 | 0.2659 | 0.9351 |
| dice | tta4 | 380 | 0.7511 | 0.2648 | 1.7907 |
| dice | tta8 | 380 | 0.7522 | 0.2653 | 1.9373 |
| boundary_f1 | baseline | 380 | 0.2912 | 0.1693 | 0.0000 |
| boundary_f1 | tta2 | 380 | 0.2977 | 0.1708 | 2.2179 |
| boundary_f1 | tta4 | 380 | 0.3111 | 0.1787 | 6.8404 |
| boundary_f1 | tta8 | 380 | 0.3147 | 0.1810 | 8.0683 |
| hd95 | baseline | 380 | 86.6858 | 92.5556 | 0.0000 |
| hd95 | tta2 | 380 | 84.3219 | 93.2874 | -2.7269 |
| hd95 | tta4 | 380 | 80.7815 | 92.9875 | -6.8111 |
| hd95 | tta8 | 380 | 79.1488 | 91.7786 | -8.6946 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 380 | 0.0069 | 0.9351 | 0.0044 | 0.0096 | 0.0000 | 0.0000 | 0.3144 |
| dice | tta4 vs baseline | 380 | 0.0132 | 1.7907 | 0.0095 | 0.0172 | 0.0000 | 0.0000 | 0.4546 |
| dice | tta8 vs baseline | 380 | 0.0143 | 1.9373 | 0.0097 | 0.0191 | 0.0000 | 0.0000 | 0.4793 |
| boundary_f1 | tta2 vs baseline | 380 | 0.0065 | 2.2179 | 0.0032 | 0.0096 | 0.0000 | 0.0001 | 0.2464 |
| boundary_f1 | tta4 vs baseline | 380 | 0.0199 | 6.8404 | 0.0150 | 0.0249 | 0.0000 | 0.0000 | 0.4658 |
| boundary_f1 | tta8 vs baseline | 380 | 0.0235 | 8.0683 | 0.0181 | 0.0289 | 0.0000 | 0.0000 | 0.4942 |
| hd95 | tta2 vs baseline | 380 | -2.3638 | -2.7269 | -3.8913 | -0.8664 | 0.0001 | 0.0001 | -0.2340 |
| hd95 | tta4 vs baseline | 380 | -5.9042 | -6.8111 | -8.0428 | -3.8079 | 0.0000 | 0.0000 | -0.3770 |
| hd95 | tta8 vs baseline | 380 | -7.5370 | -8.6946 | -9.9606 | -5.1741 | 0.0000 | 0.0000 | -0.4513 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 380 | 15 | 38 | 2.5333 | 0 | 13 | 19 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 380 | 15 | 43 | 2.8667 | 0 | 11 | 17 | 0 | 17.0000 | 12.0000 | 0.4583 | 0.9165 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 380 | 15 | 59 | 3.9333 | 0 | 14 | 20 | 1 | 32.0000 | 11.0000 | 0.0019 | 0.0057 | 1.0000 | 0.0000 | 1.0000 |
| tta8 | 380 | 15 | 73 | 4.8667 | 0 | 14 | 23 | 1 | 47.0000 | 12.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8437 | 0.7379 | -0.1058 | -12.5378 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.8513 | 0.7448 | -0.1066 | -12.5157 | 0.0077 | 0.0069 | 0.9095220225793348 | 0.935078169361923 | -0.0008 | within 0.002 |
| tta4 | 0.8541 | 0.7511 | -0.1030 | -12.0588 | 0.0104 | 0.0132 | 1.236278004755835 | 1.7907310677295232 | 0.0028 | larger externally |
| tta8 | 0.8547 | 0.7522 | -0.1026 | -11.9986 | 0.0111 | 0.0143 | 1.3126764304154008 | 1.9372824352993319 | 0.0032 | larger externally |
| tta4 -> tta8 | 0.0006 | 0.0011 | 0.0004 |  | 0.0006 | 0.0011 | <NA> | <NA> | 0.0004 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 127 | 0.6786 | 0.2748 | 0.0000 |
| tta2 | small | 127 | 0.6819 | 0.2777 | 0.0000 |
| tta4 | small | 127 | 0.6876 | 0.2749 | 0.0079 |
| tta8 | small | 127 | 0.6922 | 0.2761 | 0.0079 |
| baseline | medium | 126 | 0.8469 | 0.1668 | 0.0000 |
| tta2 | medium | 126 | 0.8536 | 0.1629 | 0.0000 |
| tta4 | medium | 126 | 0.8596 | 0.1612 | 0.0000 |
| tta8 | medium | 126 | 0.8622 | 0.1630 | 0.0000 |
| baseline | large | 127 | 0.6890 | 0.3044 | 0.0000 |
| tta2 | large | 127 | 0.6997 | 0.3023 | 0.0000 |
| tta4 | large | 127 | 0.7069 | 0.3028 | 0.0000 |
| tta8 | large | 127 | 0.7031 | 0.3016 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 127 | 0.0033 | 0.4897 | -0.0014 | 0.0081 | 0.1710 | 0.1710 |
| dice | small | tta4 vs baseline | 127 | 0.0090 | 1.3316 | 0.0026 | 0.0157 | 0.0128 | 0.0128 |
| dice | small | tta8 vs baseline | 127 | 0.0136 | 1.9989 | 0.0054 | 0.0223 | 0.0004 | 0.0004 |
| dice | medium | tta2 vs baseline | 126 | 0.0067 | 0.7893 | 0.0026 | 0.0111 | 0.0043 | 0.0087 |
| dice | medium | tta4 vs baseline | 126 | 0.0127 | 1.5021 | 0.0080 | 0.0176 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 126 | 0.0153 | 1.8024 | 0.0088 | 0.0220 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 127 | 0.0107 | 1.5515 | 0.0065 | 0.0153 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 127 | 0.0179 | 2.5949 | 0.0104 | 0.0262 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 127 | 0.0141 | 2.0412 | 0.0052 | 0.0233 | 0.0001 | 0.0001 |
| hd95 | small | tta2 vs baseline | 127 | -0.8234 | -1.0276 | -3.6062 | 2.1569 | 0.4868 | 0.4868 |
| hd95 | small | tta4 vs baseline | 127 | -5.0582 | -6.3124 | -9.1352 | -1.0108 | 0.0032 | 0.0065 |
| hd95 | small | tta8 vs baseline | 127 | -6.9275 | -8.6453 | -11.1858 | -2.7247 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 126 | -2.9460 | -5.0350 | -5.1408 | -0.9830 | 0.0030 | 0.0060 |
| hd95 | medium | tta4 vs baseline | 126 | -7.0631 | -12.0713 | -10.3340 | -3.9819 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 126 | -8.9776 | -15.3433 | -12.6512 | -5.6165 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 127 | -3.3266 | -2.7449 | -6.0194 | -0.6335 | 0.0012 | 0.0035 |
| hd95 | large | tta4 vs baseline | 127 | -5.6006 | -4.6212 | -9.3913 | -1.9488 | 0.0075 | 0.0075 |
| hd95 | large | tta8 vs baseline | 127 | -6.7171 | -5.5425 | -11.3666 | -2.1485 | 0.0025 | 0.0025 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 7.4326 | 0.0243 | 0.1459 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta4 vs baseline | 4.0398 | 0.1327 | 0.5307 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| dice | tta8 vs baseline | 1.3213 | 0.5165 | 1.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta2 vs baseline | 5.8982 | 0.0524 | 0.2619 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta4 vs baseline | 1.5859 | 0.4525 | 1.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |
| hd95 | tta8 vs baseline | 0.3760 | 0.8286 | 1.0000 | 127 | 126 | 127 | 7471.3333 | 16726.3333 |



## etis_larib / unet_resnet50

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 196 | 0.4627 | 0.2764 | 0.0000 |
| dice | tta2 | 196 | 0.5034 | 0.2800 | 8.7978 |
| dice | tta4 | 196 | 0.5284 | 0.2892 | 14.1967 |
| dice | tta8 | 196 | 0.5469 | 0.2969 | 18.1972 |
| boundary_f1 | baseline | 196 | 0.1147 | 0.0921 | 0.0000 |
| boundary_f1 | tta2 | 196 | 0.1256 | 0.1016 | 9.4907 |
| boundary_f1 | tta4 | 196 | 0.1469 | 0.1212 | 28.0212 |
| boundary_f1 | tta8 | 196 | 0.1573 | 0.1269 | 37.1617 |
| hd95 | baseline | 196 | 423.0584 | 186.2873 | 0.0000 |
| hd95 | tta2 | 196 | 385.2753 | 189.0638 | -8.9309 |
| hd95 | tta4 | 196 | 348.1784 | 200.4985 | -17.6997 |
| hd95 | tta8 | 196 | 319.3202 | 202.0276 | -24.5210 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 196 | 0.0407 | 8.7978 | 0.0346 | 0.0467 | 0.0000 | 0.0000 | 0.8457 |
| dice | tta4 vs baseline | 196 | 0.0657 | 14.1967 | 0.0567 | 0.0748 | 0.0000 | 0.0000 | 0.8648 |
| dice | tta8 vs baseline | 196 | 0.0842 | 18.1972 | 0.0732 | 0.0954 | 0.0000 | 0.0000 | 0.8849 |
| boundary_f1 | tta2 vs baseline | 196 | 0.0109 | 9.4907 | 0.0078 | 0.0141 | 0.0000 | 0.0000 | 0.5066 |
| boundary_f1 | tta4 vs baseline | 196 | 0.0321 | 28.0212 | 0.0264 | 0.0382 | 0.0000 | 0.0000 | 0.7642 |
| boundary_f1 | tta8 vs baseline | 196 | 0.0426 | 37.1617 | 0.0358 | 0.0497 | 0.0000 | 0.0000 | 0.8123 |
| hd95 | tta2 vs baseline | 196 | -37.7831 | -8.9309 | -46.3869 | -29.3245 | 0.0000 | 0.0000 | -0.6480 |
| hd95 | tta4 vs baseline | 196 | -74.8800 | -17.6997 | -86.0492 | -63.8065 | 0.0000 | 0.0000 | -0.8447 |
| hd95 | tta8 vs baseline | 196 | -103.7382 | -24.5210 | -117.3726 | -90.5541 | 0.0000 | 0.0000 | -0.9138 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 196 | 15 | 196 | 13.0667 | 1 | 29 | 51 | 5 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 196 | 15 | 222 | 14.8000 | 3 | 33 | 55 | 8 | 67.0000 | 41.0000 | 0.0157 | 0.0315 | 3.0000 | 0.0000 | 0.2500 |
| tta4 | 196 | 15 | 269 | 17.9333 | 6 | 39 | 54 | 16 | 115.0000 | 42.0000 | 0.0000 | 0.0000 | 11.0000 | 0.0000 | 0.0010 |
| tta8 | 196 | 15 | 297 | 19.8000 | 5 | 42 | 55 | 18 | 141.0000 | 40.0000 | 0.0000 | 0.0000 | 14.0000 | 1.0000 | 0.0010 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7763 | 0.4627 | -0.3135 | -40.3905 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.7908 | 0.5034 | -0.2873 | -36.3353 | 0.0145 | 0.0407 | 1.8676893843473927 | 8.797779764206393 | 0.0262 | larger externally |
| tta4 | 0.7999 | 0.5284 | -0.2715 | -33.9409 | 0.0237 | 0.0657 | 3.0471782024802123 | 14.196684096034135 | 0.0420 | larger externally |
| tta8 | 0.8046 | 0.5469 | -0.2577 | -32.0275 | 0.0284 | 0.0842 | 3.65473183788675 | 18.197174961473074 | 0.0558 | larger externally |
| tta4 -> tta8 | 0.0047 | 0.0185 | 0.0138 |  | 0.0047 | 0.0185 | <NA> | <NA> | 0.0138 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 66 | 0.2541 | 0.2481 | 0.0606 |
| tta2 | small | 66 | 0.2916 | 0.2666 | 0.0909 |
| tta4 | small | 66 | 0.3143 | 0.2843 | 0.1515 |
| tta8 | small | 66 | 0.3386 | 0.2951 | 0.1515 |
| baseline | medium | 65 | 0.5072 | 0.2304 | 0.0154 |
| tta2 | medium | 65 | 0.5567 | 0.2333 | 0.0154 |
| tta4 | medium | 65 | 0.5909 | 0.2452 | 0.0615 |
| tta8 | medium | 65 | 0.6073 | 0.2641 | 0.0769 |
| baseline | large | 65 | 0.6302 | 0.2047 | 0.0000 |
| tta2 | large | 65 | 0.6652 | 0.1913 | 0.0154 |
| tta4 | large | 65 | 0.6834 | 0.1921 | 0.0308 |
| tta8 | large | 65 | 0.6981 | 0.1967 | 0.0462 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 66 | 0.0376 | 14.7862 | 0.0271 | 0.0482 | 0.0000 | 0.0000 |
| dice | small | tta4 vs baseline | 66 | 0.0602 | 23.7066 | 0.0447 | 0.0757 | 0.0000 | 0.0000 |
| dice | small | tta8 vs baseline | 66 | 0.0845 | 33.2763 | 0.0642 | 0.1044 | 0.0000 | 0.0000 |
| dice | medium | tta2 vs baseline | 65 | 0.0496 | 9.7774 | 0.0391 | 0.0602 | 0.0000 | 0.0000 |
| dice | medium | tta4 vs baseline | 65 | 0.0837 | 16.5073 | 0.0677 | 0.0991 | 0.0000 | 0.0000 |
| dice | medium | tta8 vs baseline | 65 | 0.1002 | 19.7477 | 0.0787 | 0.1207 | 0.0000 | 0.0000 |
| dice | large | tta2 vs baseline | 65 | 0.0350 | 5.5579 | 0.0252 | 0.0457 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 65 | 0.0532 | 8.4440 | 0.0381 | 0.0696 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 65 | 0.0679 | 10.7763 | 0.0538 | 0.0845 | 0.0000 | 0.0000 |
| hd95 | small | tta2 vs baseline | 66 | -31.4637 | -6.4856 | -48.0755 | -15.4657 | 0.0001 | 0.0001 |
| hd95 | small | tta4 vs baseline | 66 | -79.4444 | -16.3759 | -100.0399 | -60.4734 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 66 | -107.9318 | -22.2480 | -130.7405 | -86.2391 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 65 | -51.4082 | -12.0577 | -66.2019 | -36.7047 | 0.0000 | 0.0000 |
| hd95 | medium | tta4 vs baseline | 65 | -94.6970 | -22.2111 | -113.0046 | -76.7293 | 0.0000 | 0.0000 |
| hd95 | medium | tta8 vs baseline | 65 | -131.5070 | -30.8448 | -154.3202 | -108.6638 | 0.0000 | 0.0000 |
| hd95 | large | tta2 vs baseline | 65 | -30.5746 | -8.5706 | -42.8914 | -18.8078 | 0.0000 | 0.0001 |
| hd95 | large | tta4 vs baseline | 65 | -50.4283 | -14.1359 | -69.0322 | -32.5871 | 0.0000 | 0.0000 |
| hd95 | large | tta8 vs baseline | 65 | -71.7113 | -20.1019 | -93.6897 | -50.7018 | 0.0000 | 0.0000 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 5.7535 | 0.0563 | 0.1126 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta4 vs baseline | 11.2642 | 0.0036 | 0.0179 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta8 vs baseline | 9.3946 | 0.0091 | 0.0274 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta2 vs baseline | 4.4215 | 0.1096 | 0.1126 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta4 vs baseline | 10.9469 | 0.0042 | 0.0179 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta8 vs baseline | 14.3107 | 0.0008 | 0.0047 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |



## etis_larib / swinunetr_2d

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 196 | 0.1407 | 0.1857 | 0.0000 |
| dice | tta2 | 196 | 0.1485 | 0.1929 | 5.5632 |
| dice | tta4 | 196 | 0.1491 | 0.1971 | 6.0168 |
| dice | tta8 | 196 | 0.1482 | 0.1987 | 5.3650 |
| boundary_f1 | baseline | 196 | 0.0126 | 0.0179 | 0.0000 |
| boundary_f1 | tta2 | 196 | 0.0137 | 0.0207 | 9.2113 |
| boundary_f1 | tta4 | 196 | 0.0144 | 0.0211 | 14.0820 |
| boundary_f1 | tta8 | 196 | 0.0146 | 0.0223 | 16.1380 |
| hd95 | baseline | 196 | 623.7297 | 178.4911 | 0.0000 |
| hd95 | tta2 | 196 | 620.5304 | 183.1528 | -0.5129 |
| hd95 | tta4 | 196 | 612.8459 | 183.3597 | -1.7449 |
| hd95 | tta8 | 196 | 607.4074 | 183.0976 | -2.6169 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 196 | 0.0078 | 5.5632 | 0.0055 | 0.0104 | 0.0000 | 0.0000 | 0.4226 |
| dice | tta4 vs baseline | 196 | 0.0085 | 6.0168 | 0.0055 | 0.0116 | 0.0001 | 0.0006 | 0.3131 |
| dice | tta8 vs baseline | 196 | 0.0075 | 5.3650 | 0.0042 | 0.0111 | 0.0060 | 0.0060 | 0.2222 |
| boundary_f1 | tta2 vs baseline | 196 | 0.0012 | 9.2113 | 0.0005 | 0.0018 | 0.0013 | 0.0026 | 0.2620 |
| boundary_f1 | tta4 vs baseline | 196 | 0.0018 | 14.0820 | 0.0010 | 0.0026 | 0.0000 | 0.0001 | 0.3526 |
| boundary_f1 | tta8 vs baseline | 196 | 0.0020 | 16.1380 | 0.0011 | 0.0030 | 0.0006 | 0.0019 | 0.2750 |
| hd95 | tta2 vs baseline | 196 | -3.1992 | -0.5129 | -6.1635 | -0.0690 | 0.0003 | 0.0011 | -0.2995 |
| hd95 | tta4 vs baseline | 196 | -10.8837 | -1.7449 | -14.5400 | -7.1957 | 0.0000 | 0.0000 | -0.5406 |
| hd95 | tta8 vs baseline | 196 | -16.3223 | -2.6169 | -20.4673 | -12.4017 | 0.0000 | 0.0000 | -0.6217 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 196 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 196 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 196 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 196 | 15 | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.5471 | 0.1407 | -0.4064 | -74.2882 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.5600 | 0.1485 | -0.4115 | -73.4855 | 0.0130 | 0.0078 | 2.3672680016008556 | 5.563189797745261 | -0.0051 | smaller externally |
| tta4 | 0.5654 | 0.1491 | -0.4162 | -73.6236 | 0.0183 | 0.0085 | 3.3454477994831624 | 6.016824427861197 | -0.0098 | smaller externally |
| tta8 | 0.5680 | 0.1482 | -0.4198 | -73.9055 | 0.0209 | 0.0075 | 3.8194110430798576 | 5.365039317349303 | -0.0133 | smaller externally |
| tta4 -> tta8 | 0.0026 | -0.0009 | -0.0035 |  | 0.0026 | -0.0009 | <NA> | <NA> | -0.0035 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 66 | 0.0270 | 0.0303 | 0.0000 |
| tta2 | small | 66 | 0.0302 | 0.0347 | 0.0000 |
| tta4 | small | 66 | 0.0304 | 0.0362 | 0.0000 |
| tta8 | small | 66 | 0.0301 | 0.0369 | 0.0000 |
| baseline | medium | 65 | 0.1030 | 0.0994 | 0.0000 |
| tta2 | medium | 65 | 0.1076 | 0.1091 | 0.0000 |
| tta4 | medium | 65 | 0.1064 | 0.1112 | 0.0000 |
| tta8 | medium | 65 | 0.1052 | 0.1129 | 0.0000 |
| baseline | large | 65 | 0.2937 | 0.2360 | 0.0000 |
| tta2 | large | 65 | 0.3095 | 0.2403 | 0.0000 |
| tta4 | large | 65 | 0.3123 | 0.2474 | 0.0000 |
| tta8 | large | 65 | 0.3112 | 0.2510 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 66 | 0.0032 | 11.8423 | 0.0014 | 0.0052 | 0.0170 | 0.0339 |
| dice | small | tta4 vs baseline | 66 | 0.0034 | 12.5467 | 0.0014 | 0.0057 | 0.0310 | 0.0621 |
| dice | small | tta8 vs baseline | 66 | 0.0030 | 11.2196 | 0.0009 | 0.0053 | 0.1392 | 0.2784 |
| dice | medium | tta2 vs baseline | 65 | 0.0046 | 4.4341 | 0.0007 | 0.0088 | 0.2287 | 0.2287 |
| dice | medium | tta4 vs baseline | 65 | 0.0034 | 3.3229 | -0.0011 | 0.0082 | 0.9349 | 0.9349 |
| dice | medium | tta8 vs baseline | 65 | 0.0022 | 2.1309 | -0.0032 | 0.0081 | 0.9922 | 0.9922 |
| dice | large | tta2 vs baseline | 65 | 0.0158 | 5.3723 | 0.0106 | 0.0211 | 0.0000 | 0.0000 |
| dice | large | tta4 vs baseline | 65 | 0.0187 | 6.3514 | 0.0117 | 0.0255 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 65 | 0.0175 | 5.9523 | 0.0098 | 0.0251 | 0.0006 | 0.0018 |
| hd95 | small | tta2 vs baseline | 66 | -6.8380 | -0.9519 | -11.7591 | -1.5082 | 0.0002 | 0.0007 |
| hd95 | small | tta4 vs baseline | 66 | -17.9384 | -2.4972 | -23.6793 | -12.3577 | 0.0000 | 0.0000 |
| hd95 | small | tta8 vs baseline | 66 | -24.1428 | -3.3609 | -30.9991 | -18.0883 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 65 | 3.5576 | 0.5277 | -2.6230 | 9.7682 | 0.7812 | 0.7812 |
| hd95 | medium | tta4 vs baseline | 65 | -2.0821 | -0.3088 | -9.5350 | 5.3607 | 0.1901 | 0.1901 |
| hd95 | medium | tta8 vs baseline | 65 | -8.3424 | -1.2374 | -15.7424 | -1.4110 | 0.0270 | 0.0270 |
| hd95 | large | tta2 vs baseline | 65 | -6.2614 | -1.3121 | -9.9882 | -2.5935 | 0.0036 | 0.0072 |
| hd95 | large | tta4 vs baseline | 65 | -12.5222 | -2.6241 | -17.9957 | -7.3613 | 0.0000 | 0.0000 |
| hd95 | large | tta8 vs baseline | 65 | -16.3612 | -3.4286 | -23.5052 | -9.7437 | 0.0000 | 0.0001 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 17.3061 | 0.0002 | 0.0009 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta4 vs baseline | 18.0885 | 0.0001 | 0.0007 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta8 vs baseline | 9.6969 | 0.0078 | 0.0157 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta2 vs baseline | 8.0995 | 0.0174 | 0.0174 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta4 vs baseline | 11.1539 | 0.0038 | 0.0139 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta8 vs baseline | 11.3291 | 0.0035 | 0.0139 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |



## etis_larib / segformer_mit_b2

### Metric per method (mean +- std over cases, % vs baseline)

| metric | method | n_cases | mean | std | change_pct |
| --- | --- | --- | --- | --- | --- |
| dice | baseline | 196 | 0.6906 | 0.2627 | 0.0000 |
| dice | tta2 | 196 | 0.7019 | 0.2635 | 1.6376 |
| dice | tta4 | 196 | 0.7128 | 0.2615 | 3.2168 |
| dice | tta8 | 196 | 0.7209 | 0.2597 | 4.3818 |
| boundary_f1 | baseline | 196 | 0.1565 | 0.0976 | 0.0000 |
| boundary_f1 | tta2 | 196 | 0.1701 | 0.1049 | 8.7350 |
| boundary_f1 | tta4 | 196 | 0.1854 | 0.1172 | 18.5000 |
| boundary_f1 | tta8 | 196 | 0.1918 | 0.1206 | 22.6058 |
| hd95 | baseline | 196 | 223.0989 | 183.4197 | 0.0000 |
| hd95 | tta2 | 196 | 215.5564 | 186.7327 | -3.3808 |
| hd95 | tta4 | 196 | 208.8853 | 188.3476 | -6.3710 |
| hd95 | tta8 | 196 | 200.5103 | 186.6984 | -10.1249 |


### Paired TTA deltas

| metric | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 196 | 0.0113 | 1.6376 | 0.0065 | 0.0163 | 0.0000 | 0.0000 | 0.4028 |
| dice | tta4 vs baseline | 196 | 0.0222 | 3.2168 | 0.0157 | 0.0289 | 0.0000 | 0.0000 | 0.5971 |
| dice | tta8 vs baseline | 196 | 0.0303 | 4.3818 | 0.0222 | 0.0387 | 0.0000 | 0.0000 | 0.6199 |
| boundary_f1 | tta2 vs baseline | 196 | 0.0137 | 8.7350 | 0.0099 | 0.0175 | 0.0000 | 0.0000 | 0.5284 |
| boundary_f1 | tta4 vs baseline | 196 | 0.0289 | 18.5000 | 0.0230 | 0.0350 | 0.0000 | 0.0000 | 0.7138 |
| boundary_f1 | tta8 vs baseline | 196 | 0.0354 | 22.6058 | 0.0287 | 0.0425 | 0.0000 | 0.0000 | 0.7555 |
| hd95 | tta2 vs baseline | 196 | -7.5425 | -3.3808 | -13.1127 | -2.1190 | 0.0000 | 0.0000 | -0.3361 |
| hd95 | tta4 vs baseline | 196 | -14.2136 | -6.3710 | -21.6417 | -6.8717 | 0.0000 | 0.0000 | -0.3923 |
| hd95 | tta8 vs baseline | 196 | -22.5886 | -10.1249 | -31.7959 | -13.7188 | 0.0000 | 0.0000 | -0.4681 |


### Missed detections counted per checkpoint

| method | n_cases | n_checkpoints | misses_total | misses_per_checkpoint | missed_cases_min | missed_cases_max | cases_missed_by_any_checkpoint | cases_majority_missed | new_misses | recovered | mcnemar_p_checkpoint | mcnemar_p_checkpoint_adjusted | cases_became_majority_miss | cases_left_majority_miss | mcnemar_p_case_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 196 | 15 | 11 | 0.7333 | 0 | 3 | 9 | 0 |  |  | 1.0000 |  |  |  | 1.0000 |
| tta2 | 196 | 15 | 18 | 1.2000 | 0 | 7 | 11 | 0 | 12.0000 | 5.0000 | 0.1435 | 0.2869 | 0.0000 | 0.0000 | 1.0000 |
| tta4 | 196 | 15 | 25 | 1.6667 | 0 | 8 | 13 | 0 | 18.0000 | 4.0000 | 0.0043 | 0.0130 | 0.0000 | 0.0000 | 1.0000 |
| tta8 | 196 | 15 | 29 | 1.9333 | 0 | 5 | 11 | 0 | 22.0000 | 4.0000 | 0.0005 | 0.0021 | 0.0000 | 0.0000 | 1.0000 |


### Internal (CVC-ClinicDB) vs external, same architecture

| method | dice_internal | dice_external | drop_abs | drop_rel_pct | gain_vs_baseline_internal | gain_vs_baseline_external | gain_rel_pct_internal | gain_rel_pct_external | gain_difference | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8437 | 0.6906 | -0.1530 | -18.1405 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0000 | within 0.002 |
| tta2 | 0.8513 | 0.7019 | -0.1494 | -17.5499 | 0.0077 | 0.0113 | 0.9095220225793348 | 1.6376205230292111 | 0.0036 | larger externally |
| tta4 | 0.8541 | 0.7128 | -0.1413 | -16.5391 | 0.0104 | 0.0222 | 1.236278004755835 | 3.216755215478519 | 0.0118 | larger externally |
| tta8 | 0.8547 | 0.7209 | -0.1339 | -15.6607 | 0.0111 | 0.0303 | 1.3126764304154008 | 4.38184982090642 | 0.0192 | larger externally |
| tta4 -> tta8 | 0.0006 | 0.0080 | 0.0074 |  | 0.0006 | 0.0080 | <NA> | <NA> | 0.0074 | increment (Dice) |


### Lesion-size subgroups (tertiles of the reference polyp area)

| method | size_bin | n_cases | mean | std | miss_rate |
| --- | --- | --- | --- | --- | --- |
| baseline | small | 66 | 0.5291 | 0.3095 | 0.0000 |
| tta2 | small | 66 | 0.5476 | 0.3163 | 0.0000 |
| tta4 | small | 66 | 0.5533 | 0.3205 | 0.0000 |
| tta8 | small | 66 | 0.5633 | 0.3232 | 0.0000 |
| baseline | medium | 65 | 0.7685 | 0.2047 | 0.0000 |
| tta2 | medium | 65 | 0.7697 | 0.2125 | 0.0000 |
| tta4 | medium | 65 | 0.7827 | 0.2029 | 0.0000 |
| tta8 | medium | 65 | 0.7873 | 0.2005 | 0.0000 |
| baseline | large | 65 | 0.7767 | 0.1744 | 0.0000 |
| tta2 | large | 65 | 0.7909 | 0.1658 | 0.0000 |
| tta4 | large | 65 | 0.8050 | 0.1518 | 0.0000 |
| tta8 | large | 65 | 0.8144 | 0.1421 | 0.0000 |


### Paired Dice/HD95 gain per size bin

| metric | size_bin | comparison | n_pairs | delta_mean | delta_rel_pct | ci_low | ci_high | p_value | p_value_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | small | tta2 vs baseline | 66 | 0.0185 | 3.4971 | 0.0094 | 0.0277 | 0.0004 | 0.0009 |
| dice | small | tta4 vs baseline | 66 | 0.0242 | 4.5737 | 0.0114 | 0.0370 | 0.0002 | 0.0005 |
| dice | small | tta8 vs baseline | 66 | 0.0342 | 6.4671 | 0.0197 | 0.0492 | 0.0002 | 0.0002 |
| dice | medium | tta2 vs baseline | 65 | 0.0012 | 0.1552 | -0.0068 | 0.0091 | 0.2541 | 0.2541 |
| dice | medium | tta4 vs baseline | 65 | 0.0142 | 1.8450 | 0.0041 | 0.0246 | 0.0003 | 0.0005 |
| dice | medium | tta8 vs baseline | 65 | 0.0188 | 2.4469 | 0.0068 | 0.0314 | 0.0001 | 0.0002 |
| dice | large | tta2 vs baseline | 65 | 0.0141 | 1.8183 | 0.0074 | 0.0218 | 0.0002 | 0.0006 |
| dice | large | tta4 vs baseline | 65 | 0.0282 | 3.6355 | 0.0184 | 0.0398 | 0.0000 | 0.0000 |
| dice | large | tta8 vs baseline | 65 | 0.0377 | 4.8541 | 0.0245 | 0.0535 | 0.0000 | 0.0000 |
| hd95 | small | tta2 vs baseline | 66 | -20.0584 | -7.0252 | -31.1066 | -8.9276 | 0.0002 | 0.0006 |
| hd95 | small | tta4 vs baseline | 66 | -31.9097 | -11.1759 | -47.1292 | -17.1678 | 0.0000 | 0.0001 |
| hd95 | small | tta8 vs baseline | 66 | -40.7764 | -14.2813 | -57.6291 | -25.1437 | 0.0000 | 0.0000 |
| hd95 | medium | tta2 vs baseline | 65 | 2.2659 | 1.1871 | -6.8725 | 11.6475 | 0.7168 | 0.7168 |
| hd95 | medium | tta4 vs baseline | 65 | -6.0743 | -3.1824 | -19.3959 | 6.5167 | 0.1176 | 0.1176 |
| hd95 | medium | tta8 vs baseline | 65 | -18.0951 | -9.4803 | -34.8244 | -3.2408 | 0.0134 | 0.0146 |
| hd95 | large | tta2 vs baseline | 65 | -4.6424 | -2.4186 | -10.9055 | 2.0779 | 0.0022 | 0.0045 |
| hd95 | large | tta4 vs baseline | 65 | -4.3846 | -2.2843 | -13.3392 | 5.1385 | 0.0127 | 0.0253 |
| hd95 | large | tta8 vs baseline | 65 | -8.6144 | -4.4880 | -22.5992 | 5.3773 | 0.0073 | 0.0146 |


### Size interaction (Kruskal-Wallis on the per-case gains)

| metric | comparison | statistic | p_value | p_value_adjusted | n_small | n_medium | n_large | cutoff_low | cutoff_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | tta2 vs baseline | 4.5914 | 0.1007 | 0.3169 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta4 vs baseline | 1.1683 | 0.5576 | 1.0000 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| dice | tta8 vs baseline | 1.3524 | 0.5085 | 1.0000 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta2 vs baseline | 11.0487 | 0.0040 | 0.0239 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta4 vs baseline | 5.5283 | 0.0630 | 0.3151 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |
| hd95 | tta8 vs baseline | 5.0711 | 0.0792 | 0.3169 | 66 | 65 | 65 | 13723.0000 | 49260.0000 |


