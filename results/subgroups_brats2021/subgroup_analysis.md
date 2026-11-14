# MELBA subgroup analysis - lesion size

CPU-only re-analysis of the per-case CSVs an existing cross-validation grid already wrote. Size bins are tertiles of the reference area computed per dataset; the cutoffs are repeated in every table (`cutoff_low_px`, `cutoff_high_px`).

## Reference lesion size per case

| fold | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 251.0000 | 22682.6853 | 17928.4404 | 0.0000 | 8986.5000 | 18530.0000 | 33744.5000 | 95541.0000 |
| 1.0000 | 250.0000 | 22535.4280 | 17938.7596 | 0.0000 | 7735.7500 | 19857.5000 | 32230.7500 | 89546.0000 |
| 2.0000 | 250.0000 | 19955.8680 | 17010.3515 | 0.0000 | 6695.7500 | 16162.0000 | 28551.2500 | 85180.0000 |
| 3.0000 | 250.0000 | 20908.0680 | 19491.1289 | 0.0000 | 6153.7500 | 15857.0000 | 29585.7500 | 111250.0000 |
| 4.0000 | 250.0000 | 21149.5840 | 17825.7321 | 0.0000 | 7633.0000 | 16664.5000 | 31366.2500 | 79831.0000 |

`gt_pixels` is the reference foreground area at the native resolution (mm^3 is in the CSV when the dataset declares a spacing).

## Metric and miss rate per subgroup (seeds averaged per case)

| model | method | metric | size_bin | n_cases | n_units | dice_mean | ci_low | ci_high | n_missed | missed_rate | missed_ci_low | missed_ci_high | hd95_mean | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats2021/segresnet_reduced | baseline | dice | small | 384 | 384 | 0.7568 | 0.7304 | 0.7820 | 6 | 0.0156 | 0.0052 | 0.0286 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | baseline | hd95 | small | 384 | 384 |  | 6.3817 | 9.2961 | 6 | 0.0156 | 0.0052 | 0.0286 | 7.7555 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | baseline | dice | medium | 417 | 417 | 0.8893 | 0.8785 | 0.8989 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | baseline | hd95 | medium | 417 | 417 |  | 3.7450 | 5.6853 | 0 | 0.0000 | 0.0000 | 0.0000 | 4.6689 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | baseline | dice | large | 417 | 417 | 0.9103 | 0.9029 | 0.9171 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | baseline | hd95 | large | 417 | 417 |  | 2.1363 | 2.9715 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.5063 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | dice | small | 384 | 384 | 0.7621 | 0.7360 | 0.7874 | 7 | 0.0182 | 0.0078 | 0.0339 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | hd95 | small | 384 | 384 |  | 5.6201 | 8.0545 | 7 | 0.0182 | 0.0078 | 0.0339 | 6.7760 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | dice | medium | 417 | 417 | 0.8914 | 0.8806 | 0.9010 | 1 | 0.0024 | 0.0000 | 0.0072 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | hd95 | medium | 417 | 417 |  | 3.3468 | 4.9054 | 1 | 0.0024 | 0.0000 | 0.0072 | 4.1067 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | dice | large | 417 | 417 | 0.9117 | 0.9045 | 0.9186 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | hd95 | large | 417 | 417 |  | 2.1192 | 2.9213 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.4771 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | dice | small | 384 | 384 | 0.7668 | 0.7410 | 0.7913 | 8 | 0.0208 | 0.0078 | 0.0365 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | hd95 | small | 384 | 384 |  | 5.1463 | 7.2643 | 8 | 0.0208 | 0.0078 | 0.0365 | 6.1571 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | dice | medium | 417 | 417 | 0.8927 | 0.8820 | 0.9024 | 1 | 0.0024 | 0.0000 | 0.0072 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | hd95 | medium | 417 | 417 |  | 3.4939 | 5.1324 | 1 | 0.0024 | 0.0000 | 0.0072 | 4.2724 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | dice | large | 417 | 417 | 0.9128 | 0.9049 | 0.9200 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | hd95 | large | 417 | 417 |  | 2.1139 | 2.9259 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.4734 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | dice | small | 384 | 384 | 0.7692 | 0.7435 | 0.7940 | 7 | 0.0182 | 0.0078 | 0.0312 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | hd95 | small | 384 | 384 |  | 5.1701 | 7.4004 | 7 | 0.0182 | 0.0078 | 0.0312 | 6.2171 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | dice | medium | 417 | 417 | 0.8937 | 0.8829 | 0.9033 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | hd95 | medium | 417 | 417 |  | 3.5098 | 5.2316 | 0 | 0.0000 | 0.0000 | 0.0000 | 4.3271 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | dice | large | 417 | 417 | 0.9140 | 0.9061 | 0.9213 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | hd95 | large | 417 | 417 |  | 2.0913 | 2.8952 | 0 | 0.0000 | 0.0000 | 0.0000 | 2.4508 | 10349.3333 | 25728.0000 |

Tertiles of the reference area (`small`/`medium`/`large`); CIs resample groups (videos), so frames of one video are not counted as independent.

## Paired TTA delta per size bin

| model | size_bin | treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats2021/segresnet_reduced | small | tta2 | dice | 384 | 384 | 0.0053 | 0.0020 | 0.0087 | 0.0000 | 0.0000 | tta2 |
| brats2021/segresnet_reduced | small | tta2 | hd95 | 377 | 377 | -0.9003 | -1.8223 | -0.2027 | 0.0786 | 0.0786 | tta2 |
| brats2021/segresnet_reduced | small | tta4 | dice | 384 | 384 | 0.0100 | 0.0055 | 0.0147 | 0.0000 | 0.0000 | tta4 |
| brats2021/segresnet_reduced | small | tta4 | hd95 | 375 | 375 | -1.4098 | -2.4043 | -0.6274 | 0.0011 | 0.0011 | tta4 |
| brats2021/segresnet_reduced | small | tta8 | dice | 384 | 384 | 0.0123 | 0.0079 | 0.0169 | 0.0000 | 0.0000 | tta8 |
| brats2021/segresnet_reduced | small | tta8 | hd95 | 375 | 375 | -1.1800 | -2.1979 | -0.3371 | 0.0042 | 0.0042 | tta8 |
| brats2021/segresnet_reduced | medium | tta2 | dice | 417 | 417 | 0.0021 | 0.0014 | 0.0029 | 0.0000 | 0.0000 | tta2 |
| brats2021/segresnet_reduced | medium | tta2 | hd95 | 416 | 416 | -0.4231 | -1.0305 | -0.0653 | 0.0358 | 0.0358 | tta2 |
| brats2021/segresnet_reduced | medium | tta4 | dice | 417 | 417 | 0.0034 | 0.0024 | 0.0044 | 0.0000 | 0.0000 | tta4 |
| brats2021/segresnet_reduced | medium | tta4 | hd95 | 416 | 416 | -0.2575 | -0.9007 | 0.1852 | 0.0303 | 0.0303 | tta4 |
| brats2021/segresnet_reduced | medium | tta8 | dice | 417 | 417 | 0.0044 | 0.0033 | 0.0056 | 0.0000 | 0.0000 | tta8 |
| brats2021/segresnet_reduced | medium | tta8 | hd95 | 417 | 417 | -0.3418 | -1.1041 | 0.2968 | 0.0065 | 0.0065 | tta8 |
| brats2021/segresnet_reduced | large | tta2 | dice | 417 | 417 | 0.0015 | 0.0010 | 0.0021 | 0.0000 | 0.0000 | tta2 |
| brats2021/segresnet_reduced | large | tta2 | hd95 | 417 | 417 | -0.0292 | -0.0779 | 0.0154 | 0.1875 | 0.1875 | tta2 |
| brats2021/segresnet_reduced | large | tta4 | dice | 417 | 417 | 0.0026 | 0.0015 | 0.0034 | 0.0000 | 0.0000 | tta4 |
| brats2021/segresnet_reduced | large | tta4 | hd95 | 417 | 417 | -0.0330 | -0.0905 | 0.0215 | 0.0664 | 0.0664 | tta4 |
| brats2021/segresnet_reduced | large | tta8 | dice | 417 | 417 | 0.0038 | 0.0026 | 0.0048 | 0.0000 | 0.0000 | tta8 |
| brats2021/segresnet_reduced | large | tta8 | hd95 | 417 | 417 | -0.0555 | -0.1133 | -0.0014 | 0.0034 | 0.0034 | tta8 |

The same paired comparison as the main tables, computed inside each bin: a negative HD95 delta means the treatment is better there.

## Missed-detection transitions per bin (exact McNemar, Holm across bins x methods)

| model | size_bin | control | treatment | unit | n_units | both_nonempty | new_misses | recovered | both_empty | net_change | mcnemar_p | mcnemar_p_adjusted | correction | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats2021/segresnet_reduced | small | baseline | tta2 | case | 384 | 377 | 1 | 0 | 6 | -1 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | small | baseline | tta4 | case | 384 | 375 | 3 | 1 | 5 | -2 | 0.6250 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | small | baseline | tta8 | case | 384 | 375 | 3 | 2 | 4 | -1 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | medium | baseline | tta2 | case | 417 | 416 | 1 | 0 | 0 | -1 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | medium | baseline | tta4 | case | 417 | 416 | 1 | 0 | 0 | -1 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | medium | baseline | tta8 | case | 417 | 417 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | large | baseline | tta2 | case | 417 | 417 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | large | baseline | tta4 | case | 417 | 417 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | large | baseline | tta8 | case | 417 | 417 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 10349.3333 | 25728.0000 |

`new_misses` counts cases the control detected and the treatment missed, `recovered` the opposite, and `mcnemar_p_adjusted` is Holm-corrected over every bin x method test shown (`correction` states the family). A treatment can gain mean Dice and still create new misses on small lesions - that is what this table is for.

## Interaction: does the TTA gain differ across size bins?

| model | treatment | metric | control | n_small | n_medium | n_large | delta_small | delta_medium | delta_large | delta_gap | gap_ci_low | gap_ci_high | kruskal_p | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats2021/segresnet_reduced | tta2 | dice | baseline | 384 | 417 | 417 | 0.0053 | 0.0021 | 0.0015 | -0.0038 | -0.0073 | -0.0004 | 0.0095 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | dice | baseline | 384 | 417 | 417 | 0.0100 | 0.0034 | 0.0026 | -0.0074 | -0.0120 | -0.0031 | 0.0168 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | dice | baseline | 384 | 417 | 417 | 0.0123 | 0.0044 | 0.0038 | -0.0085 | -0.0135 | -0.0042 | 0.0067 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta2 | hd95 | baseline | 377 | 416 | 417 | -0.9003 | -0.4231 | -0.0292 | 0.8711 | 0.1820 | 1.7671 | 0.9451 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta4 | hd95 | baseline | 375 | 416 | 417 | -1.4098 | -0.2575 | -0.0330 | 1.3768 | 0.5731 | 2.3329 | 0.2908 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | tta8 | hd95 | baseline | 375 | 417 | 417 | -1.1800 | -0.3418 | -0.0555 | 1.1245 | 0.2732 | 2.1396 | 0.8925 | 10349.3333 | 25728.0000 |

`kruskal_p` tests the per-case paired deltas across the three bins; `delta_gap` is `large - small` with a bootstrap CI over groups. A CI excluding 0 means the TTA effect genuinely depends on lesion size.

## Empty-reference subgroup

| model | subgroup | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats2021/segresnet_reduced | empty_reference | baseline | baseline | 33 | 33 | 22 | 0.6667 | 0.5152 | 0.8182 | cluster_bootstrap_group | 33 |  |  |  |  |  |  | group | 1129.9545 | 275.0000 | 5450.0000 | 24859.0000 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | empty_reference | tta2 | baseline | 33 | 33 | 19 | 0.5758 | 0.3939 | 0.7576 | cluster_bootstrap_group | 33 | 9.0000 | 2.0000 | 5.0000 | 17.0000 | 0.4531 | 1.0000 | group | 996.5263 | 211.0000 | 5381.0000 | 18934.0000 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | empty_reference | tta4 | baseline | 33 | 33 | 20 | 0.6061 | 0.4545 | 0.7576 | cluster_bootstrap_group | 33 | 9.0000 | 2.0000 | 4.0000 | 18.0000 | 0.6875 | 1.0000 | group | 916.8500 | 124.0000 | 4912.0000 | 18337.0000 | 10349.3333 | 25728.0000 |
| brats2021/segresnet_reduced | empty_reference | tta8 | baseline | 33 | 33 | 21 | 0.6364 | 0.4545 | 0.7879 | cluster_bootstrap_group | 33 | 9.0000 | 2.0000 | 3.0000 | 19.0000 | 1.0000 | 1.0000 | group | 831.2857 | 141.0000 | 4471.0000 | 17457.0000 | 10349.3333 | 25728.0000 |

`n_cases = 0` for the polyp collections: every reference contains a polyp, so the subgroup does not exist there (it is populated for BraTS ET).
