# MC-dropout summary: cvc_clinicdb_unet_resnet50_dropout

15 run(s); method `mcdropout_8`; statistics as in scripts/aggregate.py (video-level pairing, Holm).

* 77 of 612 cases fail under mcdropout_8 (Dice < 0.5 or missed); 19 of them are missed detections, for which the ROI descriptors are undefined.
* AUROC/AURC are computed against the MC prediction's own failures; the paired tests compare each MC descriptor with the tta4 ROI entropy on the same cases (delta_aurc < 0 means the MC descriptor ranks failures better).

## 1. Methods side by side (video level)

| method | metric | n_cases | n_units | mean | std |
| --- | --- | --- | --- | --- | --- |
| baseline | dice | 612 | 29 | 0.7484 | 0.1962 |
| baseline | boundary_f1 | 612 | 29 | 0.4700 | 0.1613 |
| baseline | hd95 | 612 | 29 | 52.8069 | 31.4466 |
| baseline | hd95_pooled | 612 | 29 | 49.7569 | 30.1552 |
| baseline | assd | 612 | 29 | 14.7066 | 10.2680 |
| tta2 | dice | 612 | 29 | 0.7574 | 0.2029 |
| tta2 | boundary_f1 | 612 | 29 | 0.4749 | 0.1837 |
| tta2 | hd95 | 612 | 29 | 49.3145 | 32.6342 |
| tta2 | hd95_pooled | 612 | 29 | 46.2264 | 31.2326 |
| tta2 | assd | 612 | 29 | 14.0282 | 10.2668 |
| tta4 | dice | 612 | 29 | 0.7698 | 0.2053 |
| tta4 | boundary_f1 | 612 | 29 | 0.5177 | 0.1678 |
| tta4 | hd95 | 612 | 29 | 44.8482 | 33.2434 |
| tta4 | hd95_pooled | 612 | 29 | 42.4128 | 31.9713 |
| tta4 | assd | 612 | 29 | 12.0464 | 9.4468 |
| tta8 | dice | 612 | 29 | 0.7811 | 0.2017 |
| tta8 | boundary_f1 | 612 | 29 | 0.5321 | 0.1692 |
| tta8 | hd95 | 612 | 29 | 42.0254 | 33.4687 |
| tta8 | hd95_pooled | 612 | 29 | 39.7022 | 32.2003 |
| tta8 | assd | 612 | 29 | 11.2716 | 9.5784 |
| ensemble3 | dice | 612 | 29 | 0.7702 | 0.2039 |
| ensemble3 | boundary_f1 | 612 | 29 | 0.5203 | 0.1675 |
| ensemble3 | hd95 | 612 | 29 | 45.6914 | 33.7741 |
| ensemble3 | hd95_pooled | 612 | 29 | 43.1467 | 32.2147 |
| ensemble3 | assd | 612 | 29 | 12.4718 | 9.9000 |
| mcdropout_8 | dice | 612 | 29 | 0.7475 | 0.1962 |
| mcdropout_8 | boundary_f1 | 612 | 29 | 0.4652 | 0.1589 |
| mcdropout_8 | hd95 | 612 | 29 | 52.1058 | 31.4368 |
| mcdropout_8 | hd95_pooled | 612 | 29 | 49.1914 | 30.0155 |
| mcdropout_8 | assd | 612 | 29 | 14.5712 | 9.8960 |

Mean ± std over the 29 videos; the seed ensembles are a single realisation, so their across-seed std is undefined (blank).

## 2. mcdropout_8 vs baseline / tta4 (paired, video level)

| metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | higher | mcdropout_8 | baseline | group | 29 | all_cases | 612 | 29 | 0 | 0.7484 | 0.7475 | -0.0009 | -0.0007 | -0.0018 | -0.0001 | -0.4345 | rank_biserial | 123.0000 | wilcoxon | 0.0408 | 0.1268 | baseline | two-sided | -2.1032 | 0.0446 | t_test_rel |
| boundary_f1 | higher | mcdropout_8 | baseline | group | 29 | all_cases | 601 | 29 | 0 | 0.4700 | 0.4652 | -0.0048 | -0.0047 | -0.0075 | -0.0022 | -0.7471 | rank_biserial | 55.0000 | wilcoxon | 0.0002 | 0.0010 | baseline | two-sided | -3.5136 | 0.0015 | t_test_rel |
| hd95 | lower | mcdropout_8 | baseline | group | 28 | both_nonempty | 593 | 28 | 0 | 52.9177 | 52.1602 | -0.7575 | -0.2095 | -1.3653 | -0.2034 | -0.4631 | rank_biserial | 109.0000 | wilcoxon | 0.0317 | 0.1268 | mcdropout_8 | two-sided | -2.5175 | 0.0181 | t_test_rel |
| hd95_pooled | lower | mcdropout_8 | baseline | group | 28 | both_nonempty | 593 | 28 | 0 | 49.7750 | 49.1491 | -0.6259 | -0.1030 | -1.1951 | -0.1099 | -0.3744 | rank_biserial | 127.0000 | wilcoxon | 0.0857 | 0.1713 | mcdropout_8 | two-sided | -2.2380 | 0.0337 | t_test_rel |
| assd | lower | mcdropout_8 | baseline | group | 28 | both_nonempty | 593 | 28 | 0 | 14.3204 | 14.1226 | -0.1978 | -0.0293 | -0.6674 | 0.1100 | -0.1773 | rank_biserial | 167.0000 | wilcoxon | 0.4248 | 0.4248 | mcdropout_8 | two-sided | -0.9440 | 0.3535 | t_test_rel |
| dice | higher | mcdropout_8 | tta4 | group | 29 | all_cases | 612 | 29 | 0 | 0.7698 | 0.7475 | -0.0224 | -0.0241 | -0.0336 | -0.0109 | -0.6828 | rank_biserial | 69.0000 | wilcoxon | 0.0008 | 0.0008 | tta4 | two-sided | -3.8329 | 0.0007 | t_test_rel |
| boundary_f1 | higher | mcdropout_8 | tta4 | group | 28 | all_cases | 599 | 28 | 0 | 0.5177 | 0.4760 | -0.0416 | -0.0443 | -0.0558 | -0.0262 | -0.8276 | rank_biserial | 35.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | -5.4786 | 0.0000 | t_test_rel |
| hd95 | lower | mcdropout_8 | tta4 | group | 28 | both_nonempty | 587 | 28 | 0 | 44.7100 | 52.1602 | 7.4502 | 4.8469 | 4.4725 | 10.6173 | 0.8374 | rank_biserial | 33.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | 4.6724 | 0.0001 | t_test_rel |
| hd95_pooled | lower | mcdropout_8 | tta4 | group | 28 | both_nonempty | 587 | 28 | 0 | 42.2531 | 49.1491 | 6.8960 | 4.5828 | 3.9935 | 10.0036 | 0.8128 | rank_biserial | 38.0000 | wilcoxon | 0.0000 | 0.0001 | tta4 | two-sided | 4.4136 | 0.0001 | t_test_rel |
| assd | lower | mcdropout_8 | tta4 | group | 28 | both_nonempty | 587 | 28 | 0 | 11.8991 | 14.1226 | 2.2234 | 2.0208 | 1.4446 | 3.0109 | 0.8818 | rank_biserial | 24.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 5.3731 | 0.0000 | t_test_rel |

Wilcoxon signed-rank, video bootstrap CI of the mean delta, Holm correction; distance metrics on both-non-empty cases.

## 3. Missed detections

| method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | 612 | 29 | 18 | 0.0294 | 0.0107 | 0.0567 | cluster_bootstrap_video | 29 |  |  |  |  |  |  | group |
| tta4 | baseline | 612 | 29 | 21 | 0.0343 | 0.0132 | 0.0634 | cluster_bootstrap_video | 29 | 17.0000 | 3.0000 | 2.0000 | 7.0000 | 1.0000 | 1.0000 | group |
| tta8 | baseline | 612 | 29 | 17 | 0.0278 | 0.0078 | 0.0567 | cluster_bootstrap_video | 29 | 19.0000 | 1.0000 | 3.0000 | 6.0000 | 0.6250 | 1.0000 | group |
| mcdropout_8 | baseline | 612 | 29 | 19 | 0.0310 | 0.0123 | 0.0583 | cluster_bootstrap_video | 29 | 19.0000 | 1.0000 | 0.0000 | 9.0000 | 1.0000 | 1.0000 | group |

Cluster bootstrap CI and exact McNemar.

## 4. MC descriptors as failure detectors

| predictor | description | failure_definition | auroc | ci_low | ci_high | n_used | n_failures | n_undefined | aurc | aurc_optimal | aurc_excess | aurc_ci_low | aurc_ci_high | n_used_aurc | n_undefined_aurc | auroc_if_undefined_certain | auroc_if_undefined_uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mc_entropy_mean | mcdropout: entropy of the mean sample (whole image) | all_failures | 0.4475 | 0.2898 | 0.6494 | 612 | 77 | 0 | 0.2125 | 0.0859 | 0.1266 | 0.1411 | 0.3147 | 612 | 0 | 0.4475 | 0.4475 |
| mc_entropy_mean | mcdropout: entropy of the mean sample (whole image) | non_missed_failures | 0.4580 | 0.2806 | 0.6964 | 593 | 58 | 0 | 0.1846 | 0.0818 | 0.1028 | 0.1230 | 0.2787 | 593 | 0 |  |  |
| mc_entropy_mean_roi | mcdropout: entropy of the mean sample in its ROI | all_failures | 0.8774 | 0.8009 | 0.9687 | 601 | 66 | 11 | 0.1086 | 0.0834 | 0.0252 | 0.0812 | 0.1494 | 601 | 11 | 0.7521 | 0.8949 |
| mc_entropy_mean_roi | mcdropout: entropy of the mean sample in its ROI | non_missed_failures | 0.8621 | 0.7847 | 0.9654 | 593 | 58 | 0 | 0.1072 | 0.0818 | 0.0254 | 0.0796 | 0.1477 | 593 | 0 |  |  |
| mc_variance_mean | mcdropout: across-sample variance (whole image) | all_failures | 0.4444 | 0.3381 | 0.5804 | 612 | 77 | 0 | 0.2227 | 0.0859 | 0.1368 | 0.1547 | 0.3150 | 612 | 0 | 0.4444 | 0.4444 |
| mc_variance_mean | mcdropout: across-sample variance (whole image) | non_missed_failures | 0.4654 | 0.3578 | 0.6306 | 593 | 58 | 0 | 0.1873 | 0.0818 | 0.1056 | 0.1314 | 0.2669 | 593 | 0 |  |  |
| mc_variance_mean_roi | mcdropout: across-sample variance in its ROI | all_failures | 0.8738 | 0.8244 | 0.9373 | 601 | 66 | 11 | 0.1108 | 0.0834 | 0.0273 | 0.0831 | 0.1503 | 601 | 11 | 0.7490 | 0.8919 |
| mc_variance_mean_roi | mcdropout: across-sample variance in its ROI | non_missed_failures | 0.8627 | 0.8157 | 0.9321 | 593 | 58 | 0 | 0.1091 | 0.0818 | 0.0273 | 0.0818 | 0.1473 | 593 | 0 |  |  |
| mc_view_disagreement | mcdropout: fraction of voxels where the samples disagree | all_failures | 0.4587 | 0.3382 | 0.5552 | 612 | 77 | 0 | 0.2515 | 0.0859 | 0.1656 | 0.1757 | 0.3464 | 612 | 0 | 0.4587 | 0.4587 |
| mc_view_disagreement | mcdropout: fraction of voxels where the samples disagree | non_missed_failures | 0.5750 | 0.4831 | 0.6953 | 593 | 58 | 0 | 0.1585 | 0.0818 | 0.0768 | 0.1155 | 0.2190 | 593 | 0 |  |  |
| mc_per_view_dice_mean | mcdropout: mean agreement of each sample with the consensus | all_failures | 0.2106 | 0.1261 | 0.3184 | 612 | 77 | 0 | 0.3688 | 0.0859 | 0.2828 | 0.2951 | 0.4375 | 612 | 0 | 0.2106 | 0.2106 |
| mc_per_view_dice_mean | mcdropout: mean agreement of each sample with the consensus | non_missed_failures | 0.1413 | 0.0559 | 0.2004 | 593 | 58 | 0 | 0.3355 | 0.0818 | 0.2537 | 0.2656 | 0.4060 | 593 | 0 |  |  |

AUROC and AURC (risk = 1 - Dice) against the MC prediction's own failures with video-cluster bootstrap CIs; the ROI descriptors are undefined for a missed detection, so `n_used` drops them and the NaN bounds quantify that.

## 5. MC descriptor vs tta4 ROI entropy

| predictor | compared_with | label_prediction | failure_definition | delta_auroc | ci_low | ci_high | p_value | delta_aurc | aurc_ci_low | aurc_ci_high | aurc_p_value | n_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mc_entropy_mean | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.4317 | -0.5755 | -0.2389 | 0.0000 | 0.0823 | 0.0393 | 0.1479 | 0.0000 | 593 |
| mc_entropy_mean_roi | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.0276 | -0.0613 | 0.0105 | 0.1980 | 0.0049 | 0.0008 | 0.0110 | 0.0060 | 593 |
| mc_variance_mean | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.4243 | -0.5199 | -0.2998 | 0.0000 | 0.0850 | 0.0494 | 0.1348 | 0.0000 | 593 |
| mc_variance_mean_roi | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.0270 | -0.0735 | 0.0120 | 0.1400 | 0.0068 | 0.0024 | 0.0119 | 0.0020 | 593 |
| mc_view_disagreement | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.3147 | -0.3981 | -0.2408 | 0.0000 | 0.0562 | 0.0333 | 0.0867 | 0.0000 | 593 |
| mc_per_view_dice_mean | reference_entropy_roi | mcdropout_8 | non_missed_failures | -0.7483 | -0.9047 | -0.6323 | 0.0000 | 0.2332 | 0.1769 | 0.2775 | 0.0000 | 593 |

Paired video-cluster bootstrap on both AUROC and AURC: `delta_aurc < 0` means the MC descriptor ranks failures better than the TTA view descriptor.
