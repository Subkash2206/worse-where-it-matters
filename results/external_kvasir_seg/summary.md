# MELBA external validation - Kvasir-SEG

External dataset `Kvasir-SEG` (`C:/Users/subka/Documents/datasets/polyp/Kvasir-SEG/images`) scored by 15 internal checkpoints. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 1000 | 20.2916 | 1.5199 | 18.1995 | 23.0149 | 4.8155 | external |
| baseline | boundary_f1 | 15 | 1000 | 0.3345 | 0.0250 | 0.2791 | 0.3579 | 0.0788 | external |
| baseline | dice | 15 | 1000 | 0.8428 | 0.0081 | 0.8246 | 0.8552 | 0.0306 | external |
| baseline | hd95 | 15 | 1000 | 84.0048 | 7.8539 | 70.0475 | 94.7106 | 24.6631 | external |
| tta2 | assd | 15 | 1000 | 19.8083 | 1.3724 | 18.0891 | 22.3401 | 4.2510 | external |
| tta2 | boundary_f1 | 15 | 1000 | 0.3483 | 0.0263 | 0.2870 | 0.3749 | 0.0879 | external |
| tta2 | dice | 15 | 1000 | 0.8474 | 0.0080 | 0.8286 | 0.8602 | 0.0316 | external |
| tta2 | hd95 | 15 | 1000 | 81.3758 | 7.8195 | 68.4968 | 93.1178 | 24.6210 | external |
| tta4 | assd | 15 | 1000 | 19.1699 | 1.2501 | 17.4595 | 21.2466 | 3.7871 | external |
| tta4 | boundary_f1 | 15 | 1000 | 0.3617 | 0.0286 | 0.2955 | 0.3905 | 0.0949 | external |
| tta4 | dice | 15 | 1000 | 0.8511 | 0.0077 | 0.8337 | 0.8624 | 0.0287 | external |
| tta4 | hd95 | 15 | 1000 | 78.2964 | 7.1866 | 66.1445 | 88.6331 | 22.4886 | external |
| tta8 | assd | 15 | 1000 | 18.8059 | 1.2035 | 17.1866 | 20.8456 | 3.6590 | external |
| tta8 | boundary_f1 | 15 | 1000 | 0.3655 | 0.0306 | 0.2939 | 0.3948 | 0.1010 | external |
| tta8 | dice | 15 | 1000 | 0.8528 | 0.0079 | 0.8333 | 0.8639 | 0.0306 | external |
| tta8 | hd95 | 15 | 1000 | 76.2912 | 7.3073 | 64.0845 | 86.6102 | 22.5257 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

| scope | metric | metric_direction | treatment | control | unit | n_units | restriction | n_cases_used | n_pairs | n_dropped | mean_control | mean_treatment | delta_mean | delta_median | ci_low | ci_high | effect_size | effect_size_name | statistic | test | p_value | p_value_adjusted | preferred | alternative | t_statistic | p_value_secondary | secondary_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | dice | higher | tta2 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.8428 | 0.8474 | 0.0046 | 0.0030 | 0.0036 | 0.0055 | 0.4503 | rank_biserial | 137565.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | 9.1232 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta2 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 84.0048 | 81.3946 | -2.6102 | -1.1017 | -3.3451 | -1.8841 | -0.3564 | rank_biserial | 161063.0000 | wilcoxon | 0.0000 | 0.0000 | tta2 | two-sided | -7.0632 | 0.0000 | t_test_rel |
| external | dice | higher | tta4 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.8428 | 0.8511 | 0.0083 | 0.0056 | 0.0069 | 0.0097 | 0.5406 | rank_biserial | 114972.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | 11.6463 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta4 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 84.0048 | 78.3163 | -5.6885 | -2.1869 | -6.7163 | -4.6838 | -0.4775 | rank_biserial | 130761.0000 | wilcoxon | 0.0000 | 0.0000 | tta4 | two-sided | -11.1216 | 0.0000 | t_test_rel |
| external | dice | higher | tta8 | baseline | group | 1000 | all_cases | 1000 | 1000 | 0 | 0.8428 | 0.8528 | 0.0101 | 0.0073 | 0.0086 | 0.0116 | 0.5830 | rank_biserial | 104363.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | 12.9894 | 0.0000 | t_test_rel |
| external | hd95 | lower | tta8 | baseline | group | 1000 | both_nonempty | 1000 | 1000 | 0 | 84.0048 | 76.3208 | -7.6840 | -3.0028 | -8.8574 | -6.5563 | -0.5413 | rank_biserial | 114796.0000 | wilcoxon | 0.0000 | 0.0000 | tta8 | two-sided | -13.0711 | 0.0000 | t_test_rel |

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
