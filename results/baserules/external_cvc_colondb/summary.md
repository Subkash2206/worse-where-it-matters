# MELBA external validation - CVC-ColonDB

External dataset `CVC-ColonDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/CVC-ColonDB/images`) scored by 15 internal checkpoints over dice, jaccard, precision, recall, hd, hd95, hd95_pooled, assd, boundary_f1, pred_volume_mm3. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 344 | 31.8797 | 7.8104 | 20.5903 | 47.3116 | 26.7214 | external |
| baseline | boundary_f1 | 15 | 344 | 0.3010 | 0.0313 | 0.2240 | 0.3459 | 0.1220 | external |
| baseline | dice | 15 | 380 | 0.6307 | 0.0235 | 0.5854 | 0.6715 | 0.0860 | external |
| baseline | hd95 | 15 | 344 | 93.8897 | 20.4245 | 59.7421 | 133.9072 | 74.1650 | external |
| baseline_lowdet_20 | assd | 15 | 380 | 56.4429 | 19.8369 | 20.8593 | 88.4464 | 67.5871 | external |
| baseline_lowdet_20 | boundary_f1 | 15 | 380 | 0.2710 | 0.0351 | 0.2035 | 0.3433 | 0.1399 | external |
| baseline_lowdet_20 | dice | 15 | 380 | 0.6327 | 0.0235 | 0.5881 | 0.6730 | 0.0849 | external |
| baseline_lowdet_20 | hd95 | 15 | 380 | 124.8052 | 27.1823 | 59.7725 | 152.7741 | 93.0016 | external |
| baseline_lowdet_30 | assd | 15 | 380 | 45.2215 | 22.5708 | 20.8206 | 90.9016 | 70.0809 | external |
| baseline_lowdet_30 | boundary_f1 | 15 | 380 | 0.2860 | 0.0406 | 0.2030 | 0.3442 | 0.1412 | external |
| baseline_lowdet_30 | dice | 15 | 380 | 0.6315 | 0.0234 | 0.5866 | 0.6715 | 0.0849 | external |
| baseline_lowdet_30 | hd95 | 15 | 380 | 109.1253 | 31.6715 | 59.7421 | 152.5184 | 92.7763 | external |
| baseline_lowdet_40 | assd | 15 | 380 | 34.9286 | 13.2415 | 20.8206 | 73.4977 | 52.6771 | external |
| baseline_lowdet_40 | boundary_f1 | 15 | 380 | 0.2973 | 0.0354 | 0.2027 | 0.3442 | 0.1414 | external |
| baseline_lowdet_40 | dice | 15 | 380 | 0.6308 | 0.0235 | 0.5855 | 0.6715 | 0.0860 | external |
| baseline_lowdet_40 | hd95 | 15 | 380 | 97.1212 | 24.9241 | 59.7421 | 153.2800 | 93.5379 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

_(no rows)_

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 380 | 380 | 55 | 0.1447 | 0.1105 | 0.1816 | cluster_bootstrap_video | 380 |  |  |  |  |  |  | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
