# MELBA external validation - ETIS-LaribPolypDB

External dataset `ETIS-LaribPolypDB` (`C:/Users/subka/Documents/datasets/polyp/TestDataset/ETIS-LaribPolypDB/images`) scored by 15 internal checkpoints over dice, jaccard, precision, recall, hd, hd95, hd95_pooled, assd, boundary_f1, pred_volume_mm3. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | assd | 15 | 195 | 174.2904 | 43.6392 | 104.9116 | 230.6339 | 125.7223 | external |
| baseline | boundary_f1 | 15 | 195 | 0.1209 | 0.0253 | 0.0879 | 0.1553 | 0.0674 | external |
| baseline | dice | 15 | 196 | 0.4627 | 0.0603 | 0.3836 | 0.5985 | 0.2149 | external |
| baseline | hd95 | 15 | 195 | 412.9317 | 88.9574 | 252.3550 | 542.8106 | 290.4557 | external |
| baseline_lowdet_20 | assd | 15 | 196 | 197.1266 | 34.4575 | 133.7289 | 258.1770 | 124.4481 | external |
| baseline_lowdet_20 | boundary_f1 | 15 | 196 | 0.1135 | 0.0212 | 0.0874 | 0.1433 | 0.0559 | external |
| baseline_lowdet_20 | dice | 15 | 196 | 0.4634 | 0.0605 | 0.3836 | 0.6002 | 0.2166 | external |
| baseline_lowdet_20 | hd95 | 15 | 196 | 436.1611 | 74.0153 | 325.2200 | 546.2387 | 221.0187 | external |
| baseline_lowdet_30 | assd | 15 | 196 | 185.1058 | 37.1283 | 133.2242 | 245.8189 | 112.5947 | external |
| baseline_lowdet_30 | boundary_f1 | 15 | 196 | 0.1167 | 0.0231 | 0.0879 | 0.1519 | 0.0640 | external |
| baseline_lowdet_30 | dice | 15 | 196 | 0.4630 | 0.0604 | 0.3836 | 0.5997 | 0.2161 | external |
| baseline_lowdet_30 | hd95 | 15 | 196 | 423.7457 | 81.4293 | 307.1578 | 545.4654 | 238.3076 | external |
| baseline_lowdet_40 | assd | 15 | 196 | 176.9334 | 43.2056 | 105.5272 | 235.9219 | 130.3946 | external |
| baseline_lowdet_40 | boundary_f1 | 15 | 196 | 0.1195 | 0.0251 | 0.0879 | 0.1536 | 0.0657 | external |
| baseline_lowdet_40 | dice | 15 | 196 | 0.4628 | 0.0603 | 0.3836 | 0.5987 | 0.2151 | external |
| baseline_lowdet_40 | hd95 | 15 | 196 | 414.7590 | 88.8323 | 252.5748 | 542.8106 | 290.2358 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

_(no rows)_

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 196 | 196 | 5 | 0.0255 | 0.0051 | 0.0510 | cluster_bootstrap_video | 196 |  |  |  |  |  |  | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | case |  |  |  |  |

None of the three external collections contains an empty reference mask, so this table is expected to report `n_cases = 0`; it is kept because the internal BraTS runs use the same code path and the two scopes must stay comparable.
