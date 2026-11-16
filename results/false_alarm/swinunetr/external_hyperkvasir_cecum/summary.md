# MELBA external validation - HyperKvasir - normal cecum (polyp-free)

External dataset `HyperKvasir - normal cecum (polyp-free)` (`C:/Users/subka/Documents/datasets/polyp/hyperkvasir_cecum`) scored by 15 internal checkpoints over pred_empty, pred_area_px, max_prob. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | max_prob | 15 | 1009 | 0.9968 | 0.0049 | 0.9833 | 1.0000 | 0.0167 | external |
| baseline | pred_area_px | 15 | 1009 | 35482.3599 | 8585.4202 | 20470.5619 | 52638.5005 | 32167.9386 | external |
| baseline | pred_empty | 15 | 1009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | external |
| tta2 | max_prob | 15 | 1009 | 0.9946 | 0.0076 | 0.9748 | 1.0000 | 0.0251 | external |
| tta2 | pred_area_px | 15 | 1009 | 34447.0501 | 8110.5741 | 19456.9148 | 50018.4034 | 30561.4886 | external |
| tta2 | pred_empty | 15 | 1009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | external |
| tta4 | max_prob | 15 | 1009 | 0.9912 | 0.0107 | 0.9667 | 0.9996 | 0.0329 | external |
| tta4 | pred_area_px | 15 | 1009 | 32562.8833 | 8543.7908 | 17997.8533 | 48422.3082 | 30424.4549 | external |
| tta4 | pred_empty | 15 | 1009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | external |
| tta8 | max_prob | 15 | 1009 | 0.9870 | 0.0119 | 0.9605 | 0.9970 | 0.0365 | external |
| tta8 | pred_area_px | 15 | 1009 | 31454.6877 | 8923.7199 | 16991.9851 | 47906.7255 | 30914.7403 | external |
| tta8 | pred_empty | 15 | 1009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

_(no rows)_

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_video | 0 |  |  |  |  |  |  | case |
| external | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_video | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_video | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |
| external | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_video | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 1009 | 1009 | 1009 | 1.0000 | 1.0000 | 1.0000 | cluster_bootstrap_group | 1009 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 1009 | 1009 | 1009 | 1.0000 | 1.0000 | 1.0000 | cluster_bootstrap_group | 1009 | 0.0000 | 0.0000 | 0.0000 | 1009.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta4 | baseline | 1009 | 1009 | 1009 | 1.0000 | 1.0000 | 1.0000 | cluster_bootstrap_group | 1009 | 0.0000 | 0.0000 | 0.0000 | 1009.0000 | 1.0000 | 1.0000 | case |  |  |  |  |
| external | tta8 | baseline | 1009 | 1009 | 1009 | 1.0000 | 1.0000 | 1.0000 | cluster_bootstrap_group | 1009 | 0.0000 | 0.0000 | 0.0000 | 1009.0000 | 1.0000 | 1.0000 | case |  |  |  |  |

**This is the primary table of this dataset.** Every frame is polyp-free, so every reference is empty: `false_positive_rate` is the share of frames on which the method reports a polyp at all, `correct_to_fp`/`fp_to_correct` count the frames the method invents that `baseline` does not (and vice versa), and `mcnemar_p` is the exact test on those discordant frames. There is no Dice or HD95 here: without a lesion they are undefined, not zero.
