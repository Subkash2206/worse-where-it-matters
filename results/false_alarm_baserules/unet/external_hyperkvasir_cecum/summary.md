# MELBA external validation - HyperKvasir - normal cecum (polyp-free)

External dataset `HyperKvasir - normal cecum (polyp-free)` (`C:/Users/subka/Documents/datasets/polyp/hyperkvasir_cecum`) scored by 15 internal checkpoints over pred_empty, pred_area_px, max_prob. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | max_prob | 15 | 1009 | 0.7438 | 0.0866 | 0.5770 | 0.8807 | 0.3037 | external |
| baseline | pred_area_px | 15 | 1009 | 15609.9499 | 10571.1274 | 5258.3290 | 47275.6065 | 42017.2775 | external |
| baseline | pred_empty | 15 | 1009 | 0.2918 | 0.1266 | 0.1021 | 0.5183 | 0.4163 | external |
| baseline_lowdet_20 | max_prob | 15 | 1009 | 0.7438 | 0.0866 | 0.5770 | 0.8807 | 0.3037 | external |
| baseline_lowdet_20 | pred_area_px | 15 | 1009 | 16002.8046 | 10383.1626 | 5273.4807 | 47363.6293 | 42090.1487 | external |
| baseline_lowdet_20 | pred_empty | 15 | 1009 | 0.0361 | 0.0653 | 0.0000 | 0.2101 | 0.2101 | external |
| baseline_lowdet_30 | max_prob | 15 | 1009 | 0.7438 | 0.0866 | 0.5770 | 0.8807 | 0.3037 | external |
| baseline_lowdet_30 | pred_area_px | 15 | 1009 | 15663.6162 | 10544.4032 | 5260.3280 | 47295.7463 | 42035.4182 | external |
| baseline_lowdet_30 | pred_empty | 15 | 1009 | 0.1516 | 0.1597 | 0.0000 | 0.4856 | 0.4856 | external |
| baseline_lowdet_40 | max_prob | 15 | 1009 | 0.7438 | 0.0866 | 0.5770 | 0.8807 | 0.3037 | external |
| baseline_lowdet_40 | pred_area_px | 15 | 1009 | 15613.8081 | 10569.0894 | 5258.6006 | 47276.3845 | 42017.7839 | external |
| baseline_lowdet_40 | pred_empty | 15 | 1009 | 0.2436 | 0.1467 | 0.0000 | 0.5064 | 0.5064 | external |

Every number comes from the same external cases scored by each checkpoint of the internal CVC grid; `mean`/`std`/`min`/`max`/`range` describe how much the external result depends on the checkpoint drawn, not on the dataset.

## Paired TTA comparisons on the external dataset (cases averaged over models)

_(no rows)_

Metrics are first averaged per external case across the checkpoints, so each image is one independent unit; the CI and the Wilcoxon test then describe the TTA effect on the external cases, not on the models.

## Missed detections on the external dataset

| scope | method | control | n_cases | n_videos | n_missed | missed_rate | ci_low | ci_high | unit | n | both_nonempty | nonempty_to_empty | empty_to_nonempty | both_empty | mcnemar_p | mcnemar_p_adjusted | transition_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_video | 0 |  |  |  |  |  |  | case |

An empty prediction on an external polyp is a failure to detect; the rate is over the external cases (each image is its own unit - these collections have no video structure, so no clustering is claimed).

## False positives on empty references

| scope | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external | baseline | baseline | 1009 | 1009 | 779 | 0.7721 | 0.7463 | 0.7978 | cluster_bootstrap_group | 1009 |  |  |  |  |  |  | case |  |  |  |  |

**This is the primary table of this dataset.** Every frame is polyp-free, so every reference is empty: `false_positive_rate` is the share of frames on which the method reports a polyp at all, `correct_to_fp`/`fp_to_correct` count the frames the method invents that `baseline` does not (and vice versa), and `mcnemar_p` is the exact test on those discordant frames. There is no Dice or HD95 here: without a lesion they are undefined, not zero.
