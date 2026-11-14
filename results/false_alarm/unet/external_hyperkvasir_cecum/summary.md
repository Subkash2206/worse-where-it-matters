# MELBA external validation - HyperKvasir - normal cecum (polyp-free)

External dataset `HyperKvasir - normal cecum (polyp-free)` (`C:/Users/subka/Documents/datasets/polyp/hyperkvasir_cecum`) scored by 15 internal checkpoints over pred_empty, pred_area_px, max_prob. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | max_prob | 15 | 1009 | 0.7438 | 0.0866 | 0.5770 | 0.8807 | 0.3037 | external |
| baseline | pred_area_px | 15 | 1009 | 15609.9499 | 10571.1274 | 5258.3290 | 47275.6065 | 42017.2775 | external |
| baseline | pred_empty | 15 | 1009 | 0.2918 | 0.1266 | 0.1021 | 0.5183 | 0.4163 | external |
| tta2 | max_prob | 15 | 1009 | 0.6823 | 0.0902 | 0.5194 | 0.8419 | 0.3225 | external |
| tta2 | pred_area_px | 15 | 1009 | 14150.5833 | 9547.8164 | 5082.5956 | 42650.2537 | 37567.6581 | external |
| tta2 | pred_empty | 15 | 1009 | 0.3253 | 0.1200 | 0.1338 | 0.5074 | 0.3736 | external |
| tta4 | max_prob | 15 | 1009 | 0.6289 | 0.0914 | 0.4643 | 0.8047 | 0.3405 | external |
| tta4 | pred_area_px | 15 | 1009 | 12772.8276 | 9499.0577 | 4032.2854 | 41639.0803 | 37606.7948 | external |
| tta4 | pred_empty | 15 | 1009 | 0.3869 | 0.1298 | 0.1437 | 0.5966 | 0.4529 | external |
| tta4_anyview | max_prob | 15 | 1009 | 0.6289 | 0.0914 | 0.4643 | 0.8047 | 0.3405 | external |
| tta4_anyview | pred_area_px | 15 | 1009 | 14305.8298 | 9504.6989 | 5151.3112 | 43203.2755 | 38051.9643 | external |
| tta4_anyview | pred_empty | 15 | 1009 | 0.1077 | 0.0816 | 0.0129 | 0.3003 | 0.2874 | external |
| tta4_lowdet_20 | max_prob | 15 | 1009 | 0.6289 | 0.0914 | 0.4643 | 0.8047 | 0.3405 | external |
| tta4_lowdet_20 | pred_area_px | 15 | 1009 | 15453.2348 | 9233.4889 | 5259.5064 | 43896.1179 | 38636.6115 | external |
| tta4_lowdet_20 | pred_empty | 15 | 1009 | 0.0774 | 0.0917 | 0.0000 | 0.2844 | 0.2844 | external |
| tta4_lowdet_30 | max_prob | 15 | 1009 | 0.6289 | 0.0914 | 0.4643 | 0.8047 | 0.3405 | external |
| tta4_lowdet_30 | pred_area_px | 15 | 1009 | 13371.6312 | 9448.5041 | 4131.4182 | 42306.7047 | 38175.2864 | external |
| tta4_lowdet_30 | pred_empty | 15 | 1009 | 0.1974 | 0.1548 | 0.0000 | 0.4955 | 0.4955 | external |
| tta4_lowdet_40 | max_prob | 15 | 1009 | 0.6289 | 0.0914 | 0.4643 | 0.8047 | 0.3405 | external |
| tta4_lowdet_40 | pred_area_px | 15 | 1009 | 12888.8693 | 9538.4374 | 4061.8107 | 41870.4500 | 37808.6392 | external |
| tta4_lowdet_40 | pred_empty | 15 | 1009 | 0.2891 | 0.1365 | 0.0694 | 0.5391 | 0.4698 | external |
| tta8 | max_prob | 15 | 1009 | 0.5949 | 0.0917 | 0.4350 | 0.7765 | 0.3416 | external |
| tta8 | pred_area_px | 15 | 1009 | 12106.3042 | 9733.2367 | 3425.7958 | 42292.4945 | 38866.6987 | external |
| tta8 | pred_empty | 15 | 1009 | 0.4346 | 0.1304 | 0.1685 | 0.6492 | 0.4807 | external |
| tta8_anyview | max_prob | 15 | 1009 | 0.5949 | 0.0917 | 0.4350 | 0.7765 | 0.3416 | external |
| tta8_anyview | pred_area_px | 15 | 1009 | 15457.9741 | 9693.4691 | 5848.7721 | 45669.9693 | 39821.1972 | external |
| tta8_anyview | pred_empty | 15 | 1009 | 0.0552 | 0.0562 | 0.0059 | 0.2101 | 0.2042 | external |
| tta8_lowdet_20 | max_prob | 15 | 1009 | 0.5949 | 0.0917 | 0.4350 | 0.7765 | 0.3416 | external |
| tta8_lowdet_20 | pred_area_px | 15 | 1009 | 15456.5136 | 9519.0865 | 4413.1893 | 45564.4182 | 41151.2289 | external |
| tta8_lowdet_20 | pred_empty | 15 | 1009 | 0.1054 | 0.1232 | 0.0000 | 0.4004 | 0.4004 | external |
| tta8_lowdet_30 | max_prob | 15 | 1009 | 0.5949 | 0.0917 | 0.4350 | 0.7765 | 0.3416 | external |
| tta8_lowdet_30 | pred_area_px | 15 | 1009 | 12901.3870 | 9746.3597 | 3715.1804 | 43392.3786 | 39677.1982 | external |
| tta8_lowdet_30 | pred_empty | 15 | 1009 | 0.2151 | 0.1573 | 0.0000 | 0.5015 | 0.5015 | external |
| tta8_lowdet_40 | max_prob | 15 | 1009 | 0.5949 | 0.0917 | 0.4350 | 0.7765 | 0.3416 | external |
| tta8_lowdet_40 | pred_area_px | 15 | 1009 | 12219.9845 | 9755.6334 | 3458.0912 | 42510.8712 | 39052.7800 | external |
| tta8_lowdet_40 | pred_empty | 15 | 1009 | 0.3356 | 0.1421 | 0.1001 | 0.5837 | 0.4836 | external |

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
| external | baseline | baseline | 1009 | 1009 | 779 | 0.7721 | 0.7463 | 0.7978 | cluster_bootstrap_group | 1009 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 1009 | 1009 | 722 | 0.7156 | 0.6878 | 0.7423 | cluster_bootstrap_group | 1009 | 217.0000 | 13.0000 | 70.0000 | 709.0000 | 0.0000 | 0.0000 | case |  |  |  |  |
| external | tta4 | baseline | 1009 | 1009 | 630 | 0.6244 | 0.5946 | 0.6551 | cluster_bootstrap_group | 1009 | 226.0000 | 4.0000 | 153.0000 | 626.0000 | 0.0000 | 0.0000 | case |  |  |  |  |
| external | tta8 | baseline | 1009 | 1009 | 576 | 0.5709 | 0.5411 | 0.6016 | cluster_bootstrap_group | 1009 | 227.0000 | 3.0000 | 206.0000 | 573.0000 | 0.0000 | 0.0000 | case |  |  |  |  |

**This is the primary table of this dataset.** Every frame is polyp-free, so every reference is empty: `false_positive_rate` is the share of frames on which the method reports a polyp at all, `correct_to_fp`/`fp_to_correct` count the frames the method invents that `baseline` does not (and vice versa), and `mcnemar_p` is the exact test on those discordant frames. There is no Dice or HD95 here: without a lesion they are undefined, not zero.
