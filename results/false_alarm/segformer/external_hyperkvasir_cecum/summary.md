# MELBA external validation - HyperKvasir - normal cecum (polyp-free)

External dataset `HyperKvasir - normal cecum (polyp-free)` (`C:/Users/subka/Documents/datasets/polyp/hyperkvasir_cecum`) scored by 15 internal checkpoints over pred_empty, pred_area_px, max_prob. Scope: **external** - never trained on. The internal CVC-ClinicDB tables live in `results/aggregate_cvc_clinicdb_*`; every table here is labelled `scope: external`.

## Per-model spread over the internal checkpoints (external data)

| method | metric | n_models | n_cases | mean | std | min | max | range | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | max_prob | 15 | 1009 | 0.9384 | 0.0377 | 0.8545 | 0.9801 | 0.1256 | external |
| baseline | pred_area_px | 15 | 1009 | 12452.9080 | 5353.2209 | 4598.9247 | 23941.2379 | 19342.3132 | external |
| baseline | pred_empty | 15 | 1009 | 0.0360 | 0.0336 | 0.0050 | 0.1199 | 0.1150 | external |
| tta2 | max_prob | 15 | 1009 | 0.9038 | 0.0507 | 0.7932 | 0.9652 | 0.1719 | external |
| tta2 | pred_area_px | 15 | 1009 | 12054.2605 | 5190.3622 | 4394.4579 | 23416.4866 | 19022.0287 | external |
| tta2 | pred_empty | 15 | 1009 | 0.0505 | 0.0460 | 0.0030 | 0.1685 | 0.1655 | external |
| tta4 | max_prob | 15 | 1009 | 0.8715 | 0.0617 | 0.7485 | 0.9517 | 0.2032 | external |
| tta4 | pred_area_px | 15 | 1009 | 11207.9598 | 4999.9909 | 3938.1467 | 22648.0882 | 18709.9415 | external |
| tta4 | pred_empty | 15 | 1009 | 0.0769 | 0.0703 | 0.0059 | 0.2418 | 0.2359 | external |
| tta4_anyview | max_prob | 15 | 1009 | 0.8715 | 0.0617 | 0.7485 | 0.9517 | 0.2032 | external |
| tta4_anyview | pred_area_px | 15 | 1009 | 11309.5193 | 4957.9503 | 4179.7721 | 22714.4678 | 18534.6957 | external |
| tta4_anyview | pred_empty | 15 | 1009 | 0.0034 | 0.0046 | 0.0000 | 0.0159 | 0.0159 | external |
| tta4_lowdet_20 | max_prob | 15 | 1009 | 0.8715 | 0.0617 | 0.7485 | 0.9517 | 0.2032 | external |
| tta4_lowdet_20 | pred_area_px | 15 | 1009 | 11425.5338 | 4958.7466 | 4347.4618 | 22929.3717 | 18581.9098 | external |
| tta4_lowdet_20 | pred_empty | 15 | 1009 | 0.0027 | 0.0041 | 0.0000 | 0.0119 | 0.0119 | external |
| tta4_lowdet_30 | max_prob | 15 | 1009 | 0.8715 | 0.0617 | 0.7485 | 0.9517 | 0.2032 | external |
| tta4_lowdet_30 | pred_area_px | 15 | 1009 | 11260.6638 | 4989.4636 | 4031.3003 | 22712.6551 | 18681.3548 | external |
| tta4_lowdet_30 | pred_empty | 15 | 1009 | 0.0160 | 0.0201 | 0.0000 | 0.0664 | 0.0664 | external |
| tta4_lowdet_40 | max_prob | 15 | 1009 | 0.8715 | 0.0617 | 0.7485 | 0.9517 | 0.2032 | external |
| tta4_lowdet_40 | pred_area_px | 15 | 1009 | 11217.5886 | 4996.6774 | 3953.6165 | 22658.5491 | 18704.9326 | external |
| tta4_lowdet_40 | pred_empty | 15 | 1009 | 0.0409 | 0.0430 | 0.0010 | 0.1427 | 0.1417 | external |
| tta8 | max_prob | 15 | 1009 | 0.8426 | 0.0639 | 0.7122 | 0.9257 | 0.2135 | external |
| tta8 | pred_area_px | 15 | 1009 | 10462.4050 | 4762.2186 | 3883.9187 | 21187.7166 | 17303.7978 | external |
| tta8 | pred_empty | 15 | 1009 | 0.1007 | 0.0768 | 0.0109 | 0.2686 | 0.2577 | external |
| tta8_anyview | max_prob | 15 | 1009 | 0.8426 | 0.0639 | 0.7122 | 0.9257 | 0.2135 | external |
| tta8_anyview | pred_area_px | 15 | 1009 | 10719.0656 | 4696.7209 | 4385.0733 | 21366.8157 | 16981.7423 | external |
| tta8_anyview | pred_empty | 15 | 1009 | 0.0003 | 0.0008 | 0.0000 | 0.0030 | 0.0030 | external |
| tta8_lowdet_20 | max_prob | 15 | 1009 | 0.8426 | 0.0639 | 0.7122 | 0.9257 | 0.2135 | external |
| tta8_lowdet_20 | pred_area_px | 15 | 1009 | 10792.7409 | 4767.2266 | 4340.7056 | 21598.9019 | 17258.1962 | external |
| tta8_lowdet_20 | pred_empty | 15 | 1009 | 0.0035 | 0.0051 | 0.0000 | 0.0178 | 0.0178 | external |
| tta8_lowdet_30 | max_prob | 15 | 1009 | 0.8426 | 0.0639 | 0.7122 | 0.9257 | 0.2135 | external |
| tta8_lowdet_30 | pred_area_px | 15 | 1009 | 10551.2824 | 4759.7646 | 4012.6392 | 21280.0892 | 17267.4500 | external |
| tta8_lowdet_30 | pred_empty | 15 | 1009 | 0.0227 | 0.0273 | 0.0000 | 0.0902 | 0.0902 | external |
| tta8_lowdet_40 | max_prob | 15 | 1009 | 0.8426 | 0.0639 | 0.7122 | 0.9257 | 0.2135 | external |
| tta8_lowdet_40 | pred_area_px | 15 | 1009 | 10476.0344 | 4760.9453 | 3903.9019 | 21199.7225 | 17295.8206 | external |
| tta8_lowdet_40 | pred_empty | 15 | 1009 | 0.0554 | 0.0533 | 0.0010 | 0.1863 | 0.1853 | external |

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
| external | baseline | baseline | 1009 | 1009 | 1007 | 0.9980 | 0.9950 | 1.0000 | cluster_bootstrap_group | 1009 |  |  |  |  |  |  | case |  |  |  |  |
| external | tta2 | baseline | 1009 | 1009 | 1002 | 0.9931 | 0.9871 | 0.9980 | cluster_bootstrap_group | 1009 | 2.0000 | 0.0000 | 5.0000 | 1002.0000 | 0.0625 | 0.0625 | case |  |  |  |  |
| external | tta4 | baseline | 1009 | 1009 | 987 | 0.9782 | 0.9683 | 0.9871 | cluster_bootstrap_group | 1009 | 2.0000 | 0.0000 | 20.0000 | 987.0000 | 0.0000 | 0.0000 | case |  |  |  |  |
| external | tta8 | baseline | 1009 | 1009 | 965 | 0.9564 | 0.9435 | 0.9683 | cluster_bootstrap_group | 1009 | 2.0000 | 0.0000 | 42.0000 | 965.0000 | 0.0000 | 0.0000 | case |  |  |  |  |

**This is the primary table of this dataset.** Every frame is polyp-free, so every reference is empty: `false_positive_rate` is the share of frames on which the method reports a polyp at all, `correct_to_fp`/`fp_to_correct` count the frames the method invents that `baseline` does not (and vice versa), and `mcnemar_p` is the exact test on those discordant frames. There is no Dice or HD95 here: without a lesion they are undefined, not zero.
