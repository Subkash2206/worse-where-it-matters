# MELBA subgroup analysis - lesion size

CPU-only re-analysis of the per-case CSVs an existing cross-validation grid already wrote. Size bins are tertiles of the reference area computed per dataset; the cutoffs are repeated in every table (`cutoff_low_px`, `cutoff_high_px`).

## Reference lesion size per case

| fold | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 118.0000 | 14453.7542 | 11363.5452 | 444.0000 | 5915.2500 | 12236.0000 | 20294.0000 | 54148.0000 |
| 1.0000 | 137.0000 | 9854.5109 | 8487.6240 | 953.0000 | 3089.0000 | 5984.0000 | 14277.0000 | 35211.0000 |
| 2.0000 | 126.0000 | 11178.9603 | 7208.0774 | 1963.0000 | 5592.7500 | 9458.0000 | 15511.2500 | 34386.0000 |
| 3.0000 | 129.0000 | 7894.7674 | 6253.2112 | 373.0000 | 3503.0000 | 6334.0000 | 9753.0000 | 29959.0000 |
| 4.0000 | 102.0000 | 7068.0784 | 4992.0198 | 424.0000 | 4047.0000 | 5498.0000 | 8298.0000 | 23771.0000 |

`gt_pixels` is the reference foreground area at the native resolution (mm^3 is in the CSV when the dataset declares a spacing).

## Metric and miss rate per subgroup (seeds averaged per case)

| model | method | metric | size_bin | n_cases | n_units | dice_mean | ci_low | ci_high | n_missed | missed_rate | missed_ci_low | missed_ci_high | hd95_mean | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvc_clinicdb/unet_resnet50 | baseline | dice | small | 204 | 22 | 0.7725 | 0.7082 | 0.8260 | 6 | 0.0294 | 0.0065 | 0.0553 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | baseline | hd95 | small | 204 | 22 |  | 30.7499 | 55.6898 | 6 | 0.0294 | 0.0065 | 0.0553 | 41.6802 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | baseline | dice | medium | 204 | 25 | 0.8282 | 0.7636 | 0.8789 | 6 | 0.0294 | 0.0000 | 0.0842 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | baseline | hd95 | medium | 204 | 25 |  | 31.3376 | 56.1629 | 6 | 0.0294 | 0.0000 | 0.0842 | 42.4416 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | baseline | dice | large | 204 | 21 | 0.7281 | 0.6275 | 0.8162 | 3 | 0.0147 | 0.0000 | 0.0291 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | baseline | hd95 | large | 204 | 21 |  | 53.6194 | 85.8837 | 3 | 0.0147 | 0.0000 | 0.0291 | 70.3036 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | dice | small | 204 | 22 | 0.7835 | 0.7119 | 0.8396 | 11 | 0.0539 | 0.0163 | 0.0993 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | hd95 | small | 204 | 22 |  | 23.9080 | 51.9447 | 11 | 0.0539 | 0.0163 | 0.0993 | 35.4675 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | dice | medium | 204 | 25 | 0.8425 | 0.7800 | 0.8915 | 6 | 0.0294 | 0.0000 | 0.0842 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | hd95 | medium | 204 | 25 |  | 27.0595 | 53.2706 | 6 | 0.0294 | 0.0000 | 0.0842 | 38.4815 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | dice | large | 204 | 21 | 0.7463 | 0.6456 | 0.8335 | 1 | 0.0049 | 0.0000 | 0.0159 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | hd95 | large | 204 | 21 |  | 48.8037 | 80.5696 | 1 | 0.0049 | 0.0000 | 0.0159 | 65.7545 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | dice | small | 204 | 22 | 0.7966 | 0.7309 | 0.8505 | 12 | 0.0588 | 0.0225 | 0.1072 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | hd95 | small | 204 | 22 |  | 16.4869 | 42.5857 | 12 | 0.0588 | 0.0225 | 0.1072 | 26.8977 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | dice | medium | 204 | 25 | 0.8459 | 0.7789 | 0.8984 | 6 | 0.0294 | 0.0000 | 0.0842 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | hd95 | medium | 204 | 25 |  | 22.5147 | 49.8282 | 6 | 0.0294 | 0.0000 | 0.0842 | 34.0442 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | dice | large | 204 | 21 | 0.7572 | 0.6553 | 0.8449 | 1 | 0.0049 | 0.0000 | 0.0159 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | hd95 | large | 204 | 21 |  | 44.1425 | 77.3383 | 1 | 0.0049 | 0.0000 | 0.0159 | 61.4214 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | dice | small | 204 | 22 | 0.8015 | 0.7296 | 0.8584 | 11 | 0.0539 | 0.0202 | 0.0961 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | hd95 | small | 204 | 22 |  | 16.5343 | 44.2904 | 11 | 0.0539 | 0.0202 | 0.0961 | 27.4083 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | dice | medium | 204 | 25 | 0.8539 | 0.7836 | 0.9043 | 7 | 0.0343 | 0.0000 | 0.0900 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | hd95 | medium | 204 | 25 |  | 19.6666 | 46.7241 | 7 | 0.0343 | 0.0000 | 0.0900 | 30.4829 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | dice | large | 204 | 21 | 0.7585 | 0.6556 | 0.8477 | 1 | 0.0049 | 0.0000 | 0.0159 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | hd95 | large | 204 | 21 |  | 43.7676 | 78.3772 | 1 | 0.0049 | 0.0000 | 0.0159 | 61.7542 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | dice | small | 204 | 22 | 0.4169 | 0.3358 | 0.5021 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | hd95 | small | 204 | 22 |  | 99.3924 | 126.6139 | 0 | 0.0000 | 0.0000 | 0.0000 | 112.9865 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | dice | medium | 204 | 25 | 0.6002 | 0.4955 | 0.7001 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | hd95 | medium | 204 | 25 |  | 69.0088 | 105.9915 | 0 | 0.0000 | 0.0000 | 0.0000 | 86.2557 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | dice | large | 204 | 21 | 0.6241 | 0.4945 | 0.7277 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | baseline | hd95 | large | 204 | 21 |  | 60.1417 | 103.9992 | 0 | 0.0000 | 0.0000 | 0.0000 | 79.2668 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | dice | small | 204 | 22 | 0.4285 | 0.3473 | 0.5173 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | hd95 | small | 204 | 22 |  | 93.6021 | 121.7669 | 0 | 0.0000 | 0.0000 | 0.0000 | 107.8629 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | dice | medium | 204 | 25 | 0.6181 | 0.5236 | 0.7086 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | hd95 | medium | 204 | 25 |  | 65.1408 | 100.5064 | 0 | 0.0000 | 0.0000 | 0.0000 | 82.3716 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | dice | large | 204 | 21 | 0.6335 | 0.5106 | 0.7317 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | hd95 | large | 204 | 21 |  | 59.6623 | 104.0185 | 0 | 0.0000 | 0.0000 | 0.0000 | 79.4162 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | dice | small | 204 | 22 | 0.4397 | 0.3517 | 0.5380 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | hd95 | small | 204 | 22 |  | 86.5734 | 115.3424 | 0 | 0.0000 | 0.0000 | 0.0000 | 100.7116 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | dice | medium | 204 | 25 | 0.6232 | 0.5211 | 0.7234 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | hd95 | medium | 204 | 25 |  | 60.7200 | 97.2423 | 0 | 0.0000 | 0.0000 | 0.0000 | 78.2492 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | dice | large | 204 | 21 | 0.6332 | 0.5070 | 0.7333 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | hd95 | large | 204 | 21 |  | 58.8003 | 104.7966 | 0 | 0.0000 | 0.0000 | 0.0000 | 79.0418 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | dice | small | 204 | 22 | 0.4405 | 0.3522 | 0.5401 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | hd95 | small | 204 | 22 |  | 83.8478 | 111.6464 | 0 | 0.0000 | 0.0000 | 0.0000 | 97.7323 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | dice | medium | 204 | 25 | 0.6283 | 0.5303 | 0.7277 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | hd95 | medium | 204 | 25 |  | 57.9770 | 94.8853 | 0 | 0.0000 | 0.0000 | 0.0000 | 75.8617 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | dice | large | 204 | 21 | 0.6352 | 0.5098 | 0.7359 | 0 | 0.0000 | 0.0000 | 0.0000 |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | hd95 | large | 204 | 21 |  | 58.1204 | 103.2966 | 0 | 0.0000 | 0.0000 | 0.0000 | 77.9231 | 4923.6667 | 11247.3333 |

Tertiles of the reference area (`small`/`medium`/`large`); CIs resample groups (videos), so frames of one video are not counted as independent.

## Paired TTA delta per size bin

| model | size_bin | treatment | metric | n_units | n_cases_used | delta_mean | ci_low | ci_high | p_value | p_value_adjusted | preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvc_clinicdb/unet_resnet50 | small | tta2 | dice | 22 | 204 | 0.0042 | -0.0099 | 0.0181 | 0.4628 | 0.9256 | tta2 |
| cvc_clinicdb/unet_resnet50 | small | tta2 | hd95 | 21 | 193 | 0.5945 | -5.4265 | 7.8447 | 0.4948 | 0.9256 | baseline |
| cvc_clinicdb/unet_resnet50 | small | tta4 | dice | 22 | 204 | 0.0127 | -0.0131 | 0.0322 | 0.0190 | 0.0190 | tta4 |
| cvc_clinicdb/unet_resnet50 | small | tta4 | hd95 | 21 | 192 | -8.3727 | -14.2011 | -1.8444 | 0.0090 | 0.0180 | tta4 |
| cvc_clinicdb/unet_resnet50 | small | tta8 | dice | 22 | 204 | 0.0261 | 0.0062 | 0.0445 | 0.0059 | 0.0119 | tta8 |
| cvc_clinicdb/unet_resnet50 | small | tta8 | hd95 | 21 | 193 | -7.6740 | -14.0002 | -0.5728 | 0.0263 | 0.0263 | tta8 |
| cvc_clinicdb/unet_resnet50 | medium | tta2 | dice | 25 | 204 | 0.0092 | -0.0105 | 0.0220 | 0.0061 | 0.0123 | tta2 |
| cvc_clinicdb/unet_resnet50 | medium | tta2 | hd95 | 24 | 198 | -4.9106 | -9.0428 | -1.1529 | 0.0425 | 0.0425 | tta2 |
| cvc_clinicdb/unet_resnet50 | medium | tta4 | dice | 25 | 204 | 0.0140 | -0.0058 | 0.0324 | 0.0187 | 0.0187 | tta4 |
| cvc_clinicdb/unet_resnet50 | medium | tta4 | hd95 | 24 | 198 | -10.9631 | -20.2504 | -3.2570 | 0.0048 | 0.0096 | tta4 |
| cvc_clinicdb/unet_resnet50 | medium | tta8 | dice | 25 | 204 | 0.0173 | -0.0129 | 0.0428 | 0.0160 | 0.0160 | tta8 |
| cvc_clinicdb/unet_resnet50 | medium | tta8 | hd95 | 24 | 197 | -12.3817 | -21.6865 | -4.0310 | 0.0039 | 0.0078 | tta8 |
| cvc_clinicdb/unet_resnet50 | large | tta2 | dice | 21 | 204 | 0.0227 | 0.0054 | 0.0432 | 0.0142 | 0.0283 | tta2 |
| cvc_clinicdb/unet_resnet50 | large | tta2 | hd95 | 21 | 201 | -5.2005 | -10.3858 | -0.3909 | 0.0239 | 0.0283 | tta2 |
| cvc_clinicdb/unet_resnet50 | large | tta4 | dice | 21 | 204 | 0.0313 | 0.0138 | 0.0530 | 0.0012 | 0.0024 | tta4 |
| cvc_clinicdb/unet_resnet50 | large | tta4 | hd95 | 21 | 201 | -9.6059 | -15.5371 | -3.9941 | 0.0025 | 0.0025 | tta4 |
| cvc_clinicdb/unet_resnet50 | large | tta8 | dice | 21 | 204 | 0.0366 | 0.0170 | 0.0611 | 0.0002 | 0.0003 | tta8 |
| cvc_clinicdb/unet_resnet50 | large | tta8 | hd95 | 21 | 201 | -10.4364 | -15.4246 | -5.7914 | 0.0003 | 0.0003 | tta8 |
| cvc_clinicdb/swinunetr_2d | small | tta2 | dice | 22 | 204 | 0.0135 | 0.0065 | 0.0214 | 0.0017 | 0.0017 | tta2 |
| cvc_clinicdb/swinunetr_2d | small | tta2 | hd95 | 22 | 204 | -5.4419 | -8.1729 | -3.0155 | 0.0002 | 0.0004 | tta2 |
| cvc_clinicdb/swinunetr_2d | small | tta4 | dice | 22 | 204 | 0.0287 | 0.0144 | 0.0434 | 0.0008 | 0.0008 | tta4 |
| cvc_clinicdb/swinunetr_2d | small | tta4 | hd95 | 22 | 204 | -12.2029 | -16.0574 | -8.9629 | 0.0000 | 0.0000 | tta4 |
| cvc_clinicdb/swinunetr_2d | small | tta8 | dice | 22 | 204 | 0.0226 | 0.0051 | 0.0407 | 0.0093 | 0.0093 | tta8 |
| cvc_clinicdb/swinunetr_2d | small | tta8 | hd95 | 22 | 204 | -15.1251 | -20.8076 | -10.2622 | 0.0000 | 0.0000 | tta8 |
| cvc_clinicdb/swinunetr_2d | medium | tta2 | dice | 25 | 204 | 0.0158 | 0.0054 | 0.0269 | 0.0018 | 0.0036 | tta2 |
| cvc_clinicdb/swinunetr_2d | medium | tta2 | hd95 | 25 | 204 | -4.3495 | -7.9998 | -0.7779 | 0.0125 | 0.0125 | tta2 |
| cvc_clinicdb/swinunetr_2d | medium | tta4 | dice | 25 | 204 | 0.0229 | 0.0112 | 0.0364 | 0.0006 | 0.0011 | tta4 |
| cvc_clinicdb/swinunetr_2d | medium | tta4 | hd95 | 25 | 204 | -7.8268 | -12.4056 | -3.7264 | 0.0012 | 0.0012 | tta4 |
| cvc_clinicdb/swinunetr_2d | medium | tta8 | dice | 25 | 204 | 0.0302 | 0.0159 | 0.0468 | 0.0002 | 0.0003 | tta8 |
| cvc_clinicdb/swinunetr_2d | medium | tta8 | hd95 | 25 | 204 | -10.5315 | -14.9505 | -6.5709 | 0.0002 | 0.0003 | tta8 |
| cvc_clinicdb/swinunetr_2d | large | tta2 | dice | 21 | 204 | 0.0098 | 0.0014 | 0.0196 | 0.1111 | 0.2221 | tta2 |
| cvc_clinicdb/swinunetr_2d | large | tta2 | hd95 | 21 | 204 | -0.1974 | -2.8286 | 2.5462 | 0.9729 | 0.9729 | tta2 |
| cvc_clinicdb/swinunetr_2d | large | tta4 | dice | 21 | 204 | 0.0041 | -0.0075 | 0.0147 | 0.3554 | 0.7108 | tta4 |
| cvc_clinicdb/swinunetr_2d | large | tta4 | hd95 | 21 | 204 | -2.7226 | -7.1399 | 1.3064 | 0.3926 | 0.7108 | tta4 |
| cvc_clinicdb/swinunetr_2d | large | tta8 | dice | 21 | 204 | 0.0025 | -0.0108 | 0.0171 | 0.8917 | 0.8917 | tta8 |
| cvc_clinicdb/swinunetr_2d | large | tta8 | hd95 | 21 | 204 | -3.4026 | -8.3211 | 1.1626 | 0.3038 | 0.6076 | tta8 |

The same paired comparison as the main tables, computed inside each bin: a negative HD95 delta means the treatment is better there.

## Missed-detection transitions per bin (exact McNemar, Holm across bins x methods)

| model | size_bin | control | treatment | unit | n_units | both_nonempty | new_misses | recovered | both_empty | net_change | mcnemar_p | mcnemar_p_adjusted | correction | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvc_clinicdb/unet_resnet50 | small | baseline | tta2 | case | 204 | 193 | 5 | 0 | 6 | -5 | 0.0625 | 0.5000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | small | baseline | tta4 | case | 204 | 192 | 6 | 0 | 6 | -6 | 0.0312 | 0.2812 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | small | baseline | tta8 | case | 204 | 193 | 5 | 0 | 6 | -5 | 0.0625 | 0.5000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | medium | baseline | tta2 | case | 204 | 198 | 0 | 0 | 6 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | medium | baseline | tta4 | case | 204 | 198 | 0 | 0 | 6 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | medium | baseline | tta8 | case | 204 | 197 | 1 | 0 | 6 | -1 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | large | baseline | tta2 | case | 204 | 201 | 0 | 2 | 1 | 2 | 0.5000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | large | baseline | tta4 | case | 204 | 201 | 0 | 2 | 1 | 2 | 0.5000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | large | baseline | tta8 | case | 204 | 201 | 0 | 2 | 1 | 2 | 0.5000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | small | baseline | tta2 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | small | baseline | tta4 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | small | baseline | tta8 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | medium | baseline | tta2 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | medium | baseline | tta4 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | medium | baseline | tta8 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | large | baseline | tta2 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | large | baseline | tta4 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | large | baseline | tta8 | case | 204 | 204 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | holm across 9 bin x method tests | 4923.6667 | 11247.3333 |

`new_misses` counts cases the control detected and the treatment missed, `recovered` the opposite, and `mcnemar_p_adjusted` is Holm-corrected over every bin x method test shown (`correction` states the family). A treatment can gain mean Dice and still create new misses on small lesions - that is what this table is for.

## Interaction: does the TTA gain differ across size bins?

| model | treatment | metric | control | n_small | n_medium | n_large | delta_small | delta_medium | delta_large | delta_gap | gap_ci_low | gap_ci_high | kruskal_p | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvc_clinicdb/unet_resnet50 | tta2 | dice | baseline | 204 | 204 | 204 | 0.0110 | 0.0144 | 0.0182 | 0.0072 | -0.0056 | 0.0240 | 0.7266 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | dice | baseline | 204 | 204 | 204 | 0.0241 | 0.0178 | 0.0291 | 0.0050 | -0.0110 | 0.0256 | 0.2162 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | dice | baseline | 204 | 204 | 204 | 0.0290 | 0.0258 | 0.0304 | 0.0014 | -0.0163 | 0.0254 | 0.6490 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta2 | hd95 | baseline | 198 | 199 | 204 | -5.3217 | -3.9028 | -4.5492 | 0.7725 | -5.2772 | 6.3782 | 0.5779 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta4 | hd95 | baseline | 195 | 199 | 204 | -14.0136 | -8.3402 | -8.8822 | 5.1314 | -4.6502 | 14.0882 | 0.1796 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | tta8 | hd95 | baseline | 196 | 198 | 204 | -13.4228 | -11.3685 | -8.5494 | 4.8734 | -3.2100 | 12.5757 | 0.3532 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | dice | baseline | 204 | 204 | 204 | 0.0115 | 0.0179 | 0.0094 | -0.0021 | -0.0140 | 0.0116 | 0.0155 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | dice | baseline | 204 | 204 | 204 | 0.0227 | 0.0231 | 0.0091 | -0.0136 | -0.0324 | 0.0069 | 0.0024 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | dice | baseline | 204 | 204 | 204 | 0.0235 | 0.0281 | 0.0111 | -0.0125 | -0.0374 | 0.0115 | 0.0042 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta2 | hd95 | baseline | 204 | 204 | 204 | -5.1236 | -3.8841 | 0.1494 | 5.2730 | 2.5480 | 7.9181 | 0.0000 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta4 | hd95 | baseline | 204 | 204 | 204 | -12.2749 | -8.0065 | -0.2250 | 12.0499 | 7.1498 | 16.8263 | 0.0000 | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | tta8 | hd95 | baseline | 204 | 204 | 204 | -15.2542 | -10.3940 | -1.3437 | 13.9105 | 7.3022 | 20.7576 | 0.0000 | 4923.6667 | 11247.3333 |

`kruskal_p` tests the per-case paired deltas across the three bins; `delta_gap` is `large - small` with a bootstrap CI over groups. A CI excluding 0 means the TTA effect genuinely depends on lesion size.

## Empty-reference subgroup

| model | subgroup | method | control | n_cases | n_videos | n_false_positive | false_positive_rate | ci_low | ci_high | unit | n | both_correct | correct_to_fp | fp_to_correct | both_fp | mcnemar_p | mcnemar_p_adjusted | transition_unit | pred_volume_mm3_mean | pred_volume_mm3_median | pred_volume_mm3_max | pred_volume_mm3_total | cutoff_low_px | cutoff_high_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvc_clinicdb/unet_resnet50 | empty_reference | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | empty_reference | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | empty_reference | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/unet_resnet50 | empty_reference | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | empty_reference | baseline | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 |  |  |  |  |  |  | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | empty_reference | tta2 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | empty_reference | tta4 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |
| cvc_clinicdb/swinunetr_2d | empty_reference | tta8 | baseline | 0 | 0 | 0 |  | 0.0000 | 0.0000 | cluster_bootstrap_group | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | group |  |  |  |  | 4923.6667 | 11247.3333 |

`n_cases = 0` for the polyp collections: every reference contains a polyp, so the subgroup does not exist there (it is populated for BraTS ET).
