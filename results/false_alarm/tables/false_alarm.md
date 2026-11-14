# MELBA - false-alarm cost on polyp-free colonoscopy (hyperkvasir_cecum)

Every frame of this cohort is polyp-free, so any non-empty prediction is a false positive: the tables below are the price of TTA and of the detection-safe rules, not a segmentation score. Dice and HD95 are undefined here and are not reported.

## unet: false alarms on hyperkvasir_cecum

| method | n_cases | n_checkpoints | false_alarm_rate_mean | false_alarm_rate_min | false_alarm_rate_max | false_alarm_rate_majority | n_majority_nonempty | n_any_checkpoint_nonempty | n_false_positives | median_false_area_px | median_false_area_px_per_checkpoint | mean_max_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1009 | 15 | 0.7082 | 0.4817 | 0.8979 | 0.7721 | 779 | 1007 | 10718 | 10946.5000 | 10527.0000 | 0.7438 |
| tta2 | 1009 | 15 | 0.6747 | 0.4926 | 0.8662 | 0.7156 | 722 | 999 | 10212 | 9911.0000 | 7592.0000 | 0.6823 |
| tta4 | 1009 | 15 | 0.6131 | 0.4034 | 0.8563 | 0.6244 | 630 | 984 | 9280 | 9183.0000 | 7609.0000 | 0.6289 |
| tta8 | 1009 | 15 | 0.5654 | 0.3508 | 0.8315 | 0.5709 | 576 | 958 | 8558 | 9658.0000 | 8003.0000 | 0.5949 |
| tta4_anyview | 1009 | 15 | 0.8923 | 0.6997 | 0.9871 | 0.9673 | 976 | 1009 | 13505 | 6250.0000 | 6512.0000 | 0.6289 |
| tta4_lowdet_20 | 1009 | 15 | 0.9226 | 0.7156 | 1.0000 | 0.9980 | 1007 | 1009 | 13964 | 7490.0000 | 7609.0000 | 0.6289 |
| tta4_lowdet_30 | 1009 | 15 | 0.8026 | 0.5045 | 1.0000 | 0.8850 | 893 | 1009 | 12148 | 6176.0000 | 5691.0000 | 0.6289 |
| tta4_lowdet_40 | 1009 | 15 | 0.7109 | 0.4609 | 0.9306 | 0.7473 | 754 | 1004 | 10760 | 6898.0000 | 5984.0000 | 0.6289 |
| tta8_anyview | 1009 | 15 | 0.9448 | 0.7899 | 0.9941 | 0.9941 | 1003 | 1009 | 14300 | 7147.0000 | 7207.0000 | 0.5949 |
| tta8_lowdet_20 | 1009 | 15 | 0.8946 | 0.5996 | 1.0000 | 0.9941 | 1003 | 1009 | 13540 | 8071.0000 | 8085.0000 | 0.5949 |
| tta8_lowdet_30 | 1009 | 15 | 0.7849 | 0.4985 | 1.0000 | 0.8573 | 865 | 1009 | 11880 | 6251.0000 | 5333.5000 | 0.5949 |
| tta8_lowdet_40 | 1009 | 15 | 0.6644 | 0.4163 | 0.8999 | 0.6868 | 693 | 1000 | 10056 | 7239.0000 | 5583.0000 | 0.5949 |

`false_alarm_rate_mean` is the mean over the checkpoints of the share of frames with a non-empty prediction; `..._majority` counts a frame when a majority of the checkpoints reports a polyp; `median_false_area_px` is the median area of those (false) predictions in native pixels.

## unet: exact McNemar vs baseline and TTA8

| unit | method | control | n_units | new_false_alarms | avoided_false_alarms | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| case | tta2 | baseline | 1009 | 13 | 70 | 0.0000 |
| case | tta4 | baseline | 1009 | 4 | 153 | 0.0000 |
| case | tta8 | baseline | 1009 | 3 | 206 | 0.0000 |
| case | tta4_anyview | baseline | 1009 | 197 | 0 | 0.0000 |
| case | tta4_lowdet_20 | baseline | 1009 | 228 | 0 | 0.0000 |
| case | tta4_lowdet_30 | baseline | 1009 | 123 | 9 | 0.0000 |
| case | tta4_lowdet_40 | baseline | 1009 | 33 | 58 | 0.0115 |
| case | tta8_anyview | baseline | 1009 | 224 | 0 | 0.0000 |
| case | tta8_lowdet_20 | baseline | 1009 | 224 | 0 | 0.0000 |
| case | tta8_lowdet_30 | baseline | 1009 | 104 | 18 | 0.0000 |
| case | tta8_lowdet_40 | baseline | 1009 | 15 | 101 | 0.0000 |
| case | baseline | tta8 | 1009 | 206 | 3 | 0.0000 |
| case | tta2 | tta8 | 1009 | 153 | 7 | 0.0000 |
| case | tta4 | tta8 | 1009 | 67 | 13 | 0.0000 |
| case | tta4_anyview | tta8 | 1009 | 400 | 0 | 0.0000 |
| case | tta4_lowdet_20 | tta8 | 1009 | 431 | 0 | 0.0000 |
| case | tta4_lowdet_30 | tta8 | 1009 | 317 | 0 | 0.0000 |
| case | tta4_lowdet_40 | tta8 | 1009 | 179 | 1 | 0.0000 |
| case | tta8_anyview | tta8 | 1009 | 427 | 0 | 0.0000 |
| case | tta8_lowdet_20 | tta8 | 1009 | 427 | 0 | 0.0000 |
| case | tta8_lowdet_30 | tta8 | 1009 | 289 | 0 | 0.0000 |
| case | tta8_lowdet_40 | tta8 | 1009 | 117 | 0 | 0.0000 |
| checkpoint_case | tta2 | baseline | 15135 | 741 | 1247 | 0.0000 |
| checkpoint_case | tta4 | baseline | 15135 | 636 | 2074 | 0.0000 |
| checkpoint_case | tta8 | baseline | 15135 | 512 | 2672 | 0.0000 |
| checkpoint_case | tta4_anyview | baseline | 15135 | 2787 | 0 | 0.0000 |
| checkpoint_case | tta4_lowdet_20 | baseline | 15135 | 3286 | 40 | 0.0000 |
| checkpoint_case | tta4_lowdet_30 | baseline | 15135 | 1930 | 500 | 0.0000 |
| checkpoint_case | tta4_lowdet_40 | baseline | 15135 | 1128 | 1086 | 0.3836 |
| checkpoint_case | tta8_anyview | baseline | 15135 | 3582 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_20 | baseline | 15135 | 3021 | 199 | 0.0000 |
| checkpoint_case | tta8_lowdet_30 | baseline | 15135 | 1859 | 697 | 0.0000 |
| checkpoint_case | tta8_lowdet_40 | baseline | 15135 | 940 | 1602 | 0.0000 |
| checkpoint_case | baseline | tta8 | 15135 | 2672 | 512 | 0.0000 |
| checkpoint_case | tta2 | tta8 | 15135 | 2113 | 459 | 0.0000 |
| checkpoint_case | tta4 | tta8 | 15135 | 1162 | 440 | 0.0000 |
| checkpoint_case | tta4_anyview | tta8 | 15135 | 4951 | 4 | 0.0000 |
| checkpoint_case | tta4_lowdet_20 | tta8 | 15135 | 5412 | 6 | 0.0000 |
| checkpoint_case | tta4_lowdet_30 | tta8 | 15135 | 3624 | 34 | 0.0000 |
| checkpoint_case | tta4_lowdet_40 | tta8 | 15135 | 2322 | 120 | 0.0000 |
| checkpoint_case | tta8_anyview | tta8 | 15135 | 5742 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_20 | tta8 | 15135 | 4982 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_30 | tta8 | 15135 | 3322 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_40 | tta8 | 15135 | 1498 | 0 | 0.0000 |

## segformer: false alarms on hyperkvasir_cecum

| method | n_cases | n_checkpoints | false_alarm_rate_mean | false_alarm_rate_min | false_alarm_rate_max | false_alarm_rate_majority | n_majority_nonempty | n_any_checkpoint_nonempty | n_false_positives | median_false_area_px | median_false_area_px_per_checkpoint | mean_max_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1009 | 15 | 0.9640 | 0.8801 | 0.9950 | 0.9980 | 1007 | 1009 | 14590 | 7194.5000 | 8103.5000 | 0.9384 |
| tta2 | 1009 | 15 | 0.9495 | 0.8315 | 0.9970 | 0.9931 | 1002 | 1009 | 14370 | 7091.0000 | 7691.0000 | 0.9038 |
| tta4 | 1009 | 15 | 0.9231 | 0.7582 | 0.9941 | 0.9782 | 987 | 1009 | 13971 | 6740.0000 | 6873.0000 | 0.8715 |
| tta8 | 1009 | 15 | 0.8993 | 0.7314 | 0.9891 | 0.9564 | 965 | 1009 | 13611 | 6229.0000 | 6457.0000 | 0.8426 |
| tta4_anyview | 1009 | 15 | 0.9966 | 0.9841 | 1.0000 | 1.0000 | 1009 | 1009 | 15083 | 5971.0000 | 6425.0000 | 0.8715 |
| tta4_lowdet_20 | 1009 | 15 | 0.9973 | 0.9881 | 1.0000 | 1.0000 | 1009 | 1009 | 15094 | 6085.0000 | 6484.0000 | 0.8715 |
| tta4_lowdet_30 | 1009 | 15 | 0.9840 | 0.9336 | 1.0000 | 1.0000 | 1009 | 1009 | 14893 | 6050.0000 | 6495.0000 | 0.8715 |
| tta4_lowdet_40 | 1009 | 15 | 0.9591 | 0.8573 | 0.9990 | 0.9960 | 1005 | 1009 | 14516 | 6304.5000 | 6514.0000 | 0.8715 |
| tta8_anyview | 1009 | 15 | 0.9997 | 0.9970 | 1.0000 | 1.0000 | 1009 | 1009 | 15130 | 5436.0000 | 5907.0000 | 0.8426 |
| tta8_lowdet_20 | 1009 | 15 | 0.9965 | 0.9822 | 1.0000 | 1.0000 | 1009 | 1009 | 15082 | 5523.5000 | 5911.0000 | 0.8426 |
| tta8_lowdet_30 | 1009 | 15 | 0.9773 | 0.9098 | 1.0000 | 0.9980 | 1007 | 1009 | 14792 | 5432.0000 | 5805.0000 | 0.8426 |
| tta8_lowdet_40 | 1009 | 15 | 0.9446 | 0.8137 | 0.9990 | 0.9891 | 998 | 1009 | 14297 | 5733.0000 | 5959.5000 | 0.8426 |

`false_alarm_rate_mean` is the mean over the checkpoints of the share of frames with a non-empty prediction; `..._majority` counts a frame when a majority of the checkpoints reports a polyp; `median_false_area_px` is the median area of those (false) predictions in native pixels.

## segformer: exact McNemar vs baseline and TTA8

| unit | method | control | n_units | new_false_alarms | avoided_false_alarms | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- |
| case | tta2 | baseline | 1009 | 0 | 5 | 0.0625 |
| case | tta4 | baseline | 1009 | 0 | 20 | 0.0000 |
| case | tta8 | baseline | 1009 | 0 | 42 | 0.0000 |
| case | tta4_anyview | baseline | 1009 | 2 | 0 | 0.5000 |
| case | tta4_lowdet_20 | baseline | 1009 | 2 | 0 | 0.5000 |
| case | tta4_lowdet_30 | baseline | 1009 | 2 | 0 | 0.5000 |
| case | tta4_lowdet_40 | baseline | 1009 | 2 | 4 | 0.6875 |
| case | tta8_anyview | baseline | 1009 | 2 | 0 | 0.5000 |
| case | tta8_lowdet_20 | baseline | 1009 | 2 | 0 | 0.5000 |
| case | tta8_lowdet_30 | baseline | 1009 | 2 | 2 | 1.0000 |
| case | tta8_lowdet_40 | baseline | 1009 | 0 | 9 | 0.0039 |
| case | baseline | tta8 | 1009 | 42 | 0 | 0.0000 |
| case | tta2 | tta8 | 1009 | 37 | 0 | 0.0000 |
| case | tta4 | tta8 | 1009 | 23 | 1 | 0.0000 |
| case | tta4_anyview | tta8 | 1009 | 44 | 0 | 0.0000 |
| case | tta4_lowdet_20 | tta8 | 1009 | 44 | 0 | 0.0000 |
| case | tta4_lowdet_30 | tta8 | 1009 | 44 | 0 | 0.0000 |
| case | tta4_lowdet_40 | tta8 | 1009 | 40 | 0 | 0.0000 |
| case | tta8_anyview | tta8 | 1009 | 44 | 0 | 0.0000 |
| case | tta8_lowdet_20 | tta8 | 1009 | 44 | 0 | 0.0000 |
| case | tta8_lowdet_30 | tta8 | 1009 | 42 | 0 | 0.0000 |
| case | tta8_lowdet_40 | tta8 | 1009 | 33 | 0 | 0.0000 |
| checkpoint_case | tta2 | baseline | 15135 | 172 | 392 | 0.0000 |
| checkpoint_case | tta4 | baseline | 15135 | 168 | 787 | 0.0000 |
| checkpoint_case | tta8 | baseline | 15135 | 143 | 1122 | 0.0000 |
| checkpoint_case | tta4_anyview | baseline | 15135 | 493 | 0 | 0.0000 |
| checkpoint_case | tta4_lowdet_20 | baseline | 15135 | 508 | 4 | 0.0000 |
| checkpoint_case | tta4_lowdet_30 | baseline | 15135 | 401 | 98 | 0.0000 |
| checkpoint_case | tta4_lowdet_40 | baseline | 15135 | 268 | 342 | 0.0031 |
| checkpoint_case | tta8_anyview | baseline | 15135 | 540 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_20 | baseline | 15135 | 510 | 18 | 0.0000 |
| checkpoint_case | tta8_lowdet_30 | baseline | 15135 | 374 | 172 | 0.0000 |
| checkpoint_case | tta8_lowdet_40 | baseline | 15135 | 245 | 538 | 0.0000 |
| checkpoint_case | baseline | tta8 | 15135 | 1122 | 143 | 0.0000 |
| checkpoint_case | tta2 | tta8 | 15135 | 912 | 153 | 0.0000 |
| checkpoint_case | tta4 | tta8 | 15135 | 525 | 165 | 0.0000 |
| checkpoint_case | tta4_anyview | tta8 | 15135 | 1472 | 0 | 0.0000 |
| checkpoint_case | tta4_lowdet_20 | tta8 | 15135 | 1483 | 0 | 0.0000 |
| checkpoint_case | tta4_lowdet_30 | tta8 | 15135 | 1286 | 4 | 0.0000 |
| checkpoint_case | tta4_lowdet_40 | tta8 | 15135 | 940 | 35 | 0.0000 |
| checkpoint_case | tta8_anyview | tta8 | 15135 | 1519 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_20 | tta8 | 15135 | 1471 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_30 | tta8 | 15135 | 1181 | 0 | 0.0000 |
| checkpoint_case | tta8_lowdet_40 | tta8 | 15135 | 686 | 0 | 0.0000 |
