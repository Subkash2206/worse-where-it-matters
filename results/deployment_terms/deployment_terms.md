# Deployment terms per 1000 frames / patients

Positive *fixed* and *removed* columns are benefits, positive *misses added* is a cost.  The false positives are **raw counts on the 33 lesion-free BraTS patients** (only BraTS has empty references), not a rate.  The latency column is the *extra* compute of the additional forward passes, and it is **estimated** from the measured per-pass latency - see the last column and the note below.

| dataset | unit | method | boundary failures fixed | misses added | false positives on lesion-free | false positives removed | extra latency | latency estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats | patients | tta4 | +13.6 | +2.5 | 20 of 33 | +2 | 5.25 s per volume | yes |
| brats | patients | tta8 | +16.0 | +0.8 | 21 of 33 | +1 | 12.25 s per volume | yes |
| cvc_colondb | frames | tta4 |  | +21.1 |  |  | 135 ms per frame | yes |
| cvc_colondb | frames | tta8 |  | +34.2 |  |  | 315 ms per frame | yes |
| cvc_swin | frames | tta4 | +47.4 | +0.0 |  |  | 135 ms per frame | yes |
| cvc_swin | frames | tta8 | +65.4 | +0.0 |  |  | 315 ms per frame | yes |
| cvc_unet | frames | tta4 | +99.7 | +6.5 |  |  | 135 ms per frame | yes |
| cvc_unet | frames | tta8 | +111.1 | +6.5 |  |  | 315 ms per frame | yes |
| etis_larib | frames | tta4 |  | +56.1 |  |  | 135 ms per frame | yes |
| etis_larib | frames | tta8 |  | +66.3 |  |  | 315 ms per frame | yes |
| kvasir_seg | frames | tta4 |  | +0.0 |  |  | 135 ms per frame | yes |
| kvasir_seg | frames | tta8 |  | +0.0 |  |  | 315 ms per frame | yes |

Latency is estimated, not measured end to end: the extra passes of the method times the measured per-pass latency (45 ms per pass (2-D, 352x352); 1750 ms per pass (3-D SegResNet 128^3)). The 3-D value is reported in s per volume, the 2-D values in ms per frame.
