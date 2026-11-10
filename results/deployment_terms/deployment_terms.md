# Deployment terms per 1000 frames / patients

Positive *fixed* and *removed* columns are benefits, positive *misses added* is a cost; the latency column is the extra compute of the additional forward passes, not the total.

| dataset | unit | method | boundary failures fixed | misses added | false positives removed | extra ms per frame |
| --- | --- | --- | --- | --- | --- | --- |
| brats | patients | tta4 | +13.6 | +2.5 | +60.6 | 5250 |
| brats | patients | tta8 | +16.0 | +0.8 | +30.3 | 12250 |
| cvc_colondb | frames | tta4 |  | +21.1 |  | 135 |
| cvc_colondb | frames | tta8 |  | +34.2 |  | 315 |
| cvc_swin | frames | tta4 | +47.4 | +0.0 |  | 135 |
| cvc_swin | frames | tta8 | +65.4 | +0.0 |  | 315 |
| cvc_unet | frames | tta4 | +99.7 | +6.5 |  | 135 |
| cvc_unet | frames | tta8 | +111.1 | +6.5 |  | 315 |
| etis_larib | frames | tta4 |  | +56.1 |  | 135 |
| etis_larib | frames | tta8 |  | +66.3 |  | 315 |
| kvasir_seg | frames | tta4 |  | +0.0 |  | 135 |
| kvasir_seg | frames | tta8 |  | +0.0 |  | 315 |

Latency: 45 ms per pass (2-D, 352x352); 1750 ms per pass (3-D SegResNet 128^3).
