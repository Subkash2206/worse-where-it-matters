# Deployment terms per 1000 frames / patients

Positive *fixed* and *removed* columns are benefits, positive *misses added* is a cost.  The false positives are **raw counts on the 33 lesion-free BraTS patients** (only BraTS has empty references), not a rate.  The latency column is the **absolute** cost of one image/volume (forward passes x measured per-pass latency), and it is **estimated**, not a measured end-to-end wall clock - see the ``passes`` and ``estimated`` columns and the note below.

| dataset | model | unit | method | passes | boundary failures fixed | misses added | false positives on lesion-free | false positives removed | latency (absolute) | latency estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brats | segresnet | patients | baseline | 1 | +0.0 | +0.0 | 22 of 33 | +0 | 1.75 s per volume | yes |
| brats | segresnet | patients | tta4 | 4 | +13.6 | +2.5 | 20 of 33 | +2 | 7.00 s per volume | yes |
| brats | segresnet | patients | tta8 | 8 | +16.0 | +0.8 | 21 of 33 | +1 | 14.00 s per volume | yes |
| cvc_colondb | segformer_mit_b2 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_colondb | segformer_mit_b2 | frames | tta4 | 4 |  | +2.6 |  |  | 180 ms per frame | yes |
| cvc_colondb | segformer_mit_b2 | frames | tta8 | 8 |  | +2.6 |  |  | 360 ms per frame | yes |
| cvc_colondb | swinunetr_2d | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_colondb | swinunetr_2d | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| cvc_colondb | swinunetr_2d | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |
| cvc_colondb | unet_resnet50 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_colondb | unet_resnet50 | frames | tta4 | 4 |  | +21.1 |  |  | 180 ms per frame | yes |
| cvc_colondb | unet_resnet50 | frames | tta8 | 8 |  | +34.2 |  |  | 360 ms per frame | yes |
| cvc_segformer | segformer_mit_b2 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_segformer | segformer_mit_b2 | frames | tta4 | 4 | +21.2 | +0.0 |  |  | 180 ms per frame | yes |
| cvc_segformer | segformer_mit_b2 | frames | tta8 | 8 | +21.2 | +0.0 |  |  | 360 ms per frame | yes |
| cvc_swin | swinunetr_2d | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_swin | swinunetr_2d | frames | tta4 | 4 | +47.4 | +0.0 |  |  | 180 ms per frame | yes |
| cvc_swin | swinunetr_2d | frames | tta8 | 8 | +65.4 | +0.0 |  |  | 360 ms per frame | yes |
| cvc_unet | unet_resnet50 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| cvc_unet | unet_resnet50 | frames | tta4 | 4 | +99.7 | +6.5 |  |  | 180 ms per frame | yes |
| cvc_unet | unet_resnet50 | frames | tta8 | 8 | +111.1 | +6.5 |  |  | 360 ms per frame | yes |
| etis_larib | segformer_mit_b2 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| etis_larib | segformer_mit_b2 | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| etis_larib | segformer_mit_b2 | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |
| etis_larib | swinunetr_2d | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| etis_larib | swinunetr_2d | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| etis_larib | swinunetr_2d | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |
| etis_larib | unet_resnet50 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| etis_larib | unet_resnet50 | frames | tta4 | 4 |  | +56.1 |  |  | 180 ms per frame | yes |
| etis_larib | unet_resnet50 | frames | tta8 | 8 |  | +66.3 |  |  | 360 ms per frame | yes |
| kvasir_seg | segformer_mit_b2 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| kvasir_seg | segformer_mit_b2 | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| kvasir_seg | segformer_mit_b2 | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |
| kvasir_seg | swinunetr_2d | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| kvasir_seg | swinunetr_2d | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| kvasir_seg | swinunetr_2d | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |
| kvasir_seg | unet_resnet50 | frames | baseline | 1 | +0.0 | +0.0 |  |  | 45 ms per frame | yes |
| kvasir_seg | unet_resnet50 | frames | tta4 | 4 |  | +0.0 |  |  | 180 ms per frame | yes |
| kvasir_seg | unet_resnet50 | frames | tta8 | 8 |  | +0.0 |  |  | 360 ms per frame | yes |

Latency is the absolute per-image/per-volume cost (forward passes x per-pass latency) and is estimated from the measured per-pass constant (45 ms per pass (2-D, 352x352); 1750 ms per pass (3-D SegResNet 128^3); 45 ms per pass (2-D, 352x352), SegFormer; 45 ms per pass (2-D, 352x352), SwinUNETR), not measured end to end. The 3-D value is reported in s per volume, the 2-D values in ms per frame.
