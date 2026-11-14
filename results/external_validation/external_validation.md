# MELBA external polyp datasets - validation

All three collections are used read-only and never trained on. Mask values are reported as found, not assumed.

## External dataset validation

| dataset | n_cases | image_ext | mask_ext | height_min | height_median | height_max | width_min | width_median | width_max | distinct_sizes | mask_values | threshold_used | rgb_distinct | n_empty_masks | lesion_px_median | lesion_px_min | lesion_px_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hyperkvasir_cecum | 1009 | .jpgx1009 |  | 518 | 532 | 548 | 542 | 633 | 637 | 8x50 |  |  | True | 1009 | 0.0000 | 0 | 0 |

Read-only audit: every image is paired with exactly one mask by file stem (the loader raises otherwise), frames must be genuinely RGB (three identical channels are rejected), and masks are binarised with the threshold shown here (`> 0` for a 0/1 label map, `mask_threshold` otherwise). `lesion_px_*` is the reference polyp area in native pixels before the 352x352 resize.

## hyperkvasir_cecum - details

| field | value |
| --- | --- |
| label | HyperKvasir - normal cecum (polyp-free) |
| images_dir | C:\Users\subka\Documents\datasets\polyp\hyperkvasir_cecum |
| masks_dir | (none: references are empty) |
| references | empty |
| n_cases | 1009 |
| inference_size | [352, 352] |
| spacing_mm | [1.0, 1.0] |
| mask_values | [] |
| empty_masks | 000e3fee-7f5c-4819-9f9c-4c983b68888a, 001a41c2-2a5d-40b1-8fd5-b5f2f292277b, 006af0aa-2044-4477-964d-10d9e043fb78, 00832522-ab8e-4b98-bfce-93a777929571, 012ab888-64e6-4361-9745-f52b4a03ba75, 014a6880-6306-47ab-8766-adc17507b7c3, 019975d6-14e2-4a71-9e35-934f2280fab6, 01b2ec26-e814-4053-a820-37c9b703d47a, 020945d2-ade3-4cf9-b15d-634729b2b4a2, 0225440c-73b0-4c5e-b4b4-5bfe306834c9, 022ade3a-d78e-49b6-a447-609bdc8da113, 02387cd3-612f-4358-a4d8-4f3116fb3db0, 026d749a-7017-4a08-8920-8ed44a168f7e, 02952c26-912d-447b-a6df-6dd0b15d2261, 02bd3e73-adac-40e7-a80a-16d8567b8929, 03668dee-c9d1-439f-828a-e9a77583a9a9, 03b81c36-0b59-4220-bc00-00cdee1173c2, 03dcec22-f59a-465e-b438-37e9949450fd, 042b5605-43b8-435e-85bc-0e21e4ced465, 045a9a29-7b66-4ec0-b042-4c527d880d29, 04827eb8-7b02-4e85-baa2-e55a6191f542, 0484bad7-8e24-47be-9a16-86e861c67eee, 04a2700f-b444-4b71-a956-33186e665b3e, 04b6e35b-5ece-4af3-ad0f-a09a6fdb62dc, 04dbbb77-4cd0-4a28-9260-256b95d02bc8, 052b1a76-5694-4542-8903-4f0209a6203c, 0547ea29-b470-4f81-988f-af778254f03a, 055f25cc-a2cd-48dd-9900-958f5d794e0b, 05d44203-878c-4655-ac4c-d90e890bdc0a, 05fc4f81-db11-4af5-87da-f9fc2f752842, 05fc635d-8423-4727-a3f4-367f753ba279, 06158368-8187-47d6-a724-c2d8d1a4c4e0, 068f7144-850a-4ad4-8623-2ab40e16b7c0, 06a4e549-df04-428f-a370-aa6bce87e9eb, 06dc4006-6b03-46b0-aaf6-0af47f27c68a, 06fff1c3-0d03-43c0-8850-4c6146198cb9, 071f886a-3e66-4377-bd19-b55832bbc23f, 0724d289-be4d-4ee7-b39a-9105df9b1686, 073a859a-f7e6-48ba-bc11-25bcd313c323, 077b7f3e-3a51-43e1-aa27-b414c99ce5eb, 07d5256f-0e67-4df0-a7aa-3d7116c4960b, 085623f8-d2fc-47c9-8d92-7b370f23386f, 08e91eb3-1cfd-4ef5-88f6-672ab70acb96, 0901cb71-3f85-4d4a-b5e4-67f7b614f692, 0905dab6-8295-4e9d-9e85-8ea6269ff3b9, 090d30fe-473f-4c13-9acf-d674dcfbd092, 090df806-3eaf-4621-b5ed-518323123cc7, 0924f36a-430c-4c51-ae3b-a2d970391210, 097df7d3-1ce8-4512-9b2f-d39bcce8ca5a, 099db767-3910-4e5f-bdc2-0127011b6de1 |
| lesion_px min/p25/median/p75/max | [0, 0.0, 0.0, 0.0, 0] |
