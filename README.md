# Better on Average, Not Always Where It Matters

Code, configurations, split files and result tables for

> **Better on Average, Not Always Where It Matters: A Reliability Audit of Geometric Test-Time Augmentation in Medical Image Segmentation.**
> Submitted to MIDL 2027 (Clinical Translation and Validation track).

Test-time augmentation (TTA) raises mean Dice cheaply, but means can hide missed lesions. This repository treats geometric TTA as a pre-deployment validation question. It evaluates TTA with 1, 2, 4 and 8 dihedral (D4) views on:

- three 2D polyp segmentation models (UNet with a ResNet-50 encoder, SegFormer MiT-B2, SwinUNETR trained from scratch), trained on CVC-ClinicDB with sequence-level 5-fold cross-validation and 3 seeds (15 checkpoints per model);
- a 3D SegResNet for enhancing tumour on BraTS 2021 (patient-level 5-fold cross-validation, 1 seed per fold).

The 2D models are tested without retraining on Kvasir-SEG, CVC-ColonDB and ETIS-Larib, and on 1009 polyp-free HyperKvasir cecum frames. Besides Dice, NSD and HD95, the study reports missed lesions, false alarms and error tails, audits uncertainty signals, compares against MC-Dropout and deep ensembles, and tests two detection-safe rules (any-view and a low threshold) that reuse the TTA views.

Main finding: TTA with 4 or 8 views improved Dice, NSD and HD95 for every model on every dataset with lesions, but under domain shift it also changed which lesions were found. For the weakest model (SwinUNETR), images on which most checkpoints had no overlap with the polyp rose on all three external sets (ETIS-Larib: 50 to 71 of 196) while Dice rose. For UNet, more images were left empty (ETIS-Larib: 5 to 18), but most of them had no overlap with the polyp before TTA, and the number of missed images did not change in net. The detection rules cut UNet's missed images below the single-view count at a measured cost in false alarms on polyp-free frames.

## Repository layout

```
configs/          experiment configs (YAML, with `defaults:` inheritance from configs/base.yaml)
  cvc_clinicdb/   2D models, MC-Dropout, data sweep, detection-rule and timing variants
  brats2021/      3D SegResNet (the paper uses segresnet_reduced.yaml)
  external/       external polyp sets and polyp-free HyperKvasir cecum frames
data/splits/      fold files, frame-to-sequence map, normalisation stats (committed)
docs/splits.md    why splits are by sequence/patient and how the split files are built
queues/           YAML job queues that chain grids, evaluations and reports
scripts/          entry points (training grid, evaluation, statistics, figures)
src/melba/        library code (data, models, TTA, metrics, statistics)
tests/            pytest suite
results/          every table and figure behind the paper (CSV / JSON / Markdown / PDF / PNG)
legacy_results/   tables from an earlier code base; not used in the paper (see Notes)
```

## Installation

Tested with Python 3.13.7, PyTorch 2.6.0 (CUDA 12.4, cuDNN 9.1), MONAI 1.5.2 and segmentation-models-pytorch 0.5.0 on Windows 11 with one 6 GB RTX 4050 laptop GPU.

```bash
python -m pip install -r requirements.txt   # pinned versions
python -m pip install -e .                  # installs the `melba` package from src/
python -m pytest                            # optional: run the test suite
```

`requirements.txt` pins `torch` without a CUDA tag; install the CUDA build that matches your driver if you want GPU training.

## Data

No images are distributed here. Download the datasets from their official sources:

| Dataset | Role | Where the config expects it |
|---|---|---|
| CVC-ClinicDB (612 frames, 29 sequences) | 2D training and internal test | `cvc_clinicdb/original` and `cvc_clinicdb/ground_truth`, relative to the repo root |
| BraTS 2021 training set (1251 patients) | 3D training and internal test | `dataset.root` in `configs/brats2021/segresnet.yaml` |
| Kvasir-SEG (1000) | external test | `configs/external/polyp_external.yaml` |
| CVC-ColonDB (380), ETIS-Larib (196) | external test | `configs/external/polyp_external.yaml` |
| HyperKvasir, cecum class (1009 polyp-free frames) | false alarms | `configs/external/hyperkvasir_cecum.yaml` |

**The BraTS and external configs contain absolute Windows paths from the original machine (`C:/Users/...`). Edit them to point at your copies before running anything.**

The split files in `data/splits/` are the exact ones used in the paper, so you do not need to regenerate them. To rebuild them and check them against your copy of the data:

```bash
python scripts/make_cvc_mapping.py   --config configs/cvc_clinicdb/unet_resnet50.yaml
python scripts/make_splits.py        --config configs/cvc_clinicdb/unet_resnet50.yaml
python scripts/compute_normalization.py --config configs/cvc_clinicdb/swinunetr_2d.yaml
python scripts/make_brats_splits.py  --config configs/brats2021/segresnet.yaml
python scripts/preprocess_brats.py   --config configs/brats2021/segresnet.yaml   # builds cache/brats2021_pre (large)
python scripts/validate_brats.py     --config configs/brats2021/segresnet.yaml
python scripts/validate_external.py  --config configs/external/polyp_external.yaml
```

## Reproducing the paper

### 1. Train and evaluate the internal grids

`run_grid.py` runs every (fold, seed) in its own process, resumes from `last.pt`, and skips runs that are already trained and evaluated.

```bash
# 2D models: 5 folds x 3 seeds
python scripts/run_grid.py --config configs/cvc_clinicdb/unet_resnet50.yaml    --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20
python scripts/run_grid.py --config configs/cvc_clinicdb/segformer_mit_b2.yaml --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20
python scripts/run_grid.py --config configs/cvc_clinicdb/swinunetr_2d.yaml     --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 100 --patience 20

# 3D SegResNet: 5 folds x 1 seed
python scripts/run_grid.py --config configs/brats2021/segresnet_reduced.yaml   --folds 0 1 2 3 4 --seeds 0 --epochs 100 --patience 20

# statistics per model (Wilcoxon + Holm, bootstrap CIs, misses, tails, uncertainty audit)
python scripts/aggregate.py --config configs/cvc_clinicdb/unet_resnet50.yaml
```

Checkpoints go to `checkpoints/<experiment>/`, per-run evaluations to `results/<experiment>/`, aggregate statistics to `results/aggregate_<experiment>/`.

### 2. Everything else, via queues

The remaining experiments are chained in `queues/` and run with

```bash
python scripts/run_queue.py --queue queues/<name>.yaml --dry-run   # show the steps
python scripts/run_queue.py --queue queues/<name>.yaml
python scripts/status.py                                           # progress of running grids
```

| Queue | What it produces |
|---|---|
| `overnight.yaml` | MC-Dropout grid, 25 % / 50 % data sweep, their aggregates, UNet vs SwinUNETR comparison |
| `external_finish.yaml`, `external_swin.yaml` | external evaluation on Kvasir-SEG, CVC-ColonDB and ETIS-Larib |
| `brats_finish.yaml` | BraTS aggregate and lesion-size subgroups |
| `paper_rules_segformer.yaml` | detection-rule runs (internal, external, BraTS), rule selection, rule report, SegFormer grid and per-model external tables |
| `hyperkvasir_segformer_rules.yaml` | false alarms on polyp-free cecum frames, SegFormer detection rules |

`external_recovery.yaml` and `intensity_queue.yaml` also run intensity-TTA steps, which are not part of the paper (see Notes).

The individual report scripts can also be called directly:

```bash
python scripts/evaluate_external.py --config configs/external/polyp_external.yaml \
    --models cvc_clinicdb/unet_resnet50 cvc_clinicdb/segformer_mit_b2 cvc_clinicdb/swinunetr_2d --folds 0 1 2 3 4 --seeds 0 1 2
python scripts/external_by_model.py
python scripts/compare_models.py --configs configs/cvc_clinicdb/unet_resnet50.yaml configs/cvc_clinicdb/swinunetr_2d.yaml
python scripts/compare_data_sweep.py
python scripts/summarize_mcdropout.py --config configs/cvc_clinicdb/unet_resnet50_dropout.yaml
python scripts/subgroup_analysis.py
python scripts/select_detection_rule.py
python scripts/detection_rules_report.py --external-root results/detrules --output-dir results/detection_rules
python scripts/false_alarm_report.py --run unet=results/false_alarm/unet --run segformer=results/false_alarm/segformer --output-dir results/false_alarm/tables
python scripts/measure_pass_latency.py
python scripts/deployment_terms.py
python scripts/paper_checks/brats_et_only.py   # BraTS tail and uncertainty numbers on the 1218 patients with ET
python scripts/paper_checks/overlap_misses.py  # empty predictions vs lesions with no overlap, all models and datasets (needs the per-case files)
python scripts/paper_checks/single_view_rule.py # low-threshold rule on the single view vs on TTA-8 (UNet, external; needs the per-case files)
```

Use `--help` on any script for the full argument list.

### 3. Figures

```bash
python scripts/figures/fig01_gain_vs_views.py
# ... fig02 to fig08, and sanity_false_alarms.py
```

Each script reads the committed tables in `results/` and writes a PDF and a 300 dpi PNG to `results/figures_paper/`. The figures can be regenerated without retraining anything.

## Where the paper's numbers live

| Paper content | Directory under `results/` |
|---|---|
| Internal results per model (Dice, NSD, HD95, misses, tails, uncertainty audit) | `aggregate_cvc_clinicdb_*`, `aggregate_brats2021_segresnet_reduced` |
| BraTS numbers restricted to patients with ET; missed lesions counted by overlap (Dice = 0) | `paper_checks` |
| Model comparisons and gain vs model quality | `compare_models` |
| Data-fraction sweep | `compare_data_sweep` |
| MC-Dropout | `mc_dropout_cvc_clinicdb_unet_resnet50_dropout`, `aggregate_cvc_clinicdb_unet_resnet50_dropout` |
| Deep ensembles | `ensemble_*.csv` inside each `aggregate_cvc_clinicdb_*` |
| External validation | `external_validation`, `external_by_model`, `external_kvasir_seg`, `external_cvc_colondb`, `external_etis_larib` |
| Detection-safe rules | `detection_rules`, `detection_rules_segformer`, `detrules`, `detrules_segformer` |
| Low-threshold rule on the single view (control; configs `*_baserules.yaml`) | `baserules`, `false_alarm_baserules`, `paper_checks` |
| False alarms on polyp-free / lesion-free data | `false_alarm`, `external_hyperkvasir_cecum` |
| Lesion-size subgroups | `subgroups`, `subgroups_brats2021` |
| Deployment cost | `deployment_terms` |
| Figures | `figures_paper` |

Per-case CSVs, probability maps and checkpoints are not committed (size); they are regenerated by the commands above.

## Notes

- **`legacy_results/`** holds tables from an earlier code base that split CVC-ClinicDB by frame rather than by sequence, which leaks near-duplicate frames between training and test. They are kept for reference only and are not used anywhere in the paper.
- **Intensity TTA.** The `*_intensity*` configs and `queues/intensity_queue.yaml` belong to an exploratory experiment that is not reported in the paper. Treat them as unsupported.
- **Determinism.** Runs use deterministic algorithms and fixed seeds, but exact numbers can still differ across GPUs, drivers and library versions.

