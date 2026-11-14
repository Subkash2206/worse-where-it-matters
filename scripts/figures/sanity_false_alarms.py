"""Sanity check for the false alarms on polyp-free HyperKvasir cecum (CPU only).

Before any false-alarm number goes into a figure, this script proves that the number is
real and not a pipeline artefact:

1. it reads the per-case CSV of a finished ``results/false_alarm`` run and picks false alarms
   (``pred_empty == 0`` on a polyp-free frame) for one checkpoint - ``fold0_seed0`` of the
   UNet and of the SegFormer baseline;
2. it **recomputes** those frames on the CPU with that checkpoint, through the same
   ``predict_per_view -> resample -> aggregate`` path the evaluation used, and prints the
   recomputed area and peak probability next to the values the CSV stored;
3. it attributes every false region to what it actually covers - mucosa (anatomy), black
   border/vignette, the green endoscope overlay box, or the white overlay text - by pixel
   class, so a pipeline bug (mask drawn in a corner, off-by-a-transpose) would show up as a
   region sitting on the overlay instead of on tissue;
4. it prints the preprocessing chain of the maskless (``references: empty``) loader and of
   Kvasir-SEG, and asserts numerically that the two produce the *same* network input tensor
   for the same frame file.

The overlay figure is written to ``results/figures_paper/sanity_false_alarms.png``:
image + predicted-mask contour, 8 UNet frames and 4 SegFormer frames.

Usage
-----
    python scripts/figures/sanity_false_alarms.py            # CPU only, ~1 min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402

from melba.config import Config, load_config  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.data.normalization import NormalizationSpec  # noqa: E402
from melba.data.polyp_external import (  # noqa: E402
    ExternalPolypDataset,
    ExternalPolypItem,
)
from melba.eval.runner import resample_probability_map  # noqa: E402
from melba.inference import (  # noqa: E402
    aggregate_probabilities,
    methods_from_config,
    predict_per_view,
)
from melba.models import build_model, load_checkpoint  # noqa: E402

CECUM_CONFIG = _REPO_ROOT / "configs" / "external" / "hyperkvasir_cecum.yaml"
KVASIR_CONFIG = _REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
FALSE_ALARM = style.RESULTS / "false_alarm"
#: the checkpoint both runs use for the sanity check
FOLD = 0
CHECKPOINT_SEED = 0
#: seed of the frame draw, so the figure is reproducible
FRAME_SEED = 2024
#: detected teal of the endoscope overlay box (RGB), measured on the frames
TEAL_RGB = np.array([19.0, 163.0, 136.0], dtype=np.float32)
TEAL_RADIUS = 55.0
DARK_LEVEL = 30
WHITE_LEVEL = 205
#: a connected white component larger than this is a specular highlight, not a glyph
TEXT_MAX_COMPONENT = 400


RUNS: tuple[dict[str, Any], ...] = (
    {
        "key": "unet",
        "label": "UNet",
        "runs": FALSE_ALARM
        / "unet"
        / "external_hyperkvasir_cecum"
        / "unet_resnet50_detrules"
        / "evaluation",
        "checkpoint": "checkpoints/cvc_clinicdb_unet_resnet50/fold{fold}_seed{seed}/best.pt",
        "overrides": {},
        "count": 8,
        "colour": style.MODEL_COLOURS["unet_resnet50"],
    },
    {
        "key": "segformer",
        "label": "SegFormer",
        "runs": FALSE_ALARM
        / "segformer"
        / "external_hyperkvasir_cecum"
        / "segformer_mit_b2_detrules"
        / "evaluation",
        "checkpoint": "checkpoints/cvc_clinicdb_segformer_mit_b2/fold{fold}_seed{seed}/best.pt",
        "overrides": {
            "model.architecture": "smp_segformer",
            "model.encoder": "mit_b2",
            "model.encoder_weights": "imagenet",
        },
        "count": 4,
        "colour": style.MODEL_COLOURS["segformer_mit_b2"],
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default=str(style.FIGURE_DIR / "sanity_false_alarms.png"))
    parser.add_argument("--seed", type=int, default=FRAME_SEED)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    return parser.parse_args(argv)


def _per_case_csv(run: dict[str, Any]) -> Path:
    return run["runs"] / f"hyperkvasir_cecum_test_fold{FOLD}_seed{CHECKPOINT_SEED}_per_case.csv"


def _false_alarms(run: dict[str, Any], count: int, seed: int) -> pd.DataFrame:
    """``count`` frames (seeded draw) on which the baseline invented a polyp, with the CSV values."""
    table = pd.read_csv(_per_case_csv(run))
    block = table.loc[
        (table["method"] == "baseline")
        & (table["metric"].isin(["pred_empty", "pred_area_px", "max_prob"]))
    ]
    wide = block.pivot_table(index="case_id", columns="metric", values="value", aggfunc="first")
    alarms = wide.loc[wide["pred_empty"] == 0.0].sort_index()
    if len(alarms) < count:
        raise RuntimeError(f"{run['key']}: only {len(alarms)} false alarms, need {count}")
    picked = np.sort(np.random.default_rng(seed).choice(len(alarms), size=count, replace=False))
    return alarms.iloc[picked].reset_index()


def _overlay_classes(image: np.ndarray) -> dict[str, np.ndarray]:
    """Pixel classes of one frame: green overlay box, black border, white overlay text.

    The three overlays of the HyperKvasir cecum frames are colour/structure separable: the
    endoscope's box is a saturated teal, the letterbox/vignette is near black, and the text is
    made of thin pure-white strokes (a 7x7 opening removes strokes and keeps specular blobs,
    so ``white & ~opened`` is the stroke set).  Text components are then kept only when they
    are small and lie in the two overlay panels (the left metadata strip or the teal box),
    which removes the specular highlights of the mucosa from the text class.
    """
    height, width = image.shape[:2]
    distance = np.linalg.norm(image.astype(np.float32) - TEAL_RGB, axis=2)
    green = distance < TEAL_RADIUS
    black = np.max(image, axis=2) < DARK_LEVEL
    white = np.min(image, axis=2) > WHITE_LEVEL
    opened = cv2.morphologyEx(
        (white.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
    )
    strokes = (white & (opened == 0)).astype(np.uint8)
    rows = np.nonzero(green.any(axis=1))[0]
    cols = np.nonzero(green.any(axis=0))[0]
    box = np.zeros_like(green)
    if len(rows) and len(cols):
        box[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
    panels = box.copy()
    panels[:, : int(round(0.30 * width))] = True
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    text = np.zeros_like(green)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > TEXT_MAX_COMPONENT or component_width > 80 or component_height > 40:
            continue
        component = labels == label
        if float((component & panels).sum()) / max(area, 1) < 0.6:
            continue
        text |= component

    return {"green": green, "black": black, "text": text}


def _region_report(
    mask: np.ndarray, classes: dict[str, np.ndarray], *, border_px: int = 2
) -> dict[str, Any]:
    """Composition of one predicted region and the class it mostly covers."""
    height, width = mask.shape
    total = int(mask.sum())
    fractions = {
        name: float((mask & values).sum()) / max(total, 1) for name, values in classes.items()
    }
    fractions["anatomy"] = max(0.0, 1.0 - sum(fractions.values()))
    rows, cols = np.nonzero(mask)
    bbox = (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())) if total else None
    touches_border = bool(
        total
        and (
            rows.min() < border_px
            or cols.min() < border_px
            or rows.max() >= height - border_px
            or cols.max() >= width - border_px
        )
    )
    if fractions["green"] > 0.5:
        dominant = "green scope overlay box"
    elif fractions["black"] > 0.5:
        dominant = "black border / vignette"
    elif fractions["text"] > 0.10:
        dominant = "overlay text"
    elif fractions["anatomy"] > 0.5:
        dominant = "anatomy (mucosa)"
    else:
        dominant = "mixed overlay + anatomy"
    return {
        "fractions": fractions,
        "dominant": dominant,
        "bbox": bbox,
        "touches_border": touches_border,
        "central_fraction": float(
            mask[
                int(0.10 * height) : int(0.90 * height),
                int(0.10 * width) : int(0.90 * width),
            ].sum()
        )
        / max(total, 1),
    }


def _preprocessing_chain(config: Config, *, reference: str) -> list[str]:
    """The ordered steps ``ExternalPolypDataset.__getitem__`` applies, read off the config."""
    image_size = [int(value) for value in config.require("dataset.image_size")]
    normalization = NormalizationSpec.from_config(config, fold=FOLD)
    spacing = [float(value) for value in config.require("dataset.spacing_mm")]
    threshold = config.get("dataset.mask_threshold")
    mean = ", ".join(f"{value:.4f}" for value in (normalization.mean or ()))
    std = ", ".join(f"{value:.4f}" for value in (normalization.std or ()))
    steps = [
        "read_external_case -> _read_image(path): cv2.imread(IMREAD_UNCHANGED) (tifffile for .tif), "
        "BGR planes reversed to RGB; grey/replicated frames rejected; uint8 (H, W, 3)",
        f"cv2.resize(image, (W, H) = ({image_size[1]}, {image_size[0]}), interpolation=INTER_LINEAR)",
        "np.float32(image) / 255.0 -> [0, 1]",
        "transpose(2, 0, 1) -> (3, H, W)",
        f"NormalizationSpec(kind={normalization.kind}, mean=({mean}), std=({std})).apply -> (x - mean) / std",
        "torch.from_numpy -> network input (1, 3, H, W)",
    ]
    if reference == "mask":
        steps.append(
            "reference: _read_mask(path) -> binarise_mask(threshold="
            f"{threshold if threshold is not None else 'auto'}) at the native resolution; "
            f"mask_native = that binary mask (spacing {spacing} for distances)"
        )
    else:
        steps.append(
            "reference: mask_path is None -> np.zeros(image.shape[:2]) at the native resolution "
            "(the cohort is polyp-free by definition); mask_native = that all-zero mask "
            f"(spacing {spacing} for areas)"
        )
    return steps


def _check_same_preprocessing(cecum: Config, kvasir: Config) -> bool:
    """Print both chains and assert the two loaders produce the same network input tensor."""
    cecum_dataset = build_dataset(cecum, "test", fold=FOLD)
    kvasir_dataset = build_dataset(kvasir, "test", fold=FOLD)
    print("\n--- Kvasir-SEG (masked external path, configs/external/polyp_external.yaml) ---")
    for index, step in enumerate(_preprocessing_chain(kvasir, reference="mask"), start=1):
        print(f"  {index}. {step}")
    print("\n--- HyperKvasir cecum (no-mask path, configs/external/hyperkvasir_cecum.yaml) ---")
    for index, step in enumerate(_preprocessing_chain(cecum, reference="empty"), start=1):
        print(f"  {index}. {step}")
    same_class = type(cecum_dataset) is type(kvasir_dataset) is ExternalPolypDataset
    print(
        "\n  loader class identical: "
        f"{type(kvasir_dataset).__name__} == {type(cecum_dataset).__name__} -> {same_class}"
    )
    kvasir_item = kvasir_dataset.items[0]
    probe = ExternalPolypItem(
        case_id="preprocessing_probe",
        group_id="preprocessing_probe",
        split="test",
        image_path=kvasir_item.image_path,
        mask_path=None,
    )
    maskless = ExternalPolypDataset(
        [probe],
        image_size=cecum.require("dataset.image_size"),
        mask_threshold=None
        if cecum.get("dataset.mask_threshold") is None
        else int(cecum.get("dataset.mask_threshold")),
        spacing=cecum.require("dataset.spacing_mm"),
        normalization=NormalizationSpec.from_config(cecum, fold=FOLD),
        fold=FOLD,
        split="test",
    )
    with_mask = kvasir_dataset[
        next(
            position
            for position, item in enumerate(kvasir_dataset.items)
            if item.image_path == kvasir_item.image_path
        )
    ]
    without_mask = maskless[0]
    identical = torch.equal(with_mask["image"], without_mask["image"])
    difference = float((with_mask["image"] - without_mask["image"]).abs().max())
    print(
        f"  same file ({kvasir_item.image_path.name}) through both loaders: "
        f"input tensors bit-identical -> {identical} (max |diff| = {difference:.3e})"
    )
    print(
        "  native reference: Kvasir-SEG shape "
        f"{tuple(with_mask['mask_native'].shape)}, sum {int(with_mask['mask_native'].sum())}; "
        f"no-mask shape {tuple(without_mask['mask_native'].shape)}, "
        f"sum {int(without_mask['mask_native'].sum())}"
    )
    if not (same_class and identical):
        raise AssertionError("the maskless path does not reproduce the Kvasir-SEG preprocessing")
    return True


def _predict(
    model: torch.nn.Module,
    sample: dict[str, Any],
    method: Any,
    device: torch.device,
    native_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregated probability and 0.5 mask on the native grid, exactly as the runner does."""
    image = sample["image"].unsqueeze(0)
    views = predict_per_view(model, image, device, method, activation="sigmoid", amp=False)
    native = np.stack(
        [resample_probability_map(view_probability, native_shape)[0] for view_probability in views]
    )
    probability = aggregate_probabilities(native[:, None], method.aggregator)[0]
    return probability, probability >= 0.5


def _render(path: Path, panels: list[dict[str, Any]], note: str = "") -> None:
    """Two rows of six panels: image + predicted contour + the region's class inside the frame."""
    style.apply()
    import matplotlib.pyplot as plt

    columns = 6
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.05 * columns, 1.95 * rows))
    axes = np.atleast_2d(axes)
    figure.subplots_adjust(
        left=0.004, right=0.996, top=0.88, bottom=0.004, wspace=0.03, hspace=0.03
    )
    for ax in axes.ravel():
        ax.axis("off")
    for index, panel in enumerate(panels):
        ax = axes[index // columns][index % columns]
        image = panel["image"]
        ax.imshow(image)
        mask = panel["mask"]
        if mask.any():
            ax.contour(
                mask.astype(np.float32), levels=[0.5], colors=[panel["colour"]], linewidths=1.1
            )
        region_rows, region_cols = np.nonzero(mask)
        if len(region_rows):
            # a thin dotted box keeps a small false alarm visible at print size
            ax.add_patch(
                plt.Rectangle(
                    (region_cols.min() - 3, region_rows.min() - 3),
                    region_cols.max() - region_cols.min() + 6,
                    region_rows.max() - region_rows.min() + 6,
                    fill=False,
                    edgecolor=panel["colour"],
                    linewidth=0.7,
                    linestyle=":",
                )
            )
        ax.annotate(
            f"{panel['label']} {panel['case_id'][:8]} · {panel['area']} px\n{panel['dominant']}",
            (0.015, 0.985),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=5.2,
            linespacing=1.3,
            color="#ffffff",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.2, "linewidth": 0},
        )
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    suptitle = (
        "False alarms on polyp-free HyperKvasir cecum (baseline, fold0_seed0, recomputed on CPU)\n"
        "contour = predicted mask at 0.5; blue = UNet, vermillion = SegFormer"
    )
    if note:
        suptitle = f"{suptitle}\n{note}"
    figure.suptitle(suptitle, fontsize=7.5, linespacing=1.35, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)
    print(f"\nwrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    device = torch.device("cpu")
    print(f"device {device} (CPU only), seed {args.seed}")

    cecum = load_config(CECUM_CONFIG, {"device": "cpu"})
    kvasir = load_config(
        KVASIR_CONFIG,
        {
            "device": "cpu",
            "dataset.name": "kvasir_seg",
            "dataset.images_dir": "C:/Users/subka/Documents/datasets/polyp/Kvasir-SEG/images",
            "dataset.masks_dir": "C:/Users/subka/Documents/datasets/polyp/Kvasir-SEG/masks",
            "dataset.references": "mask",
        },
    )
    _check_same_preprocessing(cecum, kvasir)

    dataset = build_dataset(cecum, "test", fold=FOLD)
    index_by_case = {str(item.case_id): position for position, item in enumerate(dataset.items)}
    methods = {method.name: method for method in methods_from_config(cecum)}
    panels: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    print(
        "\nmodel      case      csv_area  cpu_area  diff%   csv_pmax  cpu_pmax   "
        "green%  black%  text%  anatomy%  position / class"
    )
    for run in RUNS:
        alarms = _false_alarms(run, int(run["count"]), args.seed)
        config = load_config(CECUM_CONFIG, {"device": "cpu", **run["overrides"]})
        model = build_model(config)
        checkpoint = config.resolve(
            run["checkpoint"].format(fold=FOLD, seed=CHECKPOINT_SEED)
        )
        load_checkpoint(model, checkpoint, device)
        model = model.to(device).eval()
        print(f"\n[{run['label']}] checkpoint {checkpoint}")
        for _, row in alarms.iterrows():
            case_id = str(row["case_id"])
            sample = dataset[index_by_case[case_id]]
            target = sample["mask_native"][0].numpy()
            probability, mask = _predict(
                model, sample, methods["baseline"], device, tuple(target.shape)
            )
            area = int(mask.sum())
            csv_area = int(row["pred_area_px"])
            peak = float(probability.max())
            csv_peak = float(row["max_prob"])
            difference = (area - csv_area) / max(csv_area, 1) * 100.0
            image = cv2.cvtColor(
                cv2.imread(dataset.items[index_by_case[case_id]].image_path.as_posix()),
                cv2.COLOR_BGR2RGB,
            )
            classes = _overlay_classes(image)
            report = _region_report(mask, classes)
            fractions = report["fractions"]
            print(
                f"{run['label']:<10} {case_id[:8]}  {csv_area:8d}  {area:9d}  "
                f"{difference:+6.2f}  {csv_peak:8.4f}  {peak:8.4f}   "
                f"{100 * fractions['green']:6.1f}  {100 * fractions['black']:6.1f}  "
                f"{100 * fractions['text']:5.1f}  {100 * fractions['anatomy']:7.1f}  "
                f"{'border' if report['touches_border'] else 'inside'} / {report['dominant']}"
            )
            records.append(
                {
                    "model": run["key"],
                    "case_id": case_id,
                    "csv_area": csv_area,
                    "cpu_area": area,
                    "area_diff_pct": difference,
                    "csv_max_prob": csv_peak,
                    "cpu_max_prob": peak,
                    **{f"{name}_fraction": value for name, value in fractions.items()},
                    "dominant": report["dominant"],
                    "touches_border": report["touches_border"],
                    "central_fraction": report["central_fraction"],
                }
            )
            panels.append(
                {
                    "image": image,
                    "mask": mask,
                    "case_id": case_id,
                    "label": run["label"],
                    "colour": run["colour"],
                    "area": area,
                    "dominant": report["dominant"],
                }
            )
            empty_mismatch = (area == 0) != (csv_area == 0)
            if empty_mismatch or abs(difference) > 5.0:
                print(
                    "  WARNING: CPU recompute does not reproduce the stored area "
                    f"({csv_area} -> {area})"
                )
        del model

    table = pd.DataFrame(records)
    print("\nregion classes:")
    print(table.groupby("model")["dominant"].value_counts().to_string())
    print("\nmean pixel composition of the false regions [%]:")
    summary = (
        table.groupby("model")
        [["green_fraction", "black_fraction", "text_fraction", "anatomy_fraction"]]
        .mean()
        .mul(100.0)
        .round(2)
    )
    print(summary.to_string())
    print(
        f"\ntouching the image border: {int(table['touches_border'].sum())} of {len(table)} regions; "
        f"mean fraction inside the central 80 % box: {table['central_fraction'].mean():.3f}"
    )
    print(
        "match with the stored CSV: "
        f"sign(area) agrees on {int(((table['cpu_area'] > 0) == (table['csv_area'] > 0)).sum())}"
        f"/{len(table)}, median |area diff| {table['area_diff_pct'].abs().median():.2f} %, "
        "median |max_prob diff| "
        f"{(table['cpu_max_prob'] - table['csv_max_prob']).abs().median():.4f}"
    )
    _render(
        Path(args.output),
        panels,
        note=(
            f"{len(table)} of {len(table)} regions lie on mucosa; "
            f"{int((table['green_fraction'] > 0.5).sum())} on the green box, "
            f"{int((table['black_fraction'] > 0.5).sum())} on the black border, "
            f"{int((table['text_fraction'] > 0.10).sum())} on overlay text"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




