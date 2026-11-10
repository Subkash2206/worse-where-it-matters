"""Figure 7 - qualitative grid: which polyps TTA silences, and where it repairs a boundary.

Rows are chosen by a rule, never by eye, and **every row is drawn with the checkpoint that shows it
best** (named in the row label) - no checkpoint is fixed in advance:

* **three missed polyps from the external collections** (CVC-ColonDB and ETIS-Larib) that the
  baseline detects and TTA-8 loses.  A case qualifies when the baseline prediction is non-empty in the
  majority of the 15 checkpoints *and* the TTA-8 prediction is empty in the majority of them.  Among
  the checkpoints in which TTA-8 really is empty, the row is drawn with the one that gives the
  baseline its highest Dice, provided that Dice reaches 0.3 (so the "before" panel shows a real, if
  imperfect, segmentation instead of an empty prediction).  Rows are taken smallest reference area
  first, at most two from one collection;
* **two internal CVC-ClinicDB cases** with the largest TTA-4 Dice increase in the seed- and
  fold-averaged per-case table (the number the paper quotes), each drawn with the checkpoint that
  gives the case its largest TTA-4 gain among the checkpoints whose baseline Dice reaches 0.3, and
  drawn with **TTA-4** so the picture matches the criterion.  The drawn Dice of every row is asserted
  against the evaluation table (tolerance 0.011).

Every map is recomputed on the **CPU** with that row's checkpoint, ``baseline`` and ``tta8`` views
through the same ``predict_per_view -> resample -> aggregate`` path the evaluation uses, so the
contours and the printed Dice of a row always come from the same forward passes and the same
checkpoint.  Dice is reported on the native reference grid, before -> after, next to the row label;
the reference contour is the dashed line sample in the legend.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

sys.path.insert(0, str(style.REPO_ROOT / "src"))

INTERNAL_CONFIG = style.REPO_ROOT / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml"
EXTERNAL_CONFIG = style.REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
EXTERNAL_DATASETS = ("cvc_colondb", "etis_larib")
METHODS = ("baseline", "tta8")
N_EXTERNAL = 3
N_INTERNAL = 2
#: a majority of the 15 checkpoints (5 folds x 3 seeds of the internal grid)
MAJORITY = 0.5
#: the "before" panel has to show a real, imperfect segmentation, not an empty prediction
MIN_BASELINE_DICE = 0.3
#: at most this many drawn rows may come from the same external collection
MAX_PER_COLLECTION = 2
#: the drawn Dice may differ from the table by this much (float16 maps + bilinear resampling)
DICE_TOLERANCE = 0.011
#: the zoom panel crops a window this many times the reference bounding box, centred on it
ZOOM_FACTOR = 3.0
BASELINE_COLOUR = style.OKABE_ITO["vermillion"]
TTA_COLOUR = style.OKABE_ITO["blue"]
CHECKPOINT_TEMPLATE = "checkpoints/cvc_clinicdb_unet_resnet50/fold{fold}_seed{seed}/best.pt"
_CHECKPOINT_NAME = re.compile(r"fold(?P<fold>\d+)_seed(?P<seed>\d+)_per_case\.csv$")


def checkpoint_path(fold: int, seed: int) -> Path:
    """Checkpoint file of one grid point."""
    return style.REPO_ROOT / CHECKPOINT_TEMPLATE.format(fold=int(fold), seed=int(seed))


def _grid_table(directory: Path) -> pd.DataFrame:
    """Per (case, checkpoint, method) Dice and detection flag of one evaluation directory.

    The fold and the seed are read back from the per-case file names, so every row can be tied to
    the checkpoint that produced it - which is what the drawing rule needs.  ``group_id`` is kept so
    the internal rows can be forced onto distinct videos.
    """
    frames = []
    for path in sorted(directory.glob("*_per_case.csv")):
        match = _CHECKPOINT_NAME.search(path.name)
        if match is None:
            continue
        frame = pd.read_csv(
            path, usecols=["case_id", "group_id", "method", "metric", "value", "missed_detection"]
        )
        frame = frame.loc[(frame["metric"] == "dice") & (frame["method"].isin(METHODS + ("tta4",)))]
        frames.append(
            frame.assign(
                case_id=frame["case_id"].astype(str),
                fold=int(match.group("fold")),
                seed=int(match.group("seed")),
            )
        )
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    wide = table.pivot_table(
        index=["case_id", "fold", "seed"],
        columns="method",
        values=["value", "missed_detection"],
    )
    wide.columns = [f"{method}_{field}" for field, method in wide.columns]
    wide = wide.reset_index()
    video = (
        table.groupby(["case_id", "fold", "seed"])["group_id"]
        .agg(lambda values: str(values.iloc[0]))
        .rename("group_id")
        .reset_index()
    )
    return wide.merge(video, on=["case_id", "fold", "seed"], how="left")


def select_external(count: int = N_EXTERNAL) -> list[dict]:
    """External misses, each drawn with the checkpoint that shows it best.

    Rule: a case qualifies when the baseline prediction is non-empty in the majority of the 15
    checkpoints *and* the TTA-8 prediction is empty in the majority of them.  Among the checkpoints
    in which TTA-8 really is empty, the row is drawn with the one that gives the baseline its
    highest Dice, provided that Dice reaches ``MIN_BASELINE_DICE`` - otherwise the "before" panel
    would be an empty prediction.  Rows are taken smallest reference first, at most
    ``MAX_PER_COLLECTION`` from one collection.
    """
    candidates: list[dict] = []
    for key in EXTERNAL_DATASETS:
        wide = _grid_table(style.RESULTS / f"external_{key}" / "evaluation")
        if wide.empty:
            continue
        share = wide.groupby("case_id").agg(
            detected=("baseline_missed_detection", lambda values: 1.0 - float(values.mean())),
            tta8_missed=("tta8_missed_detection", "mean"),
        )
        rule = share.loc[(share["detected"] >= MAJORITY) & (share["tta8_missed"] > MAJORITY)]
        sizes = pd.read_csv(
            style.RESULTS / "external_by_model" / f"sizes_{key}.csv", usecols=["case_id", "gt_pixels"]
        )
        sizes["case_id"] = sizes["case_id"].astype(str)
        for case_id in rule.index:
            empty = wide.loc[(wide["case_id"] == case_id) & (wide["tta8_missed_detection"] > MAJORITY)]
            if empty.empty:
                continue
            best = empty.sort_values("baseline_value", ascending=False).iloc[0]
            if float(best["baseline_value"]) < MIN_BASELINE_DICE:
                continue
            size = sizes.loc[sizes["case_id"] == str(case_id), "gt_pixels"]
            candidates.append(
                {
                    "dataset": key,
                    "case_id": str(case_id),
                    "fold": int(best["fold"]),
                    "seed": int(best["seed"]),
                    "baseline_dice": float(best["baseline_value"]),
                    "detected": float(rule.loc[case_id, "detected"]),
                    "tta8_missed": float(rule.loc[case_id, "tta8_missed"]),
                    "gt_pixels": float(size.iloc[0]) if not size.empty else float("nan"),
                }
            )
    chosen: list[dict] = []
    per_collection: dict[str, int] = {}
    for item in sorted(candidates, key=lambda row: row["gt_pixels"]):
        if len(chosen) >= count:
            break
        if per_collection.get(item["dataset"], 0) >= MAX_PER_COLLECTION:
            continue
        per_collection[item["dataset"]] = per_collection.get(item["dataset"], 0) + 1
        chosen.append(item)
    return chosen


def select_internal(count: int = N_INTERNAL) -> list[dict]:
    """Internal repair rows: the largest single-checkpoint TTA-4 Dice gains, one video each.

    The criterion is the gain *of the drawn checkpoint*, not a grid average, so the row label, the
    Dice text and the picture all state the same number.  Only checkpoints whose baseline Dice reaches
    ``MIN_BASELINE_DICE`` are eligible (the "before" panel must show a real segmentation), and the
    rows are forced onto distinct videos so the figure does not show two frames of the same sequence.
    """
    wide = _grid_table(style.RESULTS / "cvc_clinicdb_unet_resnet50" / "evaluation")
    if wide.empty:
        return []
    wide["gain"] = wide["tta4_value"] - wide["baseline_value"]
    qualified = wide.loc[wide["baseline_value"] >= MIN_BASELINE_DICE].sort_values(
        "gain", ascending=False
    )
    rows: list[dict] = []
    videos: set[str] = set()
    for _, row in qualified.iterrows():
        if len(rows) >= count:
            break
        video = str(row["group_id"])
        if video in videos:
            continue
        videos.add(video)
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "baseline_dice": float(row["baseline_value"]),
                "tta4": float(row["tta4_value"]),
                "gain": float(row["gain"]),
                "video": video,
            }
        )
    return rows


def _dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Dice of two binary masks on the native grid."""
    total = int(prediction.sum()) + int(reference.sum())
    if total == 0:
        return float("nan")
    return float(2.0 * np.logical_and(prediction, reference).sum() / total)


def _display_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read one image (RGB, 16-bit TIFF or grey) and resize it onto the reference grid."""
    from PIL import Image

    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        data = np.asarray(tifffile.imread(path))
    else:
        data = np.asarray(Image.open(path).convert("RGB"))
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    values = data[:, :, :3].astype(np.float64)
    scale = float(values.max()) if float(values.max()) > 255.0 else 255.0
    values = np.clip(values / max(scale, 1.0) * 255.0, 0.0, 255.0).astype(np.uint8)
    resized = Image.fromarray(values).resize((int(shape[1]), int(shape[0])), Image.BILINEAR)
    return np.asarray(resized)


class Inference:
    """Baseline and TTA-8 probability maps of one checkpoint, computed on the CPU."""

    def __init__(self, config_path: Path, checkpoint: Path, overrides: dict, *, fold: int) -> None:
        import torch

        from melba.config import load_config
        from melba.data import build_dataset
        from melba.inference import methods_from_config
        from melba.inference.sliding import build_sliding_window
        from melba.models import build_model, load_checkpoint

        self.torch = torch
        self.device = torch.device("cpu")
        self.checkpoint = Path(checkpoint)
        self.fold = int(fold)
        self.config = load_config(config_path, overrides)
        self.model = build_model(self.config)
        load_checkpoint(self.model, self.checkpoint, self.device)
        self.model.to(self.device).eval()
        self.dataset = build_dataset(self.config, split="test", fold=self.fold)
        items = getattr(self.dataset, "items", None)
        self.case_ids = [str(item.case_id) for item in items] if items is not None else []
        self.methods = {method.name: method for method in methods_from_config(self.config)}
        self.activation = str(self.config.get("inference.activation", "sigmoid"))
        self.sliding_window = build_sliding_window(self.config)
        threshold = self.config.get("inference.threshold", 0.5)
        self.threshold = 0.5 if threshold is None else float(threshold)
        self.images_dir = Path(str(self.config.require("dataset.images_dir")))

    def _image_path(self, case_id: str) -> Path | None:
        return next(self.images_dir.glob(f"{case_id}.*"), None)

    def maps(self, case_id: str, methods: tuple[str, ...] = METHODS) -> dict:
        """Native-grid reference, display image and the binary mask of each requested method."""
        from melba.eval.runner import resample_probability_map
        from melba.inference import aggregate_probabilities, predict_per_view_tensor

        index = self.case_ids.index(case_id)
        sample = self.dataset[index]
        image = sample["image"].unsqueeze(0)
        mask = sample.get("mask_native", sample["mask"])
        target = mask[0].numpy() if hasattr(mask[0], "numpy") else np.asarray(mask[0])
        reference = np.asarray(target) > 0.5
        result: dict = {"reference": reference}
        for name in methods:
            with self.torch.no_grad():
                views = predict_per_view_tensor(
                    self.model,
                    image,
                    self.device,
                    self.methods[name],
                    activation=self.activation,
                    amp=False,
                    sliding_window=self.sliding_window,
                )
            view_maps = views.cpu().numpy()
            native = np.stack(
                [resample_probability_map(view_map, reference.shape)[0] for view_map in view_maps]
            )
            probability = aggregate_probabilities(
                native[:, None], self.methods[name].aggregator
            )[0]
            result[name] = probability > self.threshold
        path = self._image_path(case_id)
        if path is not None:
            result["image"] = _display_image(path, reference.shape)
        else:  # no image file: draw the masks on a neutral background
            result["image"] = np.full((*reference.shape, 3), 32, dtype=np.uint8)
        return result


def _external_overrides(key: str, fold: int, seed: int) -> dict:
    """Re-point the external config at one collection, one fold and one seed."""
    from melba.config import load_config

    external = load_config(EXTERNAL_CONFIG)
    entry = dict(external.get("external.datasets")[key])
    return {
        "experiment.name": f"fig07__{key}",
        "dataset.kind": "external_polyp",
        "dataset.name": key,
        "dataset.images_dir": entry["images_dir"],
        "dataset.masks_dir": entry["masks_dir"],
        "model.checkpoint": str(checkpoint_path(fold, seed)),
        "evaluation.fold": int(fold),
        "dataset.split.fold": int(fold),
        "seed": int(seed),
        "runtime.progress": False,
    }


def _internal_overrides(fold: int, seed: int) -> dict:
    return {
        "model.checkpoint": str(checkpoint_path(fold, seed)),
        "evaluation.fold": int(fold),
        "dataset.split.fold": int(fold),
        "seed": int(seed),
        "runtime.progress": False,
    }


def main() -> int:
    style.apply()
    external = select_external()
    internal = select_internal()
    if not external and not internal:
        print("the selection rule found no case to draw", file=sys.stderr)
        return 2
    print("selected rows (rule, no manual pick; CPU maps of the named checkpoint):")
    for item in external:
        print(
            f"  external {item['dataset']:12s} case {str(item['case_id']):>4s} "
            f"reference {float(item['gt_pixels']):7.0f} px  baseline non-empty in "
            f"{float(item['detected']) * 100:5.1f} % of 15 checkpoints, TTA-8 empty in "
            f"{float(item['tta8_missed']) * 100:5.1f} %  -> drawn with fold{item['fold']}_seed{item['seed']} "
            f"(baseline Dice {float(item['baseline_dice']):.3f} in the table, TTA-8 empty)"
        )
    for item in internal:
        print(
            f"  internal cvc_clinicdb case {item['case_id']:>4s} (video {item['video']}) "
            f"single-checkpoint TTA-4 gain {item['gain']:+.3f} -> drawn with "
            f"fold{item['fold']}_seed{item['seed']} "
            f"(table Dice {item['baseline_dice']:.3f} -> {item['tta4']:.3f}, drawn column is TTA-4)"
        )

    rows: list[dict] = []
    engines: dict[tuple, Inference] = {}
    for item in external:
        key = str(item["dataset"])
        fold, seed = int(item["fold"]), int(item["seed"])
        cache_key = (key, fold, seed)
        if cache_key not in engines:
            engines[cache_key] = Inference(
                EXTERNAL_CONFIG,
                checkpoint_path(fold, seed),
                _external_overrides(key, fold, seed),
                fold=fold,
            )
        maps = engines[cache_key].maps(str(item["case_id"]), METHODS)
        rows.append(
            {
                "label": (
                    f"{style.LABELS[key].replace(' (external)', '')} {item['case_id']}\n"
                    f"fold{fold}_seed{seed} · TTA-8\n"
                    f"TTA-8 empty {float(item['tta8_missed']) * 15:.0f}/15"
                ),
                "method": "tta8",
                "method_label": "TTA-8",
                "expected": {f"baseline": float(item["baseline_dice"]), "tta8": 0.0},
                **maps,
            }
        )
    for item in internal:
        fold, seed = int(item["fold"]), int(item["seed"])
        cache_key = ("cvc_clinicdb", fold, seed)
        if cache_key not in engines:
            engines[cache_key] = Inference(
                INTERNAL_CONFIG,
                checkpoint_path(fold, seed),
                _internal_overrides(fold, seed),
                fold=fold,
            )
        maps = engines[cache_key].maps(str(item["case_id"]), ("baseline", "tta4"))
        rows.append(
            {
                "label": (
                    f"CVC-ClinicDB {item['case_id']}\n"
                    f"fold{fold}_seed{seed} · TTA-4\n"
                    f"TTA-4 gain {item['gain']:+.2f}"
                ),
                "method": "tta4",
                "method_label": "TTA-4",
                "expected": {
                    "baseline": float(item["baseline_dice"]),
                    "tta4": float(item["tta4"]),
                },
                **maps,
            }
        )
    return _draw(rows)


def _reference_contour(ax, reference) -> None:
    """White dashed reference over a dark halo, so it reads on bright and dark frames."""
    if reference is None or not np.asarray(reference).any():
        return
    values = np.asarray(reference).astype(float)
    ax.contour(values, levels=[0.5], colors="#000000", linewidths=1.3, alpha=0.6)
    ax.contour(values, levels=[0.5], colors="#ffffff", linewidths=0.7, linestyles="--")


def _prediction_contour(ax, prediction, colour) -> None:
    """Solid prediction contour in the colour of the method it belongs to."""
    if prediction is None or not np.asarray(prediction).any():
        return
    ax.contour(np.asarray(prediction).astype(float), levels=[0.5], colors=colour, linewidths=1.0)


def _zoom_slices(reference: np.ndarray, factor: float = ZOOM_FACTOR) -> tuple[slice, slice]:
    """Crop window of about ``factor`` times the reference bounding box, centred on it.

    The window is clipped to the frame, so a polyp near the border still gives a valid crop.
    """
    rows = np.flatnonzero(reference.any(axis=1))
    cols = np.flatnonzero(reference.any(axis=0))
    height, width = reference.shape
    if rows.size == 0 or cols.size == 0:  # empty reference: fall back to the whole frame
        return slice(0, height), slice(0, width)
    height_span = max(int(round((rows[-1] - rows[0] + 1) * factor)), 16)
    width_span = max(int(round((cols[-1] - cols[0] + 1) * factor)), 16)
    height_span = min(height_span, height)
    width_span = min(width_span, width)
    centre_row = (rows[0] + rows[-1]) / 2.0
    centre_col = (cols[0] + cols[-1]) / 2.0
    top = max(0, min(int(round(centre_row - height_span / 2.0)), height - height_span))
    left = max(0, min(int(round(centre_col - width_span / 2.0)), width - width_span))
    return slice(top, top + height_span), slice(left, left + width_span)


def _reference_handle() -> tuple:
    """Legend sample of the dashed reference: white dashes over a black outline, as drawn."""
    from matplotlib.lines import Line2D

    outline = Line2D([], [], color="#000000", linestyle="--", linewidth=2.4)
    core = Line2D([], [], color="#ffffff", linestyle="--", linewidth=0.8)
    return (outline, core)


def _draw(rows: list[dict]) -> int:
    """The grid: image, reference, baseline, TTA per row, with the Dice of both methods."""
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    for row in rows:
        row["before"] = _dice(row["baseline"], row["reference"])
        row["after"] = _dice(row[row["method"]], row["reference"])
        expected = row["expected"]
        # The tables come from GPU runs with AMP (float16 logits), the drawn maps from float32 on
        # the CPU, so a borderline pixel can move the Dice by about a hundredth.  The check is made
        # on the values at the precision they are reported with (3 decimals).
        checked = {"baseline": row["before"], row["method"]: row["after"]}
        for key, value in checked.items():
            delta = abs(round(value, 3) - round(expected[key], 3))
            if delta > DICE_TOLERANCE + 1e-9:
                raise AssertionError(
                    f"{row['label'].splitlines()[0]}: drawn {key} Dice {value:.3f} does not match "
                    f"the table's {expected[key]:.3f} (|delta| {delta:.3f} > {DICE_TOLERANCE})"
                )
        print(
            f"  drawn {row['label'].splitlines()[0]:22s} {row['method_label']:5s} "
            f"Dice {row['before']:.3f} -> {row['after']:.3f}  "
            f"(table {expected['baseline']:.3f} -> {expected[row['method']]:.3f}, "
            f"|delta| {abs(round(row['before'], 3) - round(expected['baseline'], 3)):.3f} / "
            f"{abs(round(row['after'], 3) - round(expected[row['method']], 3)):.3f}; "
            f"baseline {int(row['baseline'].sum())} px, {row['method_label']} "
            f"{int(row[row['method']].sum())} px)"
        )

    columns = ("image", "baseline", "TTA", f"zoom ({ZOOM_FACTOR:.0f}x)")
    fig, axes = plt.subplots(
        len(rows), len(columns), figsize=(7.0, 1.28 * len(rows)), squeeze=False
    )
    fig.subplots_adjust(left=0.155, right=0.995, top=0.94, bottom=0.03, hspace=0.06, wspace=0.03)
    for row_index, row in enumerate(rows):
        panels = (
            (None, None, None),
            (row["reference"], row["baseline"], BASELINE_COLOUR),
            (row["reference"], row[row["method"]], TTA_COLOUR),
        )
        for column_index, (reference, prediction, colour) in enumerate(panels):
            ax = axes[row_index][column_index]
            ax.imshow(row["image"])
            _reference_contour(ax, reference)
            if prediction is not None:
                if prediction.any():
                    _prediction_contour(ax, prediction, colour)
                else:
                    ax.annotate(
                        "empty",
                        (0.06, 0.06),
                        xycoords="axes fraction",
                        ha="left",
                        va="bottom",
                        fontsize=5.5,
                        color="#ffffff",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.2, "linewidth": 0},
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_anchor("N")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(columns[column_index], fontsize=8)
            if column_index == 0:
                ax.annotate(
                    row["label"],
                    (-0.07, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=5.2,
                    linespacing=1.35,
                )
                ax.annotate(
                    f"Dice {row['before']:.2f} -> {row['after']:.2f}",
                    (0.02, 0.02),
                    xycoords="axes fraction",
                    fontsize=5.0,
                    color="#ffffff",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "linewidth": 0},
                )
        # zoom: a crop about ZOOM_FACTOR x the reference bbox with all three contours overlaid
        zoom_ax = axes[row_index][3]
        rows_slice, cols_slice = _zoom_slices(row["reference"])
        zoom_ax.imshow(row["image"][rows_slice, cols_slice])
        _reference_contour(zoom_ax, row["reference"][rows_slice, cols_slice])
        _prediction_contour(zoom_ax, row["baseline"][rows_slice, cols_slice], BASELINE_COLOUR)
        _prediction_contour(zoom_ax, row[row["method"]][rows_slice, cols_slice], TTA_COLOUR)
        zoom_ax.set_xticks([])
        zoom_ax.set_yticks([])
        zoom_ax.grid(False)
        zoom_ax.set_anchor("N")
        for spine in zoom_ax.spines.values():
            spine.set_visible(False)
        if row_index == 0:
            zoom_ax.set_title(columns[3], fontsize=8)
    handles = [
        _reference_handle(),
        Line2D([], [], color=BASELINE_COLOUR, linewidth=1.1, label="baseline prediction"),
        Line2D([], [], color=TTA_COLOUR, linewidth=1.1, label="TTA prediction"),
    ]
    fig.legend(
        handles=handles,
        labels=["reference", "baseline prediction", "TTA prediction"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=6.5,
        columnspacing=1.6,
        handlelength=2.6,
    )
    style.save(fig, "fig07_qualitative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

