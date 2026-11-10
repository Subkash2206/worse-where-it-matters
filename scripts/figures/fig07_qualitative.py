"""Figure 7 - qualitative grid: which polyps TTA silences, and where it repairs a boundary.

Rows are chosen by a rule, never by eye:

* **three small polyps from the external collections** (CVC-ColonDB and ETIS-Larib) that the baseline
  detects and TTA-8 loses.  "Detected" and "missed" are the detection flags of the evaluation: the
  baseline prediction is non-empty in the majority of the 15 checkpoints and the TTA-8 prediction is
  empty in the majority of them.  Only cases that the drawn checkpoint
  (``checkpoints/cvc_clinicdb_unet_resnet50/fold0_seed0/best.pt``) reproduces are eligible, one per
  collection is taken first, and within that the smallest reference area comes first - so the
  selection is reproducible instead of hand-picked;
* **two internal CVC-ClinicDB cases** with the largest TTA-4 Dice increase in that checkpoint's
  per-case table - the errors TTA repairs.

Every map is recomputed on the **CPU** with that one checkpoint, ``baseline`` and ``tta8`` views
through the same ``predict_per_view -> resample -> aggregate`` path the evaluation uses, so the
contours and the printed Dice of a row always come from the same forward passes.  Dice is reported
on the native reference grid, before -> after, next to the row label; the reference contour is the
dashed line sample in the legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

sys.path.insert(0, str(style.REPO_ROOT / "src"))

CHECKPOINT = style.REPO_ROOT / "checkpoints" / "cvc_clinicdb_unet_resnet50" / "fold0_seed0" / "best.pt"
INTERNAL_CONFIG = style.REPO_ROOT / "configs" / "cvc_clinicdb" / "unet_resnet50.yaml"
EXTERNAL_CONFIG = style.REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
EXTERNAL_DATASETS = ("cvc_colondb", "etis_larib")
FOLD = 0
SEED = 0
METHODS = ("baseline", "tta8")
N_EXTERNAL = 3
N_INTERNAL = 2
#: a majority of the 15 checkpoints (5 folds x 3 seeds of the internal grid)
MAJORITY = 0.5



def _external_candidates(key: str) -> pd.DataFrame:
    """Detection of every case in one external collection, over the 15 checkpoints.

    ``detected`` is the share of checkpoints whose baseline prediction is non-empty, ``tta8_missed``
    the share whose TTA-8 prediction is empty, and ``drawn_*`` the same two flags for the checkpoint
    the maps are actually drawn with.
    """
    directory = style.RESULTS / f"external_{key}" / "evaluation"
    frames = [
        pd.read_csv(path, usecols=["case_id", "method", "metric", "missed_detection"])
        for path in sorted(directory.glob("*_per_case.csv"))
    ]
    drawn_path = directory / f"{key}_test_fold{FOLD}_seed{SEED}_per_case.csv"
    if not frames or not drawn_path.is_file():
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[(frame["metric"] == "dice") & (frame["method"].isin(METHODS))].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["missed"] = pd.to_numeric(frame["missed_detection"], errors="coerce")
    share = frame.groupby(["case_id", "method"])["missed"].mean().unstack("method")
    drawn = pd.read_csv(drawn_path, usecols=["case_id", "method", "metric", "missed_detection"])
    drawn = drawn.loc[(drawn["metric"] == "dice") & (drawn["method"].isin(METHODS))].copy()
    drawn["case_id"] = drawn["case_id"].astype(str)
    drawn = drawn.pivot_table(index="case_id", columns="method", values="missed_detection")
    sizes = pd.read_csv(
        style.RESULTS / "external_by_model" / f"sizes_{key}.csv", usecols=["case_id", "gt_pixels"]
    )
    sizes["case_id"] = sizes["case_id"].astype(str)
    table = pd.DataFrame(
        {
            "dataset": key,
            "case_id": share.index,
            "detected": 1.0 - share["baseline"].to_numpy(),
            "tta8_missed": share["tta8"].to_numpy(),
            "drawn_detected": [
                1.0 - float(drawn.loc[case, "baseline"]) if case in drawn.index else float("nan")
                for case in share.index
            ],
            "drawn_tta8_missed": [
                float(drawn.loc[case, "tta8"]) if case in drawn.index else float("nan")
                for case in share.index
            ],
        }
    )
    return table.merge(sizes, on="case_id", how="left")


def select_external(count: int = N_EXTERNAL) -> list[dict]:
    """The documented rule: majority transition, drawn-checkpoint check, smallest area first."""
    tables = [_external_candidates(key) for key in EXTERNAL_DATASETS]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return []
    table = pd.concat(tables, ignore_index=True)
    rule = table.loc[(table["detected"] >= MAJORITY) & (table["tta8_missed"] > MAJORITY)]
    reproduced = rule.loc[(rule["drawn_detected"] >= 1.0) & (rule["drawn_tta8_missed"] > MAJORITY)]
    candidates = (reproduced if len(reproduced) >= count else rule).sort_values("gt_pixels")
    chosen: list[dict] = []
    for key in EXTERNAL_DATASETS:  # at least one case per collection
        block = candidates.loc[candidates["dataset"] == key]
        if not block.empty:
            chosen.append(block.iloc[0].to_dict())
    for _, row in candidates.iterrows():
        if len(chosen) >= count:
            break
        taken = {(item["dataset"], str(item["case_id"])) for item in chosen}
        if (row["dataset"], str(row["case_id"])) not in taken:
            chosen.append(row.to_dict())
    return sorted(chosen[:count], key=lambda item: float(item["gt_pixels"]))


def select_internal(count: int = N_INTERNAL) -> list[dict]:
    """The cases with the largest TTA-4 Dice increase in the drawn checkpoint's table."""
    path = (
        style.RESULTS
        / "cvc_clinicdb_unet_resnet50"
        / "evaluation"
        / f"cvc_clinicdb_test_fold{FOLD}_seed{SEED}_per_case.csv"
    )
    frame = pd.read_csv(path, usecols=["case_id", "method", "metric", "value"])
    frame = frame.loc[frame["metric"] == "dice"]
    wide = frame.pivot_table(index="case_id", columns="method", values="value")
    delta = (wide["tta4"] - wide["baseline"]).sort_values(ascending=False)
    return [
        {
            "case_id": str(case),
            "baseline": float(wide.loc[case, "baseline"]),
            "tta4": float(wide.loc[case, "tta4"]),
        }
        for case in delta.index[:count]
    ]


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

    def __init__(self, config_path: Path, overrides: dict) -> None:
        import torch

        from melba.config import load_config
        from melba.data import build_dataset
        from melba.inference import methods_from_config
        from melba.inference.sliding import build_sliding_window
        from melba.models import build_model, load_checkpoint

        self.torch = torch
        self.device = torch.device("cpu")
        self.config = load_config(config_path, overrides)
        self.model = build_model(self.config)
        load_checkpoint(self.model, CHECKPOINT, self.device)
        self.model.to(self.device).eval()
        self.dataset = build_dataset(self.config, split="test", fold=FOLD)
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

    def maps(self, case_id: str) -> dict:
        """Native-grid reference, display image and the binary mask of each method."""
        from melba.eval.runner import resample_probability_map
        from melba.inference import aggregate_probabilities, predict_per_view_tensor

        index = self.case_ids.index(case_id)
        sample = self.dataset[index]
        image = sample["image"].unsqueeze(0)
        mask = sample.get("mask_native", sample["mask"])
        target = mask[0].numpy() if hasattr(mask[0], "numpy") else np.asarray(mask[0])
        reference = np.asarray(target) > 0.5
        result: dict = {"reference": reference}
        for name in METHODS:
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


def _external_overrides(key: str) -> dict:
    """Re-point the external config at one collection and at the drawn checkpoint."""
    from melba.config import load_config

    external = load_config(EXTERNAL_CONFIG)
    entry = dict(external.get("external.datasets")[key])
    return {
        "experiment.name": f"fig07__{key}",
        "dataset.kind": "external_polyp",
        "dataset.name": key,
        "dataset.images_dir": entry["images_dir"],
        "dataset.masks_dir": entry["masks_dir"],
        "model.checkpoint": str(CHECKPOINT),
        "evaluation.fold": FOLD,
        "dataset.split.fold": FOLD,
        "seed": SEED,
        "runtime.progress": False,
    }


def main() -> int:
    style.apply()
    if not CHECKPOINT.is_file():
        print(f"missing checkpoint {CHECKPOINT}", file=sys.stderr)
        return 2
    external = select_external()
    internal = select_internal()
    if not external and not internal:
        print("the selection rule found no case to draw", file=sys.stderr)
        return 2
    print("selected rows (rule, no manual pick):")
    for item in external:
        print(
            f"  external {item['dataset']:12s} case {str(item['case_id']):>4s} "
            f"reference {float(item['gt_pixels']):7.0f} px  baseline non-empty in "
            f"{float(item['detected']) * 100:5.1f} % of 15 checkpoints, TTA-8 empty in "
            f"{float(item['tta8_missed']) * 100:5.1f} %"
        )
    for item in internal:
        print(
            f"  internal cvc_clinicdb case {item['case_id']:>4s} "
            f"Dice {item['baseline']:.3f} -> {item['tta4']:.3f} with TTA-4"
        )

    rows: list[dict] = []
    engines: dict[str, Inference] = {}
    for item in external:
        key = str(item["dataset"])
        if key not in engines:
            engines[key] = Inference(EXTERNAL_CONFIG, _external_overrides(key))
        maps = engines[key].maps(str(item["case_id"]))
        rows.append(
            {
                "label": (
                    f"{style.LABELS[key].replace(' (external)', '')} {item['case_id']}\n"
                    f"TTA-8 empty in {float(item['tta8_missed']) * 100:.0f} % of 15"
                ),
                **maps,
            }
        )
    engine = Inference(
        INTERNAL_CONFIG,
        {
            "model.checkpoint": str(CHECKPOINT),
            "evaluation.fold": FOLD,
            "dataset.split.fold": FOLD,
            "seed": SEED,
            "runtime.progress": False,
        },
    )
    for item in internal:
        maps = engine.maps(str(item["case_id"]))
        rows.append(
            {
                "label": (
                    f"CVC-ClinicDB {item['case_id']}\n"
                    f"largest TTA-4 gain {item['tta4'] - item['baseline']:+.2f}"
                ),
                **maps,
            }
        )
    return _draw(rows)


def _draw(rows: list[dict]) -> int:
    """The grid: image, reference, baseline, TTA-8 per row, with the Dice of both methods."""
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for row in rows:
        row["before"] = _dice(row["baseline"], row["reference"])
        row["after"] = _dice(row["tta8"], row["reference"])
        print(
            f"  drawn {row['label'].splitlines()[0]:22s} Dice {row['before']:.3f} -> "
            f"{row['after']:.3f}  (baseline {int(row['baseline'].sum())} px, "
            f"TTA-8 {int(row['tta8'].sum())} px)"
        )

    columns = ("image", "reference", "baseline", "TTA-8")
    fig, axes = plt.subplots(
        len(rows), len(columns), figsize=(7.0, 1.55 * len(rows)), squeeze=False
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.955, bottom=0.035, hspace=0.08, wspace=0.03)
    for row_index, row in enumerate(rows):
        panels = (
            (None, None),
            (row["reference"], None),
            (row["reference"], row["baseline"]),
            (row["reference"], row["tta8"]),
        )
        for column_index, (reference, prediction) in enumerate(panels):
            ax = axes[row_index][column_index]
            ax.imshow(row["image"])
            if reference is not None:
                # white dashed reference over a dark halo, so it reads on bright and dark frames
                ax.contour(
                    reference.astype(float), levels=[0.5], colors="#000000", linewidths=1.3, alpha=0.6
                )
                ax.contour(
                    reference.astype(float),
                    levels=[0.5],
                    colors="#ffffff",
                    linewidths=0.7,
                    linestyles="--",
                )
            if prediction is not None:
                if prediction.any():
                    ax.contour(
                        prediction.astype(float),
                        levels=[0.5],
                        colors=style.OKABE_ITO["vermillion"],
                        linewidths=1.0,
                    )
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
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(columns[column_index], fontsize=8)
            if column_index == 0:
                ax.set_ylabel(row["label"], fontsize=5.5)
                ax.annotate(
                    f"Dice {row['before']:.2f} -> {row['after']:.2f}",
                    (0.02, 0.02),
                    xycoords="axes fraction",
                    fontsize=5.0,
                    color="#ffffff",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "linewidth": 0},
                )
    reference_sample = Line2D([], [], color="#ffffff", linestyle="--", linewidth=0.9, label="reference")
    reference_sample.set_path_effects(
        [path_effects.Stroke(linewidth=2.0, foreground="#000000"), path_effects.Normal()]
    )
    prediction_sample = Line2D(
        [], [], color=style.OKABE_ITO["vermillion"], linewidth=1.0, label="prediction"
    )
    fig.legend(
        handles=[reference_sample, prediction_sample],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=6.5,
        columnspacing=1.6,
    )
    style.save(fig, "fig07_qualitative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

