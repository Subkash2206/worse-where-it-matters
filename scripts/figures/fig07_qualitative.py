"""Figure 7 - qualitative grid: what TTA changes at the extremes.

Panels are chosen by a rule, never by eye.  The evaluation wrote float16 probability maps for the
first Kvasir-SEG run (``evaluation/maps/fold0_seed0/``, methods ``baseline`` and ``tta4`` - the
deepest TTA available there), so the rule is applied to those ten mapped cases:

* "newly missed": baseline predicts a polyp (Dice >= 0.50 against the reference) and TTA-4 predicts
  nothing at the 0.5 threshold, ranked by the reference area (smallest first) - the small lesions
  TTA silences;
* "boundary fixed": baseline HD95 is among the two worst of the ten and TTA-4 lowers it, ranked by
  the absolute margin - the large boundary errors TTA repairs.

Each panel shows the image, the reference contour, the baseline contour and the TTA-4 contour, with
the Dice of both methods in the first column.  Reference masks are read from the dataset the
external config points at; nothing is written into the dataset directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import style  # noqa: E402

sys.path.insert(0, str(style.REPO_ROOT / "src"))

MAPS = style.RESULTS / "external_kvasir_seg" / "evaluation" / "maps" / "fold0_seed0"
CONFIG = style.REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
N_NEWLY_MISSED = 3
N_FIXED = 2


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def _hd95(pred: np.ndarray, reference: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt

    if not pred.any() or not reference.any():
        return float("nan")
    pred_surface, ref_surface = _surface(pred), _surface(reference)
    to_ref = distance_transform_edt(~ref_surface)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface)[ref_surface]
    return float(np.percentile(np.concatenate([to_ref, to_pred]), 95))


def _dice(pred: np.ndarray, reference: np.ndarray) -> float:
    total = pred.sum() + reference.sum()
    if total == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, reference).sum() / total)


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Image, reference mask and the two probability maps of one case, on the map grid."""
    from PIL import Image

    from melba.config import load_config

    maps: dict[str, np.ndarray] = {}
    for method in ("baseline", "tta4"):
        path = MAPS / f"{case_id}__{method}__mean.npy"
        if not path.is_file():
            return None
        maps[method] = np.load(path).astype(np.float32)
    shape = maps["baseline"].shape
    entry = load_config(CONFIG).get("external.datasets")["kvasir_seg"]
    image_path = next(Path(entry["images_dir"]).glob(f"{case_id}.*"), None)
    mask_path = next(Path(entry["masks_dir"]).glob(f"{case_id}.*"), None)
    if image_path is None or mask_path is None:
        return None
    image = np.asarray(Image.open(image_path).convert("RGB").resize(shape[::-1], Image.BILINEAR))
    reference = np.asarray(Image.open(mask_path).convert("L").resize(shape[::-1], Image.NEAREST)) > 0
    return image, reference, maps["baseline"], maps["tta4"]


def select(rows: list[dict]) -> list[tuple[dict, str]]:
    """The documented rule: three newly missed small polyps, two repaired boundary errors."""
    newly = [row for row in rows if row["baseline_dice"] >= 0.5 and row["tta4_empty"]]
    newly.sort(key=lambda row: row["reference_area"])
    worst = sorted(rows, key=lambda row: -row["baseline_hd95"])[:2]
    fixed = [
        row
        for row in worst
        if not row["tta4_empty"] and np.isfinite(row["tta4_hd95"]) and row["tta4_hd95"] < row["baseline_hd95"]
    ]
    if len(fixed) < N_FIXED:
        margin = sorted(
            (row for row in rows if np.isfinite(row["baseline_hd95"]) and np.isfinite(row["tta4_hd95"])),
            key=lambda row: -(row["baseline_hd95"] - row["tta4_hd95"]),
        )
        fixed = margin[:N_FIXED]
    return [(row, "newly missed") for row in newly[:N_NEWLY_MISSED]] + [
        (row, "boundary fixed") for row in fixed[:N_FIXED]
    ]


def main() -> int:
    style.apply()
    if not MAPS.is_dir():
        print(f"no probability maps in {MAPS}", file=sys.stderr)
        return 2
    case_ids = sorted({path.name.split("__")[0] for path in MAPS.glob("*__baseline__mean.npy")})
    rows: list[dict] = []
    cache: dict[str, tuple] = {}
    for case_id in case_ids:
        loaded = load_case(case_id)
        if loaded is None:
            continue
        image, reference, baseline, tta4 = loaded
        base_mask, tta_mask = baseline > 0.5, tta4 > 0.5
        cache[case_id] = (image, reference, base_mask, tta_mask)
        rows.append(
            {
                "case_id": case_id,
                "reference_area": float(reference.sum()),
                "baseline_dice": _dice(base_mask, reference),
                "tta4_dice": _dice(tta_mask, reference),
                "baseline_hd95": _hd95(base_mask, reference),
                "tta4_hd95": _hd95(tta_mask, reference),
                "tta4_empty": not tta_mask.any(),
            }
        )
    if not rows:
        print("no mapped case could be read", file=sys.stderr)
        return 2
    chosen = select(rows)
    if not chosen:
        print("the selection rule found no case to draw", file=sys.stderr)
        return 2

    import matplotlib.pyplot as plt

    columns = ("image", "reference", "baseline", "TTA-4")
    fig, axes = plt.subplots(
        len(chosen), len(columns), figsize=(1.9 * len(columns), 1.9 * len(chosen)), squeeze=False
    )
    for row_index, (row, tag) in enumerate(chosen):
        image, reference, base_mask, tta_mask = cache[row["case_id"]]
        panels = (
            (image.astype(np.uint8), None, None),
            (image.astype(np.uint8), reference, None),
            (image.astype(np.uint8), reference, base_mask),
            (image.astype(np.uint8), reference, tta_mask),
        )
        for column_index, (picture, reference_mask, prediction) in enumerate(panels):
            ax = axes[row_index][column_index]
            ax.imshow(picture)
            if reference_mask is not None:
                ax.contour(
                    reference_mask, levels=[0.5], colors="#ffffff", linewidths=0.7, linestyles="--"
                )
            if prediction is not None and prediction.any():
                ax.contour(prediction, levels=[0.5], colors="#e74c3c", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row_index == 0:
                ax.set_title(columns[column_index])
        axes[row_index][0].set_ylabel(f"{tag}\n{row['case_id'][:10]}", fontsize=5.5)
        axes[row_index][0].annotate(
            f"Dice {row['baseline_dice']:.2f} -> {row['tta4_dice']:.2f}",
            (0.02, 0.02),
            xycoords="axes fraction",
            fontsize=5.0,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "linewidth": 0},
        )
    handles = [
        plt.Line2D([], [], color="#ffffff", linestyle="--", linewidth=0.8, label="reference"),
        plt.Line2D([], [], color="#e74c3c", linewidth=0.9, label="prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    style.save(fig, "fig07_qualitative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

