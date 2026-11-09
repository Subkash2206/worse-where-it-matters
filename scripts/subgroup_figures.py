"""Figures for the subgroup analysis: small polyps that TTA newly misses (CPU only).

Reads the per-case CSVs of an existing CVC grid, finds cases that the baseline detects and a
TTA method misses (restricted to the smallest size tertile by default), and renders the
reference next to both predictions.  The predictions are recomputed **on the CPU** through
the same runner code path the GPU evaluation used (same preprocessing, same TTA views, same
thresholds), so a figure shows exactly what the CSV row refers to.

Usage
-----
    python scripts/subgroup_figures.py --model cvc_clinicdb/unet_resnet50 \
        --method tta8 --bin small --count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from melba.config import load_config  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.eval import per_case_path  # noqa: E402
from melba.eval.runner import resample_probability_map  # noqa: E402
from melba.eval.subgroups import add_size_bins, lesion_size_table  # noqa: E402
from melba.inference import (  # noqa: E402
    aggregate_probabilities,
    methods_from_config,
    predict_per_view,
)
from melba.models import build_model, load_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="cvc_clinicdb/unet_resnet50")
    parser.add_argument("--method", default="tta8")
    parser.add_argument("--bin", dest="size_bin", default="small")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def _transitions(
    config: Any,
    sizes: pd.DataFrame,
    method: str,
    size_bin: str,
    folds: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """(case, fold, seed) where the baseline detects and ``method`` misses, in one bin."""
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            path = per_case_path(config, fold, seed)
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            flags = (
                frame.loc[frame["metric"] == "dice", ["case_id", "method", "empty_flag"]]
                .drop_duplicates(["case_id", "method"])
                .pivot(index="case_id", columns="method", values="empty_flag")
            )
            if "baseline" not in flags or method not in flags:
                continue
            for case_id, row in flags.iterrows():
                if row["baseline"] == "pred_empty" or row[method] != "pred_empty":
                    continue
                rows.append({"case_id": str(case_id), "fold": fold, "seed": seed})
    if not rows:
        return []
    table = pd.DataFrame(rows)
    probe = table.assign(method="baseline", metric="dice", value=0.0, group_id=table["case_id"])
    enriched, _ = add_size_bins(probe, sizes)
    selected = enriched.loc[enriched["size_bin"] == size_bin]
    selected = selected.sort_values("gt_pixels").drop_duplicates("case_id")
    return selected.head(3).to_dict("records")


def _predict(model, image: torch.Tensor, method, device: torch.device, target_shape) -> np.ndarray:
    """Per-view TTA -> native-grid mean probability, exactly like the evaluation runner."""
    views = predict_per_view(model, image, device, method, activation="sigmoid", amp=False)
    native = np.stack([resample_probability_map(view_map, target_shape)[0] for view_map in views])
    return aggregate_probabilities(native[:, None], method.aggregator)[0]


def _render(
    path: Path,
    *,
    image: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    treatment: np.ndarray,
    title: str,
    method_name: str,
) -> None:
    """Two rows (overlay / map) x three columns (reference, baseline, TTA)."""
    rgb = np.clip(image.transpose(1, 2, 0), 0.0, 1.0)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), constrained_layout=True)
    panels = (
        ("reference", None),
        ("baseline (detected)", baseline),
        (f"{method_name} (missed)", treatment),
    )
    for column, (label, probability) in enumerate(panels):
        axes[0, column].imshow(rgb)
        if target.any():
            axes[0, column].contour(target, levels=[0.5], colors="yellow", linewidths=1.4)
        if probability is not None:
            axes[0, column].contour(
                probability, levels=[0.5], colors="cyan", linewidths=1.2, linestyles="--"
            )
            axes[0, column].set_title(
                f"{label} - predicted area {int((probability > 0.5).sum())} px"
            )
            axes[1, column].imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
        else:
            axes[0, column].set_title(f"{label} - area {int(target.sum())} px")
            axes[1, column].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(_REPO_ROOT / "configs" / f"{args.model}.yaml", {"device": "cpu"})
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "results" / "subgroups" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = lesion_size_table(config, folds=list(args.folds))
    candidates = _transitions(
        config, sizes, args.method, args.size_bin, list(args.folds), list(args.seeds)
    )
    if not candidates:
        print("no transition found for the requested bin/method")
        return 1
    print(f"{len(candidates)} case(s) to render: {[c['case_id'] for c in candidates]}")

    device = torch.device("cpu")
    methods = {method.name: method for method in methods_from_config(config)}
    for record in candidates:
        case_id = str(record["case_id"])
        fold, seed = int(record["fold"]), int(record["seed"])
        dataset = build_dataset(config, "test", fold=fold)
        index = next(
            (
                position
                for position, item in enumerate(dataset.items)
                if str(item.case_id) == case_id
            ),
            None,
        )
        if index is None:  # pragma: no cover - the CSV came from this fold
            print(f"  {case_id}: not in the fold {fold} test split, skipped")
            continue
        sample = dataset[index]
        image = sample["image"].unsqueeze(0)
        target = sample["mask_native"][0].numpy()
        checkpoint = config.resolve(
            str(config.require("model.checkpoint")).format(fold=fold, seed=seed)
        )
        model = build_model(config)
        load_checkpoint(model, checkpoint, device)
        model = model.to(device).eval()
        baseline = _predict(model, image, methods["baseline"], device, target.shape)
        treatment = _predict(model, image, methods[args.method], device, target.shape)
        _render(
            output_dir / f"{case_id}_fold{fold}_seed{seed}_baseline_vs_{args.method}.png",
            image=image[0].numpy(),
            target=target,
            baseline=baseline,
            treatment=treatment,
            title=(
                f"{case_id} (fold {fold}, seed {seed}) - reference "
                f"{int(record['gt_pixels'])} px, {args.size_bin} tertile"
            ),
            method_name=args.method,
        )
        print(
            f"  {case_id}: baseline {int((baseline > 0.5).sum())} px, "
            f"{args.method} {int((treatment > 0.5).sum())} px -> figure written"
        )
    print(f"figures in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

