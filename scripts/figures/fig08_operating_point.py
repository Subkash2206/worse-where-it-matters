"""Figure 8 - the operating point: false alarms on polyp-free cecum versus missed polyps.

Every checkpoint is scored on two cohorts that pull in opposite directions:

* **x** - the false-alarm rate on the polyp-free HyperKvasir ``cecum`` frames (1009 frames, no
  polyp at all), as the mean over the 15 checkpoints of the share of frames on which the method
  reports a polyp (``results/false_alarm/tables/<model>/summary.csv``);
* **y** - the share of cases that a **majority of the 15 checkpoints** miss on the two hard external
  collections pooled (CVC-ColonDB 380 + ETIS-Larib 196), from the detection-rule reports
  (``results/detection_rules[/_segformer]/external/<dataset>/misses_vs_tta8.csv``).

One point per method (baseline, TTA-4, TTA-8, TTA-8 + anyview, TTA-8 + lowdet 20/30/40) and one
series per architecture; the arrows follow the sequence baseline -> TTA-8 -> TTA-8 + anyview, which is
the trade the paper discusses.  Nothing is recomputed here - both axes are read from tables that were
written by the evaluation, so the figure cannot drift from the numbers it shows.  The plotted values
are printed, and so are the per-collection miss rates behind the pooled y.

Usage
-----
    python scripts/figures/fig08_operating_point.py          # CPU only, reads existing tables
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import style  # noqa: E402

METHODS = (
    "baseline",
    "tta4",
    "tta8",
    "tta8_anyview",
    "tta8_lowdet_20",
    "tta8_lowdet_30",
    "tta8_lowdet_40",
)
#: marker per method (colour is reserved for the architecture)
METHOD_MARKERS = {
    "baseline": "o",
    "tta4": "s",
    "tta8": "D",
    "tta8_anyview": "X",
    "tta8_lowdet_20": "^",
    "tta8_lowdet_30": "v",
    "tta8_lowdet_40": "P",
}
METHOD_TEXT = {
    "baseline": "baseline",
    "tta4": "TTA-4",
    "tta8": "TTA-8",
    "tta8_anyview": "anyview",
    "tta8_lowdet_20": "lowdet 0.20",
    "tta8_lowdet_30": "lowdet 0.30",
    "tta8_lowdet_40": "lowdet 0.40",
}
#: the sequence the arrows follow
ARROWS = (("baseline", "tta8"), ("tta8", "tta8_anyview"))
MISS_DATASETS = ("cvc_colondb", "etis_larib")
#: (model key, false-alarm table, detection-rule report root, colour, label)
SERIES = (
    (
        "unet",
        style.RESULTS / "false_alarm" / "tables" / "unet" / "summary.csv",
        style.RESULTS / "detection_rules" / "external",
        style.MODEL_COLOURS["unet_resnet50"],
        "UNet",
    ),
    (
        "segformer",
        style.RESULTS / "false_alarm" / "tables" / "segformer" / "summary.csv",
        style.RESULTS / "detection_rules_segformer" / "external",
        style.MODEL_COLOURS["segformer_mit_b2"],
        "SegFormer",
    ),
)
#: label anchor per (model, method), in offset points, chosen so that each label sits next to
#: its *own* marker: anyview (X) and lowdet 0.20 (up-triangle) are 5 % apart on the same y, so
#: one is labelled above and the other below instead of to the side (which made them read swapped)
LABEL_OFFSETS = {
    ("unet", "baseline"): (10, 8, "left"),
    ("unet", "tta4"): (8, 4, "left"),
    ("unet", "tta8"): (2, 13, "center"),
    ("unet", "tta8_anyview"): (-6, -14, "right"),
    ("unet", "tta8_lowdet_20"): (0, 13, "center"),
    ("unet", "tta8_lowdet_30"): (8, 4, "left"),
    ("unet", "tta8_lowdet_40"): (-8, -13, "right"),
    ("segformer", "baseline"): (10, 10, "left"),
    ("segformer", "tta8"): (-8, 8, "right"),
    ("segformer", "tta8_anyview"): (-6, -14, "right"),
}
#: in the SegFormer panel the seven points sit within 1 % of each other, so only the arrow
#: sequence is labelled there; every value is printed above and readable from the table
LABELLED = {
    "unet": METHODS,
    "segformer": ("baseline", "tta8", "tta8_anyview"),
}
#: only the UNet panel gets the baseline -> TTA-8 -> anyview arrows; the SegFormer sequence is
#: flat against the shared y axis (the point the figure makes) and arrows would just clutter it
ARROW_MODELS = ("unet",)


def collect() -> pd.DataFrame:
    """One row per (model, method): false-alarm rate [%] and pooled majority-miss rate [%]."""
    rows: list[dict] = []
    for key, false_alarm_path, misses_root, colour, label in SERIES:
        if not false_alarm_path.is_file():
            print(
                f"missing {false_alarm_path}; run scripts/false_alarm_report.py first",
                file=sys.stderr,
            )
            continue
        false_alarm = pd.read_csv(false_alarm_path).set_index("method")
        misses = {}
        for dataset in MISS_DATASETS:
            path = misses_root / dataset / "misses_vs_tta8.csv"
            if path.is_file():
                misses[dataset] = pd.read_csv(path).set_index("method")
        for method in METHODS:
            if method not in false_alarm.index:
                continue
            available = [block for block in misses.values() if method in block.index]
            missed = sum(float(block.loc[method, "units_majority_missed"]) for block in available)
            cases = sum(float(block.loc[method, "n_units"]) for block in available)
            row = {
                "model": key,
                "label": label,
                "colour": colour,
                "method": method,
                "false_alarm_pct": float(false_alarm.loc[method, "false_alarm_rate_mean"]) * 100.0,
                "missed_cases": missed,
                "miss_cases": cases,
                "miss_pct": 100.0 * missed / cases if cases else float("nan"),
            }
            for dataset in MISS_DATASETS:
                block = misses.get(dataset)
                row[f"miss_pct_{dataset}"] = (
                    100.0
                    * float(block.loc[method, "units_majority_missed"])
                    / float(block.loc[method, "n_units"])
                    if block is not None and method in block.index
                    else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _assert_label_placement(figure, axes, table, annotations) -> None:
    """Each label must be tied to its own plotted value: assert the anchor and no overlaps.

    The SegFormer points sit within 1 % of each other, so a label placed "next to" the cluster
    cannot be attributed by proximity alone.  Every label therefore carries a thin leader line
    that starts at its own marker, and this check asserts (a) the label's anchor is exactly the
    plotted (x, y) of the method it names, and (b) no two label boxes overlap.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    failures: list[bool] = []
    print("\nlabel placement check (anchor == plotted value, leader on the marker, no overlap):")
    for ax, key in zip(axes, ("unet", "segformer")):
        block = table.loc[table["model"] == key].set_index("method")
        boxes: dict[str, object] = {}
        for method, annotation in annotations.get(key, {}).items():
            expected = (
                float(block.loc[method, "false_alarm_pct"]),
                float(block.loc[method, "miss_pct"]),
            )
            anchored = bool(np.allclose(annotation.xy, expected))
            leader = annotation.arrow_patch is not None
            boxes[method] = annotation.get_window_extent(renderer)
            ok = anchored and leader
            failures.append(ok)
            print(
                f"  {key:<10} {method:<16s} anchor ({annotation.xy[0]:6.2f}, "
                f"{annotation.xy[1]:5.2f}) == plotted ({expected[0]:6.2f}, {expected[1]:5.2f}) "
                f": {anchored}, leader: {leader}"
            )
        methods = list(boxes)
        for first in range(len(methods)):
            for second in range(first + 1, len(methods)):
                left, right = methods[first], methods[second]
                overlap = boxes[left].intersection(boxes[left], boxes[right])
                collided = overlap is not None and min(overlap.width, overlap.height) > 0.5
                failures.append(not collided)
                if collided:
                    print(f"  {key:<10} label boxes overlap: {left} vs {right}")
    if not all(failures):
        raise AssertionError("a fig08 label is not anchored to its own value or overlaps another")


def _assert_arrow_targets(table, arrow_targets) -> None:
    """Each drawn arrow must end on the marker of the method it names."""
    block = table.loc[table["model"] == "unet"].set_index("method")
    print("\narrow endpoints check:")
    for name, (x, y) in arrow_targets.items():
        target = name.split("->")[1]
        expected = (
            float(block.loc[target, "false_alarm_pct"]),
            float(block.loc[target, "miss_pct"]),
        )
        ok = bool(np.allclose((x, y), expected))
        print(
            f"  {name:<24s} ends at ({x:6.2f}, {y:5.2f}) == {target} marker "
            f"({expected[0]:6.2f}, {expected[1]:5.2f}) : {'ok' if ok else 'WRONG'}"
        )
        if not ok:
            raise AssertionError(f"the arrow {name} does not end on the {target} marker")


def main() -> int:
    style.apply()
    table = collect()
    if table.empty:
        print("no false-alarm tables found", file=sys.stderr)
        return 2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print("plotted values (x = false alarms on polyp-free cecum, y = majority-missed polyps):")
    print(
        table[
            [
                "label",
                "method",
                "false_alarm_pct",
                "missed_cases",
                "miss_cases",
                "miss_pct",
                "miss_pct_cvc_colondb",
                "miss_pct_etis_larib",
            ]
        ]
        .rename(
            columns={
                "label": "model",
                "false_alarm_pct": "false_alarm_mean_%",
                "missed_cases": "majority_missed",
                "miss_cases": "cases",
                "miss_pct": "majority_miss_pooled_%",
                "miss_pct_cvc_colondb": "colondb_%",
                "miss_pct_etis_larib": "etis_%",
            }
        )
        .round(2)
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharex=True, sharey=True)
    annotations: dict[str, dict[str, object]] = {}
    arrow_targets: dict[str, tuple[float, float]] = {}
    for ax, (key, label, colour) in zip(
        axes, [(key, label, colour) for key, _, _, colour, label in SERIES]
    ):
        block = table.loc[table["model"] == key].set_index("method")
        block = block.loc[[method for method in METHODS if method in block.index]]
        annotations[key] = {}
        for method in block.index:
            row = block.loc[method]
            ax.plot(
                row["false_alarm_pct"],
                row["miss_pct"],
                linestyle="none",
                marker=METHOD_MARKERS[method],
                markersize=7.0 if method == "baseline" else 6.0,
                markerfacecolor="white" if method == "baseline" else colour,
                markeredgecolor=colour,
                markeredgewidth=1.0,
                zorder=3,
            )
            dx, dy, alignment = LABEL_OFFSETS.get((key, str(method)), (0, 8, "center"))
            if method in LABELLED.get(key, METHODS):
                annotations[key][str(method)] = ax.annotate(
                    METHOD_TEXT[method],
                    (row["false_alarm_pct"], row["miss_pct"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha=alignment,
                    fontsize=5.6,
                    color="#303030",
                    zorder=4,
                    bbox={"facecolor": "white", "alpha": 0.72, "pad": 0.5, "linewidth": 0},
                    # thin leader: the label is attached to *its* marker, not merely near it
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#606060",
                        "linewidth": 0.5,
                        "shrinkA": 0.5,
                        "shrinkB": 2.5,
                        "alpha": 0.85,
                    },
                )
        if key in ARROW_MODELS:
            for source, target in ARROWS:
                if source not in block.index or target not in block.index:
                    continue
                x, y = (
                    float(block.loc[target, "false_alarm_pct"]),
                    float(block.loc[target, "miss_pct"]),
                )
                ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=(
                        block.loc[source, "false_alarm_pct"],
                        block.loc[source, "miss_pct"],
                    ),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": colour,
                        "linewidth": 1.0,
                        "shrinkA": 8.0,
                        "shrinkB": 1.5,  # the head touches the marker it points at
                        "alpha": 0.85,
                    },
                    zorder=2,
                )
                arrow_targets[f"{key}:{source}->{target}"] = (x, y)
        ax.set_xlim(float(table["false_alarm_pct"].min()) - 6.0, 104.0)
        ax.set_title(f"({'ab'[key == 'segformer']}) {label}", loc="left")
        ax.set_xlabel("false alarms on polyp-free\ncecum frames [%] (mean over 15 checkpoints)")
    # one shared y scale for both panels: the SegFormer points sit on the floor of the same axis,
    # which is exactly the comparison the figure is about
    peak = float(table["miss_pct"].max())
    axes[0].set_ylim(-0.10 * peak - 0.02, peak * 1.22 + 0.02)
    axes[0].set_ylabel(
        "missed by a majority of the checkpoints [%]\n(ColonDB + ETIS-Larib, 576 cases)"
    )
    figure_title = "TTA trades false alarms for missed polyps; the detection rules trade them back"
    fig.suptitle(figure_title, fontsize=7.0, y=1.02)
    _assert_label_placement(fig, axes, table, annotations)
    _assert_arrow_targets(table, arrow_targets)
    method_handles = [
        Line2D(
            [],
            [],
            color="#505050",
            linestyle="none",
            marker=METHOD_MARKERS[method],
            markersize=5.0,
            label=METHOD_TEXT[method],
        )
        for method in METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        fontsize=5.8,
        title="method (marker, both panels); arrows in (a): baseline -> TTA-8 -> anyview",
        title_fontsize=5.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    style.save(fig, "fig08_operating_point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

