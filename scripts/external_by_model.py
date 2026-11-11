"""Per-architecture analysis of the external polyp validation (CPU only, read-only).

The external aggregate pools the 30 checkpoints (15 UNet + 15 Swin) into one average, which
hides the fact that the two architectures behave differently outside the training
distribution.  This script splits every external table by architecture:

1. Dice / boundary-F1 / HD95 per method (mean +- std over cases, and % change vs baseline),
   with the paired per-case delta, its bootstrap CI, the Wilcoxon test and Holm correction.
2. Missed detections counted **per checkpoint** (not after averaging): total misses over the
   15 checkpoints of one architecture, the same as a rate per checkpoint, the number of cases
   where a majority of checkpoints miss, and the McNemar test on the per-checkpoint pairs
   (plus the conservative case-level majority test).
3. Internal-vs-external per architecture: the baseline Dice drop from CVC-ClinicDB to each
   external collection and whether the TTA gain is bigger, smaller or unchanged outside,
   absolute and relative.
4. The tta4 -> tta8 increment, internal vs external.
5. Lesion-size tertiles per dataset and architecture: Dice and miss rate per bin, the paired
   delta per bin (Holm), and a Kruskal-Wallis test of whether the gain depends on the size.

Usage
-----
    python scripts/external_by_model.py                    # all three datasets
    python scripts/external_by_model.py --datasets kvasir_seg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from melba.config import ConfigError, load_config  # noqa: E402
from melba.eval.subgroups import (  # noqa: E402
    SIZE_BIN_LABELS,
    add_size_bins,
    lesion_size_table,
)

CONFIG_PATH = _REPO_ROOT / "configs" / "external" / "polyp_external.yaml"
RESULTS = _REPO_ROOT / "results"
OUTPUT = RESULTS / "external_by_model"
DATASETS = ("kvasir_seg", "cvc_colondb", "etis_larib")
FAMILIES = {
    "unet_resnet50": "cvc_clinicdb_unet_resnet50",
    "swinunetr_2d": "cvc_clinicdb_swinunetr_2d",
    # ImageNet-pretrained transformer arm: same 5 folds x 3 seeds, same TTA methods
    "segformer_mit_b2": "cvc_clinicdb_segformer_mit_b2",
}
METHODS = ("baseline", "tta2", "tta4", "tta8")
TREATMENTS = ("tta2", "tta4", "tta8")
METRICS = ("dice", "boundary_f1", "hd95")
HIGHER_IS_BETTER = {"dice": True, "boundary_f1": True, "hd95": False}
N_BOOTSTRAP = 10000
SEED = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- helpers
def holm(p_values: Iterable[float]) -> list[float]:
    """Holm step-down adjustment, returned in the input order."""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, ci: float = 0.95, seed: int = SEED) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    means = finite[draws].mean(axis=1)
    low, high = np.percentile(means, [100.0 * (1 - ci) / 2.0, 100.0 * (1 + ci) / 2.0])
    return float(low), float(high)


def wilcoxon_test(deltas: np.ndarray) -> tuple[float, float]:
    """p-value and rank-biserial effect size of a paired two-sided Wilcoxon test."""
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(values, 0.0):
        return 1.0, float("nan")
    try:
        result = sps.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
    except ValueError:  # pragma: no cover - degenerate input
        return 1.0, float("nan")
    ranks = sps.rankdata(np.abs(values))
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    total = values.size * (values.size + 1) / 2.0
    return float(result.pvalue), float((positive - negative) / total)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for the discordant counts ``b`` and ``c``."""
    total = int(b) + int(c)
    if total == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(int(b), int(c)), total, 0.5)))


def load_runs(directory: Path) -> pd.DataFrame:
    """All per-case CSVs below ``directory`` (one sub-directory per architecture after the
    output-path fix; the older layout had them directly in ``evaluation/``)."""
    paths = sorted(directory.rglob("*_per_case.csv"))
    if not paths:
        raise ConfigError(f"no per-case results in {directory}")
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["family"] = frame["run_id"].str.split("__").str[-1].str.split("-").str[0]
    return frame


def case_means(frame: pd.DataFrame, methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """One row per (case, method, metric): the mean over the checkpoints of one architecture."""
    subset = frame.loc[
        frame["metric"].isin(METRICS) & frame["method"].isin(methods),
        ["case_id", "method", "metric", "value"],
    ]
    return subset.groupby(["case_id", "method", "metric"], as_index=False)["value"].mean()


def metric_table(averaged: pd.DataFrame, dataset: str, family: str) -> pd.DataFrame:
    """Mean/std over cases per method plus the % change against baseline."""
    grouped = (
        averaged.groupby(["method", "metric"])["value"]
        .agg(n_cases="count", mean="mean", std="std")
        .reset_index()
    )
    baseline = grouped.loc[grouped["method"] == "baseline"].set_index("metric")
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        reference = baseline.loc[row["metric"], "mean"] if row["metric"] in baseline.index else np.nan
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "method": row["method"],
                "metric": row["metric"],
                "n_cases": int(row["n_cases"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "baseline_mean": float(reference),
                "change_pct": 100.0 * (float(row["mean"]) - float(reference)) / float(reference)
                if np.isfinite(reference) and reference
                else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["method"] = pd.Categorical(frame["method"], categories=list(METHODS), ordered=True)
    frame["metric"] = pd.Categorical(frame["metric"], categories=list(METRICS), ordered=True)
    return frame.sort_values(["metric", "method"]).reset_index(drop=True)


def paired_table(averaged: pd.DataFrame, dataset: str, family: str, n_bootstrap: int) -> pd.DataFrame:
    """Paired per-case deltas (treatment - baseline), CI, Wilcoxon and Holm per dataset/family."""
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        values = averaged.loc[averaged["metric"] == metric].pivot(
            index="case_id", columns="method", values="value"
        )
        if "baseline" not in values:
            continue
        for treatment in TREATMENTS:
            if treatment not in values:
                continue
            pair = values[["baseline", treatment]].dropna()
            deltas = (pair[treatment] - pair["baseline"]).to_numpy(dtype=float)
            low, high = bootstrap_ci(deltas, n_bootstrap)
            p_value, effect = wilcoxon_test(deltas)
            improved = deltas if HIGHER_IS_BETTER[metric] else -deltas
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "metric": metric,
                    "comparison": f"{treatment} vs baseline",
                    "direction": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                    "n_pairs": int(deltas.size),
                    "control_mean": float(pair["baseline"].mean()),
                    "treatment_mean": float(pair[treatment].mean()),
                    "delta_mean": float(np.mean(deltas)),
                    "delta_median": float(np.median(deltas)),
                    "delta_rel_pct": 100.0 * float(np.mean(deltas)) / float(pair["baseline"].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                    "rank_biserial": effect,
                    "preferred": treatment if float(np.mean(improved)) > 0 else "baseline",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_value_adjusted"] = holm(frame["p_value"].tolist())
    return frame


def miss_table(runs: pd.DataFrame, dataset: str, family: str) -> pd.DataFrame:
    """Misses counted **per checkpoint**, majority-miss cases and the two McNemar tests."""
    subset = runs.loc[
        (runs["metric"] == "dice") & runs["method"].isin(METHODS),
        ["case_id", "run_id", "method", "missed_detection"],
    ].copy()
    subset["missed_detection"] = subset["missed_detection"].astype(float)
    per_run = subset.groupby(["method", "run_id"])["missed_detection"].sum()
    per_case = subset.groupby(["case_id", "method"])["missed_detection"].agg(["sum", "count"])
    per_case["rate"] = per_case["sum"] / per_case["count"]
    wide = subset.pivot_table(
        index=["case_id", "run_id"], columns="method", values="missed_detection"
    )
    majority = (per_case["rate"] > 0.5).unstack("method")
    any_miss = (per_case["sum"] > 0).unstack("method")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        column = per_run.loc[method]
        row: dict[str, Any] = {
            "dataset": dataset,
            "family": family,
            "method": method,
            "n_cases": int(subset.loc[subset["method"] == method, "case_id"].nunique()),
            "n_checkpoints": int(column.size),
            "misses_total": int(column.sum()),
            "misses_per_checkpoint": float(column.mean()),
            "missed_cases_min": int(column.min()),
            "missed_cases_max": int(column.max()),
            "cases_missed_by_any_checkpoint": int(any_miss[method].sum()),
            "cases_majority_missed": int(majority[method].sum()),
        }
        if method != "baseline":
            pairs = wide[["baseline", method]].dropna()
            new_misses = int(((pairs["baseline"] == 0) & (pairs[method] == 1)).sum())
            recovered = int(((pairs["baseline"] == 1) & (pairs[method] == 0)).sum())
            became = int((~majority["baseline"] & majority[method]).sum())
            left = int((majority["baseline"] & ~majority[method]).sum())
            row.update(
                {
                    "checkpoint_pairs": int(pairs.shape[0]),
                    "new_misses": new_misses,
                    "recovered": recovered,
                    "mcnemar_p_checkpoint": mcnemar_exact(new_misses, recovered),
                    "cases_became_majority_miss": became,
                    "cases_left_majority_miss": left,
                    "mcnemar_p_case_majority": mcnemar_exact(became, left),
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("mcnemar_p_checkpoint", "mcnemar_p_case_majority"):
        frame[column] = frame[column].fillna(1.0)
    adjusted = holm(frame["mcnemar_p_checkpoint"].tolist())
    frame["mcnemar_p_checkpoint_adjusted"] = adjusted
    frame.loc[frame["method"] == "baseline", "mcnemar_p_checkpoint_adjusted"] = np.nan
    frame["method"] = pd.Categorical(frame["method"], categories=list(METHODS), ordered=True)
    return frame.sort_values("method").reset_index(drop=True)


def internal_case_dice(directory: Path) -> pd.Series:
    """Per-case internal Dice per method, averaged over that architecture's checkpoints."""
    runs = load_runs(directory)
    subset = runs.loc[(runs["metric"] == "dice") & runs["method"].isin(METHODS)]
    return subset.groupby(["case_id", "method"])["value"].mean()


def comparison_table(
    internal: pd.Series, external: pd.Series, dataset: str, family: str
) -> pd.DataFrame:
    """Internal-vs-external Dice per method, the TTA gain on both sides and tta4 -> tta8."""
    rows: list[dict[str, Any]] = []
    inner = internal.groupby(level="method").mean()
    outer = external.groupby(level="method").mean()
    inner_n = internal.groupby(level="method").size()
    outer_n = external.groupby(level="method").size()
    if "baseline" not in inner.index or "baseline" not in outer.index:
        return pd.DataFrame()
    for method in METHODS:
        if method not in inner.index or method not in outer.index:
            continue
        gain_internal = float(inner[method] - inner["baseline"])
        gain_external = float(outer[method] - outer["baseline"])
        difference = gain_external - gain_internal
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "method": method,
                "n_cases_internal": int(inner_n[method]),
                "n_cases_external": int(outer_n[method]),
                "dice_internal": float(inner[method]),
                "dice_external": float(outer[method]),
                "drop_abs": float(outer[method] - inner[method]),
                "drop_rel_pct": 100.0 * float(outer[method] - inner[method]) / float(inner[method]),
                "gain_vs_baseline_internal": gain_internal,
                "gain_vs_baseline_external": gain_external,
                "gain_rel_pct_internal": 100.0 * gain_internal / float(inner["baseline"]),
                "gain_rel_pct_external": 100.0 * gain_external / float(outer["baseline"]),
                "gain_difference": difference,
                "verdict": (
                    "larger externally"
                    if difference > 0.002
                    else "smaller externally"
                    if difference < -0.002
                    else "within 0.002"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if "tta4" in inner.index and "tta8" in inner.index:
        increment = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "family": family,
                    "method": "tta4 -> tta8",
                    "n_cases_internal": int(inner_n["tta4"]),
                    "n_cases_external": int(outer_n["tta8"]),
                    "dice_internal": float(inner["tta8"] - inner["tta4"]),
                    "dice_external": float(outer["tta8"] - outer["tta4"]),
                    "drop_abs": float(outer["tta8"] - outer["tta4"]) - float(inner["tta8"] - inner["tta4"]),
                    "drop_rel_pct": np.nan,
                    "gain_vs_baseline_internal": float(inner["tta8"] - inner["tta4"]),
                    "gain_vs_baseline_external": float(outer["tta8"] - outer["tta4"]),
                    "gain_rel_pct_internal": pd.NA,
                    "gain_rel_pct_external": pd.NA,
                    "gain_difference": float(outer["tta8"] - outer["tta4"])
                    - float(inner["tta8"] - inner["tta4"]),
                    "verdict": "increment (Dice)",
                }
            ]
        )
        frame = pd.concat([frame, increment], ignore_index=True)
    return frame


def sizes_for(dataset: str, cache_dir: Path) -> pd.DataFrame:
    """Reference polyp area per external case; cached so the masks are read only once."""
    path = cache_dir / f"sizes_{dataset}.csv"
    if path.is_file():
        sizes = pd.read_csv(path)
        sizes["case_id"] = sizes["case_id"].astype(str)
        return sizes
    config = load_config(CONFIG_PATH)
    entry = config.get("external.datasets")[dataset]
    settings = config.section("dataset")
    resolved = load_config(
        CONFIG_PATH,
        {
            "dataset.kind": "external_polyp",
            "dataset.name": dataset,
            "dataset.images_dir": entry["images_dir"],
            "dataset.masks_dir": entry["masks_dir"],
            "dataset.image_extensions": settings.get("image_extensions"),
            "dataset.mask_threshold": settings.get("mask_threshold"),
            "dataset.image_size": settings.get("image_size"),
            "dataset.spacing_mm": settings.get("spacing_mm"),
            "dataset.split.fold": 0,
            "evaluation.fold": 0,
            "evaluation.split": str(config.get("evaluation.split", "test")),
            "runtime.progress": False,
        },
    )
    sizes = lesion_size_table(resolved, folds=[0])
    sizes["case_id"] = sizes["case_id"].astype(str)
    sizes.to_csv(path, index=False)
    return sizes


def majority_miss_per_case(runs: pd.DataFrame) -> pd.DataFrame:
    """Fraction of an architecture's checkpoints that miss the polyp, per (case, method)."""
    subset = runs.loc[
        (runs["metric"] == "dice") & runs["method"].isin(METHODS),
        ["case_id", "method", "missed_detection"],
    ].copy()
    subset["case_id"] = subset["case_id"].astype(str)
    rates = subset.groupby(["case_id", "method"])["missed_detection"].mean()
    return rates.unstack("method")


def subgroup_tables(
    averaged: pd.DataFrame,
    runs: pd.DataFrame,
    sizes: pd.DataFrame,
    dataset: str,
    family: str,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dice per size bin, the paired gain per bin (Holm) and the size-interaction test."""
    enriched, cutoffs = add_size_bins(averaged, sizes)
    enriched = enriched.loc[enriched["size_bin"].notna()]
    misses = majority_miss_per_case(runs)
    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    for metric in ("dice", "hd95"):
        values = enriched.loc[enriched["metric"] == metric]
        wide = values.pivot_table(
            index=["case_id", "size_bin"], columns="method", values="value", observed=True
        )
        for bin_label in SIZE_BIN_LABELS:
            if bin_label not in wide.index.get_level_values("size_bin"):
                continue
            block = wide.xs(bin_label, level="size_bin")
            for method in METHODS:
                if method not in block:
                    continue
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "metric": metric,
                        "size_bin": bin_label,
                        "method": method,
                        "n_cases": int(block[method].notna().sum()),
                        "mean": float(block[method].mean()),
                        "std": float(block[method].std()),
                        "miss_rate": float(misses[method][block.index].gt(0.5).mean())
                        if method in misses
                        else np.nan,
                    }
                )
            for treatment in TREATMENTS:
                if treatment not in block or "baseline" not in block:
                    continue
                pair = block[["baseline", treatment]].dropna()
                deltas = (pair[treatment] - pair["baseline"]).to_numpy(dtype=float)
                low, high = bootstrap_ci(deltas, n_bootstrap)
                p_value, effect = wilcoxon_test(deltas)
                delta_rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "metric": metric,
                        "size_bin": bin_label,
                        "comparison": f"{treatment} vs baseline",
                        "n_pairs": int(deltas.size),
                        "delta_mean": float(np.mean(deltas)) if deltas.size else np.nan,
                        "delta_rel_pct": 100.0 * float(np.mean(deltas)) / float(pair["baseline"].mean())
                        if deltas.size and pair["baseline"].mean()
                        else np.nan,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": p_value,
                        "rank_biserial": effect,
                    }
                )
        for treatment in TREATMENTS:
            if treatment not in wide.columns or "baseline" not in wide.columns:
                continue
            block = wide[["baseline", treatment]].dropna()
            gains = (block[treatment] - block["baseline"]).rename("gain").reset_index()
            groups = [
                gains.loc[gains["size_bin"] == bin_label, "gain"].to_numpy(dtype=float)
                for bin_label in SIZE_BIN_LABELS
                if (gains["size_bin"] == bin_label).any()
            ]
            if len(groups) < 2:
                continue
            statistic, p_value = sps.kruskal(*groups)
            interaction_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "metric": metric,
                    "comparison": f"{treatment} vs baseline",
                    "test": "kruskal_wallis",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "n_small": int(groups[0].size),
                    "n_medium": int(groups[1].size) if len(groups) > 1 else 0,
                    "n_large": int(groups[2].size) if len(groups) > 2 else 0,
                    "cutoff_low": float(cutoffs[0]),
                    "cutoff_high": float(cutoffs[1]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    delta = pd.DataFrame(delta_rows)
    if not delta.empty:
        delta["p_value_adjusted"] = np.nan
        for (metric, comparison), block in delta.groupby(["metric", "comparison"]):
            delta.loc[block.index, "p_value_adjusted"] = holm(block["p_value"].tolist())
    interaction = pd.DataFrame(interaction_rows)
    if not interaction.empty:
        interaction["p_value_adjusted"] = holm(interaction["p_value"].tolist())
    return summary, delta, interaction


def _frame(rows: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in rows if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _md_table(frame: pd.DataFrame, columns: Sequence[str], decimals: int = 4) -> str:
    if frame.empty:
        return "_no rows_\n"
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: f"{value:.{decimals}f}" if np.isfinite(value) else ""
            )
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join("---" for _ in view.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False)
    )
    return "\n".join([header, rule, body]) + "\n"


def collect(datasets: Sequence[str], n_bootstrap: int, output: Path) -> dict[str, pd.DataFrame]:
    """Run every table for the requested datasets and return the concatenated frames."""
    buckets: dict[str, list[pd.DataFrame]] = {
        key: []
        for key in (
            "metrics",
            "paired",
            "misses",
            "comparison",
            "subgroup_summary",
            "subgroup_delta",
            "subgroup_interaction",
        )
    }
    for dataset in datasets:
        directory = RESULTS / f"external_{dataset}"
        if not directory.is_dir():
            print(f"skip {dataset}: {directory} does not exist", file=sys.stderr)
            continue
        runs = load_runs(directory)
        sizes = sizes_for(dataset, output)
        for family, experiment in FAMILIES.items():
            subset = runs.loc[runs["family"] == family]
            if subset.empty:
                print(f"skip {dataset}/{family}: no runs", file=sys.stderr)
                continue
            averaged = case_means(subset)
            buckets["metrics"].append(metric_table(averaged, dataset, family))
            buckets["paired"].append(paired_table(averaged, dataset, family, n_bootstrap))
            buckets["misses"].append(miss_table(subset, dataset, family))
            internal_dir = RESULTS / experiment / "evaluation"
            if internal_dir.is_dir():
                external_dice = averaged.loc[averaged["metric"] == "dice"].set_index(
                    ["case_id", "method"]
                )["value"]
                buckets["comparison"].append(
                    comparison_table(
                        internal_case_dice(internal_dir), external_dice, dataset, family
                    )
                )
            summary, delta, interaction = subgroup_tables(
                averaged, subset, sizes, dataset, family, n_bootstrap
            )
            buckets["subgroup_summary"].append(summary)
            buckets["subgroup_delta"].append(delta)
            buckets["subgroup_interaction"].append(interaction)
    return {key: _frame(value) for key, value in buckets.items()}


def write_markdown(tables: dict[str, pd.DataFrame], datasets: Sequence[str], path: Path) -> None:
    lines = ["# External polyp validation, split by architecture", ""]
    lines.append(
        "One architecture at a time (15 UNet checkpoints, 15 SwinUNetR checkpoints), so the "
        "external numbers are no longer an average over two different models. Metrics are averaged "
        "per case over the 15 checkpoints of the architecture and then summarised over cases; the "
        "paired tests use those per-case values, so one image stays one independent unit.\n"
    )
    for dataset in datasets:
        for family in FAMILIES:
            metrics = tables["metrics"]
            if metrics.empty:
                continue
            block = metrics.loc[(metrics["dataset"] == dataset) & (metrics["family"] == family)]
            if block.empty:
                continue
            lines.append(f"## {dataset} / {family}\n")
            lines.append("### Metric per method (mean +- std over cases, % vs baseline)\n")
            lines.append(
                _md_table(block, ["metric", "method", "n_cases", "mean", "std", "change_pct"])
            )
            paired = tables["paired"]
            if not paired.empty:
                lines.append("\n### Paired TTA deltas\n")
                lines.append(
                    _md_table(
                        paired.loc[(paired["dataset"] == dataset) & (paired["family"] == family)],
                        ["metric", "comparison", "n_pairs", "delta_mean", "delta_rel_pct",
                         "ci_low", "ci_high", "p_value", "p_value_adjusted", "rank_biserial"],
                    )
                )
            misses = tables["misses"]
            if not misses.empty:
                lines.append("\n### Missed detections counted per checkpoint\n")
                lines.append(
                    _md_table(
                        misses.loc[(misses["dataset"] == dataset) & (misses["family"] == family)],
                        ["method", "n_cases", "n_checkpoints", "misses_total",
                         "misses_per_checkpoint", "missed_cases_min", "missed_cases_max",
                         "cases_missed_by_any_checkpoint", "cases_majority_missed", "new_misses",
                         "recovered", "mcnemar_p_checkpoint", "mcnemar_p_checkpoint_adjusted",
                         "cases_became_majority_miss", "cases_left_majority_miss",
                         "mcnemar_p_case_majority"],
                    )
                )
            comparison = tables["comparison"]
            if not comparison.empty:
                lines.append("\n### Internal (CVC-ClinicDB) vs external, same architecture\n")
                lines.append(
                    _md_table(
                        comparison.loc[
                            (comparison["dataset"] == dataset) & (comparison["family"] == family)
                        ],
                        ["method", "dice_internal", "dice_external", "drop_abs", "drop_rel_pct",
                         "gain_vs_baseline_internal", "gain_vs_baseline_external",
                         "gain_rel_pct_internal", "gain_rel_pct_external", "gain_difference",
                         "verdict"],
                    )
                )
            subgroup = tables["subgroup_summary"]
            if subgroup.empty:
                lines.append("\n")
                continue
            lines.append("\n### Lesion-size subgroups (tertiles of the reference polyp area)\n")
            lines.append(
                _md_table(
                    subgroup.loc[
                        (subgroup["dataset"] == dataset)
                        & (subgroup["family"] == family)
                        & (subgroup["metric"] == "dice")
                    ],
                    ["method", "size_bin", "n_cases", "mean", "std", "miss_rate"],
                )
            )
            delta = tables["subgroup_delta"]
            lines.append("\n### Paired Dice/HD95 gain per size bin\n")
            lines.append(
                _md_table(
                    delta.loc[(delta["dataset"] == dataset) & (delta["family"] == family)]
                    if not delta.empty
                    else delta,
                    ["metric", "size_bin", "comparison", "n_pairs", "delta_mean", "delta_rel_pct",
                     "ci_low", "ci_high", "p_value", "p_value_adjusted"],
                )
            )
            interaction = tables["subgroup_interaction"]
            lines.append("\n### Size interaction (Kruskal-Wallis on the per-case gains)\n")
            lines.append(
                _md_table(
                    interaction.loc[
                        (interaction["dataset"] == dataset) & (interaction["family"] == family)
                    ]
                    if not interaction.empty
                    else interaction,
                    ["metric", "comparison", "statistic", "p_value", "p_value_adjusted",
                     "n_small", "n_medium", "n_large", "cutoff_low", "cutoff_high"],
                )
            )
            lines.append("\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = collect(args.datasets, args.n_bootstrap, output)
    for key, frame in tables.items():
        if not frame.empty:
            frame.to_csv(output / f"{key}.csv", index=False)
    write_markdown(tables, args.datasets, output / "summary_by_model.md")
    print(f"wrote {output / 'summary_by_model.md'}")
    for key, frame in tables.items():
        print(f"  {key}: {0 if frame.empty else len(frame)} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())





