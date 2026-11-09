"""Lesion-size subgroup analysis (MIDL Clinical Translation & Validation).

A method can improve the mean and still fail where it matters clinically: small polyps, or
images without a lesion at all.  This module turns a long-format per-case result frame plus
a per-case reference size into the tables a reviewer asks for:

* **size bins** - tertiles of the reference foreground area (pixels for the polyp data,
  mm^3 when the dataset declares a spacing), with the cutoffs reported, so the bins are
  reproducible rather than eyeballed;
* **per subgroup and method** - the metric mean with a cluster-bootstrap CI over groups
  (frames of one video are not independent), the missed-detection rate with its CI;
* **TTA effect per subgroup** - paired delta vs the control inside each bin, with its CI;
* **interaction test** - whether the TTA gain differs across bins at all (Kruskal-Wallis on
  the per-case deltas, plus a bootstrap CI of the delta difference between the largest and
  the smallest bin).  A TTA that helps the mean but only helps large lesions is a different
  claim from one that helps uniformly, and the mean alone cannot tell them apart;
* **empty vs non-empty reference** - the failure mode that has no size at all.  For the
  polyp data this subgroup is empty by construction (every image has a polyp); for BraTS it
  is the ~3 % of patients without enhancing tumour.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from melba.stats import analyze_long, pool_folds

__all__ = [
    "SIZE_BIN_LABELS",
    "add_size_bins",
    "empty_reference_subgroup_table",
    "lesion_size_table",
    "subgroup_delta_table",
    "subgroup_interaction_table",
    "subgroup_summary_table",
    "tertile_cutoffs",
]

#: labels of the three size bins, smallest first
SIZE_BIN_LABELS: tuple[str, ...] = ("small", "medium", "large")


def lesion_size_table(
    config: Any,
    *,
    split: str | None = None,
    folds: Sequence[int] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Reference foreground size per case, read once from the dataset masks.

    Returns ``case_id``, ``group_id``, ``fold``, ``gt_pixels`` (voxels/pixels of reference
    foreground) and ``gt_mm3`` (the same, in mm^3 when the config declares a spacing).
    Cases appear once, in the fold whose *test* split contains them.
    """
    from melba.data import build_dataset

    resolved_split = str(split or config.get("evaluation.split", "test"))
    resolved_folds = (
        [int(fold) for fold in folds]
        if folds is not None
        else list(range(int(config.get("dataset.split.n_folds", 5) or 1)))
    )
    spacing = [float(value) for value in config.get("dataset.spacing_mm", [1.0]) or [1.0]]
    voxel_volume = float(np.prod(spacing)) if spacing else 1.0
    rows: list[dict[str, Any]] = []
    for fold in resolved_folds:
        dataset = build_dataset(config, resolved_split, fold=fold)
        tags = _fold_tags(fold)
        iterator = range(len(dataset))
        if progress:  # pragma: no cover - CLI convenience
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"fold {fold} sizes", unit="case")
        for index in iterator:
            sample = dataset[index]
            mask = sample.get("mask_native", sample["mask"])
            array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
            pixels = int((np.asarray(array) > 0.5).sum())
            rows.append(
                {
                    "case_id": str(sample["case_id"]),
                    "group_id": str(sample.get("group_id", sample["case_id"])),
                    "fold": tags["fold"],
                    "split": str(sample.get("split", resolved_split)),
                    "gt_pixels": pixels,
                    "gt_mm3": float(pixels) * voxel_volume,
                }
            )
    table = pd.DataFrame(rows)
    return table.drop_duplicates("case_id", keep="first").reset_index(drop=True)


def _fold_tags(fold: int) -> dict[str, int]:
    """Fold bookkeeping that does not depend on the dataset returning a fold column."""
    return {"fold": int(fold)}


def tertile_cutoffs(values: Sequence[float]) -> list[float]:
    """The two cutoffs that split ``values`` into three equal-size groups."""
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if array.size < 3:
        raise ValueError(f"need at least three values for tertiles, got {array.size}")
    low, high = np.percentile(array, [100.0 / 3.0, 200.0 / 3.0])
    return [float(low), float(high)]


def add_size_bins(
    frame: pd.DataFrame,
    sizes: pd.DataFrame,
    *,
    column: str = "gt_pixels",
    cutoffs: Sequence[float] | None = None,
    include_empty: bool = False,
) -> tuple[pd.DataFrame, list[float]]:
    """Attach ``gt_pixels``/``gt_mm3`` and a tertile ``size_bin`` to a long-format frame.

    ``include_empty=False`` (the default) keeps cases whose reference is empty **out of the
    size bins**: they have no lesion size, so they belong to the empty-reference subgroup,
    and letting them fall into the "small" bin would mix "tiny polyp" with "no polyp at all"
    (BraTS ET has ~3 % such patients).  The tertile cutoffs are computed over the non-empty
    references only.
    """
    positive = sizes.loc[sizes[column] > 0]
    resolved = list(cutoffs) if cutoffs is not None else tertile_cutoffs(positive[column])
    if len(resolved) != 2:
        raise ValueError(f"exactly two cutoffs are needed, got {resolved}")
    mapping = sizes.set_index("case_id")[["gt_pixels", "gt_mm3"]]
    enriched = frame.copy()
    enriched["case_id"] = enriched["case_id"].astype(str)
    enriched = enriched.join(mapping, on="case_id")
    values = enriched[column].to_numpy(dtype=float)
    labels = np.where(
        values <= resolved[0],
        SIZE_BIN_LABELS[0],
        np.where(values <= resolved[1], SIZE_BIN_LABELS[1], SIZE_BIN_LABELS[2]),
    )
    enriched["size_bin"] = pd.Categorical(labels, categories=list(SIZE_BIN_LABELS), ordered=True)
    excluded = ~np.isfinite(values) | (values <= 0)
    if not include_empty:
        enriched.loc[excluded, "size_bin"] = np.nan
    return enriched, sorted(resolved)



def _cluster_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of a mean/median, resampling **groups** (videos/patients)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    if values.size == 0:
        return float("nan"), float("nan")
    unique = np.unique(groups)
    if unique.size < 2:
        return float("nan"), float("nan")
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    reduce_fn = np.nanmean if statistic == "mean" else np.nanmedian
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, unique.size, size=(int(n_bootstrap), unique.size))
    estimates = np.empty(int(n_bootstrap), dtype=float)
    for index, draw in enumerate(draws):
        rows = np.concatenate([positions[unique[position]] for position in draw])
        estimates[index] = reduce_fn(values[rows])
    finite_estimates = estimates[np.isfinite(estimates)]
    if finite_estimates.size < max(10, int(0.05 * int(n_bootstrap))):
        return float("nan"), float("nan")
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.percentile(finite_estimates, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def _case_values(frame: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    """One row per case for one method/metric (``case_id``, ``group_id``, ``value``)."""
    subset = frame.loc[
        (frame["method"] == method) & (frame["metric"] == metric),
        ["case_id", "group_id", "value", "empty_flag", "size_bin"],
    ]
    return subset.drop_duplicates("case_id")


def subgroup_summary_table(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("dice", "hd95"),
    methods: Sequence[str] | None = None,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Per (method, metric, size bin): mean with a cluster bootstrap CI, plus miss rate."""
    methods = (
        sorted(str(value) for value in methods)
        if methods is not None
        else sorted(str(value) for value in frame["method"].unique())
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        for bin_label in SIZE_BIN_LABELS:
            for metric in metrics:
                subset = _case_values(frame, method, metric)
                subset = subset.loc[subset["size_bin"] == bin_label]
                if subset.empty:
                    continue
                values = subset["value"].to_numpy(dtype=float)
                groups = subset["group_id"].to_numpy()
                low, high = _cluster_mean_ci(
                    values, groups, n_bootstrap=n_bootstrap, ci=ci, seed=seed
                )
                missed = (subset["empty_flag"].astype(str) == "pred_empty").to_numpy()
                missed_low, missed_high = _cluster_mean_ci(
                    missed.astype(float), groups, n_bootstrap=n_bootstrap, ci=ci, seed=seed
                )
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "size_bin": bin_label,
                        "n_cases": int(values.size),
                        "n_units": int(np.unique(groups).size),
                        f"{metric}_mean": float(np.nanmean(values))
                        if np.isfinite(values).any()
                        else float("nan"),
                        "ci_low": low,
                        "ci_high": high,
                        "n_missed": int(missed.sum()),
                        "missed_rate": float(missed.mean()),
                        "missed_ci_low": missed_low,
                        "missed_ci_high": missed_high,
                    }
                )
    return pd.DataFrame(rows)


def subgroup_delta_table(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    metrics: Sequence[str] = ("dice", "hd95"),
    unit: str = "group",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    correction: str = "holm",
    primary_test: str = "wilcoxon",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Paired TTA delta vs the control **inside every size bin** (CI, p, effect size).

    The same paired machinery the main tables use, repeated per subgroup: it answers "does
    TTA help small polyps as much as large ones?" with a CI and a test per bin.
    """
    rows: list[pd.DataFrame] = []
    for bin_label in SIZE_BIN_LABELS:
        subset = frame.loc[frame["size_bin"] == bin_label]
        if subset.empty:
            continue
        for treatment in treatments:
            if treatment not in set(subset["method"].astype(str)):
                continue
            try:
                table = analyze_long(
                    pool_folds([subset]),
                    treatment=treatment,
                    control=control,
                    metrics=list(metrics),
                    unit=unit,
                    primary_test=primary_test,
                    alternative=alternative,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=seed,
                    correction=correction,
                )
            except (ValueError, KeyError):
                continue
            table.insert(0, "size_bin", bin_label)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_delta_gap(
    small: np.ndarray,
    large: np.ndarray,
    small_groups: np.ndarray,
    large_groups: np.ndarray,
    *,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> tuple[float, float]:
    """CI of ``mean(delta | large) - mean(delta | small)``, resampling groups per bin."""
    rng = np.random.default_rng(int(seed))
    small_unique = np.unique(small_groups)
    large_unique = np.unique(large_groups)
    if small_unique.size < 2 or large_unique.size < 2:
        return float("nan"), float("nan")
    small_positions = {g: np.flatnonzero(small_groups == g) for g in small_unique}
    large_positions = {g: np.flatnonzero(large_groups == g) for g in large_unique}
    gaps = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        small_rows = np.concatenate(
            [
                small_positions[small_unique[position]]
                for position in rng.integers(0, small_unique.size, size=small_unique.size)
            ]
        )
        large_rows = np.concatenate(
            [
                large_positions[large_unique[position]]
                for position in rng.integers(0, large_unique.size, size=large_unique.size)
            ]
        )
        gaps[index] = np.nanmean(large[large_rows]) - np.nanmean(small[small_rows])
    finite = gaps[np.isfinite(gaps)]
    if finite.size < max(10, int(0.05 * int(n_bootstrap))):
        return float("nan"), float("nan")
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def subgroup_interaction_table(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    metrics: Sequence[str] = ("dice", "hd95"),
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Does the TTA gain differ across size bins?  (interaction / effect modification)

    Two complementary answers per (treatment, metric):

    * ``kruskal_p`` - Kruskal-Wallis across the three bins on the **per-case paired delta**.
      Distribution-free, and it uses the whole cohort instead of two bins;
    * ``delta_gap`` - mean delta in the largest bin minus mean delta in the smallest, with a
      bootstrap CI that resamples groups inside each bin.  It gives the size of the effect
      modification in the metric's own units, which a p-value alone does not.

    A metric whose delta changes sign across bins is the case worth reporting: the mean gain
    then hides a subgroup where TTA hurts.
    """
    from scipy import stats as scipy_stats

    rows: list[dict[str, Any]] = []
    for treatment in treatments:
        for metric in metrics:
            control_values = _case_values(frame, control, metric)
            treatment_values = _case_values(frame, treatment, metric)
            if control_values.empty or treatment_values.empty:
                continue
            merged = control_values.merge(
                treatment_values, on=["case_id", "group_id", "size_bin"], suffixes=("_c", "_t")
            )
            merged["delta"] = merged["value_t"] - merged["value_c"]
            merged = merged.dropna(subset=["size_bin"])
            # keep values and group ids aligned: drop non-finite deltas *before* splitting
            finite = np.isfinite(merged["delta"].to_numpy(dtype=float))
            merged = merged.loc[finite]
            bins = {
                label: merged.loc[merged["size_bin"] == label] for label in SIZE_BIN_LABELS
            }
            groups = [
                bins[label]["delta"].to_numpy(dtype=float) for label in SIZE_BIN_LABELS
            ]
            if any(values.size == 0 for values in groups):
                continue
            if sum(values.size for values in groups) < 6:
                continue
            kruskal_p = float(scipy_stats.kruskal(*groups, nan_policy="omit").pvalue)
            small, large = groups[0], groups[-1]
            small_groups = bins[SIZE_BIN_LABELS[0]]["group_id"].to_numpy()
            large_groups = bins[SIZE_BIN_LABELS[-1]]["group_id"].to_numpy()
            low, high = _bootstrap_delta_gap(
                small,
                large,
                small_groups,
                large_groups,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            rows.append(
                {
                    "treatment": treatment,
                    "metric": metric,
                    "control": control,
                    "n_small": int(small.size),
                    "n_medium": int(groups[1].size),
                    "n_large": int(large.size),
                    "delta_small": float(np.mean(small)),
                    "delta_medium": float(np.mean(groups[1])),
                    "delta_large": float(np.mean(large)),
                    "delta_gap": float(np.mean(large) - np.mean(small)),
                    "gap_ci_low": low,
                    "gap_ci_high": high,
                    "kruskal_p": kruskal_p,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["metric", "treatment"]).reset_index(drop=True)


def subgroup_mcnemar_table(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    treatments: Sequence[str] = ("tta2", "tta4", "tta8"),
    unit: str = "case",
    correction: str = "holm",
) -> pd.DataFrame:
    """Exact McNemar per size bin: does TTA change *which* cases are missed?

    For every bin and treatment the table reports the transition counts against the control
    (``nonempty_to_empty`` = **new misses**, ``empty_to_nonempty`` = **recovered cases**,
    both non-empty, both empty) and the exact two-sided McNemar p on the discordant pairs.
    ``mcnemar_p_adjusted`` is Holm-corrected **across all bins and methods shown**, so the
    family is "every bin x every TTA method", not one bin at a time - otherwise three bins
    and three methods would each get a free 5 %.

    A mean Dice gain and a significant increase in new misses are perfectly compatible; this
    table is what makes the second half visible.
    """
    from melba.stats import holm_adjust, transition_table

    rows: list[dict[str, Any]] = []
    for bin_label in SIZE_BIN_LABELS:
        subset = frame.loc[frame["size_bin"] == bin_label]
        if subset.empty:
            continue
        available = set(subset["method"].astype(str))
        for treatment in treatments:
            if treatment not in available or control not in available:
                continue
            try:
                counts = transition_table(subset, treatment, control=control, unit=unit)
            except ValueError:
                continue
            rows.append(
                {
                    "size_bin": bin_label,
                    "control": control,
                    "treatment": treatment,
                    "unit": unit,
                    "n_units": int(counts["n"]),
                    "both_nonempty": int(counts["both_nonempty"]),
                    "new_misses": int(counts["nonempty_to_empty"]),
                    "recovered": int(counts["empty_to_nonempty"]),
                    "both_empty": int(counts["both_empty"]),
                    "net_change": int(counts["empty_to_nonempty"])
                    - int(counts["nonempty_to_empty"]),
                    "mcnemar_p": float(counts["mcnemar_p"]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if correction == "holm":
        table["mcnemar_p_adjusted"] = holm_adjust(table["mcnemar_p"].to_numpy(dtype=float))
    elif correction in (None, "none"):
        table["mcnemar_p_adjusted"] = table["mcnemar_p"]
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"Unknown correction {correction!r}; expected 'holm' or 'none'")
    table["correction"] = f"{correction} across {len(table)} bin x method tests"
    return table


def empty_reference_subgroup_table(
    frame: pd.DataFrame,
    *,
    control: str = "baseline",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """The "no lesion at all" subgroup: empty-reference cases per method.

    For the polyp collections every reference mask contains a polyp, so this table reports
    ``n_cases = 0`` - which is itself the honest statement that the subgroup does not exist
    there.  For BraTS ET it lists the patients without enhancing tumour and the rate at
    which each method segments something anyway.
    """
    from melba.stats import analyze_false_positives

    methods = sorted(str(value) for value in frame["method"].unique())
    if control in methods:
        methods = [control] + [method for method in methods if method != control]
    try:
        table = analyze_false_positives(
            frame,
            control=control,
            methods=methods,
            unit="group",
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
    except ValueError:
        return pd.DataFrame()
    table.insert(0, "subgroup", "empty_reference")
    return table




