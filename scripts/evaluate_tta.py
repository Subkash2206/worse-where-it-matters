"""Evaluate an existing checkpoint grid with the TTA methods of a config (never retrains).

The intensity-TTA configs only change ``tta.methods`` (and the output directory); the
checkpoints are the ones an earlier grid trained, and their path is spelled out in the config
(``model.checkpoint`` with the *original* experiment name), so nothing is retrained here.

Resume: a (fold, seed) whose per-case CSV already exists is skipped unless ``--force`` is
given, exactly like ``scripts/evaluate_external.py``.

Usage
-----
    python scripts/evaluate_tta.py --config configs/cvc_clinicdb/unet_resnet50_intensity_tta.yaml \
        --folds 0 1 2 3 4 --seeds 0 1 2
    python scripts/evaluate_tta.py --config configs/brats2021/segresnet_intensity_tta.yaml \
        --folds 0 1 2 3 4 --seeds 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.eval import evaluate, per_case_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="TTA config (method set + output dir)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--limit", type=int, default=None, help="first N cases (smoke run)")
    parser.add_argument(
        "--force", action="store_true", help="re-evaluate runs whose per-case CSV exists"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    methods = [str(method["name"]) for method in base.get("tta.methods") or []]
    print(f"{base.require('experiment.name')}: methods {methods}")

    plan: list[tuple[int, int, Path]] = []
    for fold in args.folds:
        for seed in args.seeds:
            config = base.with_overrides(
                {
                    "seed": int(seed),
                    "evaluation.fold": int(fold),
                    "dataset.split.fold": int(fold),
                }
            )
            checkpoint = config.resolve(
                str(config.require("model.checkpoint")).format(fold=int(fold), seed=int(seed))
            )
            plan.append((int(fold), int(seed), checkpoint))
            if not checkpoint.is_file():
                print(f"  WARNING no checkpoint: fold{fold} seed{seed} -> {checkpoint}")
    if args.dry_run:
        for fold, seed, checkpoint in plan:
            print(f"  fold{fold} seed{seed}: {checkpoint.name}")
        return 0

    started = time.perf_counter()
    evaluated = 0
    skipped = 0
    for fold, seed, _checkpoint in plan:
        config = base.with_overrides(
            {
                "seed": int(seed),
                "evaluation.fold": int(fold),
                "dataset.split.fold": int(fold),
            }
        )
        expected = per_case_path(config, fold, seed)
        if expected.is_file() and not args.force:
            print(f"[fold{fold} seed{seed}] exists, skipped ({expected.name})", flush=True)
            skipped += 1
            continue
        print(f"[fold{fold} seed{seed}] evaluating {len(methods)} method(s)", flush=True)
        try:
            result = evaluate(config, limit=args.limit, progress=False)
        except (ConfigError, FileNotFoundError, RuntimeError) as error:
            print(f"ERROR [fold{fold} seed{seed}]: {error}", file=sys.stderr)
            return 2
        print(f"  -> {result.n_cases} cases -> {result.per_case_csv.name}", flush=True)
        evaluated += 1
    minutes = (time.perf_counter() - started) / 60.0
    print(f"done in {minutes:.1f} min: {evaluated} evaluated, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
