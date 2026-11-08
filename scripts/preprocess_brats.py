"""Preprocess BraTS 2021 into a brain-cropped, float16 cache (read-only on the source).

For every case this crops all four modalities and the segmentation to the brain bounding
box plus a 4-voxel margin, normalizes per modality (1-99 percentile clip + z-score over
brain voxels, exactly as the training pipeline does), and writes

    cache/brats2021_pre/<case>_image.npy   float16 (4, d, h, w)
    cache/brats2021_pre/<case>_seg.npy     uint8   (d, h, w)  (raw labels 0/1/2/4)
    cache/brats2021_pre/index.json         crop offsets, original shape, spacing

so a prediction made on the cropped grid can be placed back on the 240x240x155 grid
(``melba.data.brats.place_on_full_grid``).  Nothing inside ``dataset.root`` is written,
moved or deleted: the script only reads the NIfTI files.

Usage
-----
    python scripts/preprocess_brats.py --config configs/brats2021/segresnet.yaml
    python scripts/preprocess_brats.py --config ... --limit 20 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import discover_brats_cases  # noqa: E402
from melba.data.brats import (  # noqa: E402
    CACHE_MARGIN,
    preprocess_case,
    read_cache_index,
    write_cache_index,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="BraTS experiment YAML")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--margin", type=int, default=CACHE_MARGIN)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    section = config.section("dataset")
    normalization = dict(section.get("normalization") or {})
    low = float(normalization.get("low", 1.0))
    high = float(normalization.get("high", 99.0))
    cache_dir = config.resolve(str(section.get("cache_dir", "cache/brats2021_pre")))
    try:
        cases = discover_brats_cases(config)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: int(args.limit)]
    from melba.data.brats import BraTSItem

    root = config.resolve(str(config.require("dataset.root")))
    pending = [
        BraTSItem(case_id=case, group_id=case, split="train", directory=root / case)
        for case in cases
    ]
    print(f"root  : {root}")
    print(f"cache : {cache_dir}")
    print(f"cases : {len(pending)} (margin {args.margin} voxels, percentiles {low}-{high})")

    existing = {} if args.overwrite else read_cache_index(cache_dir)
    started = time.perf_counter()
    entries = dict(existing)
    done = 0
    failures: list[tuple[str, str]] = []

    def work(item):
        return preprocess_case(
            item,
            cache_dir=cache_dir,
            low=low,
            high=high,
            margin=int(args.margin),
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(work, item): item for item in pending}
        for future, item in futures.items():
            try:
                entry = future.result()
            except Exception as error:  # noqa: BLE001 - one bad case must not stop the run
                failures.append((item.case_id, str(error)))
                continue
            entries[item.case_id] = entry
            done += 1
            if done % 100 == 0 or done == len(pending):
                elapsed = time.perf_counter() - started
                rate = done / max(elapsed, 1e-9)
                print(
                    f"  {done}/{len(pending)} cases, {elapsed/60:.1f} min "
                    f"({rate:.1f} cases/s)",
                    flush=True,
                )
            if done % 250 == 0:
                write_cache_index(cache_dir, entries)

    write_cache_index(cache_dir, entries)
    # also leave one sidecar per case, so a reader never needs this summary file
    from melba.data.brats import cache_meta_path

    for case_id, entry in entries.items():
        meta_path = cache_meta_path(cache_dir, case_id)
        if not meta_path.is_file():
            meta_path.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for case_id, message in failures[:10]:
            print(f"  {case_id}: {message}")
    total_bytes = sum(int(entry.get("bytes", 0)) for entry in entries.values())
    shapes = [tuple(entry.get("shape") or ()) for entry in entries.values()]
    print(f"\n{len(entries)} case(s) in the cache, {elapsed/60:.1f} min "
          f"({len(pending)/max(elapsed, 1e-9):.1f} cases/s)")
    if shapes:
        mean = tuple(round(sum(axis) / len(shapes), 1) for axis in zip(*shapes))
        print(f"mean cropped shape: {mean}")
        print(f"cache size: {total_bytes/1024**3:.2f} GB (sum of the .npy payloads)")
    print(f"index: {cache_dir / 'index.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
