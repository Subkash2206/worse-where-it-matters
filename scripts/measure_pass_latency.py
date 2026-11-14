"""Measure the per-pass forward latency of the 2-D models used in the deployment table.

The deployment terms multiply the number of forward passes by a per-pass latency constant, so
that constant has to be measured on the machine that produced the table - not borrowed from
another architecture.  This script runs one real 352x352 frame through a model, warms it up
and reports the **median** of ``--repeats`` timed single-image passes (CUDA synchronised
around every pass, AMP on, exactly like the evaluation runner).

It writes ``results/deployment_terms/pass_latency.csv``; ``scripts/deployment_terms.py`` reads
that file instead of a hard-coded constant and states in its latency source that the value is
measured here.

Usage
-----
    python scripts/measure_pass_latency.py                       # all three 2-D models, CUDA
    python scripts/measure_pass_latency.py --device cpu --repeats 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from melba.config import load_config  # noqa: E402
from melba.data import build_dataset  # noqa: E402
from melba.models import build_model  # noqa: E402
from melba.utils import describe_device, git_revision  # noqa: E402

CECUM_CONFIG = _REPO_ROOT / "configs" / "external" / "hyperkvasir_cecum.yaml"
OUTPUT = _REPO_ROOT / "results" / "deployment_terms" / "pass_latency.csv"
#: one entry per 2-D model of the deployment table: key = the name the table uses,
#: config = its own training config (the model section is all the timing needs)
DEFAULT_MODELS = (
    ("unet_resnet50", "cvc_clinicdb/unet_resnet50"),
    ("segformer_mit_b2", "cvc_clinicdb/segformer_mit_b2"),
    ("swinunetr_2d", "cvc_clinicdb/swinunetr_2d"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--repeats", type=int, default=50, help="timed passes per block")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up passes per block")
    parser.add_argument(
        "--blocks",
        type=int,
        default=3,
        help="independent warm-up + timed blocks per model (median of the block medians is used, "
        "which damps the clock/thermal drift of a laptop GPU)",
    )
    parser.add_argument("--output", default=str(OUTPUT))
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _timed_passes(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    amp: bool,
) -> np.ndarray:
    """Milliseconds of ``repeats`` single-image forward passes after ``warmup`` passes."""
    cuda = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(int(warmup)):
            with torch.autocast(device_type=device.type, enabled=amp):
                model(image)
        if cuda:
            torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(int(repeats)):
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                model(image)
            if cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(timings)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    amp = device.type == "cuda"
    print(f"device {device} ({describe_device(device).get('name')}), amp {amp}, "
          f"{args.warmup} warm-up + {args.repeats} timed passes, batch 1")

    input_config = load_config(CECUM_CONFIG, {"device": device.type})
    dataset = build_dataset(input_config, "test", fold=0)
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    print(f"input {sample['case_id']}: {tuple(image.shape)} from {dataset.items[0].image_path.name}\n")

    rows: list[dict[str, Any]] = []
    for key, config_name in DEFAULT_MODELS:
        config = load_config(_REPO_ROOT / "configs" / f"{config_name}.yaml", {"device": device.type})
        model = build_model(config).to(device).eval()
        timings = np.concatenate(
            [
                _timed_passes(
                    model,
                    image,
                    device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    amp=amp,
                )
                for _ in range(int(args.blocks))
            ]
        )
        per_block = [
            float(np.median(timings[block * args.repeats : (block + 1) * args.repeats]))
            for block in range(int(args.blocks))
        ]
        median_ms = float(np.median(per_block))
        spread = (max(per_block) - min(per_block)) / median_ms * 100.0
        rows.append(
            {
                "model": key,
                "config": config_name,
                "architecture": str(config.get("model.architecture")),
                "device": device.type,
                "device_name": describe_device(device).get("name"),
                "height": height,
                "width": width,
                "batch_size": 1,
                "amp": amp,
                "warmup_passes": int(args.warmup),
                "timed_passes": int(args.repeats),
                "blocks": int(args.blocks),
                "block_medians_ms": ";".join(f"{value:.2f}" for value in per_block),
                "block_spread_pct": spread,
                "median_ms_per_pass": median_ms,
                "mean_ms_per_pass": float(timings.mean()),
                "min_ms_per_pass": float(timings.min()),
                "p10_ms_per_pass": float(np.percentile(timings, 10)),
                "p90_ms_per_pass": float(np.percentile(timings, 90)),
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "git_revision": git_revision(),
            }
        )
        print(
            f"  {key:<18s} {config.get('model.architecture'):<18s} "
            f"median {median_ms:7.2f} ms (blocks {', '.join(f'{v:.2f}' for v in per_block)}; "
            f"spread {spread:4.1f} %)  "
            f"p10 {np.percentile(timings, 10):7.2f}  p90 {np.percentile(timings, 90):7.2f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

