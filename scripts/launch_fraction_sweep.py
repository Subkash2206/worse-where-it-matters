"""Print (default) or launch the data-fraction sweep for ``unet_resnet50``.

The sweep answers "how much of the TTA gain survives less training data?".  Each
fraction has its own config with its own experiment name, so checkpoints, results and
aggregates never collide with the full-data run; only ``training.data_fraction``
differs from ``configs/cvc_clinicdb/unet_resnet50.yaml``.

Safety: nothing is started unless ``--execute`` is passed.  Run it while no other GPU
job is running — one grid at a time.

    python scripts/launch_fraction_sweep.py                 # print the commands
    python scripts/launch_fraction_sweep.py --execute       # start both grids, detached

Each launched grid writes ``logs/<experiment>_grid.log`` and
``logs/<experiment>_grid.pid``; monitor with
``python scripts/run_grid.py --config <config> --status-only``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import ConfigError, load_config  # noqa: E402

#: fraction → config (both inherit everything else from unet_resnet50.yaml)
FRACTION_CONFIGS: dict[float, str] = {
    0.25: "configs/cvc_clinicdb/unet_resnet50_data025.yaml",
    0.5: "configs/cvc_clinicdb/unet_resnet50_data050.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=sorted(FRACTION_CONFIGS),
        help=f"subset of {sorted(FRACTION_CONFIGS)} (default: all)",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="seed 0 only for now")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually start the grids (default: only print what would run)",
    )
    return parser.parse_args(argv)


def build_command(
    config_path: str, *, folds: list[int], seeds: list[int], epochs: int, patience: int, pid_file: str
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(_REPO_ROOT / "scripts" / "run_grid.py"),
        "--config",
        config_path,
        "--folds",
        *[str(fold) for fold in folds],
        "--seeds",
        *[str(seed) for seed in seeds],
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--pid-file",
        pid_file,
    ]


def launch(command: list[str], log_path: Path) -> int:
    """Start ``command`` detached so it survives this shell; returns the child pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - kept by the child
    kwargs: dict[str, object] = {"cwd": str(_REPO_ROOT), "stdout": stream, "stderr": stream}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)  # noqa: S603 - command is built here
    return int(process.pid)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [fraction for fraction in args.fractions if fraction not in FRACTION_CONFIGS]
    if unknown:
        print(
            f"ERROR: no config for fraction(s) {unknown}; known: {sorted(FRACTION_CONFIGS)}",
            file=sys.stderr,
        )
        return 2

    log_dir = Path(args.log_dir)
    plans: list[tuple[str, str, list[str], Path, Path]] = []
    for fraction in args.fractions:
        config_path = FRACTION_CONFIGS[fraction]
        try:
            config = load_config(config_path)
        except (ConfigError, FileNotFoundError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        experiment = str(config.require("experiment.name"))
        pid_file = log_dir / f"{experiment}_grid.pid"
        log_path = log_dir / f"{experiment}_grid.log"
        command = build_command(
            config_path,
            folds=args.folds,
            seeds=args.seeds,
            epochs=args.epochs,
            patience=args.patience,
            pid_file=str(pid_file),
        )
        plans.append((f"{fraction:g}", experiment, command, pid_file, log_path))

    for fraction, experiment, command, pid_file, log_path in plans:
        print(f"fraction {fraction}: {experiment}")
        print("  " + " ".join(command))
        print(f"  log: {log_path}")
        print(f"  pid: {pid_file}")

    if not args.execute:
        print(
            f"\n{len(plans)} grid(s) NOT started (dry run). Add --execute when the GPU is free."
        )
        return 0

    for fraction, experiment, command, _pid_file, log_path in plans:
        pid = launch(command, log_path)
        print(f"started {experiment} (fraction {fraction}) as pid {pid} → {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
