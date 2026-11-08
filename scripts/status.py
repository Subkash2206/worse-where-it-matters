"""One compact status report for every experiment, the queue and its ETA.

Read-only and CPU-only: it never imports torch, touches the GPU or writes a file.  A run
is classified exactly like ``scripts/run_grid.py`` does (``run_status``): an evaluation
marker means done, ``run_done.json`` without a marker means "evaluate", ``last.pt``
without ``run_done.json`` means training/resuming.

    python scripts/status.py
    python scripts/status.py --tail 5 --queue queues/overnight.yaml

Sections: per-experiment progress (done/total, failed, current run + epoch, best val
Dice, ETA from the median run time observed so far), pid-file liveness (queue, grids),
which queue step is active, and the last few lines of the active job's log.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from melba.config import load_config  # noqa: E402
from melba.training.grid import (  # noqa: E402
    STATUS_EVALUATE,
    STATUS_RESUME,
    STATUS_SKIP,
    expected_eval_paths,
    eval_marker_path,
    iter_runs,
    run_status,
)
from melba.pidfile import (  # noqa: E402
    LAUNCHER_PROCESS_NAMES,
    pid_alive,
    process_name,
    read_pid_file,
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment as the status report sees it."""

    label: str
    config: str
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (0, 1, 2)

    @property
    def path(self) -> Path:
        return _REPO_ROOT / self.config


#: every experiment the project knows about; ``brats`` is the 3D milestone and has no
#: config yet, so it is reported as "config missing" instead of silently disappearing
EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("unet", "configs/cvc_clinicdb/unet_resnet50.yaml"),
    ExperimentSpec("swin", "configs/cvc_clinicdb/swinunetr_2d.yaml"),
    ExperimentSpec("dropout", "configs/cvc_clinicdb/unet_resnet50_dropout.yaml"),
    ExperimentSpec(
        "data025", "configs/cvc_clinicdb/unet_resnet50_data025.yaml", seeds=(0,)
    ),
    ExperimentSpec(
        "data050", "configs/cvc_clinicdb/unet_resnet50_data050.yaml", seeds=(0,)
    ),
    ExperimentSpec("brats", "configs/brats2021/segresnet.yaml", seeds=(0,)),
)

#: seconds assumed for a CPU post-processing step (aggregate / compare) in the queue ETA
POST_STEP_SECONDS = 300.0


def format_duration(seconds: float | None) -> str:
    """``90`` -> ``1m 30s``, ``3720`` -> ``1h 02m``; ``None`` -> ``?``."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass
class RunState:
    """What the disk says about one (fold, seed) run."""

    fold: int
    seed: int
    status: str
    epoch: int | None = None
    total_epochs: int | None = None
    best_val_dice: float | None = None
    seconds: float | None = None
    epoch_seconds: float | None = None
    done_seconds: float | None = None

    @property
    def label(self) -> str:
        return f"fold{self.fold}/seed{self.seed}"


def read_training_log(path: Path) -> tuple[int | None, float | None, float | None]:
    """``(last epoch, best val Dice, median epoch seconds)`` of one training log."""
    if not path.is_file():
        return None, None, None
    epochs: list[int] = []
    dices: list[float] = []
    seconds: list[float] = []
    try:
        # utf-8-sig: the trainer may write a BOM, which would hide the `epoch` column
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:  # pragma: no cover - unreadable log
        return None, None, None
    header = lines[0].split(",") if lines else []
    index = {name: position for position, name in enumerate(header)}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        try:
            epochs.append(int(float(parts[index["epoch"]])))
        except (KeyError, ValueError):
            continue
        try:
            value = float(parts[index["val_dice"]])
            if value == value:  # NaN-safe
                dices.append(value)
        except (KeyError, ValueError):
            pass
        try:
            value = float(parts[index["epoch_seconds"]])
            if value == value and value > 0:
                seconds.append(value)
        except (KeyError, ValueError):
            pass
    last_epoch = max(epochs) if epochs else None
    best_dice = max(dices) if dices else None
    median_epoch = statistics.median(seconds) if seconds else None
    return last_epoch, best_dice, median_epoch
FOLD_SEED = r"fold (\d+) seed (\d+)"

#: rough wall-clock of one evaluation pass (used only when a run needs "evaluate" only)
EVAL_SECONDS = 120.0


@dataclass
class ExperimentState:
    """Aggregated progress of one experiment."""

    spec: ExperimentSpec
    available: bool
    runs: list[RunState] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    log_path: Path | None = None
    current_label: str | None = None

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def done(self) -> int:
        return sum(1 for run in self.runs if run.status == STATUS_SKIP)

    @property
    def active(self) -> RunState | None:
        """The run the grid log points at, else the one waiting for its evaluation."""
        if self.current_label:
            for run in self.runs:
                if run.label == self.current_label and run.status != STATUS_SKIP:
                    return run
        pending = [
            run for run in self.runs if run.status in (STATUS_RESUME, STATUS_EVALUATE)
        ]
        if not pending:
            return None
        return max(
            pending, key=lambda run: (run.epoch or 0, run.status == STATUS_RESUME)
        )

    @property
    def unresolved_failures(self) -> list[str]:
        """Failures whose run has still not produced an evaluation."""
        done = {run.label for run in self.runs if run.status == STATUS_SKIP}
        return [item for item in self.failures if item not in done]

    @property
    def recovered_failures(self) -> list[str]:
        """Failures that a retry or a later resume rescued (nothing to worry about)."""
        done = {run.label for run in self.runs if run.status == STATUS_SKIP}
        return [item for item in self.failures if item in done]

    @property
    def best_val_dice(self) -> tuple[float | None, str | None]:
        """``(best value, run label)`` over everything on disk."""
        scored = [run for run in self.runs if run.best_val_dice is not None]
        if not scored:
            return None, None
        best = max(scored, key=lambda run: float(run.best_val_dice))
        return float(best.best_val_dice), best.label

    @property
    def mean_best_val_dice(self) -> float | None:
        scored = [run.best_val_dice for run in self.runs if run.best_val_dice is not None]
        return float(statistics.fmean(scored)) if scored else None

    @property
    def median_run_seconds(self) -> float | None:
        """Median wall-clock of a finished training run (the ETA unit)."""
        finished = [run.done_seconds for run in self.runs if run.done_seconds]
        if finished:
            return float(statistics.median(finished))
        per_epoch = [run.epoch_seconds for run in self.runs if run.epoch_seconds]
        epochs = next((run.total_epochs for run in self.runs if run.total_epochs), None)
        if per_epoch and epochs:
            return float(statistics.median(per_epoch)) * float(epochs)
        return None

    @property
    def eta_seconds(self) -> float | None:
        """Remaining time from the median run time and the unfinished runs."""
        pending = [run for run in self.runs if run.status != STATUS_SKIP]
        if not pending:
            return 0.0
        median_run = self.median_run_seconds
        if median_run is None:
            return None
        total = 0.0
        for run in pending:
            if (
                run.status == STATUS_RESUME
                and run.epoch
                and run.total_epochs
                and run.epoch_seconds
            ):
                total += max(0.0, run.total_epochs - run.epoch) * run.epoch_seconds
            elif run.status == STATUS_EVALUATE:
                total += EVAL_SECONDS
            else:
                total += median_run
        return total


def find_grid_log(spec: ExperimentSpec, experiment_name: str) -> Path | None:
    """The newest grid log belonging to one experiment (naming varies by launcher)."""
    candidates = [
        _REPO_ROOT / "logs" / f"{experiment_name}_grid.log",
        _REPO_ROOT / "logs" / f"cvc_{spec.label}_grid.log",
        _REPO_ROOT / "logs" / f"{spec.label}_grid.log",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def parse_grid_log(path: Path | None) -> dict[str, Any]:
    """Failures, per-run durations and the *current* run's progress from a grid log.

    ``training_log.csv`` is only written when a run finishes, so mid-training progress
    (epoch, per-epoch time, best val Dice so far) has to come from the log lines
    ``[i/n] fold f seed s: ...`` followed by ``[train] epoch e/E: ... val_dice=v ...``.
    """
    info: dict[str, Any] = {
        "failures": [],
        "durations": [],
        "current": {},
        "lines": [],
    }
    if path is None or not path.is_file():
        return info
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:  # pragma: no cover - unreadable log
        return info
    info["lines"] = lines
    failures: list[str] = []
    durations: list[float] = []
    current: dict[str, Any] = {}
    for line in lines:
        if "ERROR fold" in line:
            match = re.search(FOLD_SEED, line)
            if match:
                failures.append(f"fold{match.group(1)}/seed{match.group(2)}")
        elif line.startswith("failed:"):
            for fold, seed in re.findall(r"fold(\d+)/seed(\d+)", line):
                failures.append(f"fold{fold}/seed{seed}")
        header = re.search(r"\[\d+/\d+\] fold (\d+) seed (\d+): (\w+)", line)
        if header:
            current = {
                "fold": int(header.group(1)),
                "seed": int(header.group(2)),
                "status": header.group(3),
                "epoch": None,
                "total_epochs": None,
                "epoch_seconds": None,
                "best_val_dice": None,
                "last_val_dice": None,
            }
            continue
        epoch = re.match(
            r"\[train\] epoch (\d+)/(\d+): .*?val_dice=([0-9.eE+-]+) \(([0-9.]+)s",
            line,
        )
        if epoch and current:
            current["epoch"] = int(epoch.group(1))
            current["total_epochs"] = int(epoch.group(2))
            current["epoch_seconds"] = float(epoch.group(4))
            dice = float(epoch.group(3))
            if dice == dice:  # NaN-safe
                current["last_val_dice"] = dice
                best = current["best_val_dice"]
                current["best_val_dice"] = dice if best is None else max(best, dice)
        if line.startswith("[train] done:"):
            match = re.search(r"\|\s*([0-9.]+)\s*min\s*$", line.strip())
            if match:
                durations.append(float(match.group(1)) * 60.0)
    info["failures"] = sorted(set(failures))
    info["durations"] = durations
    info["current"] = current
    return info



def scan_experiment(spec: ExperimentSpec) -> ExperimentState:
    """Read one experiment's run directories and grid log (never writes anything)."""
    pairs = iter_runs(spec.folds, spec.seeds)
    if not spec.path.is_file():
        return ExperimentState(
            spec=spec,
            available=False,
            runs=[RunState(fold, seed, "unknown") for fold, seed in pairs],
        )
    config = load_config(spec.path)
    experiment_name = str(config.get("experiment.name", spec.label))
    checkpoint_dir = config.resolve(str(config.get("training.checkpoint_dir")))
    total_epochs = int(config.get("training.epochs", 0) or 0) or None

    runs: list[RunState] = []
    for fold, seed in pairs:
        run_dir = checkpoint_dir / f"fold{fold}_seed{seed}"
        per_case, _ = expected_eval_paths(config, fold, seed)
        state = RunState(fold, seed, run_status(run_dir, eval_marker_path(per_case)))
        state.total_epochs = total_epochs
        done_file = run_dir / "run_done.json"
        if done_file.is_file():
            try:
                payload = json.loads(done_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # pragma: no cover - corrupt marker
                payload = {}
            state.best_val_dice = _as_float(payload.get("best_value"))
            state.done_seconds = _as_float(payload.get("total_seconds"))
            state.epoch_seconds = _as_float(payload.get("median_epoch_seconds"))
            state.epoch = _as_int(payload.get("epochs_run"))
        epoch, best, median_epoch = read_training_log(run_dir / "training_log.csv")
        if state.epoch is None:
            state.epoch = epoch
        if state.best_val_dice is None:
            state.best_val_dice = best
        if state.epoch_seconds is None:
            state.epoch_seconds = median_epoch
        runs.append(state)

    log_path = find_grid_log(spec, experiment_name)
    info = parse_grid_log(log_path)
    current = info.get("current") or {}
    current_label = None
    if current:
        current_label = f"fold{current.get('fold')}/seed{current.get('seed')}"
        for run in runs:
            if (run.fold, run.seed) != (current.get("fold"), current.get("seed")):
                continue
            # training_log.csv only exists once a run finished: mid-training numbers come
            # from the grid log's own epoch lines
            if run.epoch is None:
                run.epoch = current.get("epoch")
            if run.total_epochs is None:
                run.total_epochs = current.get("total_epochs")
            if run.epoch_seconds is None:
                run.epoch_seconds = current.get("epoch_seconds")
            if run.best_val_dice is None:
                run.best_val_dice = current.get("best_val_dice")
    if not any(run.done_seconds for run in runs) and info["durations"]:
        # log-only fallback: runs that finished before run_done.json existed
        fallback = float(statistics.median(info["durations"]))
        for run in runs:
            if run.status == STATUS_SKIP and run.done_seconds is None:
                run.done_seconds = fallback
    return ExperimentState(
        spec=spec,
        available=True,
        runs=runs,
        failures=info["failures"],
        log_path=log_path,
        current_label=current_label,
    )


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

        return total


@dataclass
class QueueState:
    """Liveness and current step of the overnight queue."""

    name: str
    pid: int
    alive: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    current_step: str | None = None
    progress: str | None = None
    waiting_for: str | None = None
    finished: bool = False


def read_queue_state(queue_path: Path) -> QueueState | None:
    """Load the queue file, its pid file and the log-derived current step."""
    if not queue_path.is_file():
        return None
    try:
        data = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):  # pragma: no cover - unreadable queue
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name") or queue_path.stem)
    pid = read_pid_file(_REPO_ROOT / "logs" / f"{name}.pid")
    state = QueueState(
        name=name,
        pid=pid,
        alive=bool(pid) and pid_alive(pid, names=LAUNCHER_PROCESS_NAMES),
        steps=[dict(step) for step in data.get("steps") or []],
    )
    log_path = _REPO_ROOT / "logs" / f"{name}.log"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if "queue finished in" in line:
                state.finished = True
                break
            match = re.search(r"\[(\d+/\d+)\] (\S+) \((\w+)\)", line)
            if match:
                state.progress = match.group(1)
                state.current_step = match.group(2)
                break
            match = re.search(r"waiting for (\S+) \(pid (\d+)\)", line)
            if match:
                state.waiting_for = f"{match.group(1)} (pid {match.group(2)})"
                break
    return state


def pid_files() -> list[tuple[Path, int, bool]]:
    """Every ``logs/*.pid`` with its pid and liveness (recycled pids count as dead)."""
    entries: list[tuple[Path, int, bool]] = []
    for path in sorted((_REPO_ROOT / "logs").glob("*.pid")):
        pid = read_pid_file(path)
        entries.append(
            (path, pid, bool(pid) and pid_alive(pid, names=LAUNCHER_PROCESS_NAMES))
        )
    return entries


def step_log_path(step: dict[str, Any]) -> Path:
    """Log file of one queue step (mirrors ``scripts/run_queue.py``)."""
    name = str(step.get("name"))
    kind = str(step.get("kind", "grid")).strip().lower()
    suffix = "_grid.log" if kind == "grid" else f"_{kind}.log"
    return _REPO_ROOT / "logs" / f"{name}{suffix}"


def active_log(
    queue: QueueState | None, pids: Sequence[tuple[Path, int, bool]]
) -> tuple[Path | None, str]:
    """The log of the job that is actually running right now, and its label."""
    if queue is not None and queue.alive and queue.current_step:
        for step in queue.steps:
            if str(step.get("name")) == queue.current_step:
                path = step_log_path(step)
                if path.is_file():
                    return path, f"queue step {queue.current_step}"
    for path, _pid, alive in pids:
        if alive and path.stem != "overnight":
            log = path.with_suffix(".log")
            if log.is_file():
                return log, path.stem
    newest = sorted(
        (path for path in (_REPO_ROOT / "logs").glob("*_grid.log")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if newest:
        return newest[0], newest[0].stem
    return None, "none"


def tail_lines(path: Path | None, count: int) -> list[str]:
    """The last ``count`` non-empty lines of a log."""
    if path is None or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:  # pragma: no cover - unreadable log
        return []
    return [line for line in lines if line.strip()][-count:]


def queue_eta(
    queue: QueueState | None, states: dict[str, ExperimentState]
) -> tuple[float | None, list[tuple[str, float | None]], int]:
    """Remaining time of the queue: the current step and everything after it.

    Returns ``(known_total, per-step estimates, n_steps_without_data)``; a grid whose
    experiment has no finished run yet has no median time and therefore no ETA.
    """
    if queue is None or not queue.steps:
        return None, [], 0
    by_config = {spec.config.replace("\\", "/"): spec.label for spec in EXPERIMENTS}
    if queue.finished:
        remaining: list[dict[str, Any]] = []
    elif queue.current_step:
        start = next(
            (
                index
                for index, step in enumerate(queue.steps)
                if str(step.get("name")) == queue.current_step
            ),
            0,
        )
        remaining = list(queue.steps[start:])
    else:
        remaining = list(queue.steps)
    estimates: list[tuple[str, float | None]] = []
    total = 0.0
    unknown = 0
    for step in remaining:
        kind = str(step.get("kind", "grid")).strip().lower()
        if kind == "grid":
            label = by_config.get(str(step.get("config", "")).replace("\\", "/"))
            state = states.get(label) if label else None
            value = state.eta_seconds if state is not None else None
        else:
            value = POST_STEP_SECONDS
        estimates.append((str(step.get("name")), value))
        if value is None:
            unknown += 1
        else:
            total += value
    return total, estimates, unknown
def format_epoch(run: RunState) -> str:
    """``fold2/seed1 40/100`` for the active run."""
    if run.status == STATUS_EVALUATE:
        return f"{run.label} evaluating"
    if run.status == "unknown":
        return "-"
    if run.epoch is None:
        return run.label
    total = f"/{run.total_epochs}" if run.total_epochs else ""
    return f"{run.label} epoch {run.epoch}{total}"


def render(
    states: list[ExperimentState],
    queue: QueueState | None,
    pids: Sequence[tuple[Path, int, bool]],
    *,
    tail: int,
) -> str:
    """The whole report as one string (so it can be tested and printed)."""
    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"MELBA status  {stamp}  (read-only, CPU only)")

    # ---- liveness ----------------------------------------------------------
    if queue is not None:
        if queue.finished:
            where = "finished"
        elif queue.alive and queue.current_step:
            where = f"step {queue.progress or '?'} {queue.current_step}"
        elif queue.alive and queue.waiting_for:
            where = f"waiting for {queue.waiting_for}"
        elif queue.alive:
            where = "starting"
        else:
            where = "not running"
        age = f"pid {queue.pid}" if queue.pid else "no pid file"
        lines.append(f"queue   {queue.name:<10} {'ALIVE' if queue.alive else 'dead ':<5} {age:<14} {where}")

    alive_pids = [(path, pid) for path, pid, alive in pids if alive]
    dead_pids = [(path, pid) for path, pid, alive in pids if not alive]
    for path, pid in alive_pids:
        if queue is not None and path.stem == queue.name:
            continue
        image = process_name(pid) or "?"
        lines.append(f"grid    {path.stem:<22} ALIVE pid {pid} ({image})")
    if dead_pids:
        parts = []
        for path, pid in dead_pids:
            image = process_name(pid)
            note = f", pid reused by {image}" if image else ""
            parts.append(f"{path.stem} ({pid}{note})")
        lines.append("stale   " + ", ".join(parts))

    # ---- experiments -------------------------------------------------------
    header = f"{'experiment':<34}{'runs':>8}{'failed':>8}  {'current':<26}{'valDice':>16}{'ETA':>9}"
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for state in states:
        name = state.spec.label
        if not state.available:
            lines.append(f"{name:<34}{'config missing':>24}")
            continue
        best, best_run = state.best_val_dice
        mean = state.mean_best_val_dice
        valdice = (
            f"{best:.4f}/{mean:.4f}" if best is not None and mean is not None else "-"
        )
        active = state.active
        current = format_epoch(active) if active is not None else "-"
        unresolved = len(state.unresolved_failures)
        recovered = len(state.recovered_failures)
        failed = f"{unresolved}" + (f"+{recovered}r" if recovered else "")
        lines.append(
            f"{name:<34}{f'{state.done}/{state.total}':>8}{failed:>8}  "
            f"{current:<26}{valdice:>16}{format_duration(state.eta_seconds):>9}"
        )
    if lines and any(state.best_val_dice[0] is not None for state in states):
        best_pairs = [
            (state.spec.label, *state.best_val_dice) for state in states
        ]
        highlights = [
            f"{label} {value:.4f} ({run})"
            for label, value, run in best_pairs
            if value is not None and run
        ]
        lines.append("")
        lines.append("best so far  " + "; ".join(highlights))
    lines.append("")
    lines.append(
        "legend       runs=done/total evaluated | failed=unresolved failures, +Nr = the "
        "grid retried and recovered | valDice=best/mean of the per-run bests | "
        "ETA=median run time x pending runs"
    )

    # ---- queue ETA ---------------------------------------------------------
    by_label = {state.spec.label: state for state in states}
    total_eta, estimates, unknown = queue_eta(queue, by_label)
    if estimates:
        summary = format_duration(total_eta)
        if unknown:
            summary += f" + {unknown} step(s) without a median yet"
        lines.append("")
        lines.append(
            "queue ETA    "
            + summary
            + "  ("
            + ", ".join(f"{name} {format_duration(value)}" for name, value in estimates)
            + ")"
        )
    elif queue is not None and queue.finished:
        lines.append("")
        lines.append("queue ETA    done")

    # ---- active job tail ---------------------------------------------------
    log_path, label = active_log(queue, pids)
    lines.append("")
    lines.append(f"active job   {label}  ({log_path.relative_to(_REPO_ROOT) if log_path else '-'})")
    for line in tail_lines(log_path, tail):
        lines.append(f"  {line}")
    if not tail_lines(log_path, tail):
        lines.append("  (no log lines yet)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--queue", default="queues/overnight.yaml", help="queue YAML to report on"
    )
    parser.add_argument("--tail", type=int, default=3, help="log lines of the active job")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="restrict the experiment table to these labels",
    )
    args = parser.parse_args(argv)

    specs = [
        spec for spec in EXPERIMENTS if not args.only or spec.label in set(args.only)
    ]
    if not specs:
        print(f"ERROR: no experiment matches {args.only}", file=sys.stderr)
        return 2
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = _REPO_ROOT / queue_path
    states = [scan_experiment(spec) for spec in specs]
    print(
        render(
            states,
            read_queue_state(queue_path),
            pid_files(),
            tail=max(1, int(args.tail)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

