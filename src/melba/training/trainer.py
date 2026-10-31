"""Training loop with epoch-level checkpointing and exact resume.

Every epoch writes ``last.pt`` atomically (temp file + ``os.replace``) so a process
killed mid-save can never corrupt it.  ``last.pt`` contains the model, optimizer,
scheduler, AMP scaler, epoch counter, best validation value, early-stopping counter
and all RNG states (python, numpy, torch, cuda), so resuming continues the exact
same trajectory (including the augmentation stream, which is derived from
``(seed, epoch, index)``).
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from melba.config import Config, ConfigError
from melba.data import build_dataloader, build_dataset
from melba.models import build_model, count_parameters
from melba.utils import amp_is_enabled, describe_device, get_device, seed_everything

__all__ = [
    "TrainSummary",
    "atomic_torch_save",
    "build_loss",
    "capture_rng_state",
    "restore_rng_state",
    "run_dir_for",
    "train",
]


@dataclass(frozen=True)
class TrainSummary:
    """Where a training run stored its artefacts and how it went."""

    run_dir: Path
    checkpoint: Path
    last_checkpoint: Path
    log_path: Path
    done_path: Path
    fold: int
    seed: int
    epochs_run: int
    best_epoch: int
    best_value: float
    monitor: str
    total_seconds: float
    median_epoch_seconds: float
    resumed_from_epoch: int


def run_dir_for(config: Config, fold: int, seed: int) -> Path:
    """Per-run checkpoint directory (one directory per fold/seed pair)."""
    base = config.ensure_dir("training.checkpoint_dir")
    path = base / f"fold{int(fold)}_seed{int(seed)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Save ``payload`` so that a crash can never leave a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by training in a ``weights_only``-safe structure."""
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_keys": torch.from_numpy(numpy_state[1].astype(np.uint32).copy()),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": [],
    }
    if torch.cuda.is_available():
        state["cuda"] = [tensor for tensor in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG states written by :func:`capture_rng_state`."""
    python_state = state["python"]
    random.setstate(
        (int(python_state[0]), tuple(int(value) for value in python_state[1]), python_state[2])
    )
    # Tensor states may have been moved to a GPU by torch.load(map_location=...);
    # numpy and torch's RNG setters both need host tensors.
    np.random.set_state(
        (
            "MT19937",
            state["numpy_keys"].detach().cpu().numpy().astype(np.uint32),
            int(state["numpy_pos"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].detach().cpu())
    cuda_states = [tensor.detach().cpu() for tensor in (state.get("cuda") or [])]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def build_loss(config: Config) -> nn.Module:
    """Loss function declared by ``training.loss``."""
    name = str(config.get("training.loss", "bce")).strip().lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_dice":
        from monai.losses import DiceLoss

        dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        bce = nn.BCEWithLogitsLoss()

        class BCEWithDice(nn.Module):
            def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return 0.5 * bce(logits, target) + 0.5 * dice(logits, target)

        return BCEWithDice()
    raise ConfigError(f"Unknown training.loss {name!r}; expected 'bce' or 'bce_dice'")


def batch_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, int]:
    """Summed hard Dice over a batch, for validation monitoring."""
    prediction = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(
        denominator > 0,
        2 * intersection / denominator.clamp_min(1e-8),
        torch.ones_like(denominator),
    )
    return float(dice.sum().item()), int(dice.numel())


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_function(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / max(total_samples, 1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: Any,
    loss_function: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    dice_sum = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_function(logits, masks)
        total_loss += float(loss) * images.shape[0]
        batch_sum, batch_count = batch_dice(logits, masks, threshold=threshold)
        dice_sum += batch_sum
        count += batch_count
    return total_loss / max(count, 1), dice_sum / max(count, 1)


def train(
    config: Config,
    *,
    fold: int | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    resume: bool = True,
    progress: bool = False,
) -> TrainSummary:
    """Train one fold/seed run, resuming from ``last.pt`` when present."""
    fold = int(fold if fold is not None else (config.get("dataset.split.fold", 0) or 0))
    seed = int(seed if seed is not None else config.require("seed"))
    if int(config.require("seed")) != seed:
        raise ConfigError(
            f"train() was called with seed={seed} but the config declares seed="
            f"{config.require('seed')}; override seed in the config so every artefact agrees"
        )
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    device = get_device(config.get("device", "auto"))
    print(
        f"[train] fold {fold} seed {seed} | device {describe_device(device)} | "
        f"config {config.fingerprint}"
    )

    train_dataset = build_dataset(config, "train", train=True, fold=fold)
    val_dataset = build_dataset(config, "val", train=False, fold=fold)
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    run_dir = run_dir_for(config, fold, seed)
    checkpoint_path = run_dir / str(config.get("training.checkpoint_name", "best.pt"))
    last_path = run_dir / "last.pt"
    log_path = run_dir / "training_log.csv"
    done_path = run_dir / "run_done.json"

    model = build_model(config).to(device)
    loss_function = build_loss(config).to(device)
    total_epochs = int(epochs or config.require("training.epochs"))
    use_amp = amp_is_enabled(config.get("training.amp", True), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.require("training.learning_rate")),
        weight_decay=float(config.get("training.weight_decay", 0.0)),
    )
    scheduler_name = str(config.get("training.scheduler", "cosine")).lower()
    if scheduler_name not in {"cosine", "none"}:
        raise ConfigError(
            f"Unknown training.scheduler {scheduler_name!r}; expected 'cosine' or 'none'"
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        if scheduler_name == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    monitor = str(config.get("training.monitor", "val_loss")).lower()
    if monitor not in {"val_loss", "val_dice"}:
        raise ConfigError(f"training.monitor must be 'val_loss' or 'val_dice', got {monitor!r}")
    patience = int(config.get("training.early_stopping_patience", 0))
    threshold = float(config.get("inference.threshold", 0.5))

    start_epoch = 0
    best_value = float("inf") if monitor == "val_loss" else float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    resumed_from_epoch = 0
    if resume and last_path.is_file():
        # CPU mapping: the payload also holds RNG states, which must stay on the host
        # (optimizer state is moved back to the parameter device by load_state_dict).
        payload = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"])
        best_value = float(payload["best_value"])
        best_epoch = int(payload.get("best_epoch", 0))
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        restore_rng_state(payload["rng"])
        resumed_from_epoch = start_epoch
        print(
            f"[train] resumed from {last_path.name} at epoch {start_epoch}/{total_epochs} "
            f"(best {monitor}={best_value:.4f})"
        )

    history: list[dict[str, Any]] = []
    iterator: Any = range(start_epoch, total_epochs)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, initial=start_epoch, total=total_epochs, desc="training")

    started_all = time.perf_counter()
    epochs_run = start_epoch
    for epoch in iterator:
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, scaler, device, use_amp
        )
        trained_at = time.perf_counter()
        val_loss, val_dice = validate(model, val_loader, loss_function, device, use_amp, threshold)
        validated_at = time.perf_counter()
        train_seconds = trained_at - epoch_started
        val_seconds = validated_at - trained_at
        epoch_seconds = validated_at - epoch_started
        if scheduler is not None:
            scheduler.step()

        current = val_loss if monitor == "val_loss" else val_dice
        improved = current < best_value if monitor == "val_loss" else current > best_value
        if improved:
            best_value = current
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "monitor": monitor,
                    "best_value": best_value,
                    "config_hash": config.fingerprint,
                    "config_path": str(config.path) if config.path else None,
                    "seed": seed,
                    "fold": fold,
                    "torch_version": str(torch.__version__),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "monitor": monitor,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
            }
        )
        epochs_run = epoch + 1
        print(
            f"[train] epoch {epoch + 1:03d}/{total_epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"({epoch_seconds:.1f}s: train {train_seconds:.1f}s / val {val_seconds:.1f}s)"
            + (" *" if improved else "")
        )

        # Written every epoch: kill the job and the next start continues here.
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "monitor": monitor,
                "rng": capture_rng_state(),
                "config_hash": config.fingerprint,
                "seed": seed,
                "fold": fold,
                "torch_version": str(torch.__version__),
            },
            last_path,
        )

        if patience and epochs_without_improvement >= patience:
            print(f"[train] early stopping after {patience} epochs without improvement")
            break

    total_seconds = time.perf_counter() - started_all
    median_epoch = float(np.median([row["epoch_seconds"] for row in history])) if history else 0.0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[0]) if history else ["epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    done_path.write_text(
        json.dumps(
            {
                "fold": fold,
                "seed": seed,
                "epochs_run": epochs_run,
                "resumed_from_epoch": resumed_from_epoch,
                "best_epoch": best_epoch,
                "best_value": best_value,
                "monitor": monitor,
                "total_seconds": total_seconds,
                "median_epoch_seconds": median_epoch,
                "config_hash": config.fingerprint,
                "torch_version": str(torch.__version__),
                "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[train] done: best {monitor}={best_value:.4f} at epoch {best_epoch} | "
        f"checkpoint {checkpoint_path} | {total_seconds / 60:.1f} min"
    )

    return TrainSummary(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        last_checkpoint=last_path,
        log_path=log_path,
        done_path=done_path,
        fold=fold,
        seed=seed,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_value=best_value,
        monitor=monitor,
        total_seconds=total_seconds,
        median_epoch_seconds=median_epoch,
        resumed_from_epoch=resumed_from_epoch,
    )


