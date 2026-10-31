"""Train a segmentation model with validation-only checkpoint selection.

Usage
-----
    python scripts/train.py --config configs/cvc_clinicdb/unet_resnet50.yaml

The training split is used for optimisation, the validation split for model
selection, and the test split is never touched here (``scripts/evaluate.py``
refuses to report the selection split by default).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from melba.config import Config, ConfigError, load_config, parse_set_args  # noqa: E402
from melba.data import SplitError, build_dataloader, build_dataset  # noqa: E402
from melba.models import build_model, count_parameters  # noqa: E402
from melba.utils import amp_is_enabled, describe_device, get_device, seed_everything  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML in configs/")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--fold", type=int, default=None, help="fold index (default dataset.split.fold)")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, parse_set_args(args.overrides))
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    seed = seed_everything(
        int(config.require("seed")), deterministic=bool(config.get("deterministic", True))
    )
    device = get_device(config.get("device", "auto"))
    fold = int(
        args.fold
        if args.fold is not None
        else (config.get("dataset.split.fold", 0) or 0)
    )
    print(
        f"device: {describe_device(device)} | seed: {seed} | fold: {fold} | "
        f"config: {config.fingerprint}"
    )

    try:
        train_dataset = build_dataset(config, "train", train=True, fold=fold)
        val_dataset = build_dataset(config, "val", train=False, fold=fold)
    except (ConfigError, SplitError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)
    print(
        f"train: {len(train_dataset)} cases ({len(train_loader)} batches) | "
        f"val: {len(val_dataset)} cases"
    )

    model = build_model(config).to(device)
    print(
        f"model: {config.get('model.architecture')} "
        f"({count_parameters(model):,} trainable parameters)"
    )

    loss_function = build_loss(config).to(device)
    epochs = int(args.epochs or config.require("training.epochs"))
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
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler_name == "cosine"
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = config.ensure_dir("training.checkpoint_dir")
    if config.get("dataset.split.output_pattern"):
        # one checkpoint per fold so cross-validation runs never overwrite each other
        checkpoint_dir = checkpoint_dir / f"fold{fold}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / str(config.get("training.checkpoint_name", "best.pt"))
    log_path = checkpoint_dir / "training_log.csv"
    monitor = str(config.get("training.monitor", "val_loss")).lower()
    if monitor not in {"val_loss", "val_dice"}:
        raise ConfigError(f"training.monitor must be 'val_loss' or 'val_dice', got {monitor!r}")
    patience = int(config.get("training.early_stopping_patience", 0))
    threshold = float(config.get("inference.threshold", 0.5))

    best_value = float("inf") if monitor == "val_loss" else float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    iterator: Any = range(epochs)
    if not args.no_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="training", unit="epoch")

    for epoch in iterator:
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, scaler, device, use_amp
        )
        val_loss, val_dice = validate(
            model, val_loader, loss_function, device, use_amp, threshold
        )
        if scheduler is not None:
            scheduler.step()

        current = val_loss if monitor == "val_loss" else val_dice
        improved = current < best_value if monitor == "val_loss" else current > best_value
        if improved:
            best_value = current
            epochs_without_improvement = 0
            torch.save(
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

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "monitor": monitor,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}" + (" *" if improved else "")
        )

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print(f"best {monitor}: {best_value:.4f}")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"training log : {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

