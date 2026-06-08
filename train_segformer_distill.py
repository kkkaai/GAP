from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.common.segformer_distill import (
    MaskDistillDataset,
    append_csv_row,
    average_metric_dicts,
    broadcast_arg,
    dataclass_to_dict,
    load_segformer_checkpoint,
    load_segformer_model,
    parse_multi_arg,
    save_training_manifest,
    set_seed,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SegFormer distillation from SegGPT mechanical-hand masks")
    parser.add_argument("--data_root", default="real_finger_dataset", help="Single dataset root.")
    parser.add_argument("--img_subdir", default="images", help="Image directory or subdir under data_root.")
    parser.add_argument("--mask_dir", default="", help="SegGPT pseudo-mask directory. Relative paths are under data_root.")
    parser.add_argument("--train_split", default="train.txt")
    parser.add_argument("--val_split", default="val.txt")
    parser.add_argument("--data_roots", default="", help="Comma-separated roots for multi-domain training.")
    parser.add_argument("--img_subdirs", default="", help="Comma-separated image dirs aligned with data_roots.")
    parser.add_argument("--mask_dirs", default="", help="Comma-separated mask dirs aligned with data_roots.")
    parser.add_argument("--train_splits", default="", help="Comma-separated train splits aligned with data_roots.")
    parser.add_argument("--val_splits", default="", help="Comma-separated val splits aligned with data_roots.")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--class_weight_fg", type=float, default=1.0, help="Foreground CE class weight.")
    parser.add_argument("--dice_weight", type=float, default=0.0, help="Optional foreground Dice loss weight.")
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=6)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_root", default="weights")
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--model_name", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--color_jitter", action="store_true")
    parser.add_argument("--brightness", type=float, default=0.35)
    parser.add_argument("--contrast", type=float, default=0.30)
    parser.add_argument("--saturation", type=float, default=0.15)
    parser.add_argument("--hue", type=float, default=0.03)
    parser.add_argument("--gray_prob", type=float, default=0.05)
    return parser


def make_datasets(args: argparse.Namespace) -> tuple[ConcatDataset | MaskDistillDataset, ConcatDataset | MaskDistillDataset]:
    data_roots = parse_multi_arg(args.data_roots) if args.data_roots else [args.data_root]
    n_domains = len(data_roots)
    img_subdirs = broadcast_arg(parse_multi_arg(args.img_subdirs), n_domains, "img_subdirs")
    mask_dirs = broadcast_arg(parse_multi_arg(args.mask_dirs), n_domains, "mask_dirs")
    train_splits = broadcast_arg(parse_multi_arg(args.train_splits), n_domains, "train_splits")
    val_splits = broadcast_arg(parse_multi_arg(args.val_splits), n_domains, "val_splits")

    if n_domains == 1:
        img_subdirs[0] = img_subdirs[0] or args.img_subdir
        mask_dirs[0] = mask_dirs[0] or args.mask_dir
        train_splits[0] = train_splits[0] or args.train_split
        val_splits[0] = val_splits[0] or args.val_split
    if any((mask_dir is None or str(mask_dir).strip() == "") for mask_dir in mask_dirs):
        raise ValueError("mask_dir/mask_dirs must be provided for all domains.")

    train_sets = []
    val_sets = []
    for i, root in enumerate(data_roots):
        img_subdir = img_subdirs[i] or args.img_subdir
        mask_dir = mask_dirs[i] or args.mask_dir
        train_split = train_splits[i] or args.train_split
        val_split = val_splits[i] or args.val_split
        train_set = MaskDistillDataset(
            root,
            img_subdir,
            mask_dir,
            train_split,
            img_size=args.img_size,
            mask_threshold=args.mask_threshold,
            is_training=True,
            color_jitter=args.color_jitter,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            gray_prob=args.gray_prob,
        )
        val_set = MaskDistillDataset(
            root,
            img_subdir,
            mask_dir,
            val_split,
            img_size=args.img_size,
            mask_threshold=args.mask_threshold,
            is_training=False,
        )
        print(f"[Domain {i + 1}/{n_domains}] root={root} | train={len(train_set)} | val={len(val_set)}")
        train_sets.append(train_set)
        val_sets.append(val_set)

    train_ds = train_sets[0] if n_domains == 1 else ConcatDataset(train_sets)
    val_ds = val_sets[0] if n_domains == 1 else ConcatDataset(val_sets)
    return train_ds, val_ds


def dice_loss_from_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = masks.float()
    eps = 1e-6
    intersection = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def step_loss(logits: torch.Tensor, masks: torch.Tensor, class_weights: torch.Tensor | None, dice_weight: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, masks, weight=class_weights)
    if dice_weight <= 0:
        return ce
    return ce + dice_weight * dice_loss_from_logits(logits, masks)


def run_epoch(
    *,
    model,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
    dice_weight: float,
):
    from prosthetic_grasp.common.segformer_distill import metric_from_logits

    is_train = optimizer is not None
    model.train(is_train)
    metrics = []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(pixel_values=imgs).logits
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = step_loss(logits, masks, class_weights, dice_weight)
            if is_train:
                loss.backward()
                optimizer.step()
        metrics.append(metric_from_logits(logits.detach(), masks, loss.detach()))
    return average_metric_dicts(metrics)


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_name.strip()
    name = f"SegFormerB0_mechhand_distill_{tag}_{timestamp}" if tag else f"SegFormerB0_mechhand_distill_{timestamp}"
    log_dir = Path(args.output_root) / name
    log_dir.mkdir(parents=True, exist_ok=False)

    save_training_manifest(log_dir / "config.json", vars(args))
    train_ds, val_ds = make_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_segformer_model(
        model_name=args.model_name,
        num_labels=2,
        local_files_only=args.local_files_only,
        device=device,
    )
    start_epoch = 0
    best_val = float("inf")
    if args.resume_ckpt:
        checkpoint = load_segformer_checkpoint(model, args.resume_ckpt, device)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val_loss", best_val))
        print(f"Loaded resume checkpoint: {args.resume_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    class_weights = None
    if args.class_weight_fg != 1.0:
        class_weights = torch.tensor([1.0, args.class_weight_fg], dtype=torch.float32, device=device)

    header = [
        "epoch",
        "lr",
        "train_loss",
        "train_fg_iou",
        "train_fg_dice",
        "train_fg_precision",
        "train_fg_recall",
        "val_loss",
        "val_fg_iou",
        "val_fg_dice",
        "val_fg_precision",
        "val_fg_recall",
        "val_pred_fg_ratio",
        "val_gt_fg_ratio",
    ]
    log_path = log_dir / "training_log.csv"
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            dice_weight=args.dice_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                class_weights=class_weights,
                dice_weight=args.dice_weight,
            )
        scheduler.step(val_metrics.loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f} dice {train_metrics.fg_dice:.4f} | "
            f"val loss {val_metrics.loss:.4f} iou {val_metrics.fg_iou:.4f} "
            f"dice {val_metrics.fg_dice:.4f} recall {val_metrics.fg_recall:.4f} | "
            f"lr {cur_lr:.8f}"
        )
        append_csv_row(
            log_path,
            {
                "epoch": epoch + 1,
                "lr": f"{cur_lr:.8f}",
                "train_loss": f"{train_metrics.loss:.6f}",
                "train_fg_iou": f"{train_metrics.fg_iou:.6f}",
                "train_fg_dice": f"{train_metrics.fg_dice:.6f}",
                "train_fg_precision": f"{train_metrics.fg_precision:.6f}",
                "train_fg_recall": f"{train_metrics.fg_recall:.6f}",
                "val_loss": f"{val_metrics.loss:.6f}",
                "val_fg_iou": f"{val_metrics.fg_iou:.6f}",
                "val_fg_dice": f"{val_metrics.fg_dice:.6f}",
                "val_fg_precision": f"{val_metrics.fg_precision:.6f}",
                "val_fg_recall": f"{val_metrics.fg_recall:.6f}",
                "val_pred_fg_ratio": f"{val_metrics.pred_fg_ratio:.6f}",
                "val_gt_fg_ratio": f"{val_metrics.gt_fg_ratio:.6f}",
            },
            header=header,
        )

        latest_payload = {
            "epoch": epoch + 1,
            "best_val_loss": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": dataclass_to_dict(val_metrics),
        }
        torch.save(latest_payload, log_dir / "latest_checkpoint.pth")

        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            patience_counter = 0
            best_payload = dict(latest_payload)
            best_payload["best_val_loss"] = best_val
            torch.save(best_payload, log_dir / "best_checkpoint.pth")
            torch.save(model.state_dict(), log_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"Training finished. Best val loss: {best_val:.4f}")
    print(f"Saved to: {log_dir}")


if __name__ == "__main__":
    main()
