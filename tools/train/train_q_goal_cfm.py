#!/usr/bin/env python3
"""Train a conditional flow matching q_goal baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.learning.q_goal.checkpoint import load_checkpoint, save_checkpoint
from prosthetic_grasp.learning.q_goal.dataset import (
    QGoalDataset,
    QGoalDatasetConfig,
    build_task_vocab,
    read_jsonl,
)
from prosthetic_grasp.learning.q_goal.encoders import ConditionEncoderConfig
from prosthetic_grasp.learning.q_goal.flow import build_flow_matcher
from prosthetic_grasp.learning.q_goal.model import QGoalCFMModel, QGoalCFMModelConfig


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["meta", "simple"], default="meta")
    parser.add_argument("--image-encoder", choices=["small_cnn", "torchvision_resnet18", "siglip", "precomputed", "none"], default="small_cnn")
    parser.add_argument("--lollipop-mask-encoder", choices=["small_cnn", "none"], default="small_cnn")
    parser.add_argument("--task-encoder", choices=["embedding", "siglip", "precomputed", "none"], default="embedding")
    parser.add_argument("--siglip-model-id", default="google/siglip-base-patch16-224")
    parser.add_argument("--unfreeze-siglip", action="store_true")
    parser.add_argument("--no-lollipop-params", action="store_true")
    parser.add_argument("--no-q-current", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-size", type=int, default=224)
    parser.add_argument("--image-embedding-dim", type=int, default=0)
    parser.add_argument("--task-embedding-dim", type=int, default=0)
    parser.add_argument("--condition-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--quality-min", type=float)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, help="Resume training from a checkpoint.")
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def make_dataset(path: Path, args: argparse.Namespace, task_vocab: dict[str, int]) -> QGoalDataset:
    return QGoalDataset(
        QGoalDatasetConfig(
            jsonl_path=path,
            image_size=args.image_size,
            mask_size=args.mask_size,
            require_rgb=args.image_encoder not in {"none", "precomputed"},
            use_lollipop_mask=args.lollipop_mask_encoder != "none",
            use_lollipop_params=not args.no_lollipop_params,
            use_q_current=not args.no_q_current,
            task_vocab=task_vocab,
            task_embedding_dim=args.task_embedding_dim if args.task_encoder == "precomputed" else 0,
            image_embedding_dim=args.image_embedding_dim if args.image_encoder == "precomputed" else 0,
            quality_min=args.quality_min,
        )
    )


def train_epoch(
    model: QGoalCFMModel,
    flow_matcher: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["q_goal"]
        flow_batch = flow_matcher.sample(target)
        pred = model(flow_batch.x_t, flow_batch.t, batch)
        loss = F.mse_loss(pred, flow_batch.target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * target.shape[0]
        count += target.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def eval_epoch(model: QGoalCFMModel, flow_matcher: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["q_goal"]
        flow_batch = flow_matcher.sample(target)
        pred = model(flow_batch.x_t, flow_batch.t, batch)
        loss = F.mse_loss(pred, flow_batch.target_velocity)
        total += float(loss.detach().cpu()) * target.shape[0]
        count += target.shape[0]
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = read_jsonl(args.train_jsonl)
    task_vocab = build_task_vocab(train_samples) if args.task_encoder == "embedding" else {}
    train_set = make_dataset(args.train_jsonl, args, task_vocab)
    val_set = make_dataset(args.val_jsonl, args, task_vocab) if args.val_jsonl else None

    condition_config = ConditionEncoderConfig(
        image_encoder=args.image_encoder,
        image_embedding_dim=args.image_embedding_dim,
        lollipop_mask_encoder=args.lollipop_mask_encoder,
        use_lollipop_params=not args.no_lollipop_params,
        task_encoder=args.task_encoder,
        task_vocab_size=max(len(task_vocab), 1),
        task_embedding_dim=args.task_embedding_dim,
        siglip_model_id=args.siglip_model_id,
        freeze_siglip=not args.unfreeze_siglip,
        use_q_current=not args.no_q_current,
        condition_dim=args.condition_dim,
        q_dim=train_set.q_dim,
        dropout=args.dropout,
    )
    model_config = QGoalCFMModelConfig(
        q_dim=train_set.q_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        condition=condition_config,
    )
    model = QGoalCFMModel(model_config).to(device)
    flow_matcher = build_flow_matcher(args.backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer_state = ckpt.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(ckpt.get("epoch", 0))
        if ckpt.get("val_loss") is not None:
            best_val = float(ckpt["val_loss"])
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = load_checkpoint(best_path, map_location="cpu")
            if best_ckpt.get("val_loss") is not None:
                best_val = min(best_val, float(best_ckpt["val_loss"]))
        print(f"resumed_from={args.resume} start_epoch={start_epoch} best_val={best_val:.6f}", flush=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_set is not None
        else None
    )

    run_config = {
        "args": vars(args),
        "task_vocab": task_vocab,
        "model_config": asdict(model_config),
        "q_min": train_set.normalizer.q_min.cpu().tolist(),
        "q_max": train_set.normalizer.q_max.cpu().tolist(),
    }
    run_config = as_jsonable(run_config)
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_epoch(model, flow_matcher, train_loader, optimizer, device)
        val_loss = eval_epoch(model, flow_matcher, val_loader, device) if val_loader is not None else train_loss
        print(f"epoch={epoch:04d} train_flow_mse={train_loss:.6f} val_flow_mse={val_loss:.6f}", flush=True)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": run_config,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.output_dir / "best.pt", payload)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(args.output_dir / f"epoch_{epoch:04d}.pt", payload)


if __name__ == "__main__":
    main()
