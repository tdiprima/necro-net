"""
Trains a Vision Transformer (ViT) model on a custom dataset using PyTorch,
with the option to save the best model based on validation accuracy.
Optimized for multi-core CPU training (default: 70 threads for 72-core systems).

Example usage:
  uv run train_raj_vit.py --epochs 8 --batch_size 16 --output_dir models
  uv run train_raj_vit.py --epochs 10 --cpu_threads 72 --num_workers 20

You can override the default paths if needed using the --root_dir argument.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from raj_dataset import RajDataset
from rich_argparse import RichHelpFormatter
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm


def get_vit_model(num_classes, device, pretrained=True):
    """
    Loads torchvision ViT and replaces the head.
    """
    try:
        # torchvision >= 0.13 style
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
        )
        # model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    except Exception:
        # fallback older API
        model = models.vit_b_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in tqdm(loader, desc="train", ncols=120):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="eval", ncols=120):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    logger.add("training.log", rotation="10 MB", retention="7 days", level="INFO")
    logger.info("Starting ViT training script")

    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(
        description="Train ViT on Raj dataset",
        formatter_class=RichHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=f"{home}/local_data/train",
        help="root folder with class subfolders",
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=70,
        help="Number of CPU threads for PyTorch computation",
    )
    args = parser.parse_args()

    # device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device:", device)

    # Optimize CPU threading for 72 cores
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(8)
        logger.info(f"CPU threads set to: {args.cpu_threads}, interop threads: 8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset
    ds = RajDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
    )
    n = len(ds)
    val_n = int(n * args.val_split)
    train_n = n - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    num_classes = ds.num_classes()
    logger.info(
        f"Dataset size: {n}, train: {train_n}, val: {val_n}, classes: {num_classes}"
    )
    logger.info("Classes:", ds.class_names())

    # Model
    model = get_vit_model(num_classes, device, pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    total_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        since = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - since
        logger.info(f"Epoch {epoch}/{args.epochs} — time: {elapsed:.1f}s")
        logger.info(f"  Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        logger.info(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"vit_epoch{epoch}.pth"))

        # keep best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "vit_best.pth")
            )
            logger.info("  Saved best model.")

    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training finished. Best val acc: {best_val_acc}")
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

    # With pretrained model:
    # Epoch 10/10 — time: 2493.6s
    # Train loss: 0.0088  acc: 0.9970
    # Val   loss: 0.3399  acc: 0.9290
    # Training finished. Best val acc: 0.9402219140083218


if __name__ == "__main__":
    main()
