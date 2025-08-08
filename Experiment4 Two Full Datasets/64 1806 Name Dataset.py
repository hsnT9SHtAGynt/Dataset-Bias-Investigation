#!/usr/bin/env python
"""
train_resnet18.py
=================
Standalone PyTorch training script to train ResNet-18 on a custom image classification dataset.
Uses validation loss for early stopping and allows specifying an output directory via command-line.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm


class AverageMeter:
    """Track and update running averages for metrics like loss and accuracy."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Compute top-k accuracy for the given outputs and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomImageDataset(Dataset):
    """PyTorch Dataset for images defined by a metadata CSV with columns:
        - filepath: path to the image file
        - new_label_id: integer class ID
        - split: one of 'train', 'validation', or 'test'
    """

    def __init__(self, csv_path: str, split: str, transform: transforms.Compose):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        self.df = pd.read_csv(
            csv_path,
            dtype={"cifar_fine_id": str},
            low_memory=False
        )

        for col in ("filepath", "new_label_id", "split"):
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from CSV")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples with split='{split}' found in metadata.")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["new_label_id"])
        return image, label


def train_one_epoch(model: nn.Module, criterion, optimizer, loader: DataLoader, device: torch.device):
    """Train the model for one epoch and return average loss, top-1 and top-5 accuracy."""
    model.train()
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, (1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "top1": f"{top1_meter.avg:.2f}%",
            "top5": f"{top5_meter.avg:.2f}%"
        })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def validate(model: nn.Module, criterion, loader: DataLoader, device: torch.device, split_name: str = "Val"):
    """Evaluate the model on validation or test set and return loss, top-1 and top-5 accuracy."""
    model.eval()
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc=split_name, leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, (1, 5))

            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1.item(), images.size(0))
            top5_meter.update(acc5.item(), images.size(0))
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
                "top5": f"{top5_meter.avg:.2f}%"
            })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def main():
    # Use CUDA if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train ResNet-18 (from scratch) on a custom dataset")
    parser.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/60Dataset06_Bicubic/metadata.csv",
                        help="Path to metadata CSV containing (filepath, new_label_id, split)")
    parser.add_argument("--output-dir", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1806",
                        help="Directory to save models, logs, and checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = CustomImageDataset(args.data_csv, "train", train_tf)
    val_ds   = CustomImageDataset(args.data_csv, "validation", test_tf)
    test_ds  = CustomImageDataset(args.data_csv, "test", test_tf)

    num_classes = int(pd.concat([
        train_ds.df["new_label_id"],
        val_ds.df["new_label_id"],
        test_ds.df["new_label_id"],
    ]).max() + 1)
    print(f"Detected {num_classes} classes.")

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)

    log_file = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "epoch", "lr", "train_loss", "train_top1", "train_top5",
                "val_loss", "val_top1", "val_top5",
            ])

    best_loss = float('inf')
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_top1, train_top5 = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_top1, val_top5 = validate(model, criterion, val_loader, device, split_name="Val")

        scheduler.step(val_loss)

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), epoch, f"{current_lr:.3e}",
                f"{train_loss:.4f}", f"{train_top1:.2f}", f"{train_top5:.2f}",
                f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}",
            ])

        print(
            f"Epoch {epoch}: lr={current_lr:.3e}, "
            f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}%, train_top5={train_top5:.2f}%, "
            f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}%, val_top5={val_top5:.2f}%"
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_path = os.path.join(args.output_dir, "best_resnet18.pth")
            torch.save(model.state_dict(), best_path)
            print("*** New best model saved! ***")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(f"Early stopping after {args.early_stop_patience} epochs without improvement on val_loss.")
                break

        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"resnet18_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Evaluate best model on test set
    print("\nEvaluating best model on the test set ...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_resnet18.pth"), map_location=device))
    test_loss, test_top1, test_top5 = validate(model, criterion, test_loader, device, split_name="Test")
    print(f"Test loss={test_loss:.4f}, Test top1={test_top1:.2f}%, Test top5={test_top5:.2f}%")

    print(f"\nTraining complete. Best validation epoch: {best_epoch} (Best_Val_loss={best_loss:.4f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)

