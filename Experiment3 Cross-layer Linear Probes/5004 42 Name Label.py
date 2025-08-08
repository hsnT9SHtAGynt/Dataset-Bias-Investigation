#!/usr/bin/env python
"""
train_resnet50_with_logging.py
================================
Train a ResNet-50 from scratch on a custom dataset defined via a metadata CSV.

Key features:
-------------
1. Logs all stdout/stderr output to BOTH the console and a timestamped file
   (`output_log.txt` in the output directory) for reproducibility.
2. Uses a CSV metadata file that specifies:
       - filepath (path to image)
       - new_label_id (integer label)
       - split (train / validation / test)
3. Tracks training/validation loss & accuracy (Top-1 and Top-5) each epoch.
4. Saves:
       - `training_log.csv` (per-epoch metrics)
       - Best model checkpoint (`best_resnet50.pth`)
       - Periodic checkpoints every 10 epochs.
5. Supports early stopping based on validation loss stagnation.
6. Final evaluation is performed on the test set using the best model.

Author: <your-name>
Created: 2025-XX-XX
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
from torchvision.models import resnet50
from tqdm.auto import tqdm


# -------------------------------------------------------------------------
# Utility: Running average tracker for metrics
# -------------------------------------------------------------------------
class AverageMeter:
    """
    Keeps track of a metric's current value, sum, count, and running average.
    Used for tracking loss, Top-1 accuracy, and Top-5 accuracy.
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update tracker with a new value.

        Args:
            val (float): New metric value.
            n (int): Number of samples contributing to this value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# -------------------------------------------------------------------------
# Utility: Accuracy computation
# -------------------------------------------------------------------------
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """
    Compute the top-k accuracy for the specified values of k.

    Args:
        output (Tensor): Model logits of shape [batch_size, num_classes].
        target (Tensor): Ground truth labels of shape [batch_size].
        topk (tuple[int]): Which top-k accuracies to compute (e.g., (1, 5)).

    Returns:
        list[Tensor]: Accuracies in percentage for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get indices of top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# -------------------------------------------------------------------------
# Dataset class for loading images from a CSV metadata file
# -------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    """
    PyTorch Dataset that loads image file paths & labels from a CSV file.

    CSV format must contain columns:
        filepath        : path to image file
        new_label_id    : integer class label
        split           : 'train', 'validation', or 'test'

    Args:
        csv_path (str): Path to metadata CSV file.
        split (str): Which split to load ("train" / "validation" / "test").
        transform (transforms.Compose): Torchvision transforms to apply.
    """

    def __init__(self, csv_path: str, split: str, transform: transforms.Compose):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        # Read CSV into dataframe
        self.df = pd.read_csv(csv_path, dtype={"cifar_fine_id": str}, low_memory=False)

        # Sanity check required columns
        for col in ("filepath", "new_label_id", "split"):
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from CSV")

        # Filter by split (train/val/test)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples with split='{split}' found in metadata.")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image.
            label (int): Class index.
        """
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["new_label_id"])
        return image, label


# -------------------------------------------------------------------------
# One training epoch
# -------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, criterion, optimizer, loader: DataLoader, device: torch.device):
    """
    Perform one full pass over the training dataset.

    Returns:
        tuple: (avg_loss, avg_top1_acc, avg_top5_acc)
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backpropagation & parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        acc1, acc5 = accuracy(outputs, targets, (1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))

        # Display current averages in progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "top1": f"{top1_meter.avg:.2f}%",
            "top5": f"{top5_meter.avg:.2f}%"
        })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# -------------------------------------------------------------------------
# Validation or test evaluation
# -------------------------------------------------------------------------
def validate(model: nn.Module, criterion, loader: DataLoader, device: torch.device, split_name: str = "Val"):
    """
    Evaluate model performance on a validation/test dataset.

    Returns:
        tuple: (avg_loss, avg_top1_acc, avg_top5_acc)
    """
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


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------
def main():
    # -----------------------------------------------------
    # Parse CLI arguments
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(description="Train ResNet-50 (from scratch) on a custom dataset")
    parser.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv",
                        help="Path to metadata CSV containing (filepath, new_label_id, split)")
    parser.add_argument("--output-dir", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet5004",
                        help="Directory to save models, logs, and checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    args = parser.parse_args()

    # -----------------------------------------------------
    # Prepare output directory & logging
    # -----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Duplicate stdout/stderr to a log file
    log_path = os.path.join(args.output_dir, "output_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    class Tee:
        """Redirect stdout/stderr to multiple file-like objects."""
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # -----------------------------------------------------
    # Device setup
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------------------------------------------
    # Data transforms
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Dataset & DataLoader setup
    # -----------------------------------------------------
    train_ds = CustomImageDataset(args.data_csv, "train", train_tf)
    val_ds = CustomImageDataset(args.data_csv, "validation", test_tf)
    test_ds = CustomImageDataset(args.data_csv, "test", test_tf)

    # Automatically determine number of classes
    num_classes = int(pd.concat([
        train_ds.df["new_label_id"],
        val_ds.df["new_label_id"],
        test_ds.df["new_label_id"]
    ]).max() + 1)
    print(f"[INFO] Detected {num_classes} classes.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # -----------------------------------------------------
    # Model, loss, optimizer, scheduler
    # -----------------------------------------------------
    model = resnet50(weights=None)  # From scratch (no pretrained weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for dataset
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)

    # -----------------------------------------------------
    # CSV logging setup
    # -----------------------------------------------------
    csv_log = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(csv_log):
        with open(csv_log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "epoch", "lr",
                "train_loss", "train_top1", "train_top5",
                "val_loss", "val_top1", "val_top5"
            ])

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    best_loss = float('inf')
    best_epoch = 0
    stale_epochs = 0  # For early stopping counter

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}] Starting...")
        lr_curr = optimizer.param_groups[0]["lr"]

        # Train + validate
        train_loss, train_top1, train_top5 = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_top1, val_top5 = validate(model, criterion, val_loader, device)

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Append epoch results to CSV log
        now = datetime.now().isoformat()
        with open(csv_log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                now, epoch, f"{lr_curr:.3e}",
                f"{train_loss:.4f}", f"{train_top1:.2f}", f"{train_top5:.2f}",
                f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}"
            ])

        print(f"[Stats] lr={lr_curr:.3e}, "
              f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}%, train_top5={train_top5:.2f}% | "
              f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}%, val_top5={val_top5:.2f}%")

        # Save best model if validation improves
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_path = os.path.join(args.output_dir, "best_resnet50.pth")
            torch.save(model.state_dict(), best_path)
            print(f"*** [INFO] New best model saved at epoch {epoch}! ***")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(f"[INFO] Early stopping after {stale_epochs} epochs without improvement.")
                break

        # Save periodic checkpoints
        if epoch % 10 == 0:
            ckpt = os.path.join(args.output_dir, f"resnet50_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[INFO] Checkpoint saved: {ckpt}")

    # -----------------------------------------------------
    # Final evaluation on test set
    # -----------------------------------------------------
    print("\n[INFO] Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_resnet50.pth"), map_location=device))
    test_loss, test_top1, test_top5 = validate(model, criterion, test_loader, device, split_name="Test")

    print(f"[RESULT] Test loss={test_loss:.4f}, top1={test_top1:.2f}%, top5={test_top5:.2f}%")
    print(f"[INFO] Training complete. Best epoch: {best_epoch} with val_loss={best_loss:.4f}")

    # Close log file
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user.")
        sys.exit(0)
