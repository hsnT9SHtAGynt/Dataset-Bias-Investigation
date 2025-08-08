#!/usr/bin/env python
"""
train_resnet18.py
=================
Standalone PyTorch training script to train ResNet-18 on a custom image classification dataset.

Main features:
--------------
- Reads dataset information from a CSV (metadata.csv) that contains file paths and labels.
- Splits dataset into train, validation, and test according to the "split" column in the CSV.
- Applies common image preprocessing and augmentation using torchvision.transforms.
- Implements training loop with:
    * Running metric tracking (loss, top-1, top-5 accuracy)
    * Learning rate scheduling (ReduceLROnPlateau)
    * Early stopping based on validation loss
    * Saving of best model checkpoint
    * Periodic checkpoint saving every 10 epochs
- At the end, evaluates the best model on the test split.

This script is intended to be run from the command line and accepts arguments for:
- CSV file path
- Output directory for logs and checkpoints
- Hyperparameters (batch size, learning rate, number of epochs, etc.)
"""
from __future__ import annotations  # Enables forward references in type hints
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
from tqdm.auto import tqdm  # For nice progress bars


# ======================================================================
# Helper class for metric tracking
# ======================================================================
class AverageMeter:
    """
    Keeps track of running averages for any numerical metric (e.g., loss, accuracy).
    Usage:
        meter = AverageMeter("Loss")
        meter.update(0.5, n=32)  # value, batch size
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        """Resets all statistics to their initial state."""
        self.val = 0    # most recent value seen
        self.avg = 0    # running average
        self.sum = 0    # cumulative sum of values
        self.count = 0  # total number of samples counted so far

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with a new value.
        val : new measurement (e.g., loss for a batch)
        n   : number of samples that this value represents
        """
        self.val = val
        self.sum += val * n   # accumulate sum weighted by batch size
        self.count += n       # accumulate sample count
        self.avg = self.sum / self.count if self.count else 0  # compute new average

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ======================================================================
# Accuracy computation
# ======================================================================
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """
    Computes the top-k accuracy for a given batch of predictions.
    E.g., top-1 = standard accuracy, top-5 = whether target is among top 5 predictions.

    Args:
        output : (batch_size, num_classes) tensor of raw logits.
        target : (batch_size,) tensor of class indices.
        topk   : tuple of k values to compute.

    Returns:
        List of accuracies (as torch.Tensor), each corresponding to a k in topk.
    """
    with torch.no_grad():  # no gradient computation needed
        maxk = max(topk)  # largest k to compute
        batch_size = target.size(0)

        # Get top-k predicted class indices per sample
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # transpose to shape (maxk, batch_size)

        # Compare predicted indices with ground truth (broadcasted)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[torch.Tensor] = []
        for k in topk:
            # correct[:k] contains True/False for top-k predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # convert count to percentage
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ======================================================================
# Dataset wrapper that reads from metadata CSV
# ======================================================================
class CustomImageDataset(Dataset):
    """
    A PyTorch Dataset that loads images from paths listed in a CSV file.
    The CSV must contain:
        filepath      : path to image file
        new_label_id  : integer class ID
        split         : which split the sample belongs to ("train", "validation", "test")

    The dataset is filtered to contain only samples for the requested split.
    """

    def __init__(self, csv_path: str, split: str, transform: transforms.Compose):
        # Ensure metadata file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        # Load the entire CSV into a DataFrame
        self.df = pd.read_csv(csv_path)

        # Verify that required columns exist
        for col in ("filepath", "new_label_id", "split"):
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from CSV")

        # Keep only rows for the desired split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples with split='{split}' found in metadata.")

        self.transform = transform  # torchvision transform pipeline for preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and return a single sample (image tensor, label int)."""
        row = self.df.iloc[idx]
        # Open image and ensure it's in RGB mode
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["new_label_id"])
        return image, label


# ======================================================================
# Training loop for one epoch
# ======================================================================
def train_one_epoch(model: nn.Module, criterion, optimizer, loader: DataLoader, device: torch.device):
    """
    Runs a single epoch of training:
      - forward pass
      - loss computation
      - backward pass
      - optimizer step
      - metric logging
    """
    model.train()  # enable training mode (dropout active, batchnorm updates)
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc="Train", leave=False)  # show progress bar
    for images, targets in pbar:
        # Move batch to device (CPU or GPU)
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy metrics for this batch
        acc1, acc5 = accuracy(outputs, targets, (1, 5))

        # Update running averages
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))

        # Update progress bar display
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "top1": f"{top1_meter.avg:.2f}%",
            "top5": f"{top5_meter.avg:.2f}%"
        })

    # Return average metrics for this epoch
    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# ======================================================================
# Validation loop
# ======================================================================
def validate(model: nn.Module, criterion, loader: DataLoader, device: torch.device, split_name: str = "Val"):
    """
    Evaluate the model on a validation or test set:
      - no gradient computation
      - metrics only (no parameter updates)
    """
    model.eval()  # inference mode
    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Top1")
    top5_meter = AverageMeter("Top5")

    pbar = tqdm(loader, desc=split_name, leave=False)
    with torch.no_grad():  # no backprop needed
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Accuracy metrics
            acc1, acc5 = accuracy(outputs, targets, (1, 5))

            # Update metric trackers
            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1.item(), images.size(0))
            top5_meter.update(acc5.item(), images.size(0))

            # Show progress
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
                "top5": f"{top5_meter.avg:.2f}%"
            })

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# ======================================================================
# Main function — orchestrates training, validation, checkpointing
# ======================================================================
def main():
    # Select device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Parse CLI arguments
    # -----------------------
    parser = argparse.ArgumentParser(description="Train ResNet-18 (from scratch) on a custom dataset")
    parser.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/30Dataset03_Bicubic/metadata.csv",
                        help="Path to metadata CSV containing (filepath, new_label_id, split)")
    parser.add_argument("--output-dir", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1803",
                        help="Directory to save models, logs, and checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists

    # -----------------------
    # Define data transforms
    # -----------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Data augmentation for training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),  # random crop to 224×224 at random scale/aspect
        transforms.RandomHorizontalFlip(),  # flip horizontally with 50% chance
        transforms.ToTensor(),
        normalize,
    ])
    # Minimal preprocessing for validation/testing
    test_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # -----------------------
    # Create dataset objects
    # -----------------------
    train_ds = CustomImageDataset(args.data_csv, "train", train_tf)
    val_ds   = CustomImageDataset(args.data_csv, "validation", test_tf)
    test_ds  = CustomImageDataset(args.data_csv, "test", test_tf)

    # Determine number of classes dynamically from CSV
    num_classes = int(pd.concat([
        train_ds.df["new_label_id"],
        val_ds.df["new_label_id"],
        test_ds.df["new_label_id"],
    ]).max() + 1)
    print(f"Detected {num_classes} classes.")

    # -----------------------
    # Build DataLoaders
    # -----------------------
    pin_memory = (device.type == "cuda")  # speeds up host-to-GPU transfer
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)

    # -----------------------
    # Model setup
    # -----------------------
    model = resnet18(weights=None)  # ResNet-18 with random initialization
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final FC layer
    model.to(device)  # Move model to GPU or CPU

    # Loss, optimizer, LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)

    # -----------------------
    # Training log file setup
    # -----------------------
    log_file = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "epoch", "lr", "train_loss", "train_top1", "train_top5",
                "val_loss", "val_top1", "val_top5",
            ])

    # Variables to track best performance
    best_loss = float('inf')
    best_epoch = 0
    stale_epochs = 0  # number of epochs without improvement

    # -----------------------
    # Epoch loop
    # -----------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]

        # Train for one epoch
        train_loss, train_top1, train_top5 = train_one_epoch(model, criterion, optimizer, train_loader, device)
        # Validate
        val_loss, val_top1, val_top5 = validate(model, criterion, val_loader, device, split_name="Val")

        # Step LR scheduler using validation loss
        scheduler.step(val_loss)

        # Append metrics to CSV log
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), epoch, f"{current_lr:.3e}",
                f"{train_loss:.4f}", f"{train_top1:.2f}", f"{train_top5:.2f}",
                f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}",
            ])

        # Print summary for current epoch
        print(
            f"Epoch {epoch}: lr={current_lr:.3e}, "
            f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}%, train_top5={train_top5:.2f}%, "
            f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}%, val_top5={val_top5:.2f}%"
        )

        # -----------------------
        # Check if best model so far
        # -----------------------
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

        # Save periodic checkpoints every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"resnet18_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # -----------------------
    # Final evaluation on test set
    # -----------------------
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
