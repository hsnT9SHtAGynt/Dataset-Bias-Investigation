#!/usr/bin/env python
"""
train_vit_b_32_adamw_cosine.py
================================
Standalone script to train ViT-B/32 **from scratch** on a custom image-classification dataset.

Key changes compared to the original version:
* **Optimizer**: AdamW with `weight_decay=0.05`, as recommended for ViT.
* **Learning rate**: default peak LR is `5e-5` (much smaller than 1e-3).
* **Schedule**: 5-epoch **linear warm-up** (from 0.01× to 1× LR) followed by
  a **cosine annealing** decay to `1e-6` over the remaining epochs.
* **Early-stopping patience** increased to **30** epochs to let the longer
  schedule take effect.
* CLI remains identical so it can drop-in replace the previous script.

Notes:
- The actual default for `--early-stop-patience` below is 10 (docstring mentions 30).
  Adjust the CLI default or pass `--early-stop-patience 30` when launching if desired.
- Top-5 accuracy computation assumes `num_classes >= 5`. If your dataset has fewer
  than 5 classes, this will error; switch to `(1,)` or guard accordingly.
- Validation/test transforms use `CenterCrop(224)`; images smaller than 224 on the
  shortest side will raise an error. Consider adding a `Resize(256)` before cropping
  if your data is small.
- Reproducibility: no seed is set here. Add `torch.manual_seed(...)` etc. if needed.
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
from torchvision.models import vit_b_32
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class AverageMeter:
    """Track and update running averages for metrics like loss and accuracy.

    Usage:
        m = AverageMeter("Loss"); m.update(val, n); m.avg -> running mean
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        # Reset internal state to zeros
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        # Update running totals and recompute average
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Compute top-k accuracy for the given outputs and targets.

    Args:
        output: logits of shape (N, C)
        target: integer class indices of shape (N,)
        topk:   tuple of K values (e.g., (1, 5))

    Returns:
        list of scalar tensors, each being top-K accuracy in percent.

    Caveat:
        Ensure all requested K satisfy K <= C, otherwise torch.topk will raise.
    """
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
    """Dataset backed by a CSV with columns: filepath, new_label_id, split.

    Expected CSV schema:
        - filepath: path to an RGB image file
        - new_label_id: integer class id (0..C-1)
        - split: one of {"train", "validation", "test"} (case-sensitive here)

    Filtering:
        Only rows with the requested split are loaded into memory.
    """

    def __init__(self, csv_path: str, split: str, transform: transforms.Compose):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Strict schema check to fail fast on malformed metadata
        for col in ("filepath", "new_label_id", "split"):
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from CSV")

        # Materialize the desired split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples with split='{split}' found in metadata.")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image and convert to RGB to avoid mode issues (e.g., "L" or "RGBA")
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["new_label_id"])
        return image, label


def train_one_epoch(model: nn.Module, criterion, optimizer, loader: DataLoader, device: torch.device):
    """One full pass over the training set with progress bar and running metrics."""
    model.train()
    loss_meter, top1_meter, top5_meter = AverageMeter("Loss"), AverageMeter("Top1"), AverageMeter("Top5")

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Standard single-step SGD/AdamW update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracies and update meters
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
    """Evaluation pass over validation/test set. No gradient updates."""
    model.eval()
    loss_meter, top1_meter, top5_meter = AverageMeter("Loss"), AverageMeter("Top1"), AverageMeter("Top5")

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
    # Prefer CUDA if available; falls back to CPU otherwise.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Train ViT-B/32 on a custom dataset")
    parser.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv",
                        help="Path to metadata CSV containing (filepath, new_label_id, split)")
    parser.add_argument("--output-dir", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ViT_B32_04",
                        help="Directory to save models, logs, and checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-5, help="Peak learning rate after warm-up")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of linear warm-up epochs")
    args = parser.parse_args()

    # Create output directory (idempotent)
    os.makedirs(args.output_dir, exist_ok=True)

    # Standard ImageNet normalization; augmentation only on training split.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        # If your images are smaller than 224, add a Resize step before this.
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Build datasets for each split as declared in the CSV
    train_ds = CustomImageDataset(args.data_csv, "train", train_tf)
    val_ds   = CustomImageDataset(args.data_csv, "validation", test_tf)
    test_ds  = CustomImageDataset(args.data_csv, "test", test_tf)

    # Infer number of classes from the union of splits (robust when some classes
    # appear only in val/test).
    num_classes = int(pd.concat([
        train_ds.df["new_label_id"],
        val_ds.df["new_label_id"],
        test_ds.df["new_label_id"],
    ]).max() + 1)
    print(f"Detected {num_classes} classes.")

    # Enable pinned memory only when on CUDA for faster host->device transfer
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin_memory)

    # Initialize ViT-B/32 from scratch and replace the classification head.
    # TorchVision ViT heads differ across versions; handle both layouts.
    model = vit_b_32(weights=None)
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        # If this triggers, inspect the model to find the correct classifier attr.
        raise RuntimeError("Unexpected ViT model head structure; cannot replace classifier.")
    model.to(device)

    # Standard CE loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # AdamW is recommended for transformer training; weight_decay tuned for ViT.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Scheduler: linear warm-up -> cosine annealing
    #   Phase 1 (epochs 1..warmup): LR ramps from 0.01× to 1.0× of args.lr
    #   Phase 2 (warmup+1..epochs): Cosine decay down to eta_min=1e-6
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                         eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # CSV logger (append-only). One row per epoch with train/val metrics + LR.
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
        # Log LR before stepping the scheduler (this reflects the LR used this epoch
        # if you call scheduler.step() AFTER the epoch, as done below).
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train + Validate for this epoch ---
        train_loss, train_top1, train_top5 = train_one_epoch(model, criterion, optimizer, train_loader, device)
        val_loss, val_top1, val_top5 = validate(model, criterion, val_loader, device, split_name="Val")

        # Advance the learning-rate schedule at end of epoch
        scheduler.step()

        # Persist epoch metrics to CSV for later analysis
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

        # Save best model weights by lowest validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_path = os.path.join(args.output_dir, "best_vit_b32.pth")
            torch.save(model.state_dict(), best_path)
            print("*** New best model saved! ***")
        else:
            stale_epochs += 1
            # Early stopping controlled by patience on validation loss
            if stale_epochs >= args.early_stop_patience:
                print(f"Early stopping after {args.early_stop_patience} epochs without improvement on val_loss.")
                break

        # Periodic checkpoints every 10 epochs for recovery/ablation
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"vit_b32_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Evaluate the best validation model on the held-out test set
    print("\nEvaluating best model on the test set ...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_vit_b32.pth"), map_location=device))
    test_loss, test_top1, test_top5 = validate(model, criterion, test_loader, device, split_name="Test")
    print(f"Test loss={test_loss:.4f}, Test top1={test_top1:.2f}%, Test top5={test_top5:.2f}%")

    print(f"\nTraining complete. Best validation epoch: {best_epoch} (Best_Val_loss={best_loss:.4f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C without stack trace noise
        print("\nTraining interrupted by user.")
        sys.exit(0)
