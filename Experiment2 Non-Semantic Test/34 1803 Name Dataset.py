#!/usr/bin/env python
"""
34 1803 Name Dataet.py

Fine-tune a ResNet-18 backbone (originally trained on 27 semantic classes) to classify
the *source dataset* (CIFAR-100 vs TinyImageNet), freezing all convolutional layers.

Special notes:
- Uses Dataset03 (Bicubic) and model weights from ResNet-18 model "1803".
- Adds extended logging and per-class test evaluation.
- Metadata CSV must contain:
    filepath   – full path to image
    dataset_id – "CIFAR-100" or "TinyImageNet" (the label for this fine-tuning task)
    split      – one of {"train", "validation", "test"}
    new_label  – original semantic label (used for per-class metrics in test phase)
"""

import argparse
import datetime as _dt
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class DatasetIDDataset(Dataset):
    """
    Custom PyTorch Dataset:
    - Loads images and their dataset origin labels (0=CIFAR-100, 1=TinyImageNet)
    - Also returns original semantic label (new_label) for per-class test analysis
    """

    _ID_MAP = {"CIFAR-100": 0, "TinyImageNet": 1}  # mapping from dataset_id string → integer label

    def __init__(self, csv_path: str | Path, split: str, transform=None):
        self.df = pd.read_csv(csv_path)

        # Validate split argument
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")

        # Filter DataFrame to the specified split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Ensure necessary columns exist
        required_cols = {"filepath", "dataset_id", "split"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            raise ValueError(f"CSV missing columns: {missing}")

        # If 'new_label' missing, still run but skip per-class test metrics
        if "new_label" not in self.df.columns:
            print("[WARN] 'new_label' column not found – per-class test metrics will be skipped.")
            self.df["new_label"] = "UNKNOWN"

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return (transformed_image_tensor, dataset_origin_int, original_semantic_label_str)."""
        row = self.df.loc[idx]
        img_path = row["filepath"]

        # Load and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transform if given
        if self.transform:
            image = self.transform(image)

        # Map dataset_id string to integer label
        label = self._ID_MAP[row["dataset_id"]]
        return image, label, row["new_label"]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy (percentage) for a batch."""
    _, preds = outputs.max(1)
    correct = preds.eq(targets).sum().item()
    return correct * 100.0 / targets.size(0)

def save_checkpoint(state: Dict, filename: str):
    """Save model state_dict to file."""
    torch.save(state, filename)
    print(f"[INFO] Saved checkpoint -> {filename}")

def timestamp() -> str:
    """Return current time in ISO8601 format (seconds precision)."""
    return _dt.datetime.now().isoformat(timespec="seconds")

def log_row(logfile: Path, row_dict: Dict):
    """
    Append one row of training/validation metrics to CSV log file.
    Creates header if file does not exist yet.
    """
    df_row = pd.DataFrame([row_dict])
    if logfile.exists():
        df_row.to_csv(logfile, mode="a", header=False, index=False)
    else:
        df_row.to_csv(logfile, mode="w", header=True, index=False)

# -----------------------------------------------------------------------------
# Training & validation
# -----------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Train model for one epoch on provided DataLoader.
    Returns (average_loss, average_accuracy_percent).
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_acc += accuracy(outputs, labels) * labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate model (no grad) on given DataLoader.
    Returns:
      avg_loss, avg_acc, all_predictions_list, all_targets_list
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            running_acc += accuracy(outputs, labels) * labels.size(0)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_targets

# -----------------------------------------------------------------------------
# Per-class test accuracy breakdown
# -----------------------------------------------------------------------------
def per_class_test(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Compute per-class accuracy statistics on the test split:
      - overall acc per original label
      - acc restricted to TinyImageNet images
      - acc restricted to CIFAR-100 images
    Also returns confusion matrix and sklearn classification_report for dataset_id task.
    """
    model.eval()
    per_class: Dict[str, Dict[str, int]] = {}
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels, new_labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

            # Update stats per original semantic label
            for nl, p, t in zip(new_labels, preds.cpu().tolist(), labels.cpu().tolist()):
                rec = per_class.setdefault(nl, {
                    "total": 0, "correct": 0,
                    "tot_tiny": 0, "corr_tiny": 0,
                    "tot_cifar": 0, "corr_cifar": 0
                })
                rec["total"] += 1
                if p == t:
                    rec["correct"] += 1
                if t == 1:  # TinyImageNet
                    rec["tot_tiny"] += 1
                    if p == t:
                        rec["corr_tiny"] += 1
                elif t == 0:  # CIFAR-100
                    rec["tot_cifar"] += 1
                    if p == t:
                        rec["corr_cifar"] += 1

    # Build DataFrame summarizing per-class stats
    rows = []
    for name, d in sorted(per_class.items()):
        total = d["total"]
        correct = d["correct"]
        acc = correct / total if total else 0
        tot_tiny = d["tot_tiny"]
        corr_tiny = d["corr_tiny"]
        acc_tiny = corr_tiny / tot_tiny if tot_tiny else 0
        tot_cifar = d["tot_cifar"]
        corr_cifar = d["corr_cifar"]
        acc_cifar = corr_cifar / tot_cifar if tot_cifar else 0
        rows.append({
            "new_label": name,
            "total": total,
            "correct": correct,
            "acc": acc,
            "num_images_from_Tiny": tot_tiny,
            "Tiny_correct": corr_tiny,
            "Tiny_acc": acc_tiny,
            "num_images_from_CIFAR": tot_cifar,
            "CIFAR_correct": corr_cifar,
            "CIFAR_acc": acc_cifar,
        })
    df = pd.DataFrame(rows)
    df.sort_values("new_label", inplace=True)

    # Confusion matrix & classification report for dataset origin classification
    cm = confusion_matrix(all_targets, all_preds)
    cr = classification_report(all_targets, all_preds, target_names=["CIFAR-100", "TinyImageNet"], digits=4)
    return df, cm, cr

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-18 to classify dataset origin (CIFAR-100 vs TinyImageNet) – Dataset03 version."
    )
    parser.add_argument("--data-csv", default=".../metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--checkpoint-in", default=".../best_resnet18.pth", help="Pre-trained 27-class model weights")
    parser.add_argument("--checkpoint-out", default="best_resnet18_datasetid_1803.pth", help="Save filename for fine-tuned weights")
    parser.add_argument("--output-dir", default=".../ResNet1803", help="Directory for checkpoints & logs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    return parser.parse_args(argv)

# -----------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------
def main(argv: List[str] | None = None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to both console and log file
    log_txt_path = out_dir / "output_log_1803.txt"
    log_file = open(log_txt_path, "w", encoding="utf-8")
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, data): [f.write(data) for f in self.files]
        def flush(self): [f.flush() for f in self.files]
    sys.stdout = Tee(sys.stdout, log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create dataset splits
    train_ds = DatasetIDDataset(args.data_csv, split="train", transform=train_tf)
    val_ds   = DatasetIDDataset(args.data_csv, split="validation", transform=val_tf)
    test_ds  = DatasetIDDataset(args.data_csv, split="test", transform=val_tf)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load ResNet-18 backbone and freeze convolutional layers
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 27)  # temp head to match checkpoint
    ckpt = torch.load(args.checkpoint_in, map_location="cpu")
    model.load_state_dict(ckpt)

    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    # Replace FC layer with 2-class classifier for dataset origin task
    model.fc = nn.Linear(in_features, 2)
    model.fc.requires_grad_(True)  # only FC layer is trainable
    model.to(device)

    # Optimizer and LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, verbose=True)

    best_loss = float("inf")
    epochs_no_improve = 0
    log_csv = out_dir / "dataset_ft_log_1803.csv"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        lr_current = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.2f}% (lr={lr_current:.3e})")

        log_row(log_csv, {
            "timestamp": timestamp(),
            "epoch": epoch,
            "lr": lr_current,
            "train_loss": train_loss,
            "train_top1": train_acc,
            "val_loss": val_loss,
            "val_top1": val_acc,
        })

        # Save best checkpoint
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = out_dir / args.checkpoint_out
            save_checkpoint(model.state_dict(), ckpt_path)

            # Print confusion matrix and classification report for validation set
            cm = confusion_matrix(val_targets, val_preds)
            cm_df = pd.DataFrame(cm, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"])
            cr = classification_report(val_targets, val_preds, target_names=["CIFAR-100","TinyImageNet"], digits=4)
            print("[Val Confusion Matrix]")
            print(cm_df.to_string())
            print("[Val Classification Report]\n", cr)
        else:
            epochs_no_improve += 1
            if epochs_no_improve > args.early_stop_patience:
                print("[INFO] Early stopping triggered – no improvement in val_loss.")
                break

    # Final test evaluation using best checkpoint
    print("\n[TEST] Loading best checkpoint and evaluating on test split…")
    best_model = models.resnet18(weights=None)
    best_model.fc = nn.Linear(in_features, 2)
    best_model.load_state_dict(torch.load(out_dir / args.checkpoint_out, map_location="cpu"))
    best_model.to(device)

    per_class_df, cm_test, cr_test = per_class_test(best_model, test_loader, device)

    # Print test results
    cm_test_df = pd.DataFrame(cm_test, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"])
    print("\n[Test Confusion Matrix]")
    print(cm_test_df.to_string())
    print("[Test Classification Report]\n", cr_test)
    print("[Per-class accuracy]")
    print(per_class_df.to_string(index=False))

    # Save per-class results to CSV
    per_class_csv = out_dir / "test_per_class_accuracy_1803.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"[INFO] Per-class results saved to {per_class_csv}")

    log_file.close()

if __name__ == "__main__":
    main()
