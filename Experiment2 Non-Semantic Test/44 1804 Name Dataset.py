#!/usr/bin/env python
"""
fine_tune_dataset_id_1804_modified.py

Purpose:
--------
Fine-tunes a ResNet-18 backbone (originally trained on a 27-class semantic task)
to predict the source dataset label ("CIFAR-100" ↔ "TinyImageNet").
All convolutional layers are frozen — only the final fully connected (FC) layer
is trained.

Adaptation Notes:
-----------------
This version is specific to:
    - Dataset04 (Bicubic) variant.
    - ResNet-18 model instance "1804".
It adds:
    - Extended training/validation logging.
    - Per-class accuracy breakdown in the test phase.
    - Confusion matrix and classification reports for validation/test sets.

Expected CSV schema (metadata.csv):
------------------------------------
Required columns:
    filepath        – str  – Path to the image file on disk.
    dataset_id      – str  – Either "CIFAR-100" or "TinyImageNet".
    split           – str  – One of {"train", "validation", "test"}.
Optional column:
    new_label       – str  – Original 27-class label, used only for per-class
                              breakdown in final test evaluation.

Example (single GPU execution):
-------------------------------
python fine_tune_dataset_id_1804_modified.py \
    --data-csv "path/to/metadata.csv" \
    --checkpoint-in "path/to/best_resnet18.pth" \
    --output-dir "path/to/output/"
"""

# -------------------------
# Standard library imports
# -------------------------
import argparse
import datetime as _dt
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# -------------------------
# Third-party imports
# -------------------------
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# =============================================================================
# Dataset class definition
# =============================================================================
class DatasetIDDataset(Dataset):
    """
    Custom PyTorch Dataset returning:
        - preprocessed image tensor,
        - integer dataset-origin label (0=CIFAR-100, 1=TinyImageNet),
        - original semantic label (string) for per-class analysis.

    The `_ID_MAP` dictionary maps textual dataset IDs to integer class indices
    suitable for training a binary classifier.
    """
    _ID_MAP = {"CIFAR-100": 0, "TinyImageNet": 1}

    def __init__(self, csv_path: str | Path, split: str, transform=None):
        # Load CSV and filter by split
        self.df = pd.read_csv(csv_path)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Verify required columns
        required_cols = {"filepath", "dataset_id", "split"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            raise ValueError(f"CSV missing columns: {missing}")

        # If semantic labels are absent, fill with placeholders
        if "new_label" not in self.df.columns:
            print("[WARN] 'new_label' column not found – per-class test metrics will be skipped.")
            self.df["new_label"] = "UNKNOWN"

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image from disk
        row = self.df.loc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self._ID_MAP[row["dataset_id"]]
        return image, label, row["new_label"]

# =============================================================================
# Utility functions
# =============================================================================
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch (percentage)."""
    _, preds = outputs.max(1)
    correct = preds.eq(targets).sum().item()
    return correct * 100.0 / targets.size(0)

def save_checkpoint(state: Dict, filename: str):
    """Persist model state to disk."""
    torch.save(state, filename)
    print(f"[INFO] Saved checkpoint -> {filename}")

def timestamp() -> str:
    """Return current timestamp (YYYY-MM-DDTHH:MM:SS)."""
    return _dt.datetime.now().isoformat(timespec="seconds")

def log_row(logfile: Path, row_dict: Dict):
    """
    Append a single row (dict) to a CSV log file.
    If the file does not exist, write header first.
    """
    df_row = pd.DataFrame([row_dict])
    if logfile.exists():
        df_row.to_csv(logfile, mode="a", header=False, index=False)
    else:
        df_row.to_csv(logfile, mode="w", header=True, index=False)

# =============================================================================
# Training and evaluation routines
# =============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch:
        - Forward pass
        - Compute loss
        - Backpropagation + optimizer step
        - Accumulate average loss and accuracy
    """
    model.train()
    running_loss, running_acc = 0.0, 0.0
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
    Evaluate on a given split without gradient updates.
    Returns:
        - average loss
        - average accuracy
        - list of predicted labels
        - list of ground-truth labels
    """
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    all_preds, all_targets = [], []
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

# =============================================================================
# Per-class testing (semantic labels)
# =============================================================================
def per_class_test(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Compute dataset-origin accuracy per original semantic label.
    For each semantic label:
        - total number of samples
        - number/accuracy for TinyImageNet subset
        - number/accuracy for CIFAR-100 subset
    Also returns full confusion matrix and classification report.
    """
    model.eval()
    per_class = {}
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels, new_labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

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

    # Convert to DataFrame for saving
    rows = []
    for name, d in sorted(per_class.items()):
        rows.append({
            "new_label": name,
            "total": d["total"],
            "correct": d["correct"],
            "acc": d["correct"] / d["total"] if d["total"] else 0,
            "num_images_from_Tiny": d["tot_tiny"],
            "Tiny_correct": d["corr_tiny"],
            "Tiny_acc": d["corr_tiny"] / d["tot_tiny"] if d["tot_tiny"] else 0,
            "num_images_from_CIFAR": d["tot_cifar"],
            "CIFAR_correct": d["corr_cifar"],
            "CIFAR_acc": d["corr_cifar"] / d["tot_cifar"] if d["tot_cifar"] else 0,
        })
    df = pd.DataFrame(rows).sort_values("new_label")

    cm = confusion_matrix(all_targets, all_preds)
    cr = classification_report(all_targets, all_preds, target_names=["CIFAR-100", "TinyImageNet"], digits=4)
    return df, cm, cr

# =============================================================================
# Argument parsing
# =============================================================================
def parse_args(argv: List[str]):
    """
    CLI argument parser — allows overriding default file paths,
    batch size, learning rate, epochs, etc.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-18 to classify dataset origin (CIFAR-100 vs TinyImageNet) – Dataset04."
    )
    parser.add_argument("--data-csv", default="D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv")
    parser.add_argument("--checkpoint-in", default="D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1804/best_resnet18.pth")
    parser.add_argument("--checkpoint-out", default="best_resnet18_datasetid_1804.pth")
    parser.add_argument("--output-dir", default="D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1804")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    return parser.parse_args(argv)

# =============================================================================
# Main training / evaluation routine
# =============================================================================
def main(argv: List[str] | None = None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror all prints to a log file
    log_txt_path = out_dir / "output_log_1804.txt"
    log_file = open(log_txt_path, "w", encoding="utf-8")
    class Tee:
        """Write stdout to both console and log file."""
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data)
        def flush(self):
            for f in self.files: f.flush()
    sys.stdout = Tee(sys.stdout, log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Define transforms for training (augmentation) and validation/test (deterministic)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    # Instantiate datasets & loaders
    train_ds = DatasetIDDataset(args.data_csv, split="train", transform=train_tf)
    val_ds   = DatasetIDDataset(args.data_csv, split="validation", transform=val_tf)
    test_ds  = DatasetIDDataset(args.data_csv, split="test", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load ResNet-18 backbone (trained on 27 classes), freeze all but final FC
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 27)
    ckpt = torch.load(args.checkpoint_in, map_location="cpu")
    model.load_state_dict(ckpt)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(in_features, 2)  # Replace head for binary classification
    model.fc.requires_grad_(True)
    model.to(device)

    # Loss, optimizer, LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1, verbose=True)

    # Training loop with early stopping
    best_loss, epochs_no_improve = float("inf"), 0
    log_csv = out_dir / "dataset_ft_log_1804.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        lr_current = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f}% | Val: loss={val_loss:.4f} acc={val_acc:.2f}% (lr={lr_current:.3e})")
        log_row(log_csv, {
            "timestamp": timestamp(),
            "epoch": epoch,
            "lr": lr_current,
            "train_loss": train_loss,
            "train_top1": train_acc,
            "val_loss": val_loss,
            "val_top1": val_acc,
        })

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = out_dir / args.checkpoint_out
            save_checkpoint(model.state_dict(), ckpt_path)

            # Extra logging for validation set
            cm = confusion_matrix(val_targets, val_preds)
            print("[Val Confusion Matrix]\n", pd.DataFrame(cm, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"]))
            print("[Val Classification Report]\n", classification_report(val_targets, val_preds, target_names=["CIFAR-100","TinyImageNet"], digits=4))
        else:
            epochs_no_improve += 1
            if epochs_no_improve > args.early_stop_patience:
                print("[INFO] Early stopping triggered – no improvement in val_loss.")
                break

    # Final evaluation on test split
    print("\n[TEST] Loading best checkpoint and evaluating on test split…")
    best_model = models.resnet18(weights=None)
    best_model.fc = nn.Linear(in_features, 2)
    best_model.load_state_dict(torch.load(out_dir / args.checkpoint_out, map_location="cpu"))
    best_model.to(device)

    per_class_df, cm_test, cr_test = per_class_test(best_model, test_loader, device)
    print("\n[Test Confusion Matrix]\n", pd.DataFrame(cm_test, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"]))
    print("[Test Classification Report]\n", cr_test)
    print("[Per-class accuracy]\n", per_class_df.to_string(index=False))

    per_class_df.to_csv(out_dir / "test_per_class_accuracy_1804.csv", index=False)
    print(f"[INFO] Per class results saved to {out_dir / 'test_per_class_accuracy_1804.csv'}")

    log_file.close()

if __name__ == "__main__":
    main()
