r"""
Task C (fixed): Linear Probes on **ViT‑B/32** Blocks for Dataset04
===================================================================
This version swaps the ResNet‑18 backbone for the Vision Transformer
`vit_b_32` from **torchvision 0.18+** and installs 16 probes that match
          Patch → PosEmb → Block0‑11 → Ln → CLS‑before‑FC
easily as sketched. All other training and evaluation logic is unchanged.

Key changes
-----------
* **Backbone**: `torchvision.models.vit_b_32` (conv‑patch → encoder)
* **Probe taps** (16):
  ```text
  conv_proj                   → P1
  encoder.dropout             → P2
  encoder.layers.0 – 11       → P3 … P14
  encoder.ln                  → P15
  _CLS‑token (after ln)       → P16
  ```
* Hooks now refer to `encoder.ln` instead of `ln_post`, and layer taps use `encoder.layers.{idx}`.
"""
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_32  # Vision Transformer B/32 backbone model
from tqdm.auto import tqdm

# ------------------------ utility classes & funcs ----------------------------

class AverageMeter:
    """Utility class to track running averages of metrics during training/evaluation"""
    
    def __init__(self, name: str):
        """Initialize with a descriptive name for the metric"""
        self.name = name
        self.reset()
        
    def reset(self):
        """Reset all tracked values to zero"""
        self.val = self.sum = self.count = self.avg = 0
        
    def update(self, v: float, n: int = 1):
        """Update the running average with new value(s)
        
        Args:
            v: New value to incorporate
            n: Number of samples this value represents (default: 1)
        """
        self.val = v  # Store latest value
        self.sum += v * n  # Accumulate weighted sum
        self.count += n    # Track total sample count
        self.avg = self.sum / self.count if self.count else 0  # Compute running average


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-1 accuracy percentage from logits and ground truth labels
    
    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        
    Returns:
        Accuracy as percentage (0-100)
    """
    return logits.argmax(1).eq(targets).sum().item() * 100.0 / targets.size(0)

# ------------------------------ dataset --------------------------------------

class DatasetSourceProbe(Dataset):
    """Custom dataset for training linear probes to detect dataset source bias
    
    This dataset loads images and creates binary labels to distinguish between
    CIFAR-based datasets (label 0) and other datasets (label 1).
    """
    
    def __init__(self, csv_path: str | Path, split: str, tf):
        """Initialize dataset from CSV metadata
        
        Args:
            csv_path: Path to CSV file containing image metadata
            split: Data split to use ("train", "validation", or "test")
            tf: Image transform pipeline to apply
        """
        # Load metadata and filter by split
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        
        if df.empty:
            raise ValueError(f"No rows with split={split} in {csv_path}")
            
        # Create binary source labels: 0 for CIFAR datasets, 1 for others
        df["source_label"] = df["dataset_id"].apply(
            lambda x: 0 if str(x).upper().startswith("CIFAR") else 1
        )
        
        # Clean up file paths
        df["filepath"] = df["filepath"].str.strip()
        self.df, self.tf = df, tf
        
    def __len__(self):
        """Return dataset size"""
        return len(self.df)
        
    def __getitem__(self, idx):
        """Get a single sample
        
        Returns:
            tuple: (image_tensor, source_label, class_label)
                - image_tensor: Transformed image as tensor
                - source_label: Binary dataset source label (0=CIFAR, 1=other)  
                - class_label: Original classification label
        """
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        return self.tf(img), int(r["source_label"]), int(r["new_label_id"])

# --------------------------- linear‑probe module -----------------------------

class LinearProbe(nn.Module):
    """Linear probe that extracts features from a frozen backbone and trains a binary classifier
    
    This module hooks into intermediate layers of a pre-trained model to extract features
    and trains a simple linear classifier on top to detect dataset source bias.
    """

    def __init__(self, backbone: nn.Module, tap: str, feat_dim: int):
        """Initialize linear probe
        
        Args:
            backbone: Pre-trained model to extract features from (frozen)
            tap: Name of layer/module to hook into for feature extraction
            feat_dim: Dimensionality of extracted features
        """
        super().__init__()
        self.backbone, self.tap = backbone, tap
        self._buf: torch.Tensor | None = None  # Buffer to store hooked features

        # Register forward hooks to extract intermediate features
        if tap == "cls_token":
            # Special case: extract CLS token from layer normalization output
            def cls_hook(_, __, output):
                # output: [batch_size, num_tokens, hidden_dim] after layer norm
                self._buf = output[:, 0]  # Extract CLS token (first token)
            backbone.encoder.ln.register_forward_hook(cls_hook)
        else:
            # Standard case: hook into named module
            target_module = dict(backbone.named_modules())[tap]
            target_module.register_forward_hook(
                lambda _m, _i, o: setattr(self, "_buf", o)
            )

        # Feature processing and classification layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling for conv features
        self.cls = nn.Linear(feat_dim, 2)        # Binary classifier (CIFAR vs other)

    def forward(self, x):
        """Forward pass through probe
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Logits for binary classification [batch_size, 2]
        """
        self._buf = None  # Reset feature buffer
        
        # Run backbone to trigger hooks and populate buffer
        _ = self.backbone(x)
        feats = self._buf
        
        if feats is None:
            raise RuntimeError(f"Hook at {self.tap} produced no output!")

        # Process features based on dimensionality
        if feats.dim() == 4:
            # 4D: Convolutional features [B, C, H, W] -> global average pool + flatten
            feats = self.gap(feats).flatten(1)
        elif feats.dim() == 3:
            # 3D: Transformer tokens [B, N, D] -> extract CLS token
            feats = feats[:, 0]  # CLS token is first token
        else:
            # 2D or other: flatten to feature vector
            feats = feats.flatten(1)
            
        # Apply binary classifier
        return self.cls(feats)

# --------------------------- train / eval loops -----------------------------

def run_epoch(model, loader, crit, opt, device, train: bool):
    """Run one training or evaluation epoch
    
    Args:
        model: Neural network model to train/evaluate
        loader: DataLoader providing batches
        crit: Loss criterion function
        opt: Optimizer (only used if train=True)
        device: Device to run computations on
        train: Whether to run in training mode (True) or evaluation mode (False)
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    # Set model mode
    if train:
        model.train()
    else:
        model.eval()
        
    # Initialize metric trackers
    am_loss, am_acc = AverageMeter("loss"), AverageMeter("acc")
    
    # Set gradient context
    ctx = torch.enable_grad() if train else torch.no_grad()
    
    with ctx:
        # Process all batches
        for x, y, _ in tqdm(loader, desc="train" if train else "eval", leave=False):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = crit(logits, y)
            acc = top1_accuracy(logits, y)
            
            # Update metrics
            am_loss.update(loss.item(), x.size(0))
            am_acc.update(acc, x.size(0))
            
            # Backward pass (only during training)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
    return am_loss.avg, am_acc.avg


def evaluate(model, loader, device):
    """Comprehensive evaluation with confusion matrix and per-class statistics
    
    Args:
        model: Trained model to evaluate
        loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        tuple: (confusion_matrix, overall_accuracy, per_class_stats)
    """
    model.eval()
    
    # Collect all predictions and labels
    yt, yp, yc = [], [], []  # true labels, predictions, class labels
    
    with torch.no_grad():
        for x, y, c in loader:
            # Get model predictions
            out = model(x.to(device)).argmax(1).cpu()
            yp += out.tolist()     # Predicted source labels
            yt += y.tolist()       # True source labels  
            yc += c.tolist()       # Original class labels
    
    # Compute overall metrics
    cm = confusion_matrix(yt, yp, labels=[0, 1])  # Binary confusion matrix
    acc = cm.trace() / cm.sum() * 100.0           # Overall accuracy
    
    # Compute per-class accuracy statistics
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for t, p, c in zip(yt, yp, yc):
        stats[c]["total"] += 1
        if t == p:  # Correct source prediction
            stats[c]["correct"] += 1
            
    return cm, acc, stats

# ------------------------------ main -----------------------------------------

if __name__ == "__main__":
    # Command line argument parsing
    p = argparse.ArgumentParser("Linear probes on ViT‑B/32 (Dataset04)")
    droot = r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\40Dataset04_Bicubic"
    
    # Define default paths and hyperparameters
    p.add_argument("--data-csv",   default=fr"{droot}\metadata.csv")
    p.add_argument("--checkpoint", default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ViT_B32_04\best_vit_b32.pth")
    p.add_argument("--output-dir", default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ViT_B32_04\probes")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers",type=int, default=8)
    p.add_argument("--early-stop", type=int, default=5)
    args = p.parse_args()

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define image preprocessing transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet mean
                                     [0.229, 0.224, 0.225])    # ImageNet std
    
    # Training transforms with data augmentation
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),     # Random crop and resize
        transforms.RandomHorizontalFlip(),     # Random horizontal flip
        transforms.ToTensor(),                 # Convert to tensor
        normalize,                             # Normalize
    ])
    
    # Test transforms without augmentation
    test_tf = transforms.Compose([
        transforms.CenterCrop(224),            # Center crop
        transforms.ToTensor(),                 # Convert to tensor
        normalize,                             # Normalize
    ])

    # Create data loaders for all splits
    loaders = {split: DataLoader(
            DatasetSourceProbe(args.data_csv, split,
                               train_tf if split=="train" else test_tf),
            batch_size=args.batch_size,
            shuffle=(split=="train"),          # Only shuffle training data
            num_workers=args.num_workers,
        ) for split in ("train", "validation", "test")}

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------- backbone setup ---------------------------
    
    # Load Vision Transformer backbone
    backbone = vit_b_32(weights=None)  # No pre-trained weights initially
    
    # Replace classifier head to match 27 classes from the dataset
    if isinstance(backbone.heads, nn.Linear):
        in_features = backbone.heads.in_features
    else:
        # Handle case where heads might be a more complex module
        in_features = next(m for m in backbone.heads.modules()
                           if isinstance(m, nn.Linear)).in_features
    backbone.heads = nn.Linear(in_features, 27)
    
    # Load pre-trained checkpoint and freeze all parameters
    sd = torch.load(args.checkpoint, map_location="cpu")
    backbone.load_state_dict(sd, strict=False)
    
    # Freeze backbone parameters (only train probe classifiers)
    for p_ in backbone.parameters(): 
        p_.requires_grad = False
        
    backbone.to(device).eval()

    # ------------------------- define probe locations -------------------
    
    # Define 16 probe locations throughout the ViT architecture
    probes = [
        ("probe1", "conv_proj"),           # After patch embedding convolution
        ("probe2", "encoder.dropout"),    # After positional embedding dropout
    ] + [
        # Probes after each transformer block (0-11)
        (f"probe{idx + 3}", f"encoder.layers.{idx}")  
        for idx in range(12)
    ] + [
        ("probe15", "encoder.ln"),        # After final layer normalization
        ("probe16", "cls_token"),         # CLS token after layer norm
    ]

    # ---------------------- CSV logging setup ---------------------------
    
    # Initialize CSV files for logging results
    ft_csv = out_dir / "dataset_ft_log.csv"      # Training logs
    pr_csv = out_dir / "probe_results.csv"       # Final test results
    
    # Create CSV headers if files don't exist
    if not ft_csv.exists():
        with ft_csv.open("w", newline="") as f:
            f.write("timestamp,probe,epoch,lr,train_loss,train_acc,val_loss,val_acc\n")
            
    if not pr_csv.exists():
        with pr_csv.open("w", newline="") as f:
            # Headers for confusion matrix, overall accuracy, and per-class accuracies
            headers = [
                "timestamp","probe","overall_cm_00","overall_cm_01",
                "overall_cm_10","overall_cm_11","overall_acc"
            ] + [f"class_{i}" for i in range(27)]
            f.write(",".join(headers) + "\n")

    # --------------------------- train probes ---------------------------
    
    # Train each linear probe sequentially
    for pid, tap in probes:
        print(f"\nTraining {pid} on layer: {tap}")
        
        # Dictionary to temporarily store hooked features
        feats: dict[str, torch.Tensor] = {}

        # Run a forward pass to determine feature dimensionality
        if tap == "cls_token":
            # Special hook for CLS token extraction
            def tmp_hook(_m, _i, o): 
                feats["x"] = o[:, 0]  # Extract CLS token
            hndl = backbone.encoder.ln.register_forward_hook(tmp_hook)
        else:
            # Standard hook for intermediate features
            def tmp_hook(_m, _i, o): 
                feats["x"] = o
            module = backbone.get_submodule(tap)  # Get target module by name
            hndl = module.register_forward_hook(tmp_hook)
            
        # Forward pass with dummy input to trigger hooks
        with torch.no_grad():
            backbone(torch.randn(1, 3, 224, 224, device=device))
        hndl.remove()  # Remove temporary hook
        
        # Determine feature dimensionality based on tensor shape
        feat_tensor = feats["x"]
        if feat_tensor.dim() == 4:      # Conv features: [B, C, H, W]
            fdim = feat_tensor.size(1)
        elif feat_tensor.dim() == 3:    # Transformer tokens: [B, N, D]  
            fdim = feat_tensor.size(2)
        else:                           # Flattened features
            fdim = feat_tensor.numel() // feat_tensor.size(0)

        # Create and initialize probe model
        model = LinearProbe(backbone, tap, fdim).to(device)
        
        # Setup optimizer and learning rate scheduler
        opt = optim.Adam(model.cls.parameters(), lr=1e-3)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                   patience=2, factor=0.1)
        crit = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_loss = float("inf")
        stale = 0  # Counter for epochs without improvement
        
        for ep in range(1, args.epochs + 1):
            # Get current learning rate
            lr_now = opt.param_groups[0]["lr"]
            
            # Run training and validation epochs
            tr_l, tr_a = run_epoch(model, loaders["train"], crit, opt, device, True)
            vl_l, vl_a = run_epoch(model, loaders["validation"], crit, opt, device, False)
            
            # Update learning rate based on validation loss
            sched.step(vl_l)
            
            # Log training progress to CSV
            with ft_csv.open("a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), pid, ep,
                    f"{lr_now:.3e}", f"{tr_l:.4f}", f"{tr_a:.2f}",
                    f"{vl_l:.4f}", f"{vl_a:.2f}"
                ])
            
            # Print progress
            print(f"{pid}: ep{ep} train_loss={tr_l:.4f} val_loss={vl_l:.4f} val_acc={vl_a:.2f}")
            
            # Early stopping logic
            if vl_l < best_loss - 1e-4:  # Significant improvement threshold
                best_loss, stale = vl_l, 0
                # Save best model checkpoint
                torch.save(model.cls.state_dict(), out_dir / f"{pid}.pth")
            else:
                stale += 1
                
            # Stop if no improvement for specified epochs
            if stale >= args.early_stop:
                print("Early stopping due to no improvement in validation loss")
                break

        # Ensure checkpoint exists (save current state if no improvement was made)
        ckpt = out_dir / f"{pid}.pth"
        if not ckpt.exists():
            torch.save(model.cls.state_dict(), ckpt)

        # Load best checkpoint and evaluate on test set
        model.cls.load_state_dict(torch.load(ckpt, map_location=device))
        cm, acc, stats = evaluate(model, loaders["test"], device)
        
        # Prepare results row for CSV logging
        row = [
            datetime.now().isoformat(),     # Timestamp
            pid,                            # Probe identifier
            *cm.reshape(-1),               # Flatten confusion matrix
            f"{acc:.2f}"                   # Overall accuracy
        ] + [
            # Per-class accuracy (0.00 if class not present in test set)
            f"{(stats[i]['correct']/stats[i]['total']*100 if i in stats else 0):.2f}"
            for i in range(27)
        ]
        
        # Log results to CSV
        with pr_csv.open("a", newline="") as f:
            csv.writer(f).writerow(row)
            
        print(f"{pid} completed - Test accuracy: {acc:.2f}%")

    print("\nAll probes finished successfully!")