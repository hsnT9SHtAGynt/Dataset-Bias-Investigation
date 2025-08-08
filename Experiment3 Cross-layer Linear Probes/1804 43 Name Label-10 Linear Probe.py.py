#!/usr/bin/env python
"""
probe_classification_resnet18_modified.py
========================================
Insert linear probes into a trained ResNet-18 backbone at each specified layer
and evaluate top-1 / top-5 accuracy for new_label_id classification.
Also logs training and validation metrics per epoch to CSV.

High-level idea
---------------
• A *linear probe* is a frozen backbone + a small trainable linear head attached at
  an intermediate layer. We measure how linearly separable class labels are at that layer.
• We register a forward hook on a chosen layer to capture its activations.
• If activations are 4D (B,C,H,W) we global-average-pool to (B,C,1,1), then flatten
  to (B,C), and pass through a Linear(C→num_classes) classifier.
• Backbone parameters are frozen; only the probe head is optimized.
"""

from __future__ import annotations
import argparse, os, sys, csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm.auto import tqdm


class AverageMeter:
    """Keeps track of current value, average, sum, and count.
    Useful for streaming metrics over an epoch without storing all batches.
    """
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        # Reset internal accumulators at the start of each epoch
        self.val = self.sum = self.count = self.avg = 0

    def update(self, v, n=1):
        # Update running totals: v is the batch metric; n is batch size
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k.
    Args:
      output: logits tensor of shape [B, num_classes]
      target: ground truth labels tensor of shape [B]
      topk: tuple of k values, e.g. (1,5)
    Returns:
      list of tensors containing accuracies (%) for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Take indices of top-k predictions per sample: [B, maxk]
        _, pred = output.topk(maxk, 1, True, True)
        # Transpose to [maxk, B] so slicing top-1/top-5 is easy
        pred = pred.t()
        # Compare predictions against targets broadcasted to [1,B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Count how many targets are present among the top-k predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Convert to percent of the batch
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomImageDataset(Dataset):
    """Dataset for images listed in a CSV with columns filepath, new_label_id, split.
    • Filters to the requested split.
    • Loads images via PIL and applies a torchvision transform.
    • Returns (tensor_image, int_label).
    """
    def __init__(self, csv_path: str | Path, split: str, transform):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            # Fail fast if metadata is missing
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Keep only rows that match the split; reset index for safe iloc
        df = df[df['split'] == split].reset_index(drop=True)
        if df.empty:
            # Surface mapping/split issues early
            raise ValueError(f"No rows found for split='{split}'")
        self.df = df
        self.transform = transform

    def __len__(self):
        # Standard Dataset API: total number of samples
        return len(self.df)

    def __getitem__(self, idx: int):
        # Load one image and its label; convert to RGB to avoid mode issues
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        return self.transform(img), int(row['new_label_id'])


class LinearProbe(nn.Module):
    """Hook into backbone layer output and apply a linear classifier.
    Implementation details:
      • Register a forward hook on `layer_name` to cache its output into `_feature_buffer`.
      • On forward, run the (frozen) backbone; the hook captures activations.
      • If the activation is 4D (B,C,H,W), apply global average pooling to (B,C,1,1).
      • Flatten to (B,C), then pass through a Linear(C→num_classes).
      • Only the Linear is trainable; backbone stays frozen/eval.
    """
    def __init__(self, backbone: nn.Module, layer_name: str, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.layer_name = layer_name
        self._feature_buffer: torch.Tensor | None = None
        # Look up the module object by name and register a forward hook on it
        target_module = dict(backbone.named_modules())[layer_name]
        target_module.register_forward_hook(
            # Store the raw output tensor in an attribute (no gradients needed here)
            lambda module, inp, out: setattr(self, '_feature_buffer', out)
        )
        # GAP normalizes spatial size across layers; emits one descriptor per channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear classifier that constitutes the "probe head"
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Clear any stale features from prior calls
        self._feature_buffer = None
        # Forward through the backbone triggers the hook and fills _feature_buffer
        _ = self.backbone(x)
        feat = self._feature_buffer
        if feat is None:
            # If this happens, the given layer name likely doesn't exist, or the hook failed
            raise RuntimeError(f"Hook did not capture features at layer '{self.layer_name}'")
        # If convolutional feature maps, spatially pool to (B,C,1,1)
        if feat.dim() == 4:
            feat = self.global_avg_pool(feat)
        # Flatten to (B,C) for the linear head
        feat = torch.flatten(feat, 1)
        # Output logits; CrossEntropyLoss expects raw scores
        return self.classifier(feat)


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    """Run one pass over a DataLoader (train or eval).
    Returns tuple: (avg_loss, avg_top1, avg_top5).
    """
    if training:
        model.train()   # enables grads, dropout/bn behavior (probe itself has none but good practice)
    else:
        model.eval()    # disables dropout/bn updates

    meter_loss = AverageMeter('loss')
    meter_top1 = AverageMeter('top1')
    meter_top5 = AverageMeter('top5')

    # Use grad context only for training to save memory/compute
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc='Training' if training else 'Validation', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)                 # logits for current batch
            loss = criterion(outputs, targets)      # CE loss w.r.t. labels

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute accuracy metrics for monitoring
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            meter_loss.update(loss.item(), images.size(0))
            meter_top1.update(acc1.item(), images.size(0))
            meter_top5.update(acc5.item(), images.size(0))

    return meter_loss.avg, meter_top1.avg, meter_top5.avg


def main():
    parser = argparse.ArgumentParser(description='Linear Probe Classification on ResNet-18 Backbone')
    # Default root assumes Dataset04_Bicubic layout; adjust as needed
    default_root = r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic'
    parser.add_argument('--data-csv', default=fr'{default_root}/metadata.csv')
    parser.add_argument('--checkpoint', default=r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1804/best_resnet18.pth')
    parser.add_argument('--output-dir', default=fr'{default_root}/probes_classify')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # Data normalization & transforms
    # -----------------------------
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Light augmentation for train to reduce overfitting on the probe head
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    # Deterministic center crop for validation/test
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # -----------------------------
    # Datasets & DataLoaders
    # -----------------------------
    train_ds = CustomImageDataset(args.data_csv, 'train', train_transform)
    val_ds   = CustomImageDataset(args.data_csv, 'validation', val_transform)
    test_ds  = CustomImageDataset(args.data_csv, 'test', val_transform)

    # Infer number of classes from max label id across splits (assumes labels are [0..K-1])
    num_classes = int(
        pd.concat([train_ds.df['new_label_id'], val_ds.df['new_label_id'], test_ds.df['new_label_id']]).max() + 1
    )
    pin_memory = (device.type == 'cuda')

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin_memory)
    dl_val   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin_memory)
    dl_test  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin_memory)

    # -----------------------------
    # Load backbone & freeze it
    # -----------------------------
    backbone = resnet18(weights=None)
    # Replace FC to match #classes that checkpoint expects (semantics-trained head)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    # strict=False tolerates minor mismatches (e.g., missing keys) if any
    backbone.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
    # Freeze all backbone parameters; only probe heads will learn
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device).eval()  # eval mode so BN stats, etc., remain fixed

    # -----------------------------
    # Where to tap features for probes (torchvision layer names)
    # -----------------------------
    probe_layers = [
        ('probe1', 'maxpool'),       # after stem conv/bn/relu + maxpool
        ('probe2', 'layer1.0'), ('probe3', 'layer1.1'),
        ('probe4', 'layer2.0'), ('probe5', 'layer2.1'),
        ('probe6', 'layer3.0'), ('probe7', 'layer3.1'),
        ('probe8', 'layer4.0'), ('probe9', 'layer4.1'),
        ('probe10', 'avgpool')       # just before final FC
    ]

    # CSV to store one line per probe with final test results (append-only)
    result_csv = out_dir / 'probe_classify_results.csv'
    if not result_csv.exists():
        with result_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'probe', 'test_loss', 'test_top1', 'test_top5'])

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Iterate over probes (train each head independently)
    # -----------------------------
    for probe_id, layer in probe_layers:
        # Determine feature dimension by running a dummy forward with a temporary hook
        feats: dict[str, torch.Tensor] = {}
        handle = dict(backbone.named_modules())[layer].register_forward_hook(
            lambda m, inp, out: feats.setdefault('feat', out)  # capture first time only
        )
        with torch.no_grad():
            backbone(torch.randn(1, 3, 224, 224, device=device))
        handle.remove()

        f = feats['feat']
        # If output is 4D (B,C,H,W), GAP will reduce to (B,C,1,1) → flattened (B,C) → dim=C
        # If output is already 2D, flattening preserves its feature dimension
        feat_dim = f.size(1) if f.dim() == 4 else f.view(1, -1).size(1)

        # Build the probe model for this layer (frozen backbone + trainable head)
        model = LinearProbe(backbone, layer, feat_dim, num_classes).to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
        # Reduce LR on validation plateaus to help convergence of the tiny head
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

        # Per-probe CSV capturing per-epoch train/val metrics
        log_csv = out_dir / f'{probe_id}_train_log.csv'
        if not log_csv.exists():
            with log_csv.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_top1', 'train_top5', 'val_loss', 'val_top1', 'val_top5'])

        best_val_loss = float('inf')  # best (lowest) validation loss seen
        best_epoch = -1               # epoch of best model
        no_improve_count = 0          # counter for early stopping

        # ----- Train the probe head -----
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_t1, tr_t5 = run_epoch(model, dl_train, criterion, optimizer, device, training=True)
            val_loss, val_t1, val_t5 = run_epoch(model, dl_val, criterion, optimizer, device, training=False)
            scheduler.step(val_loss)

            # Human-readable progress
            print(f"[{probe_id}] Epoch {epoch} | "
                  f"Train Loss={tr_loss:.4f}, Top1={tr_t1:.2f}%, Top5={tr_t5:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Top1={val_t1:.2f}%, Top5={val_t5:.2f}%")

            # Append a row to per-epoch CSV log
            with log_csv.open('a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_t1:.2f}", f"{tr_t5:.2f}",
                                 f"{val_loss:.4f}", f"{val_t1:.2f}", f"{val_t5:.2f}"])

            # Early-stopping logic on validation loss (with small improvement threshold)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_count = 0
                # Save only the classifier parameters for this probe head
                torch.save(model.classifier.state_dict(), out_dir / f'{probe_id}.pth')
            else:
                no_improve_count += 1
                if no_improve_count >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} for {probe_id}")
                    break

        # ----- Load best head and evaluate on the semantic Test split -----
        model.classifier.load_state_dict(torch.load(out_dir / f'{probe_id}.pth', map_location=device))
        test_loss, test_t1, test_t5 = run_epoch(model, dl_test, criterion, None, device, training=False)

        # Append one summary line per probe to the global results CSV
        with result_csv.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), probe_id,
                             f"{test_loss:.4f}", f"{test_t1:.2f}", f"{test_t5:.2f}"])

        print(f">>> Completed {probe_id} | Test Loss={test_loss:.4f}, Top1={test_t1:.2f}%, Top5={test_t5:.2f}% | Best Epoch={best_epoch}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Graceful interrupt for manual stop during long runs
        print("Execution interrupted by user.")
        sys.exit(0)
