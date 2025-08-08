#!/usr/bin/env python
"""
probe_classification_resnet50_18probes.py
=========================================
Insert linear probes into a trained ResNet-50 backbone at 18 specified layers
and evaluate top-1 / top-5 accuracy for new_label_id classification.
Also logs training and validation metrics per epoch to CSV.

What this script does (at a glance)
-----------------------------------
• Loads a ResNet-50 backbone that was trained for the dataset’s semantic labels.
• Freezes the backbone and, for each specified internal layer, attaches a tiny
  linear classifier (“probe”) on top of that layer’s activations.
• Trains ONLY the linear head (one per layer) on the train split; validates on val.
• Early-stops per probe using validation loss, saves the best head weights.
• Evaluates the best head on the test split and logs final metrics.
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
from torchvision.models import resnet50
from PIL import Image
from tqdm.auto import tqdm


class AverageMeter:
    """Keeps track of current value, average, sum, and count.
    Purpose: stream-friendly metric aggregation without storing all batches.
    """
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        # Start a fresh counter for a new epoch
        self.val = self.sum = self.count = self.avg = 0

    def update(self, v, n=1):
        # Update running totals with a batch value (v) and batch size (n)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k.
    Args:
      output: logits [B, C]
      target: ground-truth labels [B]
      topk: e.g. (1, 5)
    Returns:
      list of top-k accuracies in percentage for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Indices of top-k classes per sample: shape [B, maxk]
        _, pred = output.topk(maxk, 1, True, True)
        # Transpose to [maxk, B] for convenience when slicing by k
        pred = pred.t()
        # Compare predictions vs ground truth broadcasted to [1, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Count how many targets appear within the top-k predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Convert to % of the mini-batch
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomImageDataset(Dataset):
    """Dataset for images listed in a CSV with columns filepath, new_label_id, split.
    • Filters by split (train/validation/test).
    • Returns (transformed_image_tensor, class_id).
    """
    def __init__(self, csv_path: str | Path, split: str, transform):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Keep only the requested split; reset_index so iloc is stable
        df = df[df['split'] == split].reset_index(drop=True)
        if df.empty:
            # Useful early signal that metadata / split is wrong
            raise ValueError(f"No rows found for split='{split}'")
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        # Load the image (ensure RGB) and apply transforms
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        return self.transform(img), int(row['new_label_id'])


class LinearProbe(nn.Module):
    """Hook into backbone layer output and apply a linear classifier.
    Mechanics:
      • Register a forward hook on the given layer name -> cache its output.
      • On forward(), run the frozen backbone to populate the hook buffer.
      • If activation is 4D (B,C,H,W), global-average-pool → (B,C,1,1).
      • Flatten to (B,C) and feed a Linear(C→num_classes).
      • Only the Linear layer is learnable/optimized.
    """
    def __init__(self, backbone: nn.Module, layer_name: str, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.layer_name = layer_name
        self._feature_buffer: torch.Tensor | None = None
        # Look up the tapped module and register a forward hook on it
        target_module = dict(backbone.named_modules())[layer_name]
        target_module.register_forward_hook(
            # Store output tensor of tapped layer into an attribute
            lambda module, inp, out: setattr(self, '_feature_buffer', out)
        )
        # Global average pooling yields one vector per channel regardless of H×W
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # The actual probe classifier (trainable)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Empty the buffer in case a previous forward populated it
        self._feature_buffer = None
        # Run the frozen backbone; hook will capture intermediate features
        _ = self.backbone(x)
        feat = self._feature_buffer
        if feat is None:
            # Likely an invalid layer name or hook mishap
            raise RuntimeError(f"Hook did not capture features at layer '{self.layer_name}'")
        # If convolutional maps, spatially average to (B, C, 1, 1)
        if feat.dim() == 4:
            feat = self.global_avg_pool(feat)
        # Flatten to (B, C)
        feat = torch.flatten(feat, 1)
        # Return logits for CrossEntropyLoss
        return self.classifier(feat)


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    """One pass over the data loader.
    If training=True, update weights; otherwise pure evaluation.
    Returns averaged (loss, top1, top5).
    """
    if training:
        model.train()
    else:
        model.eval()

    meter_loss = AverageMeter('loss')
    meter_top1 = AverageMeter('top1')
    meter_top5 = AverageMeter('top5')

    # Only enable autograd in training to save memory/compute
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc='Training' if training else 'Validation', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            meter_loss.update(loss.item(), images.size(0))
            meter_top1.update(acc1.item(), images.size(0))
            meter_top5.update(acc5.item(), images.size(0))

    return meter_loss.avg, meter_top1.avg, meter_top5.avg


def main():
    parser = argparse.ArgumentParser(description='Linear Probe Classification on ResNet-50 Backbone')
    # Defaults assume your Dataset04_Bicubic layout; adjust as needed
    default_root = r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic'
    parser.add_argument('--data-csv', default=fr'{default_root}/metadata.csv')
    parser.add_argument('--checkpoint', default=r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet50/best_resnet50.pth')
    parser.add_argument('--output-dir', default=fr'{default_root}/probes50_classify')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    # • Train: mild augmentation (RandomResizedCrop + Flip) helps the linear head generalize.
    # • Val/Test: deterministic CenterCrop for stable evaluation.
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Datasets and loaders (three-way split)
    train_ds = CustomImageDataset(args.data_csv, 'train', train_transform)
    val_ds   = CustomImageDataset(args.data_csv, 'validation', val_transform)
    test_ds  = CustomImageDataset(args.data_csv, 'test', val_transform)

    # Infer number of classes by scanning label ids (assumes labels are 0..K-1)
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

    # Load and freeze backbone (ResNet-50)
    # Replace final FC to match semantic #classes (so checkpoint loads cleanly)
    backbone = resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    # strict=False tolerates minor key mismatches if present
    backbone.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device).eval()  # keep BN/Dropout fixed during probing

    # Probe definition: 18 positions across ResNet-50
    # (3 Bottleneck blocks in layer1, 4 in layer2, 6 in layer3, 3 in layer4)
    probes = [
        ('probe1',  'maxpool'),

        # stage1: layer1.0, layer1.1, layer1.2
        ('probe2',  'layer1.0'), ('probe3',  'layer1.1'), ('probe4',  'layer1.2'),

        # stage2: layer2.0, layer2.1, layer2.2, layer2.3
        ('probe5',  'layer2.0'), ('probe6',  'layer2.1'),
        ('probe7',  'layer2.2'), ('probe8',  'layer2.3'),

        # stage3: layer3.0 ~ layer3.5  (largest stage in ResNet-50)
        ('probe9',  'layer3.0'), ('probe10', 'layer3.1'),
        ('probe11', 'layer3.2'), ('probe12', 'layer3.3'),
        ('probe13', 'layer3.4'), ('probe14', 'layer3.5'),

        # stage4: layer4.0, layer4.1, layer4.2
        ('probe15', 'layer4.0'), ('probe16', 'layer4.1'),
        ('probe17', 'layer4.2'),

        ('probe18', 'avgpool'),
    ]

    # File capturing one test summary per probe (append-only)
    result_csv = out_dir / 'probe50_classify_results.csv'
    if not result_csv.exists():
        with result_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'probe', 'test_loss', 'test_top1', 'test_top5'])

    criterion = nn.CrossEntropyLoss()

    # Iterate over each probe (train a separate linear head for each tap point)
    for probe_id, layer in probes:
        # Determine feature dimension at this tap by a dry run with a one-off hook
        feats: dict[str, torch.Tensor] = {}
        handle = dict(backbone.named_modules())[layer].register_forward_hook(
            lambda m, inp, out: feats.setdefault('feat', out)  # keep first capture
        )
        with torch.no_grad():
            backbone(torch.randn(1, 3, 224, 224, device=device))
        handle.remove()

        f = feats['feat']
        # If 4D (B,C,H,W), after GAP the classifier will see C dims; else flatten to get dim
        feat_dim = f.size(1) if f.dim() == 4 else f.view(1, -1).size(1)

        # Build probe model for this layer; only classifier is trainable
        model = LinearProbe(backbone, layer, feat_dim, num_classes).to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
        # Shrink LR when validation loss plateaus to refine the tiny head
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

        # Per-probe, per-epoch CSV log
        log_csv = out_dir / f'{probe_id}_train_log.csv'
        if not log_csv.exists():
            with log_csv.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_top1', 'train_top5', 'val_loss', 'val_top1', 'val_top5'])

        best_val_loss = float('inf')  # best (lowest) validation loss so far
        best_epoch = -1               # epoch number where best was achieved
        no_improve_count = 0          # early-stopping counter

        # ---- Optimize the linear head on top of frozen features ----
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_t1, tr_t5 = run_epoch(model, dl_train, criterion, optimizer, device, training=True)
            val_loss, val_t1, val_t5 = run_epoch(model, dl_val, criterion, optimizer, device, training=False)
            scheduler.step(val_loss)

            # Console logging for quick monitoring
            print(f"[{probe_id}] Epoch {epoch} | "
                  f"Train Loss={tr_loss:.4f}, Top1={tr_t1:.2f}%, Top5={tr_t5:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Top1={val_t1:.2f}%, Top5={val_t5:.2f}%")

            # Append a CSV row with epoch metrics
            with log_csv.open('a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_t1:.2f}", f"{tr_t5:.2f}",
                                 f"{val_loss:.4f}", f"{val_t1:.2f}", f"{val_t5:.2f}"])

            # Early-stopping on validation loss (with small improvement margin)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_count = 0
                # Save only the classifier weights for this probe
                torch.save(model.classifier.state_dict(), out_dir / f'{probe_id}.pth')
            else:
                no_improve_count += 1
                if no_improve_count >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} for {probe_id}")
                    break

        # ---- Load the best head and evaluate on the (semantic) test split ----
        model.classifier.load_state_dict(torch.load(out_dir / f'{probe_id}.pth', map_location=device))
        test_loss, test_t1, test_t5 = run_epoch(model, dl_test, criterion, None, device, training=False)

        # Append a single summary line to the global results CSV for this probe
        with result_csv.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), probe_id,
                             f"{test_loss:.4f}", f"{test_t1:.2f}", f"{test_t5:.2f}"])

        print(f">>> Completed {probe_id} | Test Loss={test_loss:.4f}, Top1={test_t1:.2f}%, Top5={test_t5:.2f}% | Best Epoch={best_epoch}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Graceful stop for manual interruption during long probe sweeps
        print("Execution interrupted by user.")
        sys.exit(0)
