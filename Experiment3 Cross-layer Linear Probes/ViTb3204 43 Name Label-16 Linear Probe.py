#!/usr/bin/env python
"""
probe_classification_vitb32_16probes_fixed.py
=============================================
Insert linear probes into a trained ViT-B/32 backbone at 16 specified layers
and evaluate top-1 / top-5 accuracy for new_label_id classification.
Also logs training, validation, and final test metrics per epoch to CSV.

Overview:
- Loads a frozen ViT-B/32 backbone and attaches a lightweight linear classifier
  (the "probe") to intermediate representations captured via forward hooks.
- For each specified tap point (layer), trains only the probe on the train split,
  selects the best checkpoint by validation loss with early stopping, and reports
  test performance.
- Saves per-epoch train/val logs per probe and a summary CSV across probes.

Assumptions:
- The metadata CSV contains columns: filepath, new_label_id, split.
- The checkpoint corresponds to a ViT-B/32 trained on the same class space.
- Top-5 accuracy is meaningful only if num_classes >= 5.
"""

from __future__ import annotations
import argparse, os, sys, csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_32
from PIL import Image
from tqdm.auto import tqdm


class AverageMeter:
    """Keeps track of current value, average, sum, and count.

    Note:
        - update(v, n) accumulates a weighted sum, so pass the batch size as n.
        - avg is safe when count==0 (remains 0.0).
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.sum = self.count = self.avg = 0.0

    def update(self, v, n=1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k.

    Args:
        output: logits tensor of shape (N, C)
        target: ground-truth class indices of shape (N,)
        topk:   iterable of K values, e.g., (1, 5)

    Returns:
        List of scalar Tensors with percentages for each requested K.

    Caution:
        torch.topk requires max(topk) <= C. For small C, adjust topk accordingly.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class CustomImageDataset(Dataset):
    """Dataset for images listed in a CSV with columns filepath, new_label_id, split.

    Behavior:
        - Filters rows by the provided split ("train", "validation", "test").
        - Loads images with PIL and converts to RGB for consistency.
        - Applies the provided torchvision transform pipeline.

    Raises:
        ValueError if no rows match the requested split.
    """

    def __init__(self, csv_path: str | Path, split: str, transform):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Ensure robust RGB mode; some datasets contain grayscale/alpha images.
        img = Image.open(row['filepath']).convert('RGB')
        return self.transform(img), int(row['new_label_id'])


class LinearProbe(nn.Module):
    """Hook into backbone layer output and apply a linear classifier.

    Design:
        - A forward hook captures intermediate features into `_feature_buffer`.
        - Depending on the feature tensor rank, apply a simple global pooling:
            * 4D (N, C, H, W) → AdaptiveAvgPool2d to (N, C, 1, 1) then flatten.
            * 3D (N, L, D)    → AdaptiveAvgPool1d over sequence (L) to (N, D).
            * 2D (N, D)       → use as-is.

        - The 'cls_token' special case hooks the classifier head's input
          (i.e., the representation before the head) by registering on `heads.head`
          and reading `inp[0]` provided to that module.

    Args:
        backbone: frozen ViT-B/32 model
        layer_name: string identifier matching a named module, or "cls_token"
        feat_dim: flattened feature dimension after pooling
        num_classes: output classes for the linear classifier
    """

    def __init__(self, backbone: nn.Module, layer_name: str, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.layer_name = layer_name
        self._feature_buffer: torch.Tensor | None = None

        # Register a forward hook on the specified layer to capture features.
        # For 'cls_token', capture the input to the final head (pre-classifier embedding).
        if layer_name == "cls_token":
            target_mod = backbone.heads.head
            target_mod.register_forward_hook(
                lambda module, inp, out: setattr(self, '_feature_buffer', inp[0])
            )
        else:
            modules = dict(backbone.named_modules())
            if layer_name not in modules:
                raise ValueError(f"Module '{layer_name}' not found in backbone")
            target_mod = modules[layer_name]
            target_mod.register_forward_hook(
                lambda module, inp, out: setattr(self, '_feature_buffer', out)
            )

        # Lightweight global pooling ops (used dynamically in forward)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Run the frozen backbone; `_feature_buffer` is populated by the hook.
        self._feature_buffer = None
        _ = self.backbone(x)  # triggers hook
        feat = self._feature_buffer
        if feat is None:
            raise RuntimeError(f"Hook did not capture features at layer '{self.layer_name}'")

        # Pooling strategy depends on feature dimensionality (conv vs. sequence vs. vector).
        if feat.dim() == 4:
            pooled = self.pool2d(feat)
            feat_vec = pooled.view(pooled.size(0), -1)
        elif feat.dim() == 3:
            # (N, L, D) → permute to (N, D, L) for 1D pooling across tokens
            f = feat.permute(0, 2, 1)
            pooled = self.pool1d(f)
            feat_vec = pooled.squeeze(-1)
        elif feat.dim() == 2:
            feat_vec = feat
        else:
            # Fallback: flatten unknown ranks conservatively.
            feat_vec = feat.view(feat.size(0), -1)
        return self.classifier(feat_vec)


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    """Runs one epoch for training or validation.

    Args:
        model: LinearProbe wrapping a frozen backbone
        loader: data loader for the split
        criterion: loss function (CrossEntropyLoss)
        optimizer: optimizer for probe params (None during eval)
        device: torch.device
        training: bool flag; True for train, False for eval

    Returns:
        (avg_loss, avg_top1, avg_top5) over the provided loader.
    """
    if training:
        model.train()
    else:
        model.eval()

    meter_loss = AverageMeter('loss')
    meter_top1 = AverageMeter('top1')
    meter_top5 = AverageMeter('top5')
    # Use grad context only during training to save memory/compute.
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc='Train' if training else 'Val', leave=False):
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
    parser = argparse.ArgumentParser(description='Linear Probe Classification on ViT-B/32')
    # Default root path placeholder; adjust to your filesystem layout.
    default_root = r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic'
    parser.add_argument('--data-csv', default=fr'{default_root}/metadata.csv')
    parser.add_argument('--checkpoint', default=r'D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ViT_B32_04/best_vit_b32.pth')
    parser.add_argument('--output-dir', default=fr'{default_root}/probes_vit32_classify')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Standard ImageNet normalization; mild augmentation for training.
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])

    # Build datasets/loaders for each split.
    train_ds = CustomImageDataset(args.data_csv, 'train', train_transform)
    val_ds = CustomImageDataset(args.data_csv, 'validation', val_transform)
    test_ds = CustomImageDataset(args.data_csv, 'test', val_transform)

    # Derive class count from union across splits (handles rare classes).
    num_classes = int(
        pd.concat([train_ds.df['new_label_id'], val_ds.df['new_label_id'], test_ds.df['new_label_id']]).max() + 1
    )
    pin_memory = (device.type == 'cuda')
    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin_memory)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin_memory)
    dl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=pin_memory)

    # Construct a ViT-B/32 backbone; replace classifier with Identity to access pre-head features.
    backbone = vit_b_32(weights=None)
    backbone.heads.head = nn.Identity()
    # Check if a trained checkpoint exists; if missing, the backbone stays randomly initialized.
    if not Path(args.checkpoint).is_file():
        print(
            f"Checkpoint file not found at {args.checkpoint}. Training from scratch or using pre-trained weights might be an option.")
        # Optional: add logic to exit or to load public pretrained weights.
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # strict=False allows shape-mismatched keys to be skipped (since head was removed).
        backbone.load_state_dict(checkpoint, strict=False)

    # Freeze the entire backbone; only the linear probe will be trainable.
    for param in backbone.parameters(): param.requires_grad = False
    backbone.to(device).eval()

    # Define 16 probe tap points:
    #   - Early conv & encoder dropout, then each transformer block,
    #   - final LayerNorm, plus a special 'cls_token' (pre-head embedding).
    probes = [
                 ("probe1", "conv_proj"), ("probe2", "encoder.dropout")
             ] + [(f"probe{idx + 3}", f"encoder.layers.encoder_layer_{idx}") for idx in range(12)] + [
                 ("probe15", "encoder.ln"), ("probe16", "cls_token")
             ]

    # Global results CSV across probes; creates header once.
    result_csv = out_dir / 'probe_vit32_results.csv'
    if not result_csv.exists():
        with result_csv.open('w', newline='') as f:
            csv.writer(f).writerow(['timestamp', 'probe', 'test_loss', 'test_top1', 'test_top5'])

    criterion = nn.CrossEntropyLoss()

    for probe_id, layer in probes:
        # Infer feature dimension by running one dummy forward with a hook capturing the tensor.
        feats: dict[str, torch.Tensor] = {}
        hook_mod = backbone.heads if layer == 'cls_token' else dict(backbone.named_modules())[layer]
        handle = hook_mod.register_forward_hook(
            (lambda module, inp, out: feats.setdefault('feat', inp[0]))
            if layer == 'cls_token' else
            (lambda module, inp, out: feats.setdefault('feat', out))
        )
        with torch.no_grad():
            backbone(torch.randn(1, 3, 224, 224, device=device))
        handle.remove()

        # === FIX START ===
        # Correctly determine the feature dimension:
        # - 3D (N, L, D) → D
        # - 4D (N, C, H, W) → C
        # - 2D (N, D) → D
        f = feats['feat']
        if f.dim() == 3:
            # For 3D tensors (Batch, SequenceLength, FeatureDim), we need the last dimension.
            feat_dim = f.shape[2]
        else:
            # For 4D (Batch, Channels, H, W) and 2D (Batch, FeatureDim), we need the second dimension.
            feat_dim = f.shape[1]
        # === FIX END ===

        # Build the probe around the frozen backbone; only classifier params are optimized.
        model = LinearProbe(backbone, layer, feat_dim, num_classes).to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
        # ReduceLROnPlateau: lowers LR when validation loss stalls, improving convergence.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

        # Per-probe training log CSV (created lazily).
        log_csv = out_dir / f'{probe_id}_train_log.csv'
        if not log_csv.exists():
            with log_csv.open('w', newline='') as f:
                csv.writer(f).writerow(
                    ['epoch', 'train_loss', 'train_top1', 'train_top5', 'val_loss', 'val_top1', 'val_top5'])

        best_val_loss = float('inf');  # track the best validation loss
        no_improve = 0;                # early-stopping counter
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_t1, tr_t5 = run_epoch(model, dl_train, criterion, optimizer, device, training=True)
            val_loss, val_t1, val_t5 = run_epoch(model, dl_val, criterion, None, device, training=False)
            scheduler.step(val_loss)  # step plateau scheduler with the monitored metric
            print(
                f"[{probe_id}] Epoch {epoch} | Train Loss={tr_loss:.4f}, Top1={tr_t1:.2f}%, Top5={tr_t5:.2f}% | Val Loss={val_loss:.4f}, Top1={val_t1:.2f}%, Top5={val_t5:.2f}%")
            with log_csv.open('a', newline='') as f:
                csv.writer(f).writerow(
                    [epoch, f"{tr_loss:.4f}", f"{tr_t1:.2f}", f"{tr_t5:.2f}", f"{val_loss:.4f}", f"{val_t1:.2f}",
                     f"{val_t5:.2f}"])
            if val_loss < best_val_loss - 1e-4:
                # Small margin prevents oscillation-triggered overwrites.
                best_val_loss = val_loss; best_epoch = epoch; no_improve = 0
                torch.save(model.classifier.state_dict(), out_dir / f'{probe_id}.pth')
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience: print(
                    f"Early stopping at epoch {epoch} for {probe_id}"); break

        # Load best classifier params and evaluate once on the test set.
        model.classifier.load_state_dict(torch.load(out_dir / f'{probe_id}.pth', map_location=device))
        test_loss, test_t1, test_t5 = run_epoch(model, dl_test, criterion, None, device, training=False)
        with result_csv.open('a', newline='') as f:
            csv.writer(f).writerow(
                [datetime.now().isoformat(), probe_id, f"{test_loss:.4f}", f"{test_t1:.2f}", f"{test_t5:.2f}"])
        print(
            f">>> Completed {probe_id} | Test Loss={test_loss:.4f}, Top1={test_t1:.2f}%, Top5={test_t5:.2f}% | Best Epoch={best_epoch}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Graceful interruption on Ctrl+C
        print("Execution interrupted by user.")
        sys.exit(0)
    except FileNotFoundError as e:
        # Common failure when metadata path is wrong or missing
        print(f"Error: Data file not found. Please check the path specified in --data-csv.")
        print(e)
        sys.exit(1)
    except ValueError as e:
        # Propagate dataset/argument validation failures
        print(f"Error: {e}")
        sys.exit(1)
