r"""
Task C: Linear Probes on ResNet-18 Blocks for Dataset04
========================================================
Bicubic upsampling to 256→224 for both Tiny and CIFAR sources

Goal
----
Freeze a ResNet-18 backbone (pretrained on your 27-class semantic task) and attach
simple **linear classifiers** ("linear probes") at various internal layers ("taps").
Each probe predicts **dataset origin** (0=CIFAR-100, 1=TinyImageNet) from features
emitted at that layer. This quantifies how much dataset-source information is linearly
separable at different depths of the network.

Pipeline
--------
• Read `metadata.csv` (with columns: filepath, dataset_id, new_label_id, split)
• Create DataLoaders for train/validation/test with 224×224 center or random crops
• Load ResNet-18 weights from the 27-class model; freeze backbone parameters
• For each tap (e.g., 'layer2.0'):
    - Capture activations with a forward hook
    - Global average pool if needed; flatten
    - Train a linear classifier (2-way) with early stopping on val loss
    - Log train/val metrics and save best classifier
    - Evaluate on test set (overall confusion matrix + per-class source accuracy)
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
from torchvision.models import resnet18
from tqdm.auto import tqdm

# ------------------------ utility classes & funcs ----------------------------
class AverageMeter:
    """Keeps running average, sum, count for any scalar metric (loss/acc)."""
    def __init__(self, name: str):
        self.name = name
        self.reset()
    def reset(self):
        # Initialize/reset internal counters
        self.val = self.sum = self.count = self.avg = 0
    def update(self, v: float, n: int = 1):
        # Update running statistics with a new value v over n samples
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute standard top-1 accuracy in percent for a batch."""
    return logits.argmax(1).eq(targets).sum().item() * 100.0 / targets.size(0)

# ------------------------------ dataset --------------------------------------
class DatasetSourceProbe(Dataset):
    """
    Minimal dataset that:
      • Reads rows for a selected split from metadata.csv
      • Converts dataset_id → source_label (0=CIFAR, 1=Tiny)
      • Returns (image_tensor, source_label, new_label_id)
        where new_label_id is the original 27-class semantic label (used for
        per-class source-accuracy stats in evaluation).
    """
    def __init__(self, csv_path: str | Path, split: str, tf):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with split={split} in {csv_path}")
        # Map textual dataset_id to binary origin label
        df["source_label"] = df["dataset_id"].apply(
            lambda x: 0 if str(x).upper().startswith("CIFAR") else 1
        )
        # Strip incidental whitespace in paths (defensive)
        df["filepath"] = df["filepath"].str.strip()
        self.df, self.tf = df, tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        # Return tensor image, binary origin label, and semantic class id
        return self.tf(img), int(r["source_label"]), int(r["new_label_id"])

# --------------------------- linear-probe module -----------------------------
class LinearProbe(nn.Module):
    """
    Wraps a frozen backbone and a *single* linear head that takes features from a
    specified internal layer ("tap"). It registers a forward hook to capture the
    activations. If the tapped features are spatial (BxCxHxW), we GAP them to BxC.

    Args:
      backbone : nn.Module (frozen)
      tap      : str       name of the module to hook (e.g., 'layer2.0')
      feat_dim : int       flattened feature dimension fed to the linear layer
    """
    def __init__(self, backbone: nn.Module, tap: str, feat_dim: int):
        super().__init__()
        self.backbone, self.tap = backbone, tap
        self._buf: torch.Tensor | None = None  # holds hooked activations
        # Register forward hook on the tapped module to capture its output tensor
        dict(backbone.named_modules())[tap].register_forward_hook(
            lambda m, i, o: setattr(self, '_buf', o)
        )
        # Global average pooling for 4D feature maps (B,C,H,W) → (B,C,1,1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Linear classifier for dataset origin (2 classes)
        self.cls = nn.Linear(feat_dim, 2)

    def forward(self, x):
        # Clear previous buffer, run backbone forward to fill it via hook
        self._buf = None
        _ = self.backbone(x)
        feats = self._buf  # activations captured from the tapped layer
        # If spatial features, pool to a vector per sample
        if feats.dim() == 4:
            feats = self.gap(feats)
        # Flatten to shape (B, C) from (B, C, 1, 1) or already-flat shape
        feats = torch.flatten(feats, 1)
        # Linear classification logits (B, 2)
        return self.cls(feats)

# --------------------------- train / eval loops -----------------------------
def run_epoch(model, loader, crit, opt, device, train: bool):
    """
    One pass over a loader. If `train` is True, do SGD; otherwise, eval only.
    Returns:
      avg_loss, avg_acc (over all samples)
    """
    model.train() if train else model.eval()
    am_loss, am_acc = AverageMeter('loss'), AverageMeter('acc')
    # Use context manager to enable/disable grad as needed
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _ in tqdm(loader, desc='train' if train else 'eval', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            acc = top1_accuracy(logits, y)
            am_loss.update(loss.item(), x.size(0))
            am_acc.update(acc, x.size(0))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
    return am_loss.avg, am_acc.avg


def evaluate(model, loader, device):
    """
    Full test pass:
      • Compute predictions & confusion matrix on source labels (0/1)
      • Return overall accuracy and *per-semantic-class* source accuracy
        (how well each original 27-class group’s images are recognized by origin).
    Returns:
      cm   : 2x2 confusion matrix (rows=GT, cols=Pred)
      acc  : overall accuracy in %
      stats: dict[class_id] -> {'total': n, 'correct': k}
    """
    model.eval()
    yt, yp, yc = [], [], []  # true labels, preds, semantic class ids
    with torch.no_grad():
        for x, y, c in loader:
            out = model(x.to(device)).argmax(1).cpu()
            yp += out.tolist(); yt += y.tolist(); yc += c.tolist()
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    acc = cm.trace() / cm.sum() * 100.0
    # Aggregate per semantic class (27-way) accuracy for origin prediction
    stats = defaultdict(lambda: {'total':0, 'correct':0})
    for t, p, c in zip(yt, yp, yc):
        stats[c]['total'] += 1
        if t == p: stats[c]['correct'] += 1
    return cm, acc, stats

# ------------------------------ main -----------------------------------------
if __name__ == '__main__':
    # ------------------------ CLI arguments ------------------------
    p = argparse.ArgumentParser('Linear probes on ResNet-18 (Dataset04)')
    droot = r'D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\40Dataset04_Bicubic'
    p.add_argument('--data-csv',   default=fr'{droot}\metadata.csv', help="Path to Dataset04 metadata.csv")
    p.add_argument('--checkpoint', default=r'D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ResNet1804\best_resnet18.pth', help="27-class checkpoint to freeze as backbone")
    p.add_argument('--output-dir', default=fr'{droot}\probes', help="Directory to store probe weights and logs")
    p.add_argument('--epochs',     type=int, default=30, help="Max epochs per probe")
    p.add_argument('--batch-size', type=int, default=128, help="Batch size")
    p.add_argument('--num-workers',type=int, default=8, help="DataLoader workers")
    p.add_argument('--early-stop', type=int, default=5, help="Patience on val loss")
    args = p.parse_args()

    # ------------------------ I/O setup ------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------ transforms ------------------------
    # NOTE: Both sources were bicubic-upsampled to 256; we crop to 224 here to match
    # standard ImageNet training/test practices and stabilize features.
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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

    # ------------------------ dataloaders ------------------------
    loaders = {
        split: DataLoader(
            DatasetSourceProbe(
                args.data_csv, split,
                train_tf if split=='train' else test_tf
            ),
            batch_size=args.batch_size,
            shuffle=(split=='train'),
            num_workers=args.num_workers
        ) for split in ('train','validation','test')
    }

    # ------------------------ backbone (frozen) ------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = resnet18(weights=None)                    # architecture only
    backbone.fc = nn.Linear(backbone.fc.in_features, 27) # match checkpoint head
    sd = torch.load(args.checkpoint, map_location='cpu') # load 27-class weights
    # Use strict=False so minor non-matching keys (if any) don't break loading
    backbone.load_state_dict(sd, strict=False)
    # Freeze all backbone params; we train ONLY the linear probe’s classifier
    for p_ in backbone.parameters(): p_.requires_grad = False
    backbone.to(device).eval()

    # ------------------------ probe tap points ------------------------
    # Each tuple = (probe_id, module_name_to_hook)
    # We probe after early pooling and after each residual block pair, plus avgpool.
    probes = [
        ('probe1','maxpool'),('probe2','layer1.0'),('probe3','layer1.1'),
        ('probe4','layer2.0'),('probe5','layer2.1'),('probe6','layer3.0'),
        ('probe7','layer3.1'),('probe8','layer4.0'),('probe9','layer4.1'),
        ('probe10','avgpool')
    ]

    # ------------------------ CSV logs ------------------------
    ft_csv = out_dir / 'dataset_ft_log.csv'   # per-epoch training/val metrics for each probe
    pr_csv = out_dir / 'probe_results.csv'    # final test metrics per probe
    if not ft_csv.exists():
        ft_csv.open('w', newline='').write(
            'timestamp,probe,epoch,lr,train_loss,train_acc,val_loss,val_acc'
        )
    if not pr_csv.exists():
        # Per-probe row: timestamp, probe, flattened 2x2 CM, overall_acc, then 27 per-class accuracies
        pr_csv.open('w', newline='').write(
            ','.join([
                'timestamp','probe','overall_cm_00','overall_cm_01','overall_cm_10','overall_cm_11','overall_acc'
            ] + [f'class_{i}' for i in range(27)]) + '\n'
        )

    # ------------------------ run all probes ------------------------
    for pid, tap in probes:
        # 1) Infer feature dimension at the tap by doing a dummy forward and capturing the activation
        feats: dict[str, torch.Tensor] = {}
        def tmp_hook(m, _inp, out): feats['x'] = out
        handle = dict(backbone.named_modules())[tap].register_forward_hook(tmp_hook)
        with torch.no_grad():
            # dummy input to elicit one forward and capture hook output
            backbone(torch.randn(1,3,128,128,device=device))
        handle.remove()
        feat_tensor = feats['x']
        # If spatial (B,C,H,W), the linear head will see GAPped BxC, so feat_dim=C
        # Otherwise (e.g., after avgpool), compute flattened per-sample dimension
        fdim = feat_tensor.size(1) if feat_tensor.dim()==4 else feat_tensor.numel()//feat_tensor.size(0)

        # 2) Create the linear probe model for this tap
        model = LinearProbe(backbone, tap, fdim).to(device)
        opt = optim.Adam(model.cls.parameters(), lr=1e-3)  # only classifier params train
        # Reduce LR when validation loss plateaus; focuses on generalization
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.1)
        crit = nn.CrossEntropyLoss()

        # 3) Train with early stopping on validation loss
        best_loss = float('inf')  # best (lowest) val loss seen
        stale = 0                 # epochs since last improvement
        for ep in range(1, args.epochs + 1):
            lr_now = opt.param_groups[0]['lr']
            tr_l, tr_a = run_epoch(model, loaders['train'], crit, opt, device, True)
            vl_l, vl_a = run_epoch(model, loaders['validation'], crit, opt, device, False)
            sched.step(vl_l)

            # Append epoch metrics to CSV
            csv.writer(ft_csv.open('a', newline='')).writerow([
                datetime.now().isoformat(), pid, ep,
                f"{lr_now:.3e}", f"{tr_l:.4f}", f"{tr_a:.2f}",
                f"{vl_l:.4f}", f"{vl_a:.2f}" ]
            )
            print(f"{pid}: ep{ep} train_loss={tr_l:.4f} val_loss={vl_l:.4f} val_acc={vl_a:.2f}")

            # Early stopping logic (on validation LOSS for stability)
            if vl_l < best_loss - 1e-4:
                best_loss, stale = vl_l, 0
                # Save the probe’s classifier weights
                torch.save(model.cls.state_dict(), out_dir / f"{pid}.pth")
            else:
                stale += 1
            if stale >= args.early_stop:
                print("Early stopping due to no improvement in validation loss")
                break

        # Ensure we have a checkpoint for this probe (in case no improvement occurred)
        ckpt = out_dir / f"{pid}.pth"
        if not ckpt.exists(): torch.save(model.cls.state_dict(), ckpt)

        # 4) Load best classifier weights and evaluate on TEST split
        model.cls.load_state_dict(torch.load(ckpt, map_location=device))
        cm, acc, stats = evaluate(model, loaders['test'], device)

        # 5) Write final results: flattened confusion matrix + overall accuracy + per-27-class source accuracy
        row = [
            datetime.now().isoformat(), pid, *cm.reshape(-1), f"{acc:.2f}"
        ] + [
            # Handle classes absent in test split (avoid division by zero)
            f"{(stats[i]['correct']/stats[i]['total']*100 if i in stats else 0):.2f}"
            for i in range(27)
        ]
        csv.writer(pr_csv.open('a', newline='')).writerow(row)
        print(pid, "done, test_acc", acc)

    print("All probes finished")
