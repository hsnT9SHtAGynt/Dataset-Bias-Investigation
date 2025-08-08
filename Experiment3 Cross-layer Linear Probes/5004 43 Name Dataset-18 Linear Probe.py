r"""
Task C: Linear Probes on ResNet-50 Blocks for Dataset04
========================================================
Bicubic upsampling to 256→224 for both Tiny and CIFAR sources

Goal
----
Freeze a ResNet-50 backbone (pretrained on your 27-class semantic task) and attach
simple **linear classifiers** ("linear probes") at various internal layers ("taps").
Each probe predicts **dataset origin** (0=CIFAR-100, 1=TinyImageNet) from features
emitted at that layer. This quantifies how much dataset-source information is linearly
separable at different depths of the network.

Pipeline
--------
• Read `metadata.csv` (with columns: filepath, dataset_id, new_label_id, split)
• Create DataLoaders for train/validation/test with 224×224 center or random crops
• Load ResNet-50 weights from the 27-class model; freeze backbone parameters
• For each tap (e.g., 'layer2.0'):
    - Capture activations with a forward hook
    - Global average pool if needed; flatten
    - Train a linear classifier (2-way) with early stopping on val loss
    - Log train/val metrics and save best classifier
    - Evaluate on test set (overall confusion matrix + per-class source accuracy)
"""

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
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

# ------------------------ utility classes & funcs ----------------------------
class AverageMeter:
    # Simple running-average tracker used for per-epoch metrics.
    # Keeps the latest value `.val`, cumulative `.sum`, sample `.count`,
    # and the running average `.avg`.
    def __init__(self, name: str):
        self.name = name
        self.reset()
    def reset(self):
        self.val = self.sum = self.count = self.avg = 0
    def update(self, v: float, n: int = 1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # Compute top-1 accuracy (percentage) for a batch of logits vs integer targets.
    return logits.argmax(1).eq(targets).sum().item() * 100.0 / targets.size(0)

# ------------------------------ dataset --------------------------------------
class DatasetSourceProbe(Dataset):
    # Dataset wrapper used for *source* classification (CIFAR-100 vs TinyImageNet)
    # from the same metadata.csv used by your semantic model. It converts the
    # 'dataset_id' string into a binary label:
    #   0 = CIFAR-*  (string startswith "CIFAR")
    #   1 = Tiny*    (otherwise)
    # Returns tuples: (image_tensor, source_label, semantic_class_id)
    def __init__(self, csv_path: str | Path, split: str, tf):
        df = pd.read_csv(csv_path)
        # Keep only the requested split and make indexing stable
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with split={split} in {csv_path}")
        # Map dataset_id → {0,1}
        df["source_label"] = df["dataset_id"].apply(
            lambda x: 0 if str(x).upper().startswith("CIFAR") else 1
        )
        # Defensive cleaning: remove accidental whitespace around file paths
        df["filepath"] = df["filepath"].str.strip()
        self.df, self.tf = df, tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        # Return both the *binary* source label and the original semantic class id
        return self.tf(img), int(r["source_label"]), int(r["new_label_id"])

# --------------------------- linear-probe module -----------------------------
class LinearProbe(nn.Module):
    # A linear probe that taps into an intermediate layer of a frozen backbone.
    # It registers a forward hook at `tap`, grabs the activation, applies GAP
    # if needed, flattens, and passes through a single Linear layer to 2 classes
    # (CIFAR vs Tiny). Only the `cls` layer is trained.
    def __init__(self, backbone: nn.Module, tap: str, feat_dim: int):
        super().__init__()
        self.backbone, self.tap = backbone, tap
        self._buf: torch.Tensor | None = None
        # Forward hook to capture the activation of the tapped module.
        dict(backbone.named_modules())[tap].register_forward_hook(
            lambda m, i, o: setattr(self, '_buf', o)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # convert (B,C,H,W) → (B,C,1,1)
        self.cls = nn.Linear(feat_dim, 2)        # binary head for source prediction
    def forward(self, x):
        self._buf = None
        _ = self.backbone(x)     # run frozen backbone; fills self._buf via hook
        feats = self._buf
        if feats.dim() == 4:     # spatial maps → global average pooled vectors
            feats = self.gap(feats)
        feats = torch.flatten(feats, 1)  # (B,C,1,1) or (B,C) → (B,C)
        return self.cls(feats)

# --------------------------- train / eval loops -----------------------------
def run_epoch(model, loader, crit, opt, device, train: bool):
    # Generic pass over a loader. If train=True, optimize the probe's head.
    model.train() if train else model.eval()
    am_loss, am_acc = AverageMeter('loss'), AverageMeter('acc')
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
    # Full-dataset evaluation returning:
    #   • cm: 2×2 confusion matrix over {0=CIFAR, 1=Tiny}
    #   • acc: overall accuracy in %
    #   • stats: dict per semantic class id → {'total', 'correct'} for breakdowns
    model.eval()
    yt, yp, yc = [], [], []
    with torch.no_grad():
        for x, y, c in loader:
            out = model(x.to(device)).argmax(1).cpu()
            yp += out.tolist(); yt += y.tolist(); yc += c.tolist()
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    acc = cm.trace() / cm.sum() * 100.0
    stats = defaultdict(lambda: {'total':0, 'correct':0})
    for t, p, c in zip(yt, yp, yc):
        stats[c]['total'] += 1
        if t == p: stats[c]['correct'] += 1
    return cm, acc, stats

# ------------------------------ main -----------------------------------------
if __name__ == '__main__':
    # Parse CLI args (paths default to Dataset04 locations)
    p = argparse.ArgumentParser('Linear probes on ResNet-50 (Dataset04)')
    droot = r'D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\40Dataset04_Bicubic'
    p.add_argument('--data-csv',   default=fr'{droot}\metadata.csv')
    p.add_argument('--checkpoint', default=r'D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ResNet5004\best_resnet50.pth')
    p.add_argument('--output-dir', default=r'D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ResNet5004\NonSemantic_probes50')
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers',type=int, default=8)
    p.add_argument('--early-stop', type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bicubic upsample then crop to 224
    # NOTE: The dataset images are already upsampled to 256 in preprocessing.
    # Here we apply runtime transforms and standard ImageNet normalization.
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),   # data aug to help the tiny linear head
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(224),          # deterministic eval
        transforms.ToTensor(),
        normalize,
    ])

    # Build loaders for train/val/test. Shuffle only for train.
    loaders = {
        split: DataLoader(
            DatasetSourceProbe(
                args.data_csv, split,
                train_tf if split=='train' else test_tf
            ),
            batch_size=args.batch_size,
            shuffle=(split=='train'),
            num_workers=args.num_workers
        )
        for split in ('train','validation','test')
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare frozen ResNet-50 backbone and load 27-class semantic checkpoint.
    # We replace the final FC to match 27 semantic classes so the checkpoint
    # loads cleanly (strict=False further relaxes missing/unexpected keys).
    backbone = resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 27)
    sd = torch.load(args.checkpoint, map_location='cpu')
    backbone.load_state_dict(sd, strict=False)
    for p_ in backbone.parameters(): p_.requires_grad = False
    backbone.to(device).eval()  # freeze BN/Dropout behavior during probing

    # Tap points across ResNet-50: pool + every residual block + final avgpool.
    # Each tuple = (probe_id, module_name inside backbone.named_modules()).
    probes = [
        ('probe1',  'maxpool'),
        ('probe2',  'layer1.0'), ('probe3',  'layer1.1'), ('probe4',  'layer1.2'),
        ('probe5',  'layer2.0'), ('probe6',  'layer2.1'),
        ('probe7',  'layer2.2'), ('probe8',  'layer2.3'),
        ('probe9',  'layer3.0'), ('probe10', 'layer3.1'),
        ('probe11', 'layer3.2'), ('probe12', 'layer3.3'),
        ('probe13', 'layer3.4'), ('probe14', 'layer3.5'),
        ('probe15', 'layer4.0'), ('probe16', 'layer4.1'),
        ('probe17', 'layer4.2'),
        ('probe18', 'avgpool'),
    ]

    # CSV logs:
    #  • dataset_ft_log.csv : per-epoch train/val metrics for each probe
    #  • probe_results.csv  : final confusion matrix + overall acc + per-class acc
    ft_csv = out_dir / 'dataset_ft_log.csv'
    pr_csv = out_dir / 'probe_results.csv'
    if not ft_csv.exists():
        ft_csv.open('w', newline='').write(
            'timestamp,probe,epoch,lr,train_loss,train_acc,val_loss,val_acc\n'
        )
    if not pr_csv.exists():
        pr_csv.open('w', newline='').write(
            ','.join([
                'timestamp','probe',
                'overall_cm_00','overall_cm_01','overall_cm_10','overall_cm_11','overall_acc',
            ] + [f'class_{i}' for i in range(27)]) + '\n'
        )

    for pid, tap in probes:
        # ---- Infer feature dimensionality at the tap point via a dry run ----
        feats: dict[str, torch.Tensor] = {}
        def tmp_hook(m, _inp, out): feats['x'] = out
        handle = dict(backbone.named_modules())[tap].register_forward_hook(tmp_hook)
        with torch.no_grad():
            backbone(torch.randn(1, 3, 128, 128, device=device))  # small dummy input ok
        handle.remove()
        fdim = feats['x'].size(1) if feats['x'].dim() == 4 else feats['x'].numel() // feats['x'].size(0)

        # Build the probe model (frozen backbone + trainable 2-way classifier).
        model = LinearProbe(backbone, tap, fdim).to(device)
        opt = optim.Adam(model.cls.parameters(), lr=1e-3)
        # Scheduler reduces LR when validation loss plateaus → finer convergence.
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.1)
        crit = nn.CrossEntropyLoss()

        best_loss = float('inf')  # track best (lowest) validation loss
        stale = 0                 # epochs since last improvement (for early stop)
        for ep in range(1, args.epochs + 1):
            lr_now = opt.param_groups[0]["lr"]
            tr_l, tr_a = run_epoch(model, loaders['train'], crit, opt, device, True)
            vl_l, vl_a = run_epoch(model, loaders['validation'], crit, opt, device, False)
            # Step scheduler with validation loss signal
            sched.step(vl_l)
            csv.writer(ft_csv.open("a", newline="")).writerow([
                datetime.now().isoformat(), pid, ep,
                f"{lr_now:.3e}", f"{tr_l:.4f}", f"{tr_a:.2f}",
                f"{vl_l:.4f}", f"{vl_a:.2f}"])
            print(f"{pid}: ep{ep} train_loss={tr_l:.4f} val_loss={vl_l:.4f} val_acc={vl_a:.2f}")
            # Early stopping criterion on val loss (with tiny tolerance)
            if vl_l < best_loss - 1e-4:
                best_loss, stale = vl_l, 0
                torch.save(model.cls.state_dict(), out_dir / f"{pid}.pth")
            else:
                stale += 1
            if stale >= args.early_stop:
                print("Early stopping due to no improvement in validation loss")
                break

        # Ensure we have a checkpoint even if never improved (edge case)
        ckpt = out_dir / f"{pid}.pth"
        if not ckpt.exists(): torch.save(model.cls.state_dict(), ckpt)

        # ---- Final evaluation on the *test* split (semantic test set) ----
        model.cls.load_state_dict(torch.load(ckpt, map_location=device))
        cm, acc, stats = evaluate(model, loaders['test'], device)
        # Build CSV row: timestamp, probe id, flattened 2×2 CM, overall acc (%),
        # then per-semantic-class accuracies (%) for class ids 0..26.
        row = [
            datetime.now().isoformat(), pid, *cm.reshape(-1), f"{acc:.2f}"
        ] + [
            f"{(stats[i]['correct']/stats[i]['total']*100 if i in stats else 0):.2f}"
            for i in range(27)
        ]
        csv.writer(pr_csv.open("a", newline="")).writerow(row)
        print(pid, "done, test_acc", acc)

    print("All probes finished")
