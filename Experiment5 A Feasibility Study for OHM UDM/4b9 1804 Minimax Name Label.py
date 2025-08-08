#!/usr/bin/env python
"""
train_resnet18_adv_v2.py  ── Domain-adversarial ResNet-18 trainer (improved)
-------------------------------------------------------------------------
Key updates
===========
1. **Early-stopping on *validation classification loss***
   - The lowest `Val loss_cls` now determines the best checkpoint.
2. **Clearer metric names in logs**
   - `loss_total`, `loss_cls`, `loss_dom`, `acc_cls_top1`, `acc_cls_top5`, `acc_dom_top1`
     are used consistently.
3. **Top-5 Accuracy**
   - Top-5 classification accuracy is now tracked and reported.
4. **Save Final Metrics**
   - A `metrics.csv` file is saved with epoch-by-epoch training data
   - A `test_metrics.csv` file is saved with final test performance

Run exactly as before; CLI arguments are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm

# ────────────────────────────── Utils ──────────────────────────────────────────

class AverageMeter:
    """Tracks and updates running averages (e.g. loss, accuracy)."""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

@torch.no_grad()
def accuracy(out: torch.Tensor, tgt: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = tgt.size(0)

    _, pred = out.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(tgt.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# ─────────────────────── Gradient-Reversal Layer ──────────────────────────────

class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None

class GRL(nn.Module):
    def forward(self, x, lambd):
        return _GRLFn.apply(x, lambd)

# ───────────────────────────── Dataset ─────────────────────────────────────────

class AdvImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, transform):
        df = pd.read_csv(csv_path)

        if df["dataset_id"].dtype == object:
            df["dataset_id"] = df["dataset_id"].astype("category").cat.codes

        required = {"filepath", "new_label_id", "dataset_id", "split"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing columns: {required}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"split='{split}' is empty")

        self.t = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["filepath"]).convert("RGB")
        if self.t:
            img = self.t(img)
        return img, int(r["new_label_id"]), int(r["dataset_id"])

# ───────────────────── Domain-Adversarial ResNet-18 ────────────────────────────

class DANNResNet18(nn.Module):
    def __init__(self, num_classes: int, num_domains: int, pretrained=False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.class_head = nn.Linear(feat_dim, num_classes)
        self.domain_head = nn.Linear(feat_dim, num_domains)
        self.grl = GRL()

    def forward(self, x, lambd=0.0):
        feat = self.features(x).flatten(1)
        return self.class_head(feat), self.domain_head(self.grl(feat, lambd))

# ─────────────────────────── Train / Val loops ────────────────────────────────

def loop(model, loader, crit_c, crit_d, opt, dev, lambd, train=True):
    # NEW: Added 'acc_cls_top5' to the metrics
    meters = {n: AverageMeter(n) for n in (
        "loss_total", "loss_cls", "loss_dom",
        "acc_cls_top1", "acc_cls_top5", "acc_dom_top1"
    )}

    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, y, d in tqdm(loader, desc="Train" if train else "Val", leave=False):
            imgs, y, d = [t.to(dev) for t in (imgs, y, d)]
            if train:
                opt.zero_grad()

            ly, ld = model(imgs, lambd if train else 0.0)
            l_cls, l_dom = crit_c(ly, y), crit_d(ld, d)
            loss = l_cls + l_dom

            if train:
                loss.backward()
                opt.step()

            # NEW: Calculate both top-1 and top-5 accuracy
            acc1, acc5 = accuracy(ly, y, topk=(1, 5))
            dom_acc = accuracy(ld, d, topk=(1,))[0]

            # NEW: Update the loop to include top-5 accuracy
            metrics_to_update = {
                "loss_total": loss,
                "loss_cls": l_cls,
                "loss_dom": l_dom,
                "acc_cls_top1": acc1,
                "acc_cls_top5": acc5,
                "acc_dom_top1": dom_acc
            }
            for key, val in metrics_to_update.items():
                meters[key].update(val.item(), imgs.size(0))

    return {k: m.avg for k, m in meters.items()}

# λ schedule helper
def lambda_of(progress, lambd_max, gamma):
    """Mirrors DANN schedule:  λ(p) = λ_max * 2/(1+e^{-γp}) − 1"""
    return lambd_max * (2 / (1 + math.exp(-gamma * progress)) - 1)

# ─────────────────────────────── Main ─────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="Domain-Adversarial ResNet-18 Trainer")
    p.add_argument("--data-csv",
                   default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv",
                   help="Path to metadata CSV")
    p.add_argument("--output-dir",
                   default="D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1804/0.4_2_MINIMAX_7.27",
                   help="Save directory for model and logs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr-backbone", type=float, default=1e-3)
    p.add_argument("--lr-main", type=float, default=1e-3)
    p.add_argument("--lr-domain", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=0.4)
    p.add_argument("--lambda-gamma", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet-18 weights")
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {dev}")
    print(f"[INFO] Output directory: {out}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm])

    tr = AdvImageDataset(args.data_csv, "train", train_tf)
    va = AdvImageDataset(args.data_csv, "validation", test_tf)
    te = AdvImageDataset(args.data_csv, "test", test_tf)

    ncls = int(tr.df["new_label_id"].max() + 1)
    ndom = int(tr.df["dataset_id"].max() + 1)

    print(f"[INFO] {len(tr)} train / {len(va)} val / {len(te)} test samples")
    print(f"[INFO] {ncls} classes / {ndom} domains")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(dev.type == "cuda"))

    trL, vaL, teL = make_loader(tr, True), make_loader(va, False), make_loader(te, False)
    model = DANNResNet18(ncls, ndom, args.pretrained).to(dev)

    opt = optim.Adam([
        {"params": model.features.parameters(), "lr": args.lr_backbone},
        {"params": model.class_head.parameters(), "lr": args.lr_main},
        {"params": model.domain_head.parameters(), "lr": args.lr_domain},
    ])
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.1, patience=3)

    best_val_loss, best_ep, stall = float("inf"), 0, 0
    all_metrics: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        progress = (epoch - 1) / args.epochs
        lambd = lambda_of(progress, args.lambda_max, args.lambda_gamma)

        trm = loop(model, trL, crit, crit, opt, dev, lambd, True)
        vam = loop(model, vaL, crit, crit, None, dev, 0.0, False)

        sched.step(vam["loss_cls"])

        dur = time.time() - t0
        mem = (f", peak_mem={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
               if dev.type == "cuda" else "")

        print(f"\nEp {epoch}/{args.epochs} (λ={lambd:.3f}, "
              f"lr_back={opt.param_groups[0]['lr']:.1e}, "
              f"lr_main={opt.param_groups[1]['lr']:.1e}, "
              f"lr_dom={opt.param_groups[2]['lr']:.1e}) [{dur:.1f}s{mem}]")

        # NEW: Print top-5 accuracy
        print(f"  Train: loss_total={trm['loss_total']:.4f} loss_cls={trm['loss_cls']:.4f} loss_dom={trm['loss_dom']:.4f} | "
              f"cls_top1={trm['acc_cls_top1']:.2f}% cls_top5={trm['acc_cls_top5']:.2f}% dom_acc={trm['acc_dom_top1']:.2f}%")
        print(f"  Val  : loss_total={vam['loss_total']:.4f} loss_cls={vam['loss_cls']:.4f} loss_dom={vam['loss_dom']:.4f} | "
              f"cls_top1={vam['acc_cls_top1']:.2f}% cls_top5={vam['acc_cls_top5']:.2f}% dom_acc={vam['acc_dom_top1']:.2f}%")

        if vam["loss_cls"] < best_val_loss - 1e-4:
            best_val_loss, best_ep, stall = vam["loss_cls"], epoch, 0
            torch.save(model.state_dict(), out / "best.pth")
            print(f"  *** Best model saved (val_loss_cls={best_val_loss:.4f}) ***")
        else:
            stall += 1
            if stall >= args.early_stop_patience:
                print(f"Early stopping after {args.early_stop_patience} epochs of no improvement.")
                break

        # Collect metrics for CSV saving
        all_metrics.append({
            "epoch": epoch,
            "train_loss_total": trm["loss_total"],
            "train_loss_cls":   trm["loss_cls"],
            "train_loss_dom":   trm["loss_dom"],
            "train_acc_top1":   trm["acc_cls_top1"],
            "train_acc_top5":   trm["acc_cls_top5"],
            "train_dom_acc":    trm["acc_dom_top1"],
            "val_loss_total":   vam["loss_total"],
            "val_loss_cls":     vam["loss_cls"],
            "val_loss_dom":     vam["loss_dom"],
            "val_acc_top1":     vam["acc_cls_top1"],
            "val_acc_top5":     vam["acc_cls_top5"],
            "val_dom_acc":      vam["acc_dom_top1"],
        })

    # Save epoch metrics to CSV
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv = out / "metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"[INFO] all epoch metrics saved to {metrics_csv}")

    print("\n--- Testing Best Model ---")
    best_model = DANNResNet18(ncls, ndom)
    best_model.load_state_dict(torch.load(out / "best.pth", map_location=dev))
    best_model.to(dev)
    tm = loop(best_model, teL, crit, crit, None, dev, 0.0, False)

    # NEW: Print top-5 accuracy for the test set
    print(f"  Test Results (from epoch {best_ep}):")
    print(f"    Losses: total_loss={tm['loss_total']:.4f} cls_loss={tm['loss_cls']:.4f} dom_loss={tm['loss_dom']:.4f}")
    print(f"    Accs  : cls_top1={tm['acc_cls_top1']:.2f}% cls_top5={tm['acc_cls_top5']:.2f}% dom_acc={tm['acc_dom_top1']:.2f}%")

    # NEW: Save final test metrics to CSV instead of JSON
    test_metrics = {
        "best_epoch": best_ep,
        "test_loss_total": tm["loss_total"],
        "test_loss_cls":   tm["loss_cls"],
        "test_loss_dom":   tm["loss_dom"],
        "test_acc_top1":   tm["acc_cls_top1"],
        "test_acc_top5":   tm["acc_cls_top5"],
        "test_dom_acc":    tm["acc_dom_top1"],
    }

    df_test = pd.DataFrame([test_metrics])
    test_csv = out / "test_metrics.csv"
    df_test.to_csv(test_csv, index=False)
    print(f"[INFO] test metrics saved to {test_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)