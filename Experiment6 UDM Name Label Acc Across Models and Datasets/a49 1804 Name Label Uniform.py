#!/usr/bin/env python3
r"""
train_resnet18_uniform.py  ── Domain-alignment ResNet-18 trainer (uniform target)
-------------------------------------------------------------------------
Implements **strategy 2**: jointly optimize feature extractor and domain head
to push domain predictions toward a uniform distribution.

Usage is identical to the adversarial version; just call with your arguments.
"""
from __future__ import annotations
import argparse, sys, time
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm

# ────────────────────────────── Utils ──────────────────────────────────────────
class AverageMeter:
    """Keeps track of running averages for a given metric."""
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
    """Computes Top-*k* classification accuracy."""
    maxk, bs = max(topk), tgt.size(0)
    _, pred = out.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(tgt.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / bs))
    return res

# ───────────────────────────── Dataset ─────────────────────────────────────────
class AdvImageDataset(Dataset):
    """Reads a metadata CSV and loads images lazily."""
    def __init__(self, csv_path: str | Path, split: str, transform):
        df = pd.read_csv(csv_path)
        if df["dataset_id"].dtype == object:
            df["dataset_id"] = df["dataset_id"].astype("category").cat.codes
        required = {"filepath", "new_label_id", "dataset_id", "split"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing columns: {required}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"split='{split}' is empty in CSV {csv_path}")
        self.t = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["filepath"]).convert("RGB")
        if self.t:
            img = self.t(img)
        return img, int(r["new_label_id"]), int(r["dataset_id"])

# ───────────────────── Uniform-alignment ResNet-18 ────────────────────────────
class UniformResNet18(nn.Module):
    """Backbone + two heads (classification & domain) with no GRL."""
    def __init__(self, num_classes: int, num_domains: int, pretrained=False):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.class_head = nn.Linear(feat_dim, num_classes)
        self.domain_head = nn.Linear(feat_dim, num_domains)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.class_head(feat), self.domain_head(feat)

# ───────────────────── Loss helpers ───────────────────────────────────────────
def uniform_domain_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy to the uniform distribution."""
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs.mean()

# ─────────────────────────── Train / Val loops ────────────────────────────────
def loop(model, loader, crit_cls, lambda_w, device, train=True):
    meters = {n: AverageMeter(n) for n in (
        "loss_total", "loss_cls", "loss_dom", "cls_top1", "cls_top5", "dom_acc")}
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, y, d in tqdm(loader, desc="Train" if train else "Val", leave=False):
            imgs, y, d = [t.to(device) for t in (imgs, y, d)]
            if train:
                opt.zero_grad()

            ly, ld = model(imgs)
            l_cls = crit_cls(ly, y)
            l_dom = uniform_domain_loss(ld)
            loss = l_cls + lambda_w * l_dom

            if train:
                loss.backward()
                opt.step()

            acc1, acc5 = accuracy(ly, y, topk=(1, 5))
            dom_acc = accuracy(ld, d, topk=(1,))[0]

            for key, val in zip((
                "loss_total", "loss_cls", "loss_dom", "cls_top1", "cls_top5", "dom_acc"),
                (loss, l_cls, l_dom, acc1, acc5, dom_acc)):
                meters[key].update(val.item(), imgs.size(0))

    return {k: m.avg for k, m in meters.items()}

# ─────────────────────────────── Main ─────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train ResNet-18 with uniform domain alignment")
    ap.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv",
                   help="Path to metadata CSV containing (filepath, new_label_id, split)")
    ap.add_argument("--output-dir", default="D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ResNet1804/exp_Uni7.9",
                   help="Save directory")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=1e-3)
    ap.add_argument("--lr-main", type=float, default=1e-3)
    ap.add_argument("--lr-domain", type=float, default=1e-3)
    ap.add_argument("--lambda-weight", type=float, default=0.4,
                    help="Coefficient for uniform domain loss")
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="Terminate if Val classification loss has not improved")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use ImageNet-pretrained weights for backbone")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device}")

    # Data transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm])

    tr = AdvImageDataset(args.data_csv, "train", train_tf)
    va = AdvImageDataset(args.data_csv, "validation", test_tf)
    te = AdvImageDataset(args.data_csv, "test", test_tf)

    ncls = int(max(tr.df["new_label_id"].max(),
                   va.df["new_label_id"].max(),
                   te.df["new_label_id"].max()) + 1)
    ndom = int(max(tr.df["dataset_id"].max(),
                   va.df["dataset_id"].max(),
                   te.df["dataset_id"].max()) + 1)

    print(f"[INFO] {len(tr)} train / {len(va)} val / {len(te)} test "
          f"(classes={ncls}, domains={ndom})")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.workers,
                          pin_memory=(device.type=="cuda"))

    trL, vaL, teL = make_loader(tr, True), make_loader(va, False), make_loader(te, False)

    global model, opt
    model = UniformResNet18(ncls, ndom, args.pretrained).to(device)
    opt = optim.Adam([
        {"params": model.features.parameters(), "lr": args.lr_backbone},
        {"params": model.class_head.parameters(),  "lr": args.lr_main},
        {"params": model.domain_head.parameters(), "lr": args.lr_domain},
    ])
    crit_cls = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 factor=0.1, patience=3)

    best_val_cls, best_ep, stall = float("inf"), 0, 0
    all_metrics: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        trm = loop(model, trL, crit_cls, args.lambda_weight, device, train=True)
        vam = loop(model, vaL, crit_cls, args.lambda_weight, device, train=False)
        sched.step(vam["loss_cls"])

        dur = time.time() - start
        mem = (f", peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
               if device.type=="cuda" else "")

        print(f"\nEpoch {epoch}/{args.epochs}  (λ={args.lambda_weight}  "
              f"lr_back={opt.param_groups[0]['lr']:.1e} "
              f"lr_main={opt.param_groups[1]['lr']:.1e} "
              f"lr_dom={opt.param_groups[2]['lr']:.1e}, {dur:.1f}s){mem}")
        print(f"  Train: loss_total={trm['loss_total']:.4f}  "
              f"loss_cls={trm['loss_cls']:.4f}  "
              f"loss_dom={trm['loss_dom']:.4f}  "
              f"cls_top1={trm['cls_top1']:.2f}%  "
              f"cls_top5={trm['cls_top5']:.2f}%  "
              f"dom_acc={trm['dom_acc']:.2f}%")
        print(f"  Val  : loss_total={vam['loss_total']:.4f}  "
              f"loss_cls={vam['loss_cls']:.4f}  "
              f"loss_dom={vam['loss_dom']:.4f}  "
              f"cls_top1={vam['cls_top1']:.2f}%  "
              f"cls_top5={vam['cls_top5']:.2f}%  "
              f"dom_acc={vam['dom_acc']:.2f}%")

        # Save best
        if vam["loss_cls"] < best_val_cls - 1e-4:
            best_val_cls, best_ep, stall = vam["loss_cls"], epoch, 0
            torch.save(model.state_dict(), out_dir / "best_uniform.pth")
            print("*** best model saved ***")
        else:
            stall += 1
            if stall >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

        # Collect metrics
        all_metrics.append({
            "epoch": epoch,
            "train_loss_total": trm["loss_total"],
            "train_loss_cls":   trm["loss_cls"],
            "train_loss_dom":   trm["loss_dom"],
            "train_acc_top1":   trm["cls_top1"],
            "train_acc_top5":   trm["cls_top5"],
            "train_dom_acc":    trm["dom_acc"],
            "val_loss_total":   vam["loss_total"],
            "val_loss_cls":     vam["loss_cls"],
            "val_loss_dom":     vam["loss_dom"],
            "val_acc_top1":     vam["cls_top1"],
            "val_acc_top5":     vam["cls_top5"],
            "val_dom_acc":      vam["dom_acc"],
        })

    # Save epoch metrics
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv = out_dir / "metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"[INFO] all epoch metrics saved to {metrics_csv}")

    # Test best model
    best = UniformResNet18(ncls, ndom)
    best.load_state_dict(torch.load(out_dir / "best_uniform.pth",
                                    map_location=device))
    best.to(device)
    tm = loop(best, teL, crit_cls, args.lambda_weight, device, train=False)

    print(f"\nTest : loss_total={tm['loss_total']:.4f}  "
          f"loss_cls={tm['loss_cls']:.4f}  "
          f"loss_dom={tm['loss_dom']:.4f}  "
          f"cls_top1={tm['cls_top1']:.2f}%  "
          f"cls_top5={tm['cls_top5']:.2f}%  "
          f"dom_acc={tm['dom_acc']:.2f}%  "
          f"(best epoch {best_ep})")

    # Save test metrics
    test_metrics = {
        "best_epoch": best_ep,
        "test_loss_total": tm["loss_total"],
        "test_loss_cls":   tm["loss_cls"],
        "test_loss_dom":   tm["loss_dom"],
        "test_acc_top1":   tm["cls_top1"],
        "test_acc_top5":   tm["cls_top5"],
        "test_dom_acc":    tm["dom_acc"],
    }
    df_test = pd.DataFrame([test_metrics])
    test_csv = out_dir / "test_metrics.csv"
    df_test.to_csv(test_csv, index=False)
    print(f"[INFO] test metrics saved to {test_csv}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
