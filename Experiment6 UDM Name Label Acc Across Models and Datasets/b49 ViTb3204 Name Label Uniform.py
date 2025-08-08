#!/usr/bin/env python3
r"""
train_vit_b_32_uniform.py  ── Domain-alignment ViT-B/32 trainer (uniform target)
-------------------------------------------------------------------------
Implements **strategy 2**: jointly optimize feature extractor and domain head
to push domain predictions toward a uniform distribution using ViT-B/32.

Usage is identical to the ResNet-18 version; just call with your arguments.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ────────────────────────────── Utils ─────────────────────────────────────────
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

# ───────────────────── Uniform-alignment ViT-B/32 ─────────────────────────────
class UniformViTB32(nn.Module):
    """Backbone ViT-B/32 + two heads (classification & domain) with no GRL."""
    def __init__(self, num_classes: int, num_domains: int, pretrained=False):
        super().__init__()
        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_32(weights=weights)
        backbone.heads = nn.Identity()
        self.backbone = backbone
        feat_dim = backbone.hidden_dim
        self.class_head = nn.Linear(feat_dim, num_classes)
        self.domain_head = nn.Linear(feat_dim, num_domains)

    def forward(self, x):
        feat = self.backbone(x)
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
    ap = argparse.ArgumentParser(description="Train ViT-B/32 with uniform domain alignment")
    ap.add_argument("--data-csv", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/40Dataset04_Bicubic/metadata.csv",
                        help="Path to metadata CSV containing (filepath, new_label_id, split)")
    ap.add_argument("--output-dir", default=r"D:/DeepLearning/Bias/User Files/ResNet18DatasetBias/Model Record/ViT_B32_04",
                        help="Directory to save models, logs, and checkpoints")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--lr-main", type=float, default=5e-5)
    ap.add_argument("--lr-domain", type=float, default=5e-5)
    ap.add_argument("--lambda-weight", type=float, default=0.4,
                    help="Coefficient for uniform domain loss")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="Number of linear warm‑up epochs")
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="Terminate if Val classification loss has not improved")
    ap.add_argument("--pretrained", action="store_true", default=False,
                    help="Use ImageNet-pretrained weights for backbone")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device}")

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
    model = UniformViTB32(ncls, ndom, args.pretrained).to(device)
    opt = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.class_head.parameters(),  "lr": args.lr_main},
        {"params": model.domain_head.parameters(), "lr": args.lr_domain},
    ], weight_decay=0.05)

    crit_cls = nn.CrossEntropyLoss()
    # Scheduler: linear warm-up -> cosine annealing
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0,
                      total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs,
                               eta_min=1e-6)
    sched = SequentialLR(opt, schedulers=[warmup, cosine],
                        milestones=[args.warmup_epochs])

    best_val_cls, best_ep, stall = float("inf"), 0, 0
    all_metrics: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()

        trm = loop(model, trL, crit_cls, args.lambda_weight, device, train=True)
        vam = loop(model, vaL, crit_cls, args.lambda_weight, device, train=False)
        sched.step()

        dur = time.time() - start
        mem = (f", peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB" if device.type=="cuda" else "")
        print(f"\nEpoch {epoch}/{args.epochs} (λ={args.lambda_weight} lr_back={opt.param_groups[0]['lr']:.1e} lr_main={opt.param_groups[1]['lr']:.1e} lr_dom={opt.param_groups[2]['lr']:.1e}, {dur:.1f}s){mem}")
        print(f"  Train: loss_total={trm['loss_total']:.4f} loss_cls={trm['loss_cls']:.4f} loss_dom={trm['loss_dom']:.4f} cls_top1={trm['cls_top1']:.2f}% cls_top5={trm['cls_top5']:.2f}% dom_acc={trm['dom_acc']:.2f}%")
        print(f"  Val  : loss_total={vam['loss_total']:.4f} loss_cls={vam['loss_cls']:.4f} loss_dom={vam['loss_dom']:.4f} cls_top1={vam['cls_top1']:.2f}% cls_top5={vam['cls_top5']:.2f}% dom_acc={vam['dom_acc']:.2f}%")

        if vam["loss_cls"] < best_val_cls - 1e-6:
            best_val_cls, best_ep, stall = vam["loss_cls"], epoch, 0
            torch.save(model.state_dict(), out_dir / "best_uniform.pth")
            print("*** best model saved ***")
        else:
            stall += 1
            if stall >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

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

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(f"[INFO] all epoch metrics saved to {out_dir/'metrics.csv'}")

    # Test best model
    best = UniformViTB32(ncls, ndom, args.pretrained).to(device)
    best.load_state_dict(torch.load(out_dir / "best_uniform.pth", map_location=device))
    tm = loop(best, teL, crit_cls, args.lambda_weight, device, train=False)
    print(f"\nTest : loss_total={tm['loss_total']:.4f}  "
          f"loss_cls={tm['loss_cls']:.4f}  "
          f"loss_dom={tm['loss_dom']:.4f}  "
          f"cls_top1={tm['cls_top1']:.2f}%  "
          f"cls_top5={tm['cls_top5']:.2f}%  "
          f"dom_acc={tm['dom_acc']:.2f}%  "
          f"(best epoch {best_ep})")
    df_test = pd.DataFrame([{"best_epoch": best_ep, **{f"test_{k}": v for k,v in tm.items()}}])
    df_test.to_csv(out_dir / "test_metrics.csv", index=False)
    print(f"[INFO] test metrics saved to {out_dir/'test_metrics.csv'}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
