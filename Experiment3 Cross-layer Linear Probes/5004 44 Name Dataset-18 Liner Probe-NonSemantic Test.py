# evaluate_linear_probes_dataset04.py
"""
Evaluate 18 linear probes (Dataset-04) on BOTH the semantic Test split and the
held-out Non-Semantic Test split.

For each probe we:
  • print Test vs Non-Semantic accuracy side-by-side
  • save confusion matrices (CSV), sklearn classification reports (TXT)
    and a compact metrics.json
  • tee all console output to stdout_log.txt

Usage (defaults already filled in):

    python evaluate_linear_probes_dataset04.py
"""

from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self, name): self.name, self.reset()  # type: ignore
    def reset(self): self.val = self.sum = self.count = self.avg = 0.0
    def update(self, v, n=1):                           # type: ignore
        self.val, self.sum, self.count = v, self.sum + v * n, self.count + n
        self.avg = self.sum / self.count if self.count else 0.0


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return logits.argmax(1).eq(targets).sum().item() * 100. / targets.size(0)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DatasetSourceProbe(Dataset):
    """Semantic Test or Non-Semantic Test, returns (img, source_label, class_id)"""
    def __init__(self, csv_path: Path, split: str, tf):
        df = (pd.read_csv(csv_path)
                .query("split == @split")
                .reset_index(drop=True))
        if df.empty:
            raise ValueError(f"No rows with split='{split}' in {csv_path}")
        df["source_label"] = df["dataset_id"].apply(
            lambda x: 0 if str(x).upper().startswith("CIFAR") else 1)
        df["filepath"] = df["filepath"].str.strip()
        df["new_label_id"] = df["new_label_id"].fillna(-1).astype(int)

        self.df, self.tf = df, tf

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        return self.tf(img), int(r["source_label"]), int(r["new_label_id"])


# --------------------------------------------------------------------------- #
# Linear-probe wrapper
# --------------------------------------------------------------------------- #
class LinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module, tap: str, feat_dim: int):
        super().__init__()
        self.backbone, self.tap = backbone, tap
        self._buf: torch.Tensor | None = None
        dict(backbone.named_modules())[tap].register_forward_hook(
            lambda _m, _i, o: setattr(self, "_buf", o))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(feat_dim, 2)

    def forward(self, x):
        self._buf = None
        _ = self.backbone(x)
        feats = self._buf
        if feats.dim() == 4: feats = self.gap(feats)
        feats = torch.flatten(feats, 1)
        return self.cls(feats)


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(model, loader, device):
    preds, targets, class_names = [], [], []
    model.eval()
    for x, y, c in loader:
        logits = model(x.to(device)).argmax(1).cpu()
        preds.extend(logits.tolist())
        targets.extend(y.tolist())
        # ensure each class-id is a plain int, not a 0-D tensor
        class_names.extend(int(v) for v in c)
    return preds, targets, class_names



def _metrics(preds, targets, cls_names):
    arr_p, arr_t = np.array(preds), np.array(targets)

    out = {"overall_accuracy": float(accuracy_score(arr_t, arr_p))}

    # CIFAR vs Tiny breakdown …
    # (unchanged)

    # per-class stats
    per_class = defaultdict(lambda: {"tot": 0, "cor": 0})
    for cid, p, t in zip(cls_names, arr_p, arr_t):
        d = per_class[cid]
        d["tot"] += 1
        if p == t:
            d["cor"] += 1

    out["class_metrics"] = [
        {"class_id": k, "accuracy": round(v["cor"] / v["tot"], 4)}
        for k in sorted(per_class, key=int)
        for v in (per_class[k],)
    ]
    return out



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("Probe evaluation (Dataset04)")
    parser.add_argument("--root", default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\40Dataset04_Bicubic",
                        help="Path to 40Dataset04_Bicubic")
    parser.add_argument("--backbone_ckpt",
                        default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ResNet5004\best_resnet50.pth",
                        help="ResNet-50 checkpoint used during probe training")
    parser.add_argument("--probes-dir",
                        default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ResNet5004\NonSemantic_probes50",
                        help="Folder with *.pth heads for the 18 probes")
    parser.add_argument("--output-dir", default="./probe_comparison_results",
                        help="Where to dump confusion-matrices / reports / logs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- tee stdout to file ------------------------------------------------
    tee_f = open(out_root / "stdout_log.txt", "w", encoding="utf-8")
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data): [f.write(data) for f in self.files]
        def flush(self):       [f.flush() for f in self.files]
    sys.stdout = _Tee(sys.stdout, tee_f)

    print("======== Linear-Probe Two-Split Evaluation ========")
    print("Timestamp :", datetime.now().isoformat(timespec='seconds'))
    print("Data root :", args.root)
    print("Backbone  :", args.backbone_ckpt)
    print("Probes dir:", args.probes_dir)
    print("Out dir   :", out_root.resolve(), "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device    :", device, "\n")

    # Backbone
    backbone = resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 27)
    sd = torch.load(args.backbone_ckpt, map_location="cpu")
    backbone.load_state_dict(sd, strict=False)
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.to(device).eval()

    # Data
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    test_tf = transforms.Compose([transforms.CenterCrop(224),
                                  transforms.ToTensor(), normalize])

    csv_superset = Path(args.root) / "Newmetadata.csv"
    loaders = {
        "test": DataLoader(
            DatasetSourceProbe(csv_superset, "test", test_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
        "nonsem": DataLoader(
            DatasetSourceProbe(csv_superset, "Non-Semantic Test", test_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()),
    }

    # Probe definition
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


    # Header
    hdr = "{:8s} {:>8s} {:>8s} {:>8s}".format("Probe", "Test", "NonSem", "Δ")
    print(hdr)
    print("-" * len(hdr))

    # Loop
    for pid, tap in probes:
        ckpt_path = Path(args.probes_dir) / f"{pid}.pth"
        if not ckpt_path.exists():
            print(f"{pid:<8}  [checkpoint missing]")
            continue

        # infer feature-dim
        holder = {}
        handle = dict(backbone.named_modules())[tap].register_forward_hook(
            lambda _m, _i, o: holder.setdefault("x", o))
        _ = backbone(torch.randn(1, 3, 128, 128, device=device))
        handle.remove()
        ft = holder["x"]
        feat_dim = ft.size(1) if ft.dim() == 4 else ft.numel() // ft.size(0)

        # build probe & load head
        probe = LinearProbe(backbone, tap, feat_dim).to(device)
        probe.cls.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

        # Evaluate on both splits
        preds_t, tgt_t, cid_t   = _evaluate(probe, loaders["test"],   device)
        preds_n, tgt_n, cid_n   = _evaluate(probe, loaders["nonsem"], device)

        acc_t  = accuracy_score(tgt_t, preds_t) * 100
        acc_n  = accuracy_score(tgt_n, preds_n) * 100
        print(f"{pid:<8} {acc_t:8.2f} {acc_n:8.2f} {acc_t-acc_n:8.2f}")

        # -------- save artefacts --------------------------------------------
        probe_out = out_root / pid
        probe_out.mkdir(exist_ok=True)

        # confusion matrices
        cm_t = confusion_matrix(tgt_t, preds_t, labels=[0, 1])
        cm_n = confusion_matrix(tgt_n, preds_n, labels=[0, 1])
        pd.DataFrame(cm_t, index=["CIFAR-100","TinyImageNet"],
                           columns=["CIFAR-100","TinyImageNet"]
                     ).to_csv(probe_out / "cm_test.csv")
        pd.DataFrame(cm_n, index=["CIFAR-100","TinyImageNet"],
                           columns=["CIFAR-100","TinyImageNet"]
                     ).to_csv(probe_out / "cm_nonsem.csv")

        # classification reports
        with open(probe_out / "classification_report_test.txt", "w") as f:
            f.write(classification_report(tgt_t, preds_t,
                                          target_names=["CIFAR-100","TinyImageNet"],
                                          digits=4))
        with open(probe_out / "classification_report_nonsem.txt", "w") as f:
            f.write(classification_report(tgt_n, preds_n,
                                          target_names=["CIFAR-100","TinyImageNet"],
                                          digits=4))

        # metrics JSON
        metrics = {
            "test":   _metrics(preds_t, tgt_t, cid_t),
            "nonsem": _metrics(preds_n, tgt_n, cid_n),
            "delta_accuracy": acc_t - acc_n,
        }
        with open(probe_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nFinished —", datetime.now().isoformat(timespec='seconds'))
    tee_f.close()


if __name__ == "__main__":
    main()
