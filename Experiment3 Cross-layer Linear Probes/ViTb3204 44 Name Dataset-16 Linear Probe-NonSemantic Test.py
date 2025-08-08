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

import argparse, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_32


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()
    def reset(self):
        self.val = self.sum = self.count = self.avg = 0.0
    def update(self, v: float, n: int = 1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return logits.argmax(1).eq(targets).sum().item() * 100.0 / targets.size(0)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DatasetSourceProbe(Dataset):
    """Semantic Test or Non-Semantic Test, returns (img, source_label, class_id)"""
    def __init__(self, csv_path: Path, split: str, tf):
        df = (
            pd.read_csv(csv_path)
              .query("split == @split")
              .reset_index(drop=True)
        )
        if df.empty:
            raise ValueError(f"No rows with split='{split}' in {csv_path}")
        df["source_label"] = df["dataset_id"].apply(
            lambda x: 0 if str(x).upper().startswith("CIFAR") else 1
        )
        df["filepath"] = df["filepath"].str.strip()
        df["new_label_id"] = df["new_label_id"].fillna(-1).astype(int)

        self.df = df
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["filepath"]).convert("RGB")
        return self.tf(img), int(r["source_label"]), int(r["new_label_id"])


# --------------------------------------------------------------------------- #
# Linear-probe wrapper for ViT-B/32
# --------------------------------------------------------------------------- #
class LinearProbe(nn.Module):
    """Freezes backbone and attaches a 2-way classifier at given tap."""
    def __init__(self, backbone: nn.Module, tap: str, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.tap = tap
        self._buf: torch.Tensor | None = None

        # Register hook
        if tap == "cls_token":
            # Virtual hook: grab CLS token output after encoder.ln
            def cls_hook(_m, _i, output):
                # output: [B, N, D]
                self._buf = output[:, 0]
            backbone.encoder.ln.register_forward_hook(cls_hook)
        else:
            module = dict(backbone.named_modules()).get(tap)
            if module is None:
                raise ValueError(f"Module {tap} not found in backbone")
            module.register_forward_hook(
                lambda _m, _i, output: setattr(self, '_buf', output)
            )

        # Pooler & classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._buf = None
        _ = self.backbone(x)
        feats = self._buf
        if feats is None:
            raise RuntimeError(f"Hook at {self.tap} produced no output")

        # Handle 4-D conv maps, 3-D token sequences, and 2-D features
        if feats.dim() == 4:
            feats = self.gap(feats).flatten(1)
        elif feats.dim() == 3:
            # token sequence: CLS token already extracted for cls_token tap
            if self.tap != "cls_token":
                feats = feats[:, 0]
        else:
            feats = feats.flatten(1)

        return self.cls(feats)


# --------------------------------------------------------------------------- #
# Evaluation utils
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    preds, targets, class_ids = [], [], []
    model.eval()
    for x, y, c in loader:
        logits = model(x.to(device))
        pred = logits.argmax(1).cpu()
        preds.extend(pred.tolist())
        targets.extend(y.tolist())
        class_ids.extend(int(v) for v in c)
    return preds, targets, class_ids


def _metrics(preds: list[int], targets: list[int], cls_ids: list[int]):
    arr_p = np.array(preds)
    arr_t = np.array(targets)
    out = {"overall_accuracy": float(accuracy_score(arr_t, arr_p))}

    per_class = defaultdict(lambda: {"tot": 0, "cor": 0})
    for cid, p, t in zip(cls_ids, arr_p, arr_t):
        d = per_class[cid]
        d["tot"] += 1
        if p == t:
            d["cor"] += 1
    out["class_metrics"] = [
        {"class_id": k, "accuracy": round(v["cor"]/v["tot"], 4)}
        for k, v in sorted(per_class.items(), key=lambda x: int(x[0]))
    ]
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("Probe evaluation (Dataset04, ViT-B/32)")
    parser.add_argument("--root", default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\40Dataset04_Bicubic",
                        help="Path to 40Dataset04_Bicubic")
    parser.add_argument("--backbone_ckpt",
                        default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ViT_B32_04\best_vit_b32.pth",
                        help="ViT-B/32 checkpoint used during probe training")
    parser.add_argument("--probes_dir",
                        default=r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\Model Record\ViT_B32_04\probes",
                        help="Folder with *.pth heads for the 16 probes")
    parser.add_argument("--output_dir", default="./probe_comparison_results_vitb32",
                        help="Where to dump confusion-matrices / reports / logs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tee_f = open(out_root / "stdout_log.txt", "w", encoding="utf-8")
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data)
        def flush(self):
            for f in self.files: f.flush()
    sys.stdout = _Tee(sys.stdout, tee_f)

    print("======== ViT-B/32 Linear-Probe Two-Split Evaluation ========")
    print("Timestamp :", datetime.now().isoformat(timespec='seconds'))
    print("Data root :", args.root)
    print("Backbone  :", args.backbone_ckpt)
    print("Probes dir:", args.probes_dir)
    print("Out dir   :", out_root.resolve(), "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device    :", device, "\n")

    # Backbone
    backbone = vit_b_32(weights=None)
    # Replace classification head with original 27 outputs
    if isinstance(backbone.heads, nn.Linear):
        in_feat = backbone.heads.in_features
    else:
        in_feat = next(m for m in backbone.heads.modules() if isinstance(m, nn.Linear)).in_features
    backbone.heads = nn.Linear(in_feat, 27)
    sd = torch.load(args.backbone_ckpt, map_location="cpu")
    backbone.load_state_dict(sd, strict=False)
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.to(device).eval()

    # Transforms & DataLoaders
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    csv_sup = Path(args.root) / "Newmetadata.csv"
    loaders = {
        "test": DataLoader(
            DatasetSourceProbe(csv_sup, "test", test_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        ),
        "nonsem": DataLoader(
            DatasetSourceProbe(csv_sup, "Non-Semantic Test", test_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )
    }

    # Define 16 probes
    probes = [
        ('probe1',  'conv_proj'),
        ('probe2',  'encoder.dropout'),
    ] + [
        (f'probe{idx+3}', f'encoder.layers.{idx}')
        for idx in range(12)
    ] + [
        ('probe15', 'encoder.ln'),
        ('probe16', 'cls_token'),
    ]

    # Header
    hdr = "{:8s} {:>8s} {:>8s} {:>8s}".format("Probe", "Test", "NonSem", "Δ")
    print(hdr)
    print("-" * len(hdr))

    # Evaluate each probe
    for pid, tap in probes:
        ckpt_path = Path(args.probes_dir) / f"{pid}.pth"
        if not ckpt_path.exists():
            print(f"{pid:<8}  [checkpoint missing]")
            continue

        # Infer feature dim
        holder = {}
        if tap == 'cls_token':
            handle = backbone.encoder.ln.register_forward_hook(lambda _, __, o: holder.setdefault('x', o[:, 0]))
            _ = backbone(torch.randn(1, 3, 224, 224, device=device))
            handle.remove()
            ft = holder['x']
            feat_dim = ft.size(1)
        else:
            handle = dict(backbone.named_modules())[tap].register_forward_hook(lambda _, __, o: holder.setdefault('x', o))
            _ = backbone(torch.randn(1, 3, 224, 224, device=device))
            handle.remove()
            ft = holder['x']
            if ft.dim() == 4:
                feat_dim = ft.size(1)
            elif ft.dim() == 3:
                feat_dim = ft.size(2)
            else:
                feat_dim = ft.numel() // ft.size(0)

        # Build probe & load head
        probe = LinearProbe(backbone, tap, feat_dim).to(device)
        probe.cls.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)

        # Evaluate
        preds_t, tgt_t, cid_t = _evaluate(probe, loaders['test'], device)
        preds_n, tgt_n, cid_n = _evaluate(probe, loaders['nonsem'], device)

        acc_t = accuracy_score(tgt_t, preds_t) * 100
        acc_n = accuracy_score(tgt_n, preds_n) * 100
        print(f"{pid:<8} {acc_t:8.2f} {acc_n:8.2f} {acc_t-acc_n:8.2f}")

        # Save artifacts
        p_out = out_root / pid
        p_out.mkdir(exist_ok=True)
        # Confusion matrices
        cm_t = confusion_matrix(tgt_t, preds_t, labels=[0,1])
        cm_n = confusion_matrix(tgt_n, preds_n, labels=[0,1])
        pd.DataFrame(cm_t, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"]).to_csv(p_out / "cm_test.csv")
        pd.DataFrame(cm_n, index=["CIFAR-100","TinyImageNet"], columns=["CIFAR-100","TinyImageNet"]).to_csv(p_out / "cm_nonsem.csv")

        # Classification reports
        with open(p_out / "classification_report_test.txt", "w") as f:
            f.write(classification_report(tgt_t, preds_t, target_names=["CIFAR-100","TinyImageNet"], digits=4))
        with open(p_out / "classification_report_nonsem.txt", "w") as f:
            f.write(classification_report(tgt_n, preds_n, target_names=["CIFAR-100","TinyImageNet"], digits=4))

        # Metrics JSON
        metrics = {
            "test": _metrics(preds_t, tgt_t, cid_t),
            "nonsem": _metrics(preds_n, tgt_n, cid_n),
            "delta_accuracy": acc_t - acc_n,
        }
        with open(p_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nFinished —", datetime.now().isoformat(timespec='seconds'))


if __name__ == '__main__':
    main()
