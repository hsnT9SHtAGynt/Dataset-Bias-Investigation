"""nonsemantic_test_eval_1803.py

Purpose:
--------
Run inference using a fine-tuned ResNet-18 model (binary classification head:
CIFAR-100 vs TinyImageNet) on the **NonSemanticTest** split of Dataset03_Bicubic.

This script is meant for testing how well the model distinguishes between
datasets when semantic information is controlled (i.e., a non-semantic bias test).

Outputs:
--------
In `output-dir` (defaults to a sibling folder of this script):
    - metrics.json                  : Overall accuracy, per-dataset accuracy, and per-class accuracy
    - confusion_matrix.csv          : 2×2 confusion matrix with labelled rows and columns
    - classification_report.txt     : Detailed sklearn classification report (precision, recall, f1-score)
    - stdout_log.txt                 : Copy of all console output for provenance

Example usage (default paths):
    python nonsemantic_test_eval_1803.py

Override defaults with CLI args:
    python nonsemantic_test_eval_1803.py --data-root /path/to/NonSemanticTest --checkpoint model.pth
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Helper to map folder name → dataset label
# -----------------------------------------------------------------------------
def _infer_dataset_id(folder_name: str) -> int:
    """
    Infer dataset origin from folder name.
    Returns:
        0 if the folder name starts with a digit (CIFAR-100 style label),
        1 if it starts with a letter (TinyImageNet synset).
    """
    return 0 if folder_name[0].isdigit() else 1


# -----------------------------------------------------------------------------
# Custom Dataset for the NonSemanticTest split
# -----------------------------------------------------------------------------
class NonSemanticDataset(Dataset):
    """
    Dataset that yields:
        image        : Tensor (transformed RGB image)
        dataset_label: int (0=CIFAR-100, 1=TinyImageNet)
        class_name   : str (folder name, used for per-class accuracy)

    Directory structure assumed:
        NonSemanticTest/
            000_classname/   # CIFAR-100 style
            n01443537/       # TinyImageNet synset style
    """

    def __init__(self, root: str | Path, transform=None):
        self.samples: List[Tuple[Path, int, str]] = []
        self.transform = transform
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"NonSemanticTest root not found: {root}")

        # Walk 1-level deep: each folder = a class
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = _infer_dataset_id(class_dir.name)
            # Include common image extensions
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label, class_dir.name))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, class_name = self.samples[idx]
        image = Image.open(path).convert("RGB")  # Ensure 3-channel RGB
        if self.transform:
            image = self.transform(image)
        return image, label, class_name


# -----------------------------------------------------------------------------
# Metric computation helper
# -----------------------------------------------------------------------------
def _compute_metrics(preds: List[int], targets: List[int], class_names: List[str]) -> Dict:
    """
    Compute:
        - Overall accuracy
        - Per-dataset accuracy (CIFAR-100, TinyImageNet)
        - Per-class accuracy (with dataset info)
    """
    metrics: Dict = {}
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    # Overall accuracy
    metrics["overall_accuracy"] = float(accuracy_score(targets_arr, preds_arr))

    # Per-dataset accuracy
    dataset_breakdown: Dict[str, Dict[str, float | int]] = {}
    for ds_id, ds_name in [(0, "CIFAR-100"), (1, "TinyImageNet")]:
        mask = targets_arr == ds_id
        if mask.sum() == 0:
            continue
        acc = accuracy_score(targets_arr[mask], preds_arr[mask])
        dataset_breakdown[ds_name] = {"accuracy": float(acc), "support": int(mask.sum())}
    metrics["dataset_breakdown"] = dataset_breakdown

    # Per-class accuracy
    per_class: Dict[str, Dict] = {}
    for cl_name, p, t in zip(class_names, preds, targets):
        d = per_class.setdefault(
            cl_name,
            {"total": 0, "correct": 0, "dataset": "CIFAR-100" if t == 0 else "TinyImageNet"},
        )
        d["total"] += 1
        if p == t:
            d["correct"] += 1

    metrics["class_metrics"] = [
        {
            "class": k,
            "accuracy": round(v["correct"] / v["total"], 4),
            "dataset": v["dataset"],
        }
        for k, v in sorted(per_class.items())
    ]
    return metrics


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Run inference on the given dataloader.
    Returns:
        preds       : list of predicted dataset IDs
        targets     : list of ground-truth dataset IDs
        class_names : list of original class folder names
    """
    preds, targets, class_names = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels, cls_names in loader:
            images = images.to(device)
            out = model(images)
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
            class_names.extend(cls_names)
    return preds, targets, class_names


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Run Non-Semantic Test evaluation for ResNet-18 dataset-ID model.")
    parser.add_argument(
        "--data-root",
        default=r"D:\\DeepLearning\\Bias\\User Files\\ResNet18DatasetBias\\30Dataset03_Bicubic\\NonSemanticTest",
        help="Path to NonSemanticTest directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"D:\\DeepLearning\\Bias\\User Files\\ResNet18DatasetBias\\Model Record\\ResNet1803\\best_resnet18_datasetid_1803.pth",
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="./30Dataset03_Bicubic/nonsemantic_eval_results",
        help="Directory to store outputs",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    # Prepare output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Duplicate stdout to both console and log file
    log_path = out_dir / "stdout_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data): [f.write(data) for f in self.files]
        def flush(self): [f.flush() for f in self.files]
    import sys
    sys.stdout = _Tee(sys.stdout, log_file)

    # Logging configuration
    print("[INFO] Starting Non-Semantic Test evaluation")
    print(f"[INFO] Data root    : {args.data_root}")
    print(f"[INFO] Checkpoint   : {args.checkpoint}")
    print(f"[INFO] Output dir   : {out_dir.resolve()}")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device       : {device}")

    # Load model (ResNet-18 with binary classification head)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("[INFO] Checkpoint loaded.")

    # Image preprocessing (center crop, normalize)
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = NonSemanticDataset(args.data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[INFO] Loaded {len(dataset)} images.")

    # Inference
    preds, targets, class_names = _evaluate(model, loader, device)

    # Compute metrics
    cm = confusion_matrix(targets, preds)
    cr = classification_report(targets, preds, target_names=["CIFAR-100", "TinyImageNet"], digits=4)
    metrics = _compute_metrics(preds, targets, class_names)

    # Print metrics
    print("\n[Confusion Matrix]")
    print(cm)
    print("\n[Classification Report]\n")
    print(cr)
    print("\n[Metrics Summary]")
    print(json.dumps(metrics, indent=2))

    # Save outputs
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=["CIFAR-100", "TinyImageNet"], columns=["CIFAR-100", "TinyImageNet"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cr)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Results saved to {out_dir.resolve()}")
    log_file.close()


if __name__ == "__main__":
    main()
