"""nonsemantic_test_eval_1804.py

Run inference with a fine-tuned ResNet-18 (binary head: CIFAR-100 vs TinyImageNet)
on the **NonSemanticTest** split of Dataset04_Bicubic.

Default paths are set for the local workstation, so the script can be executed
without additional CLI arguments:

    python nonsemantic_test_eval_1804.py

If you need to override anything (e.g., a different checkpoint), simply pass
`--data-root`, `--checkpoint`, or any other flag as usual.

Outputs in ``output-dir`` (defaults to a sibling folder of this script):
    - metrics.json                  JSON with overall / per-dataset / per-class metrics
    - confusion_matrix.csv          2×2 labelled confusion matrix
    - classification_report.txt     Full sklearn classification report
    - stdout_log.txt                Console log duplicate (useful for provenance)

Author: <your-name>
Created: 2025-06-21
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
# Helper: dataset ID inference
# -----------------------------------------------------------------------------
def _infer_dataset_id(folder_name: str) -> int:
    """
    Infer whether a given folder corresponds to CIFAR-100 or TinyImageNet
    based on its name.

    CIFAR-100 class folders are named with numeric prefixes (e.g., '000_cat'),
    whereas TinyImageNet folders follow synset-style names (non-numeric).

    Args:
        folder_name: Name of the folder containing the images.

    Returns:
        0 if folder is from CIFAR-100, 1 if from TinyImageNet.
    """
    return 0 if folder_name[0].isdigit() else 1


# -----------------------------------------------------------------------------
# Custom Dataset class for NonSemanticTest split
# -----------------------------------------------------------------------------
class NonSemanticDataset(Dataset):
    """
    A dataset that yields (image_tensor, dataset_label, class_name) tuples.

    dataset_label: 0 for CIFAR-100, 1 for TinyImageNet
    class_name:    name of the folder containing the image (used for per-class metrics)
    """

    def __init__(self, root: str | Path, transform=None):
        self.samples: List[Tuple[Path, int, str]] = []
        self.transform = transform
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"NonSemanticTest root not found: {root}")

        # Iterate through each class folder (only one level deep)
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = _infer_dataset_id(class_dir.name)  # 0 or 1 depending on dataset origin
            # Collect all supported image formats
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label, class_dir.name))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Load an image and return it with its dataset label and class name.
        """
        path, label, class_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, class_name


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------
def _compute_metrics(
    preds: List[int],
    targets: List[int],
    class_names: List[str],
) -> Dict:
    """
    Compute evaluation metrics at three levels:
        1. Overall accuracy
        2. Per-dataset accuracy (CIFAR-100 vs TinyImageNet)
        3. Per-class accuracy

    Args:
        preds: List of predicted dataset IDs (0 or 1).
        targets: List of ground truth dataset IDs.
        class_names: Class folder names corresponding to each sample.

    Returns:
        A dictionary with 'overall_accuracy', 'dataset_breakdown', and 'class_metrics'.
    """
    metrics: Dict = {}
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    # Overall accuracy across all samples
    metrics["overall_accuracy"] = float(accuracy_score(targets_arr, preds_arr))

    # Dataset-level accuracy (CIFAR vs TinyImageNet separately)
    dataset_breakdown: Dict[str, Dict[str, float | int]] = {}
    for ds_id, ds_name in [(0, "CIFAR-100"), (1, "TinyImageNet")]:
        mask = targets_arr == ds_id
        if mask.sum() == 0:
            continue
        acc = accuracy_score(targets_arr[mask], preds_arr[mask])
        dataset_breakdown[ds_name] = {"accuracy": float(acc), "support": int(mask.sum())}
    metrics["dataset_breakdown"] = dataset_breakdown

    # Per-class accuracy
    per_class: Dict[str, Dict[str, float | int | str]] = {}
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
    Run inference over the DataLoader and collect predictions, targets, and class names.

    Args:
        model: The trained PyTorch model.
        loader: DataLoader for evaluation data.
        device: torch.device to run the model on.

    Returns:
        preds: List of predicted labels (0 or 1).
        targets: List of ground truth labels.
        class_names: List of class folder names for each sample.
    """
    preds: List[int] = []
    targets: List[int] = []
    class_names: List[str] = []
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
# Main script
# -----------------------------------------------------------------------------
def main(argv: List[str] | None = None):
    # Argument parser for configurable paths and settings
    parser = argparse.ArgumentParser(
        description="Run Non-Semantic Test evaluation for ResNet-18 dataset-ID model."
    )
    parser.add_argument(
        "--data-root",
        default=r"D:\\DeepLearning\\Bias\\User Files\\ResNet18DatasetBias\\40Dataset04_Bicubic\\NonSemanticTest",
        help="Path to NonSemanticTest directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"D:\\DeepLearning\\Bias\\User Files\\ResNet18DatasetBias\\Model Record\\ResNet1804\\best_resnet18_datasetid_1804.pth",
        help="Path to fine tuned model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="./40Dataset04_Bicubic/nonsemantic_eval_results",
        help="Directory to store outputs",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    # Prepare output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log file setup: duplicate stdout to both console and file
    log_path = out_dir / "stdout_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    import sys
    sys.stdout = _Tee(sys.stdout, log_file)

    # Info header
    print("[INFO] Starting Non-Semantic Test evaluation")
    print(f"[INFO] Data root    : {args.data_root}")
    print(f"[INFO] Checkpoint   : {args.checkpoint}")
    print(f"[INFO] Output dir   : {out_dir.resolve()}")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device       : {device}")

    # Load model architecture and weights
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # Binary head: CIFAR-100 vs TinyImageNet
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("[INFO] Checkpoint loaded.")

    # Data transforms: match training preprocessing (center crop + normalization)
    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset and DataLoader
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

    # Compute evaluation metrics
    cm = confusion_matrix(targets, preds)
    cr = classification_report(
        targets, preds, target_names=["CIFAR-100", "TinyImageNet"], digits=4
    )
    metrics = _compute_metrics(preds, targets, class_names)

    # Print results to stdout/log
    print("\n[Confusion Matrix]")
    print(cm)
    print("\n[Classification Report]\n")
    print(cr)
    print("\n[Metrics Summary]")
    print(json.dumps(metrics, indent=2))

    # Save results to files
    import pandas as pd
    cm_df = pd.DataFrame(
        cm,
        index=["CIFAR-100", "TinyImageNet"],
        columns=["CIFAR-100", "TinyImageNet"],
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cr)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Results saved to {out_dir.resolve()}")

    log_file.close()


if __name__ == "__main__":
    main()
