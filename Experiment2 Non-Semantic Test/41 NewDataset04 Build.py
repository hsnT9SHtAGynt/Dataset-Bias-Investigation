#!/usr/bin/env python3
"""
This script builds a unified **256×256** image dataset (Dataset04, Bicubic)
by merging TinyImageNet and CIFAR-100 according to a provided mapping file.
It outputs separate train, validation, and test splits, along with a
`metadata.csv` file summarising each image's provenance and label information.

Pipeline details:
- **TinyImageNet** (original 64×64) → _resize_ **32×32** (BICUBIC) → _resize_ **256×256** (BICUBIC)
- **CIFAR-100**    (original 32×32)  → _resize_ **256×256** (BICUBIC)

_No random crop is performed any more; every saved image is exactly 256×256._

Data splits:
- **test**       = CIFAR-100 test + TinyImageNet validation
- **train/val**  = 90 % / 10 % random split of (CIFAR-100 train + TinyImageNet train)

The random seed is fixed so the split is reproducible.
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torchvision.transforms.functional import to_pil_image

# -------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------
# Root directories for sources and mapping CSV.
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")
DEFAULT_MAP_FILE = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\ChoosenO3.txt")

# Destination dataset root (this script will recreate it from scratch).
OUTPUT_ROOT = Path("40Dataset04_Bicubic")

# Deterministic shuffling for reproducible 90/10 split within each class.
SEED = 42

# Two-stage TinyImageNet resizing: simulate 32×32 footprint before upscaling to 256.
# Rationale: CIFAR images natively have 32×32 acquisition/preprocessing artifacts.
# Downsampling TinyImageNet to 32×32 first better aligns non-semantic footprints.
TINY_INTERMEDIATE_SIZE = 32  # downsample TinyImageNet first
IMG_SIZE = 256  # final square resolution for every saved image
# -------------------------------------------------------------------

def read_mapping(path: Path):
    """
    Read the chosen-class mapping CSV and build lookup dicts:
      - tiny2new: Tiny synset ID -> unified textual label
      - cifar2new: CIFAR fine ID -> unified textual label
      - new2id: unified textual label -> integer class ID
    Expected CSV columns include:
      tiny_synset_id, tiny_synset_name, cifar_fine_id, cifar_fine_name, new_label, new_label_id
    """
    df = pd.read_csv(path, sep=",", encoding="latin1", engine="python")
    # Normalize potential BOM/whitespace in headers to avoid KeyErrors.
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    tiny2new = dict(zip(df["tiny_synset_id"], df["new_label"]))
    cifar2new = dict(zip(df["cifar_fine_id"], df["new_label"]))
    new2id = dict(zip(df["new_label"], df["new_label_id"]))
    return df, tiny2new, cifar2new, new2id


def gather_tiny_images(tiny2new: dict):
    """
    Traverse TinyImageNet directories and collect absolute image paths grouped by unified label.

    Returns:
      tiny_train_map: new_label -> [TinyImageNet TRAIN image paths]
      tiny_val_map  : new_label -> [TinyImageNet VAL image paths]
    Note:
      - TRAIN images live under train/<synset>/images/*.JPEG
      - VAL images live under val/images with mapping in val_annotations.txt
    """
    tiny_train_map = defaultdict(list)
    tiny_val_map = defaultdict(list)

    train_dir = DEFAULT_TINY_ROOT / "train"
    for syn_dir in train_dir.iterdir():
        # Only process class folders that are mapped to the unified space
        if not syn_dir.is_dir() or syn_dir.name not in tiny2new:
            continue
        new_label = tiny2new[syn_dir.name]
        # Collect all JPEG files for this synset
        for img_path in (syn_dir / "images").glob("*.JPEG"):
            tiny_train_map[new_label].append(img_path)

    # Validation: annotations file provides (image_name, synset) pairs
    val_ann = DEFAULT_TINY_ROOT / "val" / "val_annotations.txt"
    if val_ann.exists():
        with open(val_ann, encoding="latin1") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, syn = parts[0], parts[1]
                if syn not in tiny2new:
                    continue  # skip classes not in mapping
                new_label = tiny2new[syn]
                tiny_val_map[new_label].append(DEFAULT_TINY_ROOT / "val" / "images" / img_name)
    return tiny_train_map, tiny_val_map


def gather_cifar_images(cifar2new: dict):
    """
    Index CIFAR-100 samples by unified label without loading image arrays.

    Returns:
      cifar_map: new_label -> list of tuples (split_tag, idx)
                 split_tag in {'tr','te'} for train/test respectively.
    """
    cifar_map = defaultdict(list)
    # Torchvision provides .targets with per-sample fine-label IDs.
    cifar_train = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False)
    cifar_test = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False)
    # Train indices for chosen classes
    for idx, fid in enumerate(cifar_train.targets):
        if fid in cifar2new:
            cifar_map[cifar2new[fid]].append(("tr", idx))
    # Test indices for chosen classes
    for idx, fid in enumerate(cifar_test.targets):
        if fid in cifar2new:
            cifar_map[cifar2new[fid]].append(("te", idx))
    return cifar_map


def resize_and_save(img: Image.Image, dest: Path, is_tiny: bool = False):
    """Resize `img` appropriately and save as high-quality JPEG
    Steps:
      - If TinyImageNet image: downsample to TINY_INTERMEDIATE_SIZE (32×32) first
        to match CIFAR’s nominal resolution footprint.
      - Upsample to IMG_SIZE (256×256) using bicubic interpolation.
      - Save as JPEG (quality=95). Create directories if needed.
    """
    # If TinyImageNet, first downsample to intermediate size (simulating CIFAR-like footprint)
    if is_tiny:
        img = img.resize((TINY_INTERMEDIATE_SIZE, TINY_INTERMEDIATE_SIZE), Image.BICUBIC)
    # Then upsample to final standardized resolution
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=95)


def process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id):
    """
    Main routine:
      - Pre-create output folders for each split/class
      - Materialize TEST from (CIFAR test + Tiny val)
      - Materialize TRAIN/VAL by 90/10 split of (CIFAR train + Tiny train)
      - Perform two-stage Tiny resize and single-stage CIFAR resize
      - Write metadata.csv and print per-class stats
    """
    random.seed(SEED)

    # Lazy CIFAR loaders so we can fetch numpy arrays at specific indices on demand
    cifar_ds = {
        "tr": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False),
        "te": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False),
    }

    # Prepare output directories: <OUTPUT_ROOT>/<split>/<id>_<label>/
    for split in ["train", "validation", "test"]:
        for _, row in df.iterrows():
            (OUTPUT_ROOT / split / f"{row['new_label_id']}_{row['new_label']}").mkdir(parents=True, exist_ok=True)

    # Accumulate rows for metadata.csv
    meta_records = []
    # stats[split][label] = {'total': int, 'TinyImageNet': int, 'CIFAR-100': int}
    stats = {s: {l: {"total": 0, "TinyImageNet": 0, "CIFAR-100": 0} for l in new2id} for s in ["train", "validation", "test"]}
    # Deterministic class order by numeric ID
    labels = [l for l, _ in sorted(new2id.items(), key=lambda x: x[1])]

    # Per-class loop with progress bar
    for new_label in tqdm(labels, desc="Processing classes"):
        # Look up display info for metadata rows
        row = df[df["new_label"] == new_label].iloc[0]
        nl_id = row["new_label_id"]
        tiny_id, tiny_name = row["tiny_synset_id"], row["tiny_synset_name"]
        cif_id, cif_name = row["cifar_fine_id"], row["cifar_fine_name"]

        # ---------------------- TEST SPLIT ----------------------
        # Test split composition follows the paper protocol:
        #   CIFAR-100 test  + TinyImageNet val
        test_items = []
        # CIFAR test indices for this label
        for tag, idx in cifar_map.get(new_label, []):
            if tag == "te":
                test_items.append(((tag, idx), "CIFAR-100"))
        # Tiny val images for this label
        for img_path in tiny_val_map.get(new_label, []):
            test_items.append((img_path, "TinyImageNet"))

        # Save all test items with appropriate resize rules
        for src, dsid in test_items:
            if dsid == "TinyImageNet":
                # Tiny val: load from disk, RGB, two-stage resize
                img = Image.open(src).convert("RGB")
                filename = f"tiny_{src.stem}.jpg"
                dest = OUTPUT_ROOT / "test" / f"{nl_id}_{new_label}" / filename
                resize_and_save(img, dest, is_tiny=True)
            else:
                # CIFAR test: fetch array by index, one-stage resize
                tag, idx = src
                np_img = cifar_ds[tag].data[idx]
                img = to_pil_image(np_img).convert("RGB")
                filename = f"cifar_te_{idx:05d}.jpg"
                dest = OUTPUT_ROOT / "test" / f"{nl_id}_{new_label}" / filename
                resize_and_save(img, dest)

            # Record metadata row
            meta_records.append({
                "image_id": dest.stem,
                "filepath": dest.as_posix(),
                "new_label_id": nl_id,
                "new_label": new_label,
                "tiny_synset_id": tiny_id,
                "tiny_synset_name": tiny_name,
                "cifar_fine_id": cif_id,
                "cifar_fine_name": cif_name,
                "dataset_id": dsid,
                "split": "test",
            })
            stats["test"][new_label]["total"] += 1
            stats["test"][new_label][dsid] += 1

        # ---------------- TRAIN + VALIDATION POOL ----------------
        # Pool consists of CIFAR-100 train + TinyImageNet train for this unified class
        pool_items = []
        for tag, idx in cifar_map.get(new_label, []):
            if tag == "tr":
                pool_items.append(((tag, idx), "CIFAR-100"))
        for img_path in tiny_train_map.get(new_label, []):
            pool_items.append((img_path, "TinyImageNet"))

        # Deterministic shuffle, then 10% → validation
        random.shuffle(pool_items)
        n_val = int(len(pool_items) * 0.1)
        val_items, tr_items = pool_items[:n_val], pool_items[n_val:]

        # Save validation and training items
        for split_tag, items in [("validation", val_items), ("train", tr_items)]:
            for src, dsid in items:
                if dsid == "TinyImageNet":
                    # Tiny train: two-stage resize path
                    img = Image.open(src).convert("RGB")
                    filename = f"tiny_{src.stem}.jpg"
                    dest = OUTPUT_ROOT / split_tag / f"{nl_id}_{new_label}" / filename
                    resize_and_save(img, dest, is_tiny=True)
                else:
                    # CIFAR train: one-stage resize path
                    tag, idx = src
                    np_img = cifar_ds[tag].data[idx]
                    img = to_pil_image(np_img).convert("RGB")
                    filename = f"cifar_tr_{idx:05d}.jpg"
                    dest = OUTPUT_ROOT / split_tag / f"{nl_id}_{new_label}" / filename
                    resize_and_save(img, dest)

                # Record metadata for each saved image
                meta_records.append({
                    "image_id": dest.stem,
                    "filepath": dest.as_posix(),
                    "new_label_id": nl_id,
                    "new_label": new_label,
                    "tiny_synset_id": tiny_id,
                    "tiny_synset_name": tiny_name,
                    "cifar_fine_id": cif_id,
                    "cifar_fine_name": cif_name,
                    "dataset_id": dsid,
                    "split": split_tag,
                })
                stats[split_tag][new_label]["total"] += 1
                stats[split_tag][new_label][dsid] += 1

    # ---------------------- WRITE METADATA ----------------------
    pd.DataFrame(meta_records).to_csv(OUTPUT_ROOT / "metadata.csv", index=False)

    # ---------------------- SUMMARY STATS ----------------------
    print("=== Split Statistics ===")
    for split in ["test", "validation", "train"]:
        print(f"Split: {split}")
        for lbl in labels:
            s = stats[split][lbl]
            print(
                f"  {lbl} (id={new2id[lbl]}): total={s['total']}, TI={s['TinyImageNet']}, CIF={s['CIFAR-100']}"
            )


if __name__ == "__main__":
    # Rebuild from a clean slate to avoid stale files from older mappings or runs.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    # 1) Parse mapping & build lookups
    df, tiny2new, cifar2new, new2id = read_mapping(DEFAULT_MAP_FILE)

    # 2) Collect TinyImageNet train/val image paths for mapped classes
    tiny_train_map, tiny_val_map = gather_tiny_images(tiny2new)

    # 3) Collect CIFAR-100 train/test indices for mapped classes (no pixels loaded yet)
    cifar_map = gather_cifar_images(cifar2new)

    # 4) Materialize splits, perform resize pipeline, save images + metadata
    process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id)
