#!/usr/bin/env python3
"""
Build **40Dataset04_Bicubic** including:
    - An extra **NonSemanticTest** split (contains classes NOT chosen for training/testing)
    - A combined **Newmetadata.csv** containing all four splits (train, val, test, NonSemanticTest)
    - Per-class and per-origin statistics file: `40 Num of origin for each class.txt`

Processing logic follows Dataset03’s structure but uses updated bicubic resizing.

---------------------------------------------------------------------
PREPROCESSING PIPELINE:
---------------------------------------------------------------------
TinyImageNet images:
    1. Original resolution: 64×64
    2. Downsample to 32×32 (bicubic) — removes some high-frequency artifacts
    3. Upsample to final 256×256 (bicubic)

CIFAR-100 images:
    1. Original resolution: 32×32
    2. Upsample directly to 256×256 (bicubic)

---------------------------------------------------------------------
SPLIT DEFINITIONS:
---------------------------------------------------------------------
1. **train** / **validation** / **test**:
    - Classes listed in `ChoosenO3.txt` mapping file
    - Train/validation: random 90% / 10% split of the merged Tiny train + CIFAR train
    - Test: Tiny val + CIFAR test (same classes)

2. **NonSemanticTest**:
    - All excluded classes (i.e., not in `ChoosenO3.txt`)
    - From Tiny val + CIFAR test
    - Saved in per-class folders (id_name format)

---------------------------------------------------------------------
OUTPUT FILES:
---------------------------------------------------------------------
- `metadata.csv`         → only the three main splits
- `Newmetadata.csv`      → includes NonSemanticTest entries
- `40 Num of origin for each class.txt` → per-split origin counts
"""

from __future__ import annotations
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torchvision.transforms.functional import to_pil_image

# -------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -------------------------------------------------------------------
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")
DEFAULT_MAP_FILE = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\ChoosenO3.txt")
OUTPUT_ROOT = Path("40Dataset04_Bicubic")
SEED = 42  # ensures reproducible shuffling of train/val split

# Resize parameters
TINY_INTERMEDIATE = 32   # TinyImageNet is first downsampled to this size
FINAL_SIZE = 256         # final resolution for all saved images
# -------------------------------------------------------------------


def read_mapping(path: Path):
    """
    Reads the mapping file linking TinyImageNet synsets and CIFAR-100 fine IDs
    to unified new labels.

    Returns:
        df       → full mapping DataFrame
        tiny2new → dict: tiny_synset_id → new_label
        cifar2new→ dict: cifar_fine_id  → new_label
        new2id   → dict: new_label      → new_label_id
    """
    df = pd.read_csv(path, sep=",", encoding="latin1", engine="python")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)  # clean BOM/whitespace
    tiny2new = dict(zip(df["tiny_synset_id"], df["new_label"]))
    cifar2new = dict(zip(df["cifar_fine_id"], df["new_label"]))
    new2id = dict(zip(df["new_label"], df["new_label_id"]))
    return df, tiny2new, cifar2new, new2id


def load_tiny_words(root: Path) -> Dict[str, str]:
    """
    Loads TinyImageNet's 'words.txt' mapping: synset → human-readable description.
    Used for naming NonSemanticTest folders.
    """
    mapping = {}
    f = root / "words.txt"
    if f.exists():
        with open(f, encoding="latin1") as fin:
            for line in fin:
                syn, *desc = line.strip().split("\t")
                mapping[syn] = " ".join(desc)
    return mapping


def gather_tiny(root: Path, tiny2new: Dict[str, str]):
    """
    Gathers TinyImageNet images for the chosen classes.

    Returns:
        train_map → new_label → [Path to image in train split]
        val_map   → new_label → [Path to image in validation split]
    """
    train_map = defaultdict(list)
    val_map = defaultdict(list)

    # Training set (original TinyImageNet split)
    for syn_dir in (root / "train").iterdir():
        if not syn_dir.is_dir():
            continue
        syn = syn_dir.name
        if syn not in tiny2new:
            continue  # skip non-selected classes
        for img in (syn_dir / "images").glob("*.JPEG"):
            train_map[tiny2new[syn]].append(img)

    # Validation set
    ann = root / "val" / "val_annotations.txt"
    if ann.exists():
        with open(ann, encoding="latin1") as fin:
            for line in fin:
                name, syn, *_ = line.strip().split("\t")
                if syn not in tiny2new:
                    continue
                val_map[tiny2new[syn]].append(root / "val" / "images" / name)

    return train_map, val_map


def gather_cifar(root: Path, cifar2new: Dict[int, str]):
    """
    Gathers CIFAR-100 images for the chosen classes.

    Returns:
        cmap → new_label → list of tuples (tag, idx)
               tag = 'tr' for train, 'te' for test
               idx = image index in that split
    """
    cmap = defaultdict(list)
    tr = CIFAR100(root=root, train=True, download=False)
    te = CIFAR100(root=root, train=False, download=False)

    for idx, fid in enumerate(tr.targets):
        if fid in cifar2new:
            cmap[cifar2new[fid]].append(("tr", idx))
    for idx, fid in enumerate(te.targets):
        if fid in cifar2new:
            cmap[cifar2new[fid]].append(("te", idx))
    return cmap


def preprocess_resize(img: Image.Image, is_tiny: bool) -> Image.Image:
    """
    Applies the dataset-specific resizing procedure.
    TinyImageNet: downsample to TINY_INTERMEDIATE, then upsample to FINAL_SIZE
    CIFAR-100: upsample directly to FINAL_SIZE
    """
    if is_tiny:
        img = img.resize((TINY_INTERMEDIATE, TINY_INTERMEDIATE), Image.BICUBIC)
    return img.resize((FINAL_SIZE, FINAL_SIZE), Image.BICUBIC)


def save_img(img: Image.Image, dst: Path):
    """Saves the image as high-quality JPEG, creating parent dirs if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=95)


def process_splits(df, t_train, t_val, c_map, new2id):
    """
    Builds the train, validation, and test splits for the selected classes.

    Returns:
        meta   → list of metadata dicts (rows for metadata.csv)
        stats  → dict of per-split per-class counts
        labels → list of new_labels in ID order
    """
    random.seed(SEED)
    cifar_ds = {
        "tr": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False),
        "te": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False)
    }
    splits = ["train", "validation", "test"]
    labels = [l for l, _ in sorted(new2id.items(), key=lambda x: x[1])]
    stats = {sp: {lbl: {'total': 0, 'TI': 0, 'CIF': 0} for lbl in labels} for sp in splits}

    # Create output folders for all splits/classes
    for sp in splits:
        for lbl in labels:
            (OUTPUT_ROOT / sp / f"{new2id[lbl]}_{lbl}").mkdir(parents=True, exist_ok=True)

    meta = []

    for lbl in tqdm(labels, desc="Chosen classes"):
        row = df[df['new_label'] == lbl].iloc[0]
        nl, tid, tname, cid, cname = (
            row['new_label_id'], row['tiny_synset_id'], row['tiny_synset_name'],
            row['cifar_fine_id'], row['cifar_fine_name']
        )

        # 1. TEST split
        items = [((tag, idx), 'CIFAR-100') for tag, idx in c_map.get(lbl, []) if tag == 'te']
        items += [(p, 'TinyImageNet') for p in t_val.get(lbl, [])]
        for src, ds in items:
            if ds == 'TinyImageNet':
                img = Image.open(src).convert('RGB')
                fn = f"tiny_{src.stem}.jpg"
                proc = preprocess_resize(img, True)
            else:
                tag, idx = src
                img = to_pil_image(cifar_ds[tag].data[idx]).convert('RGB')
                fn = f"cifar_te_{idx:05d}.jpg"
                proc = preprocess_resize(img, False)
            dst = OUTPUT_ROOT / 'test' / f"{nl}_{lbl}" / fn
            save_img(proc, dst)
            stats['test'][lbl]['total'] += 1
            stats['test'][lbl]['TI' if ds == 'TinyImageNet' else 'CIF'] += 1
            meta.append({
                'image_id': dst.stem, 'filepath': dst.as_posix(),
                'new_label_id': nl, 'new_label': lbl,
                'tiny_synset_id': tid, 'tiny_synset_name': tname,
                'cifar_fine_id': cid, 'cifar_fine_name': cname,
                'dataset_id': ds, 'split': 'test'
            })

        # 2. TRAIN/VALIDATION split (90/10 from pool)
        pool = [((tag, idx), 'CIFAR-100') for tag, idx in c_map.get(lbl, []) if tag == 'tr']
        pool += [(p, 'TinyImageNet') for p in t_train.get(lbl, [])]
        random.shuffle(pool)
        n_val = int(len(pool) * 0.1)
        for sp, sub in [('validation', pool[:n_val]), ('train', pool[n_val:])]:
            for src, ds in sub:
                if ds == 'TinyImageNet':
                    img = Image.open(src).convert('RGB')
                    fn = f"tiny_{src.stem}.jpg"
                    proc = preprocess_resize(img, True)
                else:
                    tag, idx = src
                    img = to_pil_image(cifar_ds[tag].data[idx]).convert('RGB')
                    fn = f"cifar_tr_{idx:05d}.jpg"
                    proc = preprocess_resize(img, False)
                dst = OUTPUT_ROOT / sp / f"{nl}_{lbl}" / fn
                save_img(proc, dst)
                stats[sp][lbl]['total'] += 1
                stats[sp][lbl]['TI' if ds == 'TinyImageNet' else 'CIF'] += 1
                meta.append({
                    'image_id': dst.stem, 'filepath': dst.as_posix(),
                    'new_label_id': nl, 'new_label': lbl,
                    'tiny_synset_id': tid, 'tiny_synset_name': tname,
                    'cifar_fine_id': cid, 'cifar_fine_name': cname,
                    'dataset_id': ds, 'split': sp
                })

    pd.DataFrame(meta).to_csv(OUTPUT_ROOT / 'metadata.csv', index=False)
    return meta, stats, labels


def build_nonsemantic(tiny_root, cifar_root, tiny2new, cifar2new, tiny_words):
    """
    Builds the NonSemanticTest split:
    - Tiny val images from classes NOT in chosen mapping
    - CIFAR test images from classes NOT in chosen mapping
    Returns metadata list and per-class stats.
    """
    ns_meta = []
    ns_stats = defaultdict(lambda: {'total': 0, 'TI': 0, 'CIF': 0})
    base = OUTPUT_ROOT / 'NonSemanticTest'

    # Tiny val excluded classes
    ann = tiny_root / 'val' / 'val_annotations.txt'
    if ann.exists():
        with open(ann, encoding='latin1') as fin:
            for line in fin:
                name, syn, *_ = line.strip().split('\t')
                if syn in tiny2new:
                    continue
                src = tiny_root / 'val' / 'images' / name
                folder = f"{syn}_{tiny_words.get(syn, '').replace(' ', '_')}"
                dst_dir = base / folder
                img = Image.open(src).convert('RGB')
                proc = preprocess_resize(img, True)
                fn = f"tiny_{name}.jpg"
                save_img(proc, dst_dir / fn)
                ns_stats[folder]['total'] += 1
                ns_stats[folder]['TI'] += 1
                ns_meta.append({
                    'image_id': (dst_dir / fn).stem, 'filepath': (dst_dir / fn).as_posix(),
                    'new_label_id': '', 'new_label': '',
                    'tiny_synset_id': syn, 'tiny_synset_name': tiny_words.get(syn, ''),
                    'cifar_fine_id': '', 'cifar_fine_name': '',
                    'dataset_id': 'TinyImageNet', 'split': 'Non-Semantic Test'
                })

    # CIFAR test excluded classes
    cifar_test = CIFAR100(root=cifar_root, train=False, download=False)
    for idx, fid in enumerate(cifar_test.targets):
        if fid in cifar2new:
            continue
        cname = cifar_test.classes[fid]
        folder = f"{fid}_{cname}"
        dst_dir = base / folder
        img = to_pil_image(cifar_test.data[idx]).convert('RGB')
        proc = preprocess_resize(img, False)
        fn = f"cifar_te_{idx:05d}.jpg"
        save_img(proc, dst_dir / fn)
        ns_stats[folder]['total'] += 1
        ns_stats[folder]['CIF'] += 1
        ns_meta.append({
            'image_id': (dst_dir / fn).stem, 'filepath': (dst_dir / fn).as_posix(),
            'new_label_id': '', 'new_label': '',
            'tiny_synset_id': '', 'tiny_synset_name': '',
            'cifar_fine_id': fid, 'cifar_fine_name': cname,
            'dataset_id': 'CIFAR-100', 'split': 'Non-Semantic Test'
        })

    return ns_meta, ns_stats


def write_stats(path, stats, ns_stats, labels, new2id):
    """
    Writes per-split and per-class origin counts to a text file.
    """
    with open(path, 'w') as f:
        f.write('=== Split Statistics ===\n')
        for sp in ['test', 'train', 'validation']:
            f.write(f'Split: {sp}\n')
            for lbl in labels:
                st = stats[sp][lbl]
                f.write(f"  {lbl} (id={new2id[lbl]}): total={st['total']}, TI={st['TI']}, CIF={st['CIF']}\n")
        f.write('Split: Non-Semantic Test\n')
        for folder, st in ns_stats.items():
            f.write(f"  {folder}: total={st['total']}, TI={st['TI']}, CIF={st['CIF']}\n")


if __name__ == '__main__':
    # Clean output dir if exists
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    # Step 1: Load mapping and Tiny words
    df, tiny2new, cifar2new, new2id = read_mapping(DEFAULT_MAP_FILE)
    tiny_words = load_tiny_words(DEFAULT_TINY_ROOT)

    # Step 2: Gather paths/indexes for chosen classes
    t_train, t_val = gather_tiny(DEFAULT_TINY_ROOT, tiny2new)
    c_map = gather_cifar(DEFAULT_CIFAR_ROOT, cifar2new)

    # Step 3: Build main splits
    chosen_meta, stats, labels = process_splits(df, t_train, t_val, c_map, new2id)

    # Step 4: Build NonSemanticTest split
    ns_meta, ns_stats = build_nonsemantic(
        DEFAULT_TINY_ROOT, DEFAULT_CIFAR_ROOT, tiny2new, cifar2new, tiny_words
    )

    # Step 5: Save metadata files
    pd.DataFrame(chosen_meta).to_csv(OUTPUT_ROOT / 'metadata.csv', index=False)
    pd.DataFrame(chosen_meta + ns_meta).to_csv(OUTPUT_ROOT / 'Newmetadata.csv', index=False)

    # Step 6: Save stats
    write_stats(OUTPUT_ROOT / '40 Num of origin for each class.txt', stats, ns_stats, labels, new2id)

    print("40Dataset04_Bicubic built with NonSemanticTest & stats.")
