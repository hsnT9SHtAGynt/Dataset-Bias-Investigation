#!/usr/bin/env python3
"""
Build **30Dataset03_Bicubic** including an additional **NonSemanticTest** split, a
combined **Newmetadata.csv** file, and a statistics report:

1. **train** / **validation** / **test** splits for chosen classes.
2. **NonSemanticTest**—TinyImageNet *validation* + CIFAR-100 *test* for classes **not** in `ChoosenO3.txt`,
   with per-class subfolders.
3. **Newmetadata.csv**—superset of metadata plus NonSemanticTest rows.
4. **30 Num of origin for each class.txt**—text file printing origin counts for each class in every split.

Pipeline:
---------
* TinyImageNet 64×64 → 256×256 (BICUBIC)
* CIFAR-100   32×32 → 256×256 (BICUBIC)

Random seed fixed for reproducibility.
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
# Config
# -------------------------------------------------------------------
# Root directory of TinyImageNet (standard folder structure expected).
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")
# Root directory where torchvision keeps CIFAR-100 files.
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")
# CSV file that enumerates the subset of classes ("chosen") to include in train/val/test.
# Expected columns include: tiny_synset_id, tiny_synset_name, cifar_fine_id, cifar_fine_name, new_label, new_label_id
DEFAULT_MAP_FILE = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\ChoosenO3.txt")
# Output dataset root; the script will (re)create this directory tree.
OUTPUT_ROOT = Path("30Dataset03_Bicubic")
# PRNG seed used for reproducible shuffling (train/val split).
SEED = 42
# Final image side length (pixels) after resizing with bicubic interpolation.
IMG_SIZE = 256
# -------------------------------------------------------------------


def read_mapping(path: Path):
    """
    Load the chosen-class mapping file and construct lookup tables.

    Parameters
    ----------
    path : Path
        Filesystem path to the mapping CSV (ChoosenO3.txt).

    Returns
    -------
    df : pandas.DataFrame
        Full mapping dataframe with normalized column names.
    tiny2new : Dict[str, str]
        Maps TinyImageNet synset ID -> unified class label used in output folders.
    cifar2new : Dict[int, str]
        Maps CIFAR-100 fine label ID (0..99) -> unified class label.
    new2id : Dict[str, int]
        Maps unified class label -> integer class ID (used in folder prefix).
    """
    # Read as CSV; latin1+python engine helps tolerate BOMs or odd encodings.
    df = pd.read_csv(path, sep=",", encoding="latin1", engine="python")
    # Normalize header names: remove leading/trailing spaces and stray BOMs.
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    # Build fast label mapping lookups for TinyImageNet and CIFAR-100.
    tiny2new = dict(zip(df["tiny_synset_id"], df["new_label"]))
    cifar2new = dict(zip(df["cifar_fine_id"], df["new_label"]))
    # Map unified label to stable integer ID (controls class ordering in output).
    new2id = dict(zip(df["new_label"], df["new_label_id"]))
    return df, tiny2new, cifar2new, new2id


def load_tiny_words(tiny_root: Path) -> Dict[str, str]:
    """
    Read TinyImageNet's 'words.txt' to map synset -> human-readable description.

    Used only for the NonSemanticTest folders (classes not present in ChoosenO3.txt),
    so that their folder names include a readable gloss when available.

    Parameters
    ----------
    tiny_root : Path
        Root of the TinyImageNet dataset.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping synset string (e.g., "n01641577") to textual description.
        Missing or unreadable files result in an empty mapping.
    """
    mapping = {}
    words_file = tiny_root / "words.txt"
    if words_file.exists():
        with open(words_file, encoding="latin1") as f:
            for line in f:
                # Each line: "<synset>\t<desc words...>"
                syn, *desc = line.strip().split("\t")
                mapping[syn] = " ".join(desc)
    return mapping


def gather_tiny_images(root: Path, tiny2new: Dict[str, str]):
    """
    Enumerate TinyImageNet training and validation images for the CHOSEN classes only.

    - Training images are found in:   train/<synset>/images/*.JPEG
    - Validation images are listed in: val/val_annotations.txt with image->synset

    Parameters
    ----------
    root : Path
        TinyImageNet root directory.
    tiny2new : Dict[str, str]
        Mapping from synset to unified label (only synsets present here are collected).

    Returns
    -------
    train_map : Dict[str, List[Path]]
        new_label -> list of absolute paths to TinyImageNet TRAIN images.
    val_map : Dict[str, List[Path]]
        new_label -> list of absolute paths to TinyImageNet VAL images.
    """
    train_map = defaultdict(list)
    val_map = defaultdict(list)
    # Iterate synset directories inside TinyImageNet/train
    for syn_dir in (root / "train").iterdir():
        if not syn_dir.is_dir():
            continue  # Skip stray files
        syn = syn_dir.name
        if syn not in tiny2new:
            # Ignore non-chosen synsets
            continue
        # Collect all JPEGs for this synset into the chosen unified label.
        for img in (syn_dir / "images").glob("*.JPEG"):
            train_map[tiny2new[syn]].append(img)
    # Validation images are centralized under 'val/images', with annotations linking to synset.
    val_ann = root / "val" / "val_annotations.txt"
    if val_ann.exists():
        with open(val_ann, encoding="latin1") as f:
            for line in f:
                # Format: "<image>\t<synset>\t<x1>\t<y1>\t<x2>\t<y2>\t<w>\t<h>" (we only need first two)
                name, syn, *_ = line.strip().split("\t")
                if syn not in tiny2new:
                    continue  # Skip non-chosen synsets
                val_map[tiny2new[syn]].append(root / "val" / "images" / name)
    return train_map, val_map


def gather_cifar_images(root: Path, cifar2new: Dict[int, str]):
    """
    Build CIFAR-100 index lists (train/test) for the CHOSEN classes only.

    This function avoids loading image arrays into memory; it records (split_tag, index)
    pairs so pixels can be loaded lazily downstream.

    Parameters
    ----------
    root : Path
        CIFAR-100 root directory (as used by torchvision).
    cifar2new : Dict[int, str]
        Fine-label ID -> unified label mapping (defines chosen classes).

    Returns
    -------
    cmap : Dict[str, List[Tuple[str, int]]]
        Mapping from unified label -> list of ("tr"|"te", dataset_index) tuples.
    """
    cmap = defaultdict(list)
    # Torchvision datasets provide .data (numpy arrays) and .targets (fine-label ids).
    tr_ds = CIFAR100(root=root, train=True, download=False)
    te_ds = CIFAR100(root=root, train=False, download=False)
    # Collect TRAIN indices for chosen fine-labels.
    for idx, fid in enumerate(tr_ds.targets):
        if fid in cifar2new:
            cmap[cifar2new[fid]].append(("tr", idx))
    # Collect TEST indices likewise.
    for idx, fid in enumerate(te_ds.targets):
        if fid in cifar2new:
            cmap[cifar2new[fid]].append(("te", idx))
    return cmap


def resize_and_save(img: Image.Image, dst: Path):
    """
    Resize an image to IMG_SIZE×IMG_SIZE using bicubic interpolation and save as JPEG.

    Parameters
    ----------
    img : PIL.Image.Image
        Loaded PIL image to be resized.
    dst : Path
        Destination path including filename (directories will be created as needed).

    Notes
    -----
    - JPEG quality=95 chosen as a good trade-off between size and fidelity.
    - Bicubic ensures consistent interpolation footprints across sources.
    """
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=95)


def process_chosen(
    df: pd.DataFrame,
    t_train: Dict[str, List[Path]],
    t_val: Dict[str, List[Path]],
    c_map: Dict[str, List[Tuple[str, int]]],
    new2id: Dict[str, int],
) -> Tuple[List[dict], Dict[str, Dict[str, int]], List[str]]:
    """
    Materialize the CHOSEN classes into train/validation/test splits, write images + metadata, and
    compute per-split statistics.

    Split definition for CHOSEN labels:
      - test: CIFAR-100 test + TinyImageNet validation (for chosen classes)
      - train/validation: 90/10 split of (CIFAR-100 train + TinyImageNet train) per class

    Parameters
    ----------
    df : pandas.DataFrame
        Mapping dataframe (provides human-readable names and IDs).
    t_train : Dict[str, List[Path]]
        TinyImageNet train images grouped by new_label.
    t_val : Dict[str, List[Path]]
        TinyImageNet val images grouped by new_label.
    c_map : Dict[str, List[Tuple[str, int]]]
        CIFAR items grouped by new_label as (split_tag, index).
    new2id : Dict[str, int]
        Unified label -> numeric ID.

    Returns
    -------
    records : List[dict]
        Metadata rows for all saved CHOSEN images (to be written into metadata.csv).
    stats : Dict[str, Dict[str, int]]
        Nested counters: stats[split][label] = {'total':..., 'TI':..., 'CIF':...}
    labels : List[str]
        Ordered list of unified labels (sorted by new_label_id).
    """
    random.seed(SEED)
    # Lazily load CIFAR pixel arrays only at save time (memory-friendly).
    cifar_ds = {
        "tr": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False),
        "te": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False),
    }
    # Define the three canonical splits.
    splits = ["train", "validation", "test"]
    # Stable label ordering by numeric class id ensures deterministic folder layout.
    labels = [l for l, _ in sorted(new2id.items(), key=lambda x: x[1])]
    # Pre-allocate stats dict: counts of total/TI/CIF per label and split.
    # Shorthand keys 'TI' and 'CIF' are used here for compactness in the text report.
    stats = {sp: {lbl: {'total': 0, 'TI': 0, 'CIF': 0} for lbl in labels} for sp in splits}
    # Ensure output directories exist: <OUTPUT_ROOT>/<split>/<id>_<label>/
    for sp in splits:
        for lbl in labels:
            (OUTPUT_ROOT / sp / f"{new2id[lbl]}_{lbl}").mkdir(parents=True, exist_ok=True)
    # Accumulate metadata dicts, then write one CSV at the end for speed.
    records = []

    # Iterate chosen labels with a progress bar.
    for lbl in tqdm(labels, desc="Chosen classes"):
        # Row for this label: includes Tiny/CIFAR IDs and names for metadata fields.
        row = df[df["new_label"] == lbl].iloc[0]
        nl_id, t_id, t_name, c_id, c_name = (
            row["new_label_id"], row["tiny_synset_id"], row["tiny_synset_name"],
            row["cifar_fine_id"], row["cifar_fine_name"],
        )
        # --------------------- TEST SPLIT ---------------------
        # Combine CIFAR test indices and Tiny val files mapped to this label.
        items = [((tag, idx), "CIFAR-100") for tag, idx in c_map.get(lbl, []) if tag == "te"]
        items += [(p, "TinyImageNet") for p in t_val.get(lbl, [])]
        for src, ds in items:
            if ds == "TinyImageNet":
                # Load Tiny image from disk; enforce RGB.
                img = Image.open(src).convert("RGB")
                fn = f"tiny_{src.stem}.jpg"
            else:
                # CIFAR image is a numpy array; convert to PIL then RGB.
                tag, idx = src  # type: ignore
                img = to_pil_image(cifar_ds[tag].data[idx]).convert("RGB")
                fn = f"cifar_te_{idx:05d}.jpg"
            # Save under test/<id>_<label>/
            dst = OUTPUT_ROOT / "test" / f"{nl_id}_{lbl}" / fn
            resize_and_save(img, dst)
            # Update aggregate stats for this label.
            stats['test'][lbl]['total'] += 1
            stats['test'][lbl]['TI' if ds == 'TinyImageNet' else 'CIF'] += 1
            # Append one metadata row for this saved image.
            records.append({
                'image_id': dst.stem, 'filepath': dst.as_posix(),
                'new_label_id': nl_id, 'new_label': lbl,
                'tiny_synset_id': t_id, 'tiny_synset_name': t_name,
                'cifar_fine_id': c_id, 'cifar_fine_name': c_name,
                'dataset_id': ds, 'split': 'test',
            })
        # ---------------- TRAIN + VALIDATION ----------------
        # Pool = CIFAR train + Tiny train for this label.
        pool = [((tag, idx), 'CIFAR-100') for tag, idx in c_map.get(lbl, []) if tag == 'tr']
        pool += [(p, 'TinyImageNet') for p in t_train.get(lbl, [])]
        # Deterministic shuffle prior to 10% validation split selection.
        random.shuffle(pool)
        n_val = int(len(pool) * 0.1)  # floor of 10%
        # First 10% -> validation ; remainder -> train
        for split, subset in [('validation', pool[:n_val]), ('train', pool[n_val:])]:
            for src, ds in subset:
                if ds == "TinyImageNet":
                    img = Image.open(src).convert("RGB")
                    fn = f"tiny_{src.stem}.jpg"
                else:
                    tag, idx = src  # type: ignore
                    img = to_pil_image(cifar_ds[tag].data[idx]).convert("RGB")
                    fn = f"cifar_tr_{idx:05d}.jpg"
                dst = OUTPUT_ROOT / split / f"{nl_id}_{lbl}" / fn
                resize_and_save(img, dst)
                stats[split][lbl]['total'] += 1
                stats[split][lbl]['TI' if ds == 'TinyImageNet' else 'CIF'] += 1
                records.append({
                    'image_id': dst.stem, 'filepath': dst.as_posix(),
                    'new_label_id': nl_id, 'new_label': lbl,
                    'tiny_synset_id': t_id, 'tiny_synset_name': t_name,
                    'cifar_fine_id': c_id, 'cifar_fine_name': c_name,
                    'dataset_id': ds, 'split': split,
                })
    # Persist the CHOSEN-only metadata as 'metadata.csv' (baseline metadata).
    pd.DataFrame(records).to_csv(OUTPUT_ROOT / 'metadata.csv', index=False)
    return records, stats, labels


def build_nonsemantic(
    tiny_root: Path,
    cifar_root: Path,
    tiny2new: Dict[str, str],
    cifar2new: Dict[int, str],
    tiny_words: Dict[str, str],
) -> Tuple[List[dict], Dict[str, Dict[str, int]]]:
    """
    Construct the NonSemanticTest split by harvesting images from classes NOT present
    in the chosen mapping (i.e., complementary to ChoosenO3.txt).

    Composition:
      - TinyImageNet validation images whose synsets are NOT in tiny2new
      - CIFAR-100 test images whose fine-label IDs are NOT in cifar2new

    Output layout:
      <OUTPUT_ROOT>/NonSemanticTest/<id_and_name>/image_files...
        * For Tiny: folder name "<syn>_<words-desc with underscores>"
        * For CIFAR: folder name "<fid>_<class_name>"

    Returns
    -------
    ns_records : List[dict]
        Metadata rows for each NonSemanticTest image (new_label fields intentionally blank).
    ns_stats : Dict[str, Dict[str, int]]
        Aggregate counts per NonSemanticTest folder: {'total', 'TI', 'CIF'}.
    """
    ns_records = []
    # ns_stats keys are folder identifiers (e.g., "n016..._some_desc" or "23_bear").
    ns_stats = defaultdict(lambda: {'total': 0, 'TI': 0, 'CIF': 0})
    base = OUTPUT_ROOT / 'NonSemanticTest'

    # ------------------ TinyImageNet validation (non-chosen synsets) ------------------
    val_ann = tiny_root / 'val' / 'val_annotations.txt'
    if val_ann.exists():
        with open(val_ann, encoding='latin1') as f:
            for line in f:
                # Each line associates a val image with its synset; ignore bbox fields.
                name, syn, *_ = line.strip().split('\t')
                if syn in tiny2new: continue  # Skip synsets that are chosen (we only want non-chosen)
                src = tiny_root / 'val' / 'images' / name
                # Folder uses synset plus a readable gloss (underscored for filesystem safety).
                folder = f"{syn}_{tiny_words.get(syn,'').replace(' ','_')}"
                dst_dir = base / folder
                img = Image.open(src).convert('RGB')
                fn = f"tiny_{name}.jpg"
                resize_and_save(img, dst_dir / fn)
                # Update stats for this NonSemanticTest class bucket.
                ns_stats[folder]['total'] += 1
                ns_stats[folder]['TI'] += 1
                # Record metadata row; chosen label fields remain empty by design.
                ns_records.append({
                    'image_id': (dst_dir / fn).stem, 'filepath': (dst_dir / fn).as_posix(),
                    'new_label_id': '', 'new_label': '',
                    'tiny_synset_id': syn, 'tiny_synset_name': tiny_words.get(syn,''),
                    'cifar_fine_id': '', 'cifar_fine_name': '',
                    'dataset_id': 'TinyImageNet', 'split': 'Non-Semantic Test'
                })

    # ------------------ CIFAR-100 test (non-chosen fine labels) ------------------
    cifar_test = CIFAR100(root=cifar_root, train=False, download=False)
    for idx, fid in enumerate(cifar_test.targets):
        if fid in cifar2new: continue  # Skip chosen labels; only collect the complement
        cname = cifar_test.classes[fid]
        folder = f"{fid}_{cname}"
        dst_dir = base / folder
        img = to_pil_image(cifar_test.data[idx]).convert('RGB')
        fn = f"cifar_te_{idx:05d}.jpg"
        resize_and_save(img, dst_dir / fn)
        ns_stats[folder]['total'] += 1
        ns_stats[folder]['CIF'] += 1
        ns_records.append({
            'image_id': (dst_dir / fn).stem, 'filepath': (dst_dir / fn).as_posix(),
            'new_label_id': '', 'new_label': '',
            'tiny_synset_id': '', 'tiny_synset_name': '',
            'cifar_fine_id': fid, 'cifar_fine_name': cname,
            'dataset_id': 'CIFAR-100', 'split': 'Non-Semantic Test'
        })

    # Write a combined metadata file that includes CHOSEN (pre-collected) + NonSemanticTest.
    # NOTE: 'chosen_meta' must exist in the outer scope when this function is called.
    pd.DataFrame(chosen_meta + ns_records).to_csv(OUTPUT_ROOT / 'Newmetadata.csv', index=False)
    return ns_records, ns_stats


def write_stats(report_path: Path, stats: Dict[str, Dict[str, Dict[str, int]]],
                ns_stats: Dict[str, Dict[str, int]], chosen_labels: List[str], new2id: Dict[str, int]):
    """
    Emit a human-readable text report summarizing image counts per class, per split.

    File format:
      === Split Statistics ===
      Split: test
        <label> (id=<id>): total=<>, TI=<>, CIF=<>
      Split: train
        ...
      Split: validation
        ...
      Split: Non-Semantic Test
        <ns_folder>: total=<>, TI=<>, CIF=<>

    Parameters
    ----------
    report_path : Path
        Output text file path ("30 Num of origin for each class.txt").
    stats : Dict[str, Dict[str, Dict[str, int]]]
        CHOSEN splits aggregate counts: stats[split][label] -> {'total','TI','CIF'}.
    ns_stats : Dict[str, Dict[str, int]]
        NonSemanticTest aggregate counts: ns_stats[folder] -> {'total','TI','CIF'}.
    chosen_labels : List[str]
        Ordered unified labels (for deterministic report ordering).
    new2id : Dict[str, int]
        Unified label -> class ID, for annotating lines in the report.
    """
    with open(report_path, 'w') as f:
        f.write('=== Split Statistics ===\n')
        # Use a consistent split order for readability.
        for sp in ['test', 'train', 'validation']:
            f.write(f'Split: {sp}\n')
            for lbl in chosen_labels:
                st = stats[sp][lbl]
                f.write(f"  {lbl} (id={new2id[lbl]}): total={st['total']}, TI={st['TI']}, CIF={st['CIF']}\n")
        # NonSemanticTest section—summarize each folder (class) present there.
        f.write('Split: Non-Semantic Test\n')
        for folder, st in ns_stats.items():
            f.write(f"  {folder}: total={st['total']}, TI={st['TI']}, CIF={st['CIF']}\n")


if __name__ == '__main__':
    # ---------------------- CLEAN START ----------------------
    # Remove the existing output tree (if any) to avoid stale files and ensure a fresh rebuild.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    # ---------------------- LOAD MAPPINGS ----------------------
    # df: full chosen-class mapping; tiny2new/cifar2new/new2id: lookups into the unified label space.
    df, tiny2new, cifar2new, new2id = read_mapping(DEFAULT_MAP_FILE)

    # Load synset -> textual gloss for TinyImageNet (used to name NonSemanticTest folders for Tiny).
    tiny_words = load_tiny_words(DEFAULT_TINY_ROOT)

    # ---------------------- ENUMERATE SOURCES ----------------------
    # For CHOSEN classes only: collect Tiny train/val images and CIFAR train/test indices.
    t_train, t_val = gather_tiny_images(DEFAULT_TINY_ROOT, tiny2new)
    c_map = gather_cifar_images(DEFAULT_CIFAR_ROOT, cifar2new)

    # ---------------------- BUILD CHOSEN SPLITS ----------------------
    # Create train/validation/test for the CHOSEN classes, save images, and write baseline metadata.csv.
    chosen_meta, stats, labels = process_chosen(df, t_train, t_val, c_map, new2id)

    # ---------------------- BUILD NON-SEMANTIC TEST ----------------------
    # Harvest the COMPLEMENT sets (non-chosen) from Tiny val + CIFAR test, save images, and create combined metadata.
    ns_records, ns_stats = build_nonsemantic(
        DEFAULT_TINY_ROOT, DEFAULT_CIFAR_ROOT, tiny2new, cifar2new, tiny_words
    )

    # ---------------------- WRITE COMBINED METADATA ----------------------
    # Ensure the unified 'Newmetadata.csv' includes both chosen and non-semantic entries.
    pd.DataFrame(chosen_meta + ns_records).to_csv(OUTPUT_ROOT / 'Newmetadata.csv', index=False)

    # ---------------------- WRITE STATS REPORT ----------------------
    # Human-readable per-class/source breakdown for all splits, plus NonSemanticTest.
    write_stats(OUTPUT_ROOT / '30 Num of origin for each class.txt', stats, ns_stats, labels, new2id)

    # Final status line for console users.
    print("30Dataset03_Bicubic rebuilt with NonSemanticTest & stats report.")
