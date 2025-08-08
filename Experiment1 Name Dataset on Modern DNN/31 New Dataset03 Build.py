#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script builds a unified **256×256** image dataset (Dataset03, Bicubic)
by merging TinyImageNet and CIFAR-100 according to a provided mapping file.
It outputs separate train, validation, and test splits, along with a
`metadata.csv` file summarising each image's provenance and label information.

Pipeline details:
- **TinyImageNet** (original 64×64)  → _resize_ **256×256** (BICUBIC)
- **CIFAR-100**    (original 32×32)  → _resize_ **256×256** (BICUBIC)

_No random crop is performed any more; every saved image is exactly 256×256._

Data splits:
- **test**       = CIFAR-100 test  + TinyImageNet validation
- **train/val**  = 90 % / 10 % random split of (CIFAR-100 train + TinyImageNet train)

The random seed is fixed so the split is reproducible.

Notes on outputs
----------------
Folder structure:
    Dataset03_Bicubic/
        train/
            <class_id>_<class_label>/
                *.jpg
        validation/
            <class_id>_<class_label>/
                *.jpg
        test/
            <class_id>_<class_label>/
                *.jpg
        metadata.csv

`metadata.csv` columns (one row per saved image):
    image_id, filepath, new_label_id, new_label,
    tiny_synset_id, tiny_synset_name, cifar_fine_id, cifar_fine_name,
    dataset_id, split
"""

# -------------------------------------------------------------------
# Standard library imports
# -------------------------------------------------------------------
import shutil                         # For recursively deleting an existing output directory
import random                         # For deterministic shuffling given a fixed seed
from pathlib import Path              # Path object for robust cross-platform file paths
from collections import defaultdict   # Dict that auto-creates list values for unseen keys

# -------------------------------------------------------------------
# Third-party imports
# -------------------------------------------------------------------
import pandas as pd                   # Tabular data handling + CSV I/O
from PIL import Image                 # PIL.Image for loading, converting, resizing, saving images
from tqdm import tqdm                 # Progress bar for per-class processing loop
from torchvision.datasets import CIFAR100                   # Official CIFAR-100 dataset interface
from torchvision.transforms.functional import to_pil_image  # Convert numpy arrays to PIL.Image

# -------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------
# Paths below are *defaults*; change if your local dataset layout differs.
# TinyImageNet directory must follow the standard structure:
#     tiny-imagenet-200/
#         train/<synset>/images/*.JPEG
#         val/images/*.JPEG
#         val/val_annotations.txt
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")

# CIFAR-100 is managed by torchvision; 'root' is where CIFAR-100 tar files / extracted data live.
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")

# Mapping file that aligns TinyImageNet synset IDs and CIFAR-100 fine-label IDs to a unified label space.
# Expected CSV columns (no header order dependence as long as names match):
#   tiny_synset_id, tiny_synset_name, cifar_fine_id, cifar_fine_name, new_label, new_label_id
DEFAULT_MAP_FILE = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\ChoosenO3.txt")

# Output directory where the merged dataset and metadata.csv will be created
OUTPUT_ROOT = Path("Dataset03_Bicubic")

# Global, fixed random seed -> ensures split reproducibility across runs
SEED = 42

# Final square resolution for every saved image (resized using bicubic interpolation)
IMG_SIZE = 256

# -------------------------------------------------------------------
# I/O + mapping utilities
# -------------------------------------------------------------------
def read_mapping(path: Path):
    """
    Read the mapping CSV that defines the unified label space.

    Parameters
    ----------
    path : Path
        Location of the CSV file that maps dataset-specific labels to a shared space.

    Returns
    -------
    df : pandas.DataFrame
        Entire mapping table, with any BOM/whitespace stripped from headers.
        Must contain columns:
            - "tiny_synset_id" (e.g., 'n01443537')
            - "tiny_synset_name" (human-readable name)
            - "cifar_fine_id" (int in [0..99])
            - "cifar_fine_name" (human-readable name)
            - "new_label" (unified textual label used for folder names)
            - "new_label_id" (unique int used as class ID)
    tiny2new : dict[str, str]
        Map TinyImageNet synset ID → unified textual label.
        Used to route TinyImageNet images to the correct unified class.
    cifar2new : dict[int, str]
        Map CIFAR-100 fine-label ID → unified textual label.
        Used to route CIFAR images to the correct unified class.
    new2id : dict[str, int]
        Map unified textual label → unified integer class ID.
        Used to name output folders and fill metadata.
    """
    # Read using a permissive engine and Latin-1 in case the CSV has mixed encodings.
    df = pd.read_csv(path, sep=",", encoding="latin1", engine="python")

    # Normalize column names to avoid subtle issues: strip whitespace and BOM
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    # Create the three essential lookups for later processing
    tiny2new = dict(zip(df["tiny_synset_id"], df["new_label"]))
    cifar2new = dict(zip(df["cifar_fine_id"], df["new_label"]))
    new2id = dict(zip(df["new_label"], df["new_label_id"]))
    return df, tiny2new, cifar2new, new2id


def gather_tiny_images(tiny2new: dict):
    """
    Scan TinyImageNet directories and collect absolute image paths grouped
    by the unified label (text).

    TinyImageNet split specifics
    ----------------------------
    - Train split:
        tiny-imagenet-200/train/<synset>/images/*.JPEG
      Each <synset> folder corresponds to one TinyImageNet class.

    - Validation split:
        tiny-imagenet-200/val/images/*.JPEG
        tiny-imagenet-200/val/val_annotations.txt
      The annotations file maps image filename → synset.

    Parameters
    ----------
    tiny2new : dict[str, str]
        Mapping from Tiny synset to unified textual label.

    Returns
    -------
    tiny_train_map : dict[str, list[Path]]
        new_label → list of absolute image paths for TinyImageNet **train** images.
    tiny_val_map : dict[str, list[Path]]
        new_label → list of absolute image paths for TinyImageNet **validation** images.
    """
    tiny_train_map = defaultdict(list)
    tiny_val_map = defaultdict(list)

    train_dir = DEFAULT_TINY_ROOT / "train"
    # Iterate through synset directories in TinyImageNet train
    for syn_dir in train_dir.iterdir():
        if not syn_dir.is_dir():
            # Skip non-directories (defensive)
            continue
        syn = syn_dir.name
        # Only keep synsets that are actually mapped to a unified label
        if syn not in tiny2new:
            continue
        new_label = tiny2new[syn]
        # Collect all JPEGs for this synset
        for img_path in (syn_dir / "images").glob("*.JPEG"):
            tiny_train_map[new_label].append(img_path)

    # Validation split: same folder, labels supplied by 'val_annotations.txt'
    val_ann = DEFAULT_TINY_ROOT / "val" / "val_annotations.txt"
    if val_ann.exists():
        with open(val_ann, encoding="latin1") as f:
            for line in f:
                # Format: "<image_name>\t<synset>\t<bbox...>" — we only need first two fields
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, syn = parts[0], parts[1]
                if syn not in tiny2new:
                    # Ignore unmapped synsets
                    continue
                new_label = tiny2new[syn]
                tiny_val_map[new_label].append(DEFAULT_TINY_ROOT / "val" / "images" / img_name)

    return tiny_train_map, tiny_val_map


def gather_cifar_images(cifar2new: dict):
    """
    Build a per-class list of CIFAR items, each annotated with the originating split.

    The function returns a mapping:
        new_label → list[(split_tag, idx)]
    where:
        split_tag ∈ {'tr', 'te'} for CIFAR train vs. test
        idx is the index into the corresponding torchvision CIFAR100 instance.

    We DO NOT load actual image arrays here (only indices) to keep memory usage low.

    Parameters
    ----------
    cifar2new : dict[int, str]
        CIFAR-100 fine-label ID → unified label mapping.

    Returns
    -------
    cifar_map : dict[str, list[tuple[str, int]]]
        Per unified class, a list of (split_tag, dataset_index) tuples.
    """
    cifar_map = defaultdict(list)

    # Instantiate CIFAR-100 datasets (download=False assumes data already present).
    cifar_train = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False)
    cifar_test = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False)

    # Enumerate CIFAR-100 TRAIN targets and keep indices that map to our unified space.
    for idx, fid in enumerate(cifar_train.targets):
        if fid in cifar2new:
            cifar_map[cifar2new[fid]].append(("tr", idx))

    # Enumerate CIFAR-100 TEST targets likewise.
    for idx, fid in enumerate(cifar_test.targets):
        if fid in cifar2new:
            cifar_map[cifar2new[fid]].append(("te", idx))

    return cifar_map


def resize_and_save(img: Image.Image, dest: Path):
    """
    Resize a PIL.Image to IMG_SIZE×IMG_SIZE using bicubic interpolation,
    create the destination directory if needed, and save as high-quality JPEG.

    Parameters
    ----------
    img : PIL.Image.Image
        The loaded image to be resized and saved.
    dest : Path
        Final path of the saved .jpg file (including filename).

    Notes
    -----
    - Bicubic is chosen to *standardize* low-level interpolation footprints across sources.
    - JPEG quality=95 balances fidelity and storage size.
    """
    # Force a deterministic output resolution to remove any downstream randomness.
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    # Ensure the parent directories exist (mkdir -p behavior)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Save as JPEG; extension normalization is handled by PIL based on format argument
    img.save(dest, "JPEG", quality=95)


def process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id):
    """
    Merge sources, apply split logic, resize & save images, and write metadata.

    High-level algorithm
    --------------------
    For each unified class (ordered by new_label_id):
      1) TEST set = CIFAR-100 test + TinyImageNet val
         - Load pixels (Tiny via disk, CIFAR via numpy array → PIL)
         - Resize to 256×256 (bicubic) and save under: test/<id>_<label>/
         - Write one metadata row per saved image
      2) TRAIN/VALIDATION pool = CIFAR-100 train + TinyImageNet train
         - Shuffle with fixed seed
         - First 10% → validation/<id>_<label>/ ; remaining 90% → train/<id>_<label>/
         - Resize/save + metadata rows

    Parameters
    ----------
    df : pandas.DataFrame
        Full mapping table (human-readable names and ids).
    tiny_train_map : dict[str, list[Path]]
        TinyImageNet training images grouped by unified label.
    tiny_val_map : dict[str, list[Path]]
        TinyImageNet validation images grouped by unified label.
    cifar_map : dict[str, list[tuple[str, int]]]
        CIFAR-100 images as (split_tag, index) grouped by unified label.
    new2id : dict[str, int]
        Unified label → class ID mapping.

    Side effects
    ------------
    - Creates folder trees under OUTPUT_ROOT for each split and class.
    - Writes resized .jpg images.
    - Emits `metadata.csv` with one row per saved image.
    - Prints per-class counts for sanity checking.
    """
    # Fix PRNG state so that shuffling and thus train/val split is reproducible.
    random.seed(SEED)

    # Lazily access CIFAR image arrays only when needed.
    cifar_ds = {
        "tr": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False),
        "te": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False),
    }

    # ------------------------------------------------------------------
    # Pre-create the directory tree:
    #   <OUTPUT_ROOT>/<split>/<class_id>_<class_label>/
    # Doing this up-front avoids repeated existence checks in the hot loop.
    # ------------------------------------------------------------------
    for split in ["train", "validation", "test"]:
        for _, row in df.iterrows():
            class_dir = OUTPUT_ROOT / split / f"{row['new_label_id']}_{row['new_label']}"
            class_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate metadata rows here, then write a single CSV at the end for speed.
    meta_records = []

    # Prepare a nested stats dict to track image counts per split/dataset/class.
    # stats[split][label] = {'total': int, 'TinyImageNet': int, 'CIFAR-100': int}
    stats = {
        s: {l: {"total": 0, "TinyImageNet": 0, "CIFAR-100": 0} for l in new2id}
        for s in ["train", "validation", "test"]
    }

    # Deterministic class ordering: increasing new_label_id
    labels = [l for l, _ in sorted(new2id.items(), key=lambda x: x[1])]

    # Progress bar for user feedback during potentially long per-class processing
    for new_label in tqdm(labels, desc="Processing classes"):
        # Lookup all human-readable + numeric info for this unified class
        row = df[df["new_label"] == new_label].iloc[0]
        nl_id = row["new_label_id"]
        tiny_id, tiny_name = row["tiny_synset_id"], row["tiny_synset_name"]
        cif_id, cif_name = row["cifar_fine_id"], row["cifar_fine_name"]

        # ------------------------------  TEST SPLIT  ------------------------------
        # TEST is formed by:
        #   - CIFAR-100 test images that map to this unified label
        #   - TinyImageNet validation images that map to this unified label
        test_items = []

        # Collect CIFAR test items ((split_tag, idx), "CIFAR-100")
        for tag, idx in cifar_map.get(new_label, []):
            if tag == "te":
                test_items.append(((tag, idx), "CIFAR-100"))

        # Collect Tiny val items (Path, "TinyImageNet")
        for img_path in tiny_val_map.get(new_label, []):
            test_items.append((img_path, "TinyImageNet"))

        # Process all test items; branch on dataset source to load pixels appropriately
        for src, dsid in test_items:
            if dsid == "TinyImageNet":
                # src is a concrete file path on disk
                img = Image.open(src).convert("RGB")
                # Include a prefix so filenames from different sources don't collide
                filename = f"tiny_{src.stem}.jpg"
            else:
                # dsid == "CIFAR-100": src is a (split_tag, index) tuple
                tag, idx = src
                # cifar_ds[tag].data[idx] -> numpy array (H, W, 3), uint8, RGB order
                np_img = cifar_ds[tag].data[idx]
                img = to_pil_image(np_img).convert("RGB")
                filename = f"cifar_te_{idx:05d}.jpg"

            # Save under: test/<id>_<label>/<filename>
            dest = OUTPUT_ROOT / "test" / f"{nl_id}_{new_label}" / filename
            resize_and_save(img, dest)

            # Append one row to metadata for this saved image
            meta_records.append({
                "image_id": dest.stem,                 # filename without extension
                "filepath": dest.as_posix(),           # POSIX-style path for portability
                "new_label_id": nl_id,
                "new_label": new_label,
                "tiny_synset_id": tiny_id,
                "tiny_synset_name": tiny_name,
                "cifar_fine_id": cif_id,
                "cifar_fine_name": cif_name,
                "dataset_id": dsid,                    # "TinyImageNet" or "CIFAR-100"
                "split": "test",
            })
            stats["test"][new_label]["total"] += 1
            stats["test"][new_label][dsid] += 1

        # -------------------  TRAIN + VALIDATION POOL  -------------------
        # Pool is the union of:
        #   - CIFAR-100 train items for this class
        #   - TinyImageNet train items for this class
        pool_items = []

        # Add CIFAR train items
        for tag, idx in cifar_map.get(new_label, []):
            if tag == "tr":
                pool_items.append(((tag, idx), "CIFAR-100"))

        # Add Tiny train items
        for img_path in tiny_train_map.get(new_label, []):
            pool_items.append((img_path, "TinyImageNet"))

        # Shuffle deterministically; then take 10% for validation
        random.shuffle(pool_items)
        n_val = int(len(pool_items) * 0.1)    # floor(0.1 * N) per-class
        val_items = pool_items[:n_val]
        tr_items = pool_items[n_val:]

        # ----- VALIDATION -----
        for src, dsid in val_items:
            if dsid == "TinyImageNet":
                img = Image.open(src).convert("RGB")
                filename = f"tiny_{src.stem}.jpg"
            else:
                tag, idx = src
                np_img = cifar_ds[tag].data[idx]
                img = to_pil_image(np_img).convert("RGB")
                filename = f"cifar_tr_{idx:05d}.jpg"

            dest = OUTPUT_ROOT / "validation" / f"{nl_id}_{new_label}" / filename
            resize_and_save(img, dest)

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
                "split": "validation",
            })
            stats["validation"][new_label]["total"] += 1
            stats["validation"][new_label][dsid] += 1

        # ----- TRAIN -----
        for src, dsid in tr_items:
            if dsid == "TinyImageNet":
                img = Image.open(src).convert("RGB")
                filename = f"tiny_{src.stem}.jpg"
            else:
                tag, idx = src
                np_img = cifar_ds[tag].data[idx]
                img = to_pil_image(np_img).convert("RGB")
                filename = f"cifar_tr_{idx:05d}.jpg"

            dest = OUTPUT_ROOT / "train" / f"{nl_id}_{new_label}" / filename
            resize_and_save(img, dest)

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
                "split": "train",
            })
            stats["train"][new_label]["total"] += 1
            stats["train"][new_label][dsid] += 1

    # ------------------------------------------------------------------
    # Persist metadata and show split statistics
    # ------------------------------------------------------------------
    pd.DataFrame(meta_records).to_csv(OUTPUT_ROOT / "metadata.csv", index=False)

    # Print a compact summary so you can eyeball class balance and source mix
    print("=== Split Statistics ===")
    for split in ["test", "validation", "train"]:
        print(f"Split: {split}")
        for lbl in labels:
            s = stats[split][lbl]
            print(
                f"  {lbl} (id={new2id[lbl]}): total={s['total']}, "
                f"TI={s['TinyImageNet']}, CIF={s['CIFAR-100']}"
            )


# ----------------------------------------------------------------------
# Entry-point guard — prevents accidental execution on import
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # If the output directory already exists, remove it to guarantee a clean rebuild.
    # This ensures old files (e.g., from a different mapping) don't linger.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    # --- Pipeline execution (strict order) ---
    # 1) Read mapping: establishes the unified label space and id lookups.
    df, tiny2new, cifar2new, new2id = read_mapping(DEFAULT_MAP_FILE)

    # 2) Scan TinyImageNet directories; collect paths for train/val images per unified class.
    tiny_train_map, tiny_val_map = gather_tiny_images(tiny2new)

    # 3) Index CIFAR-100 by (split, index) for each unified class, without loading pixel data yet.
    cifar_map = gather_cifar_images(cifar2new)

    # 4) Materialize the dataset: split, load, resize (bicubic to 256×256), save, and log metadata.
    process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id)
