#!/usr/bin/env python3
"""
Build Dataset06 (Bicubic, 256×256) by merging Tiny-ImageNet-200 and
CIFAR-100 following the class list in **05dataset_new_label_list.csv**.

This script creates train/validation/test splits and writes a metadata.csv
with columns in the exact order:

  image_id,filepath,new_label_id,new_label,
  tiny_synset_id,tiny_synset_name,
  cifar_fine_id,cifar_fine_name,
  dataset_id,split

Each saved image is 256×256 using bicubic resampling. For Tiny-ImageNet,
images are first downsampled to 32×32 then upsampled to 256×256 (both
with bicubic).
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
# Configuration
# -------------------------------------------------------------------
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")
DEFAULT_MAP_FILE   = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\05dataset_new_label_list.csv")
OUTPUT_ROOT        = Path("60Dataset06_Bicubic")  # MODIFIED: new output folder for Dataset06
IMG_SIZE           = 256
SEED               = 42
# -------------------------------------------------------------------


def read_mapping(path: Path):
    df = pd.read_csv(path, encoding="latin1", engine="python")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    tiny2new = {syn: lbl for syn, lbl in zip(df.tiny_synset_id, df.new_label) if str(syn).upper() != "NON"}
    cifar2new = {int(fid): lbl for fid, lbl in zip(df.cifar_fine_id, df.new_label) if str(fid).upper() != "NON"}
    new2id = dict(zip(df.new_label, df.new_label_id))
    return df, tiny2new, cifar2new, new2id


def gather_tiny_images(tiny2new: dict):
    train_map = defaultdict(list)
    val_map = defaultdict(list)
    for syn_dir in (DEFAULT_TINY_ROOT / "train").iterdir():
        if syn_dir.is_dir() and syn_dir.name in tiny2new:
            lbl = tiny2new[syn_dir.name]
            train_map[lbl].extend((syn_dir / "images").glob("*.JPEG"))
    ann_file = DEFAULT_TINY_ROOT / "val" / "val_annotations.txt"
    if ann_file.exists():
        with open(ann_file, encoding="latin1") as fh:
            for line in fh:
                img, syn, *_ = line.strip().split("\t")
                if syn in tiny2new:
                    val_map[tiny2new[syn]].append(DEFAULT_TINY_ROOT / "val" / "images" / img)
    return train_map, val_map


def gather_cifar_images(cifar2new: dict):
    mapping = defaultdict(list)
    cifar_train = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False)
    cifar_test  = CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False)
    for idx, fid in enumerate(cifar_train.targets):
        if fid in cifar2new:
            mapping[cifar2new[fid]].append(("tr", idx))
    for idx, fid in enumerate(cifar_test.targets):
        if fid in cifar2new:
            mapping[cifar2new[fid]].append(("te", idx))
    return mapping


def resize_and_save(img: Image.Image, dest: Path):
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=95)


def process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id):
    random.seed(SEED)
    cifar_ds = {
        "tr": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=True, download=False),
        "te": CIFAR100(root=DEFAULT_CIFAR_ROOT, train=False, download=False),
    }
    # Create class directories
    for split in ["train", "validation", "test"]:
        for _, row in df.iterrows():
            (OUTPUT_ROOT / split / f"{row.new_label_id}_{row.new_label}").mkdir(parents=True, exist_ok=True)
    meta = []
    stats = {sp: defaultdict(lambda: {"total":0, "TinyImageNet":0, "CIFAR-100":0}) for sp in ["train","validation","test"]}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classes"):
        lbl = row.new_label
        lbl_id = row.new_label_id
        tiny_syn = row.tiny_synset_id
        cifar_fine = row.cifar_fine_id
        # Test split
        test_items = []
        for tag, idx in cifar_map.get(lbl, []):
            if tag == "te": test_items.append(((tag, idx), "CIFAR-100"))
        for p in tiny_val_map.get(lbl, []): test_items.append((p, "TinyImageNet"))
        for src, origin in test_items:
            # MODIFIED: down-up pipeline for TinyImageNet
            if origin == "TinyImageNet":
                img = Image.open(src).convert("RGB")
                img = img.resize((32, 32), Image.BICUBIC)  # downsample to 32×32
                fn = f"tiny_{src.stem}.jpg"
            else:
                tag, idx = src
                img = to_pil_image(cifar_ds[tag].data[idx]).convert("RGB")
                fn = f"cifar_te_{idx:05d}.jpg"
            dst = OUTPUT_ROOT / "test" / f"{lbl_id}_{lbl}" / fn
            resize_and_save(img, dst)
            meta.append({
                "image_id": dst.stem,
                "filepath": dst.as_posix(),
                "new_label_id": lbl_id,
                "new_label": lbl,
                "tiny_synset_id": tiny_syn,
                "tiny_synset_name": row.tiny_synset_name,
                "cifar_fine_id": cifar_fine,
                "cifar_fine_name": row.cifar_fine_name,
                "dataset_id": origin,
                "split": "test",
            })
            stats["test"][lbl]["total"] += 1
            stats["test"][lbl][origin] += 1
        # Train/Validation pool
        pool = []
        for tag, idx in cifar_map.get(lbl, []):
            if tag == "tr": pool.append(((tag, idx), "CIFAR-100"))
        for p in tiny_train_map.get(lbl, []): pool.append((p, "TinyImageNet"))
        random.shuffle(pool)
        n_val = int(len(pool)*0.1)
        val_items, train_items = pool[:n_val], pool[n_val:]
        def dump(sp, items):
            for src, origin in items:
                # MODIFIED: apply down-up only for TinyImageNet
                if origin == "TinyImageNet":
                    img = Image.open(src).convert("RGB")
                    img = img.resize((32, 32), Image.BICUBIC)  # downsample to 32×32
                    fn = f"tiny_{src.stem}.jpg"
                else:
                    tag, idx = src
                    img = to_pil_image(cifar_ds[tag].data[idx]).convert("RGB")
                    fn = f"cifar_{tag}_{idx:05d}.jpg"
                dst = OUTPUT_ROOT / sp / f"{lbl_id}_{lbl}" / fn
                resize_and_save(img, dst)
                meta.append({
                    "image_id": dst.stem,
                    "filepath": dst.as_posix(),
                    "new_label_id": lbl_id,
                    "new_label": lbl,
                    "tiny_synset_id": tiny_syn,
                    "tiny_synset_name": row.tiny_synset_name,
                    "cifar_fine_id": cifar_fine,
                    "cifar_fine_name": row.cifar_fine_name,
                    "dataset_id": origin,
                    "split": sp,
                })
                stats[sp][lbl]["total"] += 1
                stats[sp][lbl][origin] += 1
        dump("validation", val_items)
        dump("train",      train_items)

    # Write metadata.csv with fixed column order
    cols = ["image_id","filepath","new_label_id","new_label",
            "tiny_synset_id","tiny_synset_name",
            "cifar_fine_id","cifar_fine_name",
            "dataset_id","split"]
    pd.DataFrame(meta).to_csv(OUTPUT_ROOT / "metadata.csv", index=False, columns=cols)

    # Print and save stats summary
    lines = ["=== Split Statistics ==="]
    for sp in ["test","train","validation"]:
        lines.append(f"Split: {sp}")
        for lbl, s in stats[sp].items():
            lines.append(f"  {lbl} (id={new2id[lbl]}): total={s['total']}, TI={s['TinyImageNet']}, CIF={s['CIFAR-100']}")
    summary = "\n".join(lines)
    print(summary)
    (OUTPUT_ROOT / "60 Num of origin for each class.txt").write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    df, tiny2new, cifar2new, new2id = read_mapping(DEFAULT_MAP_FILE)
    tiny_train_map, tiny_val_map = gather_tiny_images(tiny2new)
    cifar_map = gather_cifar_images(cifar2new)
    process_and_save(df, tiny_train_map, tiny_val_map, cifar_map, new2id)
