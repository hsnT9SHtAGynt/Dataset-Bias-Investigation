# Dataset-Bias-Investigation

Investigating dataset bias via **Name the Dataset** (Torralba & Efros, *CVPR 2011*) and **Name the Label**, decomposing it into **semantic** (generalizable and transferable) and **non-semantic** components. Proposes two DANN variants, improving semantic accuracy by ~2% (**ResNet-18**) and ~0.5% (**ViT-B/32**).

> **Note:** Reported values are from individual runs — due to stochastic training variability, results of accuracy can fluctuate by about ±1%. Fortunately, in this setting, the non-semantic component dominates.

---

## Repository Description

This repository contains the code accompanying the MSc dissertation:  
**"Name the Dataset or Name the Label: Dataset Bias Investigation"**.

It includes **six experiments**, each stored in a separate folder.

The experiments involve **four dataset variants** (**Dataset 03 – Dataset 06**), all created by merging **TinyImageNet** and **CIFAR-100** with different preprocessing pipelines. Full construction details are available in **Appendix A** of the dissertation.

---

## Dataset Variants (Summary)

| Feature | Dataset03 | Dataset04 | Dataset05 | Dataset06 |
|---------|-----------|-----------|-----------|-----------|
| **Preprocessing** (TinyImageNet) | 64×64 → 256×256 (Bicubic) → 224×224 (Random Crop) | 64×64 → 32×32 (Bicubic) → 256×256 (Bicubic) → 224×224 (Random Crop) | Same as Dataset03 | Same as Dataset04 |
| **Preprocessing** (CIFAR-100) | 32×32 → 256×256 (Bicubic) → 224×224 (Random Crop) | Same as Dataset03 | Same as Dataset03 | Same as Dataset03 |
| **Training Classes** | 27 overlapping classes | 27 overlapping classes | 273 total: 27 overlapping, 73 CIFAR-only, 173 Tiny-only | Same as Dataset05 |
| **Image Format** | JPEG (quality=95) | Same as Dataset03 | Same as Dataset03 | Same as Dataset03 |
| **Test Split** | Normal test (overlap) + Non-semantic test (non-overlap) | Same as Dataset03 | Three groups: overlap, CIFAR-only, Tiny-only | Same as Dataset05 |
| **Train/Val Split** | 90%/10% per class | Same as Dataset03 | 90%/10% per group | Same as Dataset05 |
| **Key Characteristics** | Pure overlap, single preprocessing, unified test | Overlap + extra Tiny downscaling, unified test | Mixed groups, separate tests, full utilization | Mixed groups + extra Tiny downscaling, full utilization |

> **Note:** The 224×224 random crop is applied dynamically before each training epoch; images are stored at 256×256.

---

## Models Used

- **ResNet-18**
- **ResNet-50**
- **ViT-B/32**

---

## Code Naming Convention

All `.py` scripts follow the format:

<sort_id> <model_id><dataset_id> <task_name>.py


**Example:**
32 1803 Name Label.py


Where:

- **sort_id** (`32`) — for sorting in PyCharm; also implicitly indicates dataset (3) and task type (2)
- **model_id** (`18`) — model type (`ResNet-18`, `50` for ResNet-50, `VITB32` for ViT-B/32)
- **dataset_id** (`03`) — dataset variant
- **task_name** (`Name Label`) — the experimental task

---

## How to Use

> **Default project root**
> `D:\DeepLearning\Bias\User Files\ResNet18DatasetBias`

### 1) Mapping file

Ensure `data\ChoosenO3.txt` exists.
This file defines the semantic overlap mapping between **TinyImageNet** synsets and **CIFAR-100** fine labels and is required to construct **Dataset03–Dataset06**.

### 2) Prepare raw datasets

Download and place the raw datasets under the project root:

* **TinyImageNet**:
  Unzip to `data\tiny-imagenet-200\` so that folders like
  `data\tiny-imagenet-200\train\`, `data\tiny-imagenet-200\val\images\` exist.

* **CIFAR-100**:
  If your scripts set `download=False`, place CIFAR data under `data\` as expected by `torchvision.datasets.CIFAR100`.
  (If you prefer auto-download, set `download=True` where `CIFAR100(...)` is called.)

Double-check in each dataset-builder script that these constants match your paths:

```python
DEFAULT_TINY_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\tiny-imagenet-200")  # TinyImageNet root directory
DEFAULT_CIFAR_ROOT = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data")                   # CIFAR-100 root directory
DEFAULT_MAP_FILE = Path(r"D:\DeepLearning\Bias\User Files\ResNet18DatasetBias\data\ChoosenO3.txt")       # Mapping between datasets
OUTPUT_ROOT = Path("Dataset03_Bicubic")      # Destination folder for the merged dataset
```

### 3) Build the datasets (03–06)

Run the corresponding builder scripts to materialize images at **256×256** and write a `metadata.csv` per dataset directory.
(Each dataset variant follows the preprocessing summarized above; 224×224 random crop is applied **at training time**, not when saving.)

After each build, verify that:

* `DatasetXX_*/train|validation|test/<id>_<label>/...` exists
* `DatasetXX_*/metadata.csv` is present

### 4) Run experiments

Sort the `.py` files **by filename (ascending)** and execute them **top-to-bottom**.
The naming convention is:

```
<sort_id> <model_id><dataset_id> <task_name>.py
# e.g., 32 1803 Name Label.py
```

Before running each script:

* Confirm all dataset paths are correct for the chosen Dataset**03–06**.
* (Optional) If your training scripts support logging, keep per-epoch metrics (e.g., `metrics.csv`) and final test results (e.g., `test_metrics.csv`) for later analysis.

### 5) Reproducibility

* Dataset builders fix a global RNG seed for deterministic train/val splits.
* Due to stochastic training, final accuracy typically varies by about **±1%** per run; consider averaging multiple runs for stable numbers.

## License

This project is released under the MIT License.
