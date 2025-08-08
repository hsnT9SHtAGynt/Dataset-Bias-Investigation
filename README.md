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

## License

This project is released under the MIT License.
